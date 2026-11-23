"""Is a CrowdGPT round's delta low-rank?  AdaRankGrad (2410.17881) says gradients
become near rank-1 as training proceeds; if that holds for our uploads, a
low-rank factorisation beats top-k sparsity (no index cost, error spread over
every coordinate).  Measure it on real deltas instead of assuming it.

Compares, at MATCHED BYTES, three ways to ship one 512x512 bf16 matrix:
  dense  = 2 bytes/param
  sparse = top 2% of |entries|, 6 bytes/entry (uint32 index + bf16 value)
  lowrank= r*(d0+d1) bf16 factors
and reports the relative Frobenius error each recovery leaves behind.
"""
import sys, json
sys.path.insert(0, "/app/crowdgpt/CrowdGPT")
sys.path.insert(0, "/app/crowdgpt/CrowdGPT/experiments/v2")
import torch
import scale_306m as S

S.D, S.L, S.H = 512, 12, 8
S.B = 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S.DEV = DEV
model = S.build().to(DEV)
print("params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

import crowd_mosaic as M
cfg = M.MosaicConfig(slices=2, local_steps=5, outer_lr=0.15, outer_momentum=0.9)
desc = M.attach(model, cfg)
splits = desc["slices"]

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
state = None
opt_state = {}
last = None
for r in range(ROUNDS):
    current = {n: p.detach().clone() for n, p in model.named_parameters()}
    deltas = []
    for k in range(2):
        x, y = S.batch(r * 10 + k)
        deltas.append(M.local_delta(model, slices=splits, k=k, x=x, y=y, steps=5,
                                    lr=1e-3, opt_state=opt_state, share_trunk=True))
    new, state = M.aggregate(current, deltas, cfg=cfg, state=state)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in new:
                p.copy_(new[n])
    last = deltas

# merged view: what the server would apply for the round
merged = {}
for d in last:
    for n, v in d.items():
        merged[n] = merged.get(n, 0) + v / len(last)

# a backprop delta over the same number of local steps, for contrast
bp = S.build().to(DEV)
opt = torch.optim.AdamW(bp.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
before = {n: p.detach().clone() for n, p in bp.named_parameters()}
for s in range(5):
    x, y = S.batch(900 + s)
    opt.zero_grad(set_to_none=True)
    S.ce(bp(x), y).backward()
    torch.nn.utils.clip_grad_norm_(bp.parameters(), 1.0)
    opt.step()
bpd = {n: p.detach() - before[n] for n, p in bp.named_parameters()}

def analyse(tag, delta):
    rows = []
    for n, d in delta.items():
        if d.ndim < 2:
            continue
        D = d.detach().float().reshape(d.shape[0], -1)
        if min(D.shape) < 8:
            continue
        s = torch.linalg.svdvals(D)
        e = (s ** 2).sum()
        if float(e) <= 0:
            continue
        frac = (s ** 2).cumsum(0) / e
        get = lambda k: float(frac[min(k, len(frac)) - 1])
        n_el = D.numel()
        dense_b = n_el * 2
        # sparse: top 2% of entries by magnitude, 6 bytes each (uint32 + bf16)
        k = max(1, int(0.02 * n_el))
        flat = D.reshape(-1)
        thr = torch.topk(flat.abs(), k).values.min()
        kept = torch.where(flat.abs() >= thr, flat, torch.zeros_like(flat))
        sp_err = float((flat - kept).pow(2).sum() / flat.pow(2).sum()) ** 0.5
        sparse_b = k * 6
        # low rank at the SAME byte budget
        r_match = max(1, sparse_b // (2 * (D.shape[0] + D.shape[1])))
        lr_err = (max(0.0, 1.0 - get(r_match))) ** 0.5
        rows.append(dict(name=n, fro=float(torch.sqrt(e)), r1=get(1), r4=get(4),
                         r8=get(8), r16=get(16), rankMatch=r_match,
                         sparseErr=sp_err, lowrankErr=lr_err,
                         denseB=dense_b, sparseB=sparse_b,
                         rank8B=8 * (D.shape[0] + D.shape[1]) * 2))
    if not rows:
        return None
    tot = sum(r["fro"] ** 2 for r in rows)
    w = lambda key: sum(r[key] * r["fro"] ** 2 for r in rows) / tot
    print(f"--- {tag}: {len(rows)} matrices, fro={sum(r['fro'] for r in rows):.3f}")
    print(f"    energy@1 {w('r1'):.3f}  @4 {w('r4'):.3f}  @8 {w('r8'):.3f}  @16 {w('r16'):.3f}")
    print(f"    at equal bytes (sparse 2% = {sum(r['sparseB'] for r in rows)/1e6:.1f}MB, "
          f"rank ~{int(w('rankMatch'))}): sparse err {w('sparseErr'):.3f}  "
          f"lowrank err {w('lowrankErr'):.3f}")
    print(f"    rank8: {sum(r['rank8B'] for r in rows)/1e6:.1f}MB, err "
          f"{(1-w('r8'))**0.5:.3f}  |  dense {sum(r['denseB'] for r in rows)/1e6:.1f}MB")
    return rows

out = {"rounds": ROUNDS}
out["mosaic"] = analyse("MOSAIC round (merged deltas)", merged)
out["backprop"] = analyse("backprop 5 AdamW steps", bpd)
json.dump({"rounds": ROUNDS, "mosaic": out["mosaic"], "backprop": out["backprop"]},
          open("/tmp/opencode/rank/rank.json", "w"), indent=1)
print("wrote /tmp/opencode/rank/rank.json")
