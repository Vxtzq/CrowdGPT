"""Staleness bite (ROCm only): naive averaging vs two staleness corrections.

Worker 0's contribution is always `STALE` rounds old. We compare what the server
does with that stale pseudo-gradient. HeLoCo's point (arXiv 2606.00271) is that a
uniform negative-momentum extrapolation (MLA, Ajanthan et al. 2025) is too coarse
because tensor blocks align differently with the outer trajectory, so it corrects
per block. Two data regimes: IID (all workers see the whole corpus) and non-IID
(worker k sees a disjoint slice) -- the regime HeLoCo says is worst.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device("cuda")
K = 4
INNER = int(os.environ.get("SB_INNER", 25))
STALE = int(os.environ.get("SB_STALE", 2))
ARM = os.environ.get("SB_ARM", "naive")          # naive | mla | block
NONIID = os.environ.get("SB_NONIID", "0") == "1"
TARGET = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 2_000_000
TRAIN, VAL = "/tmp/v15data/tok/train.bin", "/tmp/v15data/tok/val.bin"
train = np.memmap(TRAIN, dtype=np.uint16, mode="r")
val = np.memmap(VAL, dtype=np.uint16, mode="r")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D); self.up = nn.Linear(D, 4 * D); self.down = nn.Linear(4 * D, D)

    def forward(self, x):
        b, t, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(b, t, H, D // H).transpose(1, 2)
        k = k.view(b, t, H, D // H).transpose(1, 2)
        v = v.view(b, t, H, D // H).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + self.down(torch.nn.functional.silu(self.up(self.ln2(x))))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight
        with torch.no_grad():
            self.wte.weight.normal_(0.0, 0.02)

    def forward(self, idx):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def span_for(worker):
    """Data slice: whole corpus when IID, a disjoint band when non-IID."""
    if not NONIID:
        return 0, train.size - T - 2
    band = (train.size - T - 2) // K
    return worker * band, (worker + 1) * band


def batch(seed, worker):
    lo, hi = span_for(worker)
    g = torch.Generator().manual_seed(seed * 1_000_003 + worker * 7919)
    off = (torch.randint(0, hi - lo, (B,), generator=g) + lo).numpy()
    x = np.stack([train[o:o + T] for o in off]).astype(np.int64)
    y = np.stack([train[o + 1:o + T + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


EVAL = []
for i in range(8):
    o = i * (T + 1)
    EVAL.append((torch.from_numpy(np.stack([val[o:o + T]]).astype(np.int64)).to(DEV),
                 torch.from_numpy(np.stack([val[o + 1:o + T + 1]]).astype(np.int64)).to(DEV)))


def ce(logits, y):
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, V), y[:, :-1].reshape(-1))


def evaluate(model):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for x, y in EVAL:
            tot += float(ce(model(x), y))
    model.train(); return tot / len(EVAL)


rounds = max(1, TARGET // (INNER * K * B * T))
torch.manual_seed(0)
globe = Tiny().to(DEV)
opt_o = torch.optim.SGD(globe.parameters(), lr=0.7, momentum=0.9, nesterov=False)
workers = [Tiny().to(DEV) for _ in range(K)]
inners = [torch.optim.AdamW(w.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01) for w in workers]

curve = [evaluate(globe)]
history = []          # worker 0 states, one per round
prev_grad = None      # last accepted pseudo-gradient, for the momentum extrapolation
t0 = time.perf_counter()

for r in range(rounds):
    for w, opt, wk in zip(workers, inners, range(K)):
        w.load_state_dict(globe.state_dict())
        for _ in range(INNER):
            x, y = batch(r * INNER + _, wk)
            opt.zero_grad(set_to_none=True)
            ce(w(x), y).backward()
            torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
            opt.step()

    stale_state = history[-STALE] if (STALE > 0 and len(history) >= STALE) else None
    grads, deltas = {}, {}
    with torch.no_grad():
        for name, p in globe.named_parameters():
            acc = torch.zeros_like(p)
            for wi, w in enumerate(workers):
                src = stale_state if (wi == 0 and stale_state is not None) else w.state_dict()
                acc += src[name].float()
            mean = acc / K
            g = (p.float() - mean).reshape(-1)          # DiLoCo pseudo-gradient
            deltas[name] = mean - p.float()             # what the stale worker would have moved
            grads[name] = g

        # ---- correction ------------------------------------------------------
        if ARM == "mla" and prev_grad is not None:
            # uniform negative-momentum extrapolation over the whole vector (MLA)
            for name in grads:
                grads[name] = grads[name] + 0.9 * (grads[name] - prev_grad[name])
        elif ARM == "block" and prev_grad is not None:
            # per-block: extrapolate only along each tensor's alignment with the
            # outer momentum, which is HeLoCo's objection to a uniform correction.
            for name in grads:
                g, pv = grads[name], prev_grad[name]
                dot = float((g * pv).sum())
                nrm = float(pv.norm()) ** 2 + 1e-12
                align = max(0.0, dot / nrm)             # 0 when opposed, 1 when aligned
                grads[name] = g + align * 0.9 * (g - pv)

        flat_prev = {n: g.detach().clone() for n, g in grads.items()}
        for name, p in globe.named_parameters():
            p.grad = grads[name].reshape(p.shape).to(p.dtype)
        opt_o.step()
        prev_grad = flat_prev

    if STALE > 0:
        history.append({k: v.detach().clone() for k, v in workers[0].state_dict().items()})
    if (r + 1) % max(1, rounds // 5) == 0:
        curve.append(evaluate(globe))

tok = rounds * INNER * K * B * T
dt = time.perf_counter() - t0
loss = evaluate(globe)
print(f"arm={ARM:5s} noniid={int(NONIID)} stale={STALE} inner={INNER} rounds={rounds} "
      f"tokens={tok:,} curve={['%.4f' % c for c in curve]} held-out {loss:.4f} "
      f"{tok/dt:9,.0f} tok/s {dt:6.1f}s", flush=True)
