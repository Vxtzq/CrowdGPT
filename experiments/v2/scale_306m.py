"""Does the tiny-model paradigm result survive at 306 M?

``paradigm_e2e.py`` compares the five paradigms on a 1 M-parameter toy with 40
rounds. That answers "is the wiring right", not "does this converge at scale".
This harness takes the 306 M preset from ``diloco_306m.py`` (V=4096, d=1024,
24 layers, 16 heads, B*T=2048 tokens per step) and runs two arms to the *same
token budget* on the same held-out windows:

* ``backprop`` — the stock path: one AdamW over the whole model, every step.
* ``mosaic``   — two slices, H local steps per slice per round, median clip,
  one outer step. Same code path the client uses (``crowd_mosaic``), including
  the SOLO head on the non-terminal slice.

Equal tokens is the honest comparison: MOSAIC spends more steps for the same
data (2 slices x H), which is exactly its cost. Wall clock is reported so the
price is visible next to the loss.

1.01 B is deliberately not attempted here: 1.2 GB of weights plus 2.4 GB of
Adam moments plus activations does not leave room for evaluation copies on a
16 GB card, and the point is the paradigm, not the size.

    python experiments/v2/scale_306m.py --paradigm backprop --tokens 2000000
    python experiments/v2/scale_306m.py --paradigm both --tokens 2000000 --json out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import crowd_mosaic as mosaic_mod  # noqa: E402

V, D, L, H, T, B = 4096, 1024, 24, 16, 512, 4
DEV = torch.device(os.environ.get("SOLO_DEV") or ("cuda" if torch.cuda.is_available() else "cpu"))
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


class Big(nn.Module):
    """The 306 M preset, wearing the client's paradigm-facing surface."""

    def __init__(self):
        super().__init__()
        self.main_device = DEV
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight  # tied, as in the real client

    # --- the three methods crowd_mosaic/crowd_solo drive ---------------------
    def embed(self, idx, head_w=None):
        return self.wte(idx)

    def run_blocks(self, x, start=0, stop=None, use_chunked=True, use_ckpt=True, stream_dtype=None):
        for blk in self.blocks[start:stop]:
            x = blk(x)
        return x

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        return w if dtype is None else w.to(dtype)

    def forward(self, idx):
        return self.lm_head(self.ln_f(self.run_blocks(self.embed(idx))))


def batch(step, seed=0, n=None):
    n = B if n is None else int(n)
    g = torch.Generator().manual_seed(seed * 1_000_003 + step * 7919)
    off = torch.randint(0, train.size - T - 1, (n,), generator=g).numpy()
    x = np.stack([train[o:o + T] for o in off]).astype(np.int64)
    y = np.stack([train[o + 1:o + T + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


EVAL = []
for _i in range(8):
    _o = _i * (T + 1)
    EVAL.append((torch.from_numpy(np.stack([val[_o:_o + T]]).astype(np.int64)).to(DEV),
                 torch.from_numpy(np.stack([val[_o + 1:_o + T + 1]]).astype(np.int64)).to(DEV)))


def ce(logits, y):
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, V), y[:, :-1].reshape(-1))


def evaluate(model):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x, y in EVAL:
            tot += float(ce(model(x), y))
    model.train()
    return tot / len(EVAL)


def build():
    torch.manual_seed(0)
    model = Big().to(DEV)
    with torch.no_grad():                 # GPT-2 init, or an untrained 306 M scores ~700
        model.wte.weight.normal_(0.0, 0.02)
    return model


def run_backprop(tokens, lr=3e-4, eval_every=500_000, grad_accum=1):
    model = build()
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    curve, seen, nxt = [evaluate(model)], 0, eval_every
    # ``grad_accum`` exists because the round-matched control needs ONE update over
    # 4096 tokens at the 306 M preset, and B=8 does not fit in 16 GB. Accumulating
    # two B=4 micro-batches gives the identical optimisation step at B=4's
    # activation footprint, so the control stays memory-feasible.
    per_step = B * T * max(1, int(grad_accum))
    steps = max(1, tokens // per_step)
    t0 = time.perf_counter()
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        for m in range(max(1, int(grad_accum))):
            x, y = batch(s * max(1, int(grad_accum)) + m)
            ce(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += per_step
        if seen >= nxt:
            curve.append(evaluate(model))
            nxt += eval_every
    dt = time.perf_counter() - t0
    return {"paradigm": "backprop", "params": n, "tokens": seen, "steps": steps,
            "uniqueTokens": seen, "rounds": steps, "gradSteps": steps,
            "gradAccum": int(grad_accum), "tokensPerStep": per_step,
            "curve": curve, "final": curve[-1], "seconds": round(dt, 1),
            "tokensPerSecond": round(seen / dt), "device": str(DEV)}


def run_mosaic(tokens, slices=2, local_steps=30, inner_lr=3e-3,
               outer_lr=0.7, clip=True, eval_every=500_000, budget="unique",
               freeze_head=False, outer_momentum=0.9, inner_state=False,
               trunk_owner="first", share_trunk=False, trust_ratio=False,
               head_lr_mult=1.0, head_skip=False, overlap=0, head_mlp=False,
               fresh_local_batches=False, calib_every=0,
               outer_lr_final=None, schedule="cosine",
               anchor_every=0, anchor_lr=3e-4, anchor_batch=0,
               calib_steps=20, calib_lr=1e-2):
    model = build()
    n = sum(p.numel() for p in model.parameters())
    # Defaults from crowd_paradigm.ParadigmConfig (slices 2, local 30, inner 3e-3),
    # plus MosaicConfig's own outer_lr 0.7 / momentum 0.9: the same configuration the
    # tiny-model table was measured with, so the comparison is like for like.
    cfg = mosaic_mod.MosaicConfig(slices=slices, local_steps=local_steps,
                                  outer_lr=outer_lr, clip_median=clip,
                                  outer_momentum=outer_momentum,
                                  head_mlp=head_mlp,
                                  trust_ratio=trust_ratio)
    desc = mosaic_mod.attach(model, cfg)
    # The tied readout IS wte, so whoever owns wte trains the output layer. On the
    # first slice it sees only a local SOLO surrogate; on the last slice it sees
    # the terminal slice's true CE -- the role it plays under SOLO, where one
    # optimiser covers both. Measured, not assumed.
    if trunk_owner == "last":
        _inner = mosaic_mod.slice_of
        def _slice_of(name, slices_=None, _orig=_inner, _k=slices):
            if name.startswith("wte."):
                return _k - 1
            return _orig(name, slices_)
        mosaic_mod.slice_of = _slice_of
    if freeze_head:
        # Reproduces the pre-fix local_delta: the SOLO head is built but never
        # trained, so every non-terminal slice chases a frozen random projection.
        for p in model.solo_heads.parameters():
            p.requires_grad_(False)
    # Two defensible budgets, and they answer different questions:
    #   "unique"  -> equal *data*: rounds = tokens / (slices*B*T), so H local
    #                steps cost proportionally more compute (the method's price)
    #                but nothing is starved of tokens. This is the fair test.
    #   "inner"   -> equal *inner-step tokens*: rounds = tokens/(slices*H*B*T),
    #                which silently gives the arm H times fewer unique tokens.
    # Default is "unique"; "inner" is kept only to reproduce the first run.
    per_round = (slices * local_steps * B * T
                 if (budget == "inner" or fresh_local_batches)
                 else slices * B * T)
    rounds = max(1, tokens // per_round)
    curve, seen, nxt = [evaluate(model)], 0, eval_every
    state = None
    t0 = time.perf_counter()
    opt_state: dict | None = {} if inner_state else None
    # Periodic anchoring: every `anchor_every` rounds, take ONE ordinary AdamW
    # step on the true terminal loss over the WHOLE model. The point is to ask
    # whether a cheap global gradient once in a while repairs what a round of
    # greedy local steps cannot. Note the accounting mismatch: an anchor round
    # processes one batch (B*T tokens) and updates every parameter once, which is
    # exactly what an ordinary backprop step costs, so at anchor_every=10 it adds
    # 10% to the token bill of a round and 10% to the parameter-update count.
    anchor_opt = None
    if anchor_every:
        anchor_opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                       lr=float(anchor_lr), betas=(0.9, 0.95),
                                       weight_decay=0.01)
    for r in range(rounds):
        # Snapshot BEFORE the local steps: aggregate() adds outer_lr*delta to the
        # base it is given, so passing the post-step weights applies the local
        # step (1 + outer_lr) times. The production path captures it here too.
        current = {nm: p.detach().clone() for nm, p in model.named_parameters()}
        deltas = []
        for k in range(slices):
            x, y = batch(r * slices * local_steps + k * local_steps)
            xy_seq = ([batch(r * slices * local_steps + k * local_steps + s)
                       for s in range(local_steps)] if fresh_local_batches else None)
            deltas.append(mosaic_mod.local_delta(model, slices=desc["slices"], k=k, x=x, y=y,
                                                 xy_seq=xy_seq,
                                                 steps=local_steps, lr=inner_lr,
                                                 opt_state=opt_state,
                                                 share_trunk=share_trunk,
                                                 head_lr_mult=head_lr_mult,
                                                 head_skip=head_skip,
                                                 overlap=overlap))
        # Teacher lag: the head is fitted online while the trunk it supervises
        # keeps moving. Stopping to fit it (trunk frozen) is the cheap fix the
        # lr sweep pointed at -- see crowd_mosaic.fit_head's docstring.
        if calib_every and (r + 1) % int(calib_every) == 0:
            for kk in range(slices - 1):
                cxk, cyk = batch(50_000_000 + r * slices + kk)
                mosaic_mod.fit_head(model, slices=desc["slices"], k=kk, x=cxk, y=cyk,
                                    steps=calib_steps, lr=calib_lr)
        # The outer rate has an interior optimum (0.15 measured), which usually
        # means the optimum moves during training: a decaying schedule can spend
        # the early rounds where a big step is safe and the late rounds where only
        # a small one is. `cfg` is a frozen dataclass, so rebuild it per round.
        if outer_lr_final is not None:
            frac = r / max(1, rounds - 1)
            if schedule == "cosine":
                import math as _m
                o = outer_lr_final + (outer_lr - outer_lr_final) * 0.5 * (1.0 + _m.cos(_m.pi * frac))
            else:
                o = outer_lr + (outer_lr_final - outer_lr) * frac
            cfg = dataclasses.replace(cfg, outer_lr=o)
        new, state = mosaic_mod.aggregate(current, deltas, cfg=cfg, state=state)
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if nm in new:
                    p.copy_(new[nm])
        if anchor_opt is not None and (r + 1) % int(anchor_every) == 0:
            # ``anchor_batch`` exists for the 306 M preset: the anchor optimiser's
            # own state (2 x params fp32) plus its gradients pushed a B=4 anchor
            # step over 15.5 GB, so the step can run at a smaller batch there while
            # the 65 M ladder stays at the preset batch.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ax, ay = batch(10_000_000 + r, n=(int(anchor_batch) or B))
            anchor_opt.zero_grad(set_to_none=True)
            ce(model(ax), ay).backward()
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters()
                                            if q.requires_grad], 1.0)
            anchor_opt.step()
        seen += per_round
        if seen >= nxt:
            curve.append(evaluate(model))
            nxt += eval_every
    dt = time.perf_counter() - t0
    # ``tokens`` counts inner-step tokens, so it overstates the data the model
    # actually saw: each slice spends ``local_steps`` steps on ONE batch. With
    # local_steps=30 that is 30x reuse, and the run only gets ``rounds`` outer
    # aggregations. Both numbers are reported so the comparison cannot be read
    # as equal-data when it is equal-inner-step-tokens.
    return {"paradigm": "mosaic", "params": n, "tokens": seen,
            "uniqueTokens": rounds * slices * B * T,
            "rounds": rounds, "gradSteps": rounds * slices * local_steps,
            "slices": desc["slices"], "localSteps": local_steps, "innerLr": inner_lr,
            "frozenHead": bool(freeze_head), "trunkOwner": trunk_owner,
            "shareTrunk": bool(share_trunk), "trustRatio": bool(trust_ratio),
            "headLrMult": float(head_lr_mult), "headSkip": bool(head_skip), "overlap": int(overlap),
            "headMlp": bool(head_mlp), "freshLocalBatches": bool(fresh_local_batches),
            "outerLrFinal": outer_lr_final, "schedule": schedule,
            "anchorEvery": int(anchor_every), "anchorLr": float(anchor_lr),
            "calibEvery": int(calib_every), "calibSteps": int(calib_steps),
            "outerMomentum": outer_momentum,
            "innerState": bool(inner_state),
            "outerLr": outer_lr, "curve": curve, "final": curve[-1],
            "seconds": round(dt, 1), "tokensPerSecond": round(seen / dt), "device": str(DEV)}


def run_solo(tokens, modules=2, lr=3e-4, eval_every=500_000):
    """The SOLO paradigm exactly as the client drives it.

    Same local law as MOSAIC's non-terminal slices (learn through a SOLO head
    projected through a detached readout), but the optimiser is the client's own
    and it persists across steps. This separates "splitting the model costs
    accuracy" from "MOSAIC's outer aggregation costs accuracy".
    """
    import crowd_paradigm as CP

    torch.manual_seed(0)
    model = build().to(DEV)
    n = sum(p.numel() for p in model.parameters())
    par = CP.select("solo", solo_modules=modules)
    desc = par.attach(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    curve = [evaluate(model)]
    nxt = eval_every
    t0 = time.perf_counter()
    steps = 0
    seen = 0
    while seen < tokens:
        x, y = batch(steps)
        opt.zero_grad(set_to_none=True)
        par.step(model, x, y)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        steps += 1
        seen += B * T
        if seen >= nxt:
            curve.append(evaluate(model))
            nxt += eval_every
    dt = time.perf_counter() - t0
    return {"paradigm": "solo", "params": n, "tokens": seen, "uniqueTokens": seen,
            "rounds": steps, "gradSteps": steps, "modules": modules, "lr": lr,
            "splits": desc.get("splits"), "curve": curve, "final": curve[-1],
            "seconds": round(dt, 1), "tokensPerSecond": round(seen / dt), "device": str(DEV)}


def main():
    global D, L, H, B
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", default="both",
                    choices=("backprop", "mosaic", "solo", "both"))
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--slices", type=int, default=2)
    ap.add_argument("--local-steps", type=int, default=30)
    ap.add_argument("--inner-lr", type=float, default=3e-3)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--budget", default="unique", choices=("unique", "inner"),
                    help="how --tokens is spent: unique data (default) or inner-step tokens")
    ap.add_argument("--json", default="")
    ap.add_argument("--dim", type=int, default=0, help="override D (0 = the 306 M preset)")
    ap.add_argument("--layers", type=int, default=0, help="override L")
    ap.add_argument("--heads", type=int, default=0, help="override H")
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--inner-state", action="store_true",
                    help="keep each slice's AdamW state across rounds")
    ap.add_argument("--trunk-owner", choices=["first", "last"], default="first",
                    help="which slice owns the tied readout (wte)")
    ap.add_argument("--trust-ratio", action="store_true",
                    help="per-tensor relative outer step (outer_lr becomes relative)")
    ap.add_argument("--batch", type=int, default=B,
                    help="sequences per local step (default: the preset's B)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step (keeps the round-matched 4096-token step inside 16 GB)")
    ap.add_argument("--outer-lr-final", type=float, default=None,
                    help="decay the outer rate from --outer-lr to this")
    ap.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    ap.add_argument("--fresh-local-batches", action="store_true",
                    help="each inner step trains on its own batch (DiLoCo inner loop)")
    ap.add_argument("--head-mlp", action="store_true",
                    help="give each auxiliary head a hidden layer")
    ap.add_argument("--overlap", type=int, default=0,
                    help="extra blocks each side of a slice boundary, trained by both")
    ap.add_argument("--anchor-every", type=int, default=0,
                    help="every N rounds, one full-model AdamW step on the true loss")
    ap.add_argument("--anchor-lr", type=float, default=3e-4,
                    help="lr for the periodic full-model anchoring step")
    ap.add_argument("--anchor-batch", type=int, default=0,
                    help="sequences per anchor step (0 = the preset batch; the 306 M preset needs a smaller one)")
    ap.add_argument("--calib-every", type=int, default=0,
                    help="re-fit each SOLO head to the frozen trunk every N rounds")
    ap.add_argument("--calib-steps", type=int, default=20,
                    help="head-only AdamW steps per recalibration")
    ap.add_argument("--calib-lr", type=float, default=1e-2,
                    help="lr for head-only recalibration")
    ap.add_argument("--head-skip", action="store_true",
                    help="let the SOLO head see the slice input as well as its output")
    ap.add_argument("--head-lr-mult", type=float, default=1.0,
                    help="local lr multiplier for the SOLO head (the teacher)")
    ap.add_argument("--share-trunk", action="store_true",
                    help="let the terminal slice train the tied readout too")
    ap.add_argument("--freeze-head", action="store_true",
                    help="leave the SOLO head untrained (pre-fix behaviour)")
    a = ap.parse_args()

    B = a.batch
    if a.dim or a.layers or a.heads:
        D = a.dim or D
        L = a.layers or L
        H = a.heads or H

    print(f"device={DEV} D={D} L={L} H={H} B*T={B * T} tokens={a.tokens:,} "
          f"budget={a.budget} freeze_head={a.freeze_head}", flush=True)
    results = []
    if a.paradigm in ("backprop", "both"):
        results.append(run_backprop(a.tokens, grad_accum=a.grad_accum))
        r = results[-1]
        print(f"backprop curve={['%.4f' % c for c in r['curve']]} final {r['final']:.4f} "
              f"{r['seconds']}s {r['tokensPerSecond']:,} tok/s "
              f"steps={r['gradSteps']} unique={r['uniqueTokens']:,}", flush=True)
    if a.paradigm in ("solo", "both"):
        results.append(run_solo(a.tokens, modules=a.slices, lr=a.inner_lr))
        r = results[-1]
        print(f"solo(modules={r['modules']}) curve={['%.4f' % c for c in r['curve']]} "
              f"final {r['final']:.4f} {r['seconds']}s {r['tokensPerSecond']:,} tok/s "
              f"steps={r['gradSteps']}", flush=True)
    if a.paradigm in ("mosaic", "both"):
        results.append(run_mosaic(a.tokens, slices=a.slices, local_steps=a.local_steps,
                                  inner_lr=a.inner_lr, outer_lr=a.outer_lr,
                                  budget=a.budget, freeze_head=a.freeze_head,
                                  outer_momentum=a.outer_momentum,
                                  inner_state=a.inner_state,
                                  trunk_owner=a.trunk_owner,
                                  share_trunk=a.share_trunk,
                                  head_lr_mult=a.head_lr_mult,
                                  head_skip=a.head_skip,
                                  overlap=a.overlap,
                                  anchor_every=a.anchor_every, anchor_batch=a.anchor_batch,
                                  anchor_lr=a.anchor_lr,
                                  head_mlp=a.head_mlp,
                                  fresh_local_batches=a.fresh_local_batches,
                                  outer_lr_final=a.outer_lr_final,
                                  schedule=a.schedule,
                                  calib_every=a.calib_every,
                                  calib_steps=a.calib_steps,
                                  calib_lr=a.calib_lr,
                                  trust_ratio=a.trust_ratio))
        r = results[-1]
        print(f"mosaic(K={a.slices},H={a.local_steps},frozen={int(a.freeze_head)}) curve={['%.4f' % c for c in r['curve']]} "
              f"final {r['final']:.4f} {r['seconds']}s {r['tokensPerSecond']:,} tok/s "
              f"rounds={r['rounds']} steps={r['gradSteps']} "
              f"unique={r['uniqueTokens']:,}", flush=True)
    if len(results) == 2:
        d = results[1]["final"] - results[0]["final"]
        print(f"delta(mosaic - backprop) = {d:+.4f}  (init {results[0]['curve'][0]:.4f}, "
              f"ln(V)={math.log(V):.4f})", flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
