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


def batch(step, seed=0):
    g = torch.Generator().manual_seed(seed * 1_000_003 + step * 7919)
    off = torch.randint(0, train.size - T - 1, (B,), generator=g).numpy()
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


def run_backprop(tokens, lr=3e-4, eval_every=500_000):
    model = build()
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    curve, seen, nxt = [evaluate(model)], 0, eval_every
    steps = max(1, tokens // (B * T))
    t0 = time.perf_counter()
    for s in range(steps):
        x, y = batch(s)
        opt.zero_grad(set_to_none=True)
        ce(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += B * T
        if seen >= nxt:
            curve.append(evaluate(model))
            nxt += eval_every
    dt = time.perf_counter() - t0
    return {"paradigm": "backprop", "params": n, "tokens": seen, "steps": steps,
            "uniqueTokens": seen, "rounds": steps, "gradSteps": steps,
            "curve": curve, "final": curve[-1], "seconds": round(dt, 1),
            "tokensPerSecond": round(seen / dt), "device": str(DEV)}


def run_mosaic(tokens, slices=2, local_steps=30, inner_lr=3e-3,
               outer_lr=0.7, clip=True, eval_every=500_000, budget="unique"):
    model = build()
    n = sum(p.numel() for p in model.parameters())
    # Defaults from crowd_paradigm.ParadigmConfig (slices 2, local 30, inner 3e-3),
    # plus MosaicConfig's own outer_lr 0.7 / momentum 0.9: the same configuration the
    # tiny-model table was measured with, so the comparison is like for like.
    cfg = mosaic_mod.MosaicConfig(slices=slices, local_steps=local_steps,
                                  outer_lr=outer_lr, clip_median=clip)
    desc = mosaic_mod.attach(model, cfg)
    # Two defensible budgets, and they answer different questions:
    #   "unique"  -> equal *data*: rounds = tokens / (slices*B*T), so H local
    #                steps cost proportionally more compute (the method's price)
    #                but nothing is starved of tokens. This is the fair test.
    #   "inner"   -> equal *inner-step tokens*: rounds = tokens/(slices*H*B*T),
    #                which silently gives the arm H times fewer unique tokens.
    # Default is "unique"; "inner" is kept only to reproduce the first run.
    per_round = slices * local_steps * B * T if budget == "inner" else slices * B * T
    rounds = max(1, tokens // per_round)
    curve, seen, nxt = [evaluate(model)], 0, eval_every
    state = None
    t0 = time.perf_counter()
    for r in range(rounds):
        deltas = []
        for k in range(slices):
            x, y = batch(r * slices * local_steps + k * local_steps)
            deltas.append(mosaic_mod.local_delta(model, slices=desc["slices"], k=k, x=x, y=y,
                                                 steps=local_steps, lr=inner_lr))
        current = {nm: p.detach().clone() for nm, p in model.named_parameters()
                   if any(nm in d for d in deltas)}
        new, state = mosaic_mod.aggregate(current, deltas, cfg=cfg, state=state)
        with torch.no_grad():
            for nm, p in model.named_parameters():
                if nm in new:
                    p.copy_(new[nm])
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
            "outerLr": outer_lr, "curve": curve, "final": curve[-1],
            "seconds": round(dt, 1), "tokensPerSecond": round(seen / dt), "device": str(DEV)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", default="both", choices=("backprop", "mosaic", "both"))
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--slices", type=int, default=2)
    ap.add_argument("--local-steps", type=int, default=30)
    ap.add_argument("--inner-lr", type=float, default=3e-3)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--budget", default="unique", choices=("unique", "inner"),
                    help="how --tokens is spent: unique data (default) or inner-step tokens")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    print(f"device={DEV} params~306M B*T={B * T} tokens={a.tokens:,} "
          f"budget={a.budget}", flush=True)
    results = []
    if a.paradigm in ("backprop", "both"):
        results.append(run_backprop(a.tokens))
        r = results[-1]
        print(f"backprop curve={['%.4f' % c for c in r['curve']]} final {r['final']:.4f} "
              f"{r['seconds']}s {r['tokensPerSecond']:,} tok/s "
              f"steps={r['gradSteps']} unique={r['uniqueTokens']:,}", flush=True)
    if a.paradigm in ("mosaic", "both"):
        results.append(run_mosaic(a.tokens, slices=a.slices, local_steps=a.local_steps,
                                  inner_lr=a.inner_lr, outer_lr=a.outer_lr,
                                  budget=a.budget))
        r = results[-1]
        print(f"mosaic(K={a.slices},H={a.local_steps}) curve={['%.4f' % c for c in r['curve']]} "
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
