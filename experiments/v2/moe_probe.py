"""Does this project's federated loop actually train a mixture of experts?

Three questions, measured on one tiny model, all against real tokens:

  1. equal *active* parameters: MoE vs dense SwiGLU
  2. expert-column parallelism: each column trains only its own experts, on the
     tokens the router sends it — the shape a volunteer can actually hold
  3. router ownership: trained by a global anchor step, or owned by one column

Why a probe rather than a production flag: `crowd_mosaic` slices a model by
**contiguous layer range** (`local_delta` re-runs `run_blocks(h0, lo, hi)`), so it
cannot express "all layers, one expert". Expert parallelism is the other axis and
needs its own execution path before any 4B/7B config can exist. This script is
that path, at a size that finishes in minutes.

Runs on the ROCm box (the local 16 GB card is busy). No synthetic data: it reads
the same uint16 BPE-4096 tokens the other harnesses use.

    SOLO_DEV=cuda python moe_probe.py --rounds 200 --local-steps 5 --json out.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crowd_mosaic  # noqa: E402  (production merge rule: mean + outer_lr*step)

DEV = torch.device(os.environ.get("SOLO_DEV", "cuda" if torch.cuda.is_available() else "cpu"))

V = 4096          # matches the BPE-4096 bins on the box
D = 256
L = 4
T = 256
B = 4
N_EXP = 8         # experts per layer = columns
TOP_K = 2         # active routed experts
H_EXP = 64        # per-expert hidden; active MLP rows = TOP_K*H_EXP + H_EXP(shared)
H_DENSE = (TOP_K + 1) * H_EXP   # dense control, matched on *active* compute

TOK = os.environ.get("V15_TOK", "/tmp/v15data/tok")


# ---------------------------------------------------------------- model

class Expert(nn.Module):
    """SwiGLU MLP, the shape SotaGPT uses, so dense and MoE are comparable."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """Routed experts + one always-on shared expert, top-k over the routed set.

    ``only=j`` is the column-parallel form: only tokens whose top-k includes j
    contribute, and only expert j is evaluated. That is the most a volunteer
    holding one column can compute — and it is why the router must be owned by
    something global, since a column cannot see the routing decisions it would
    need to learn from.
    """

    def __init__(self, d: int, h: int, n_exp: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d, n_exp, bias=False)
        self.experts = nn.ModuleList([Expert(d, h) for _ in range(n_exp)])
        self.shared = Expert(d, h)

    def _route(self, x, detach):
        logits = self.router(x)
        if detach:
            logits = logits.detach()
        probs = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        return probs, probs.topk(self.top_k, dim=-1)

    def forward(self, x, only=None, detach_router=False, return_mask=False):
        probs, (topv, topi) = self._route(x, detach_router)

        if only is None:
            out = self.shared(x)
            for e, expert in enumerate(self.experts):
                sel = (topi == e)
                mask = sel.any(dim=-1)
                if not bool(mask.any()):
                    continue
                w = (topv * sel.to(topv.dtype)).sum(dim=-1, keepdim=True)
                part = torch.zeros_like(x)
                part[mask] = w[mask] * expert(x[mask])
                out = out + part
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            return (out, mask) if return_mask else out

        sel = (topi == only)
        mask = sel.any(dim=-1)
        w = (topv * sel.to(topv.dtype)).sum(dim=-1, keepdim=True)
        out = self.shared(x)
        if bool(mask.any()):
            part = torch.zeros_like(x)
            part[mask] = w[mask] * self.experts[only](x[mask])
            out = out + part
        return (out, mask) if return_mask else out


class DenseLayer(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.mlp = Expert(d, h)

    def forward(self, x, only=None, detach_router=False, return_mask=False):
        out = self.mlp(x)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return (out, mask) if return_mask else out


class Block(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.ln2 = nn.LayerNorm(D)
        self.layer = layer

    def forward(self, x, **kw):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        a = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x = x + self.proj(a)
        out = self.layer(self.ln2(x), **kw)
        if isinstance(out, tuple):
            y, mask = out
            return x + y, mask
        return x + out


class Net(nn.Module):
    def __init__(self, moe: bool):
        super().__init__()
        self.moe = moe
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([
            Block(MoELayer(D, H_EXP, N_EXP, TOP_K) if moe else DenseLayer(D, H_DENSE))
            for _ in range(L)
        ])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight
        with torch.no_grad():
            self.wte.weight.normal_(0.0, 0.02)

    def forward(self, idx, **kw):
        x = self.wte(idx)
        mask = None
        for blk in self.blocks:
            out = blk(x, **kw)
            if isinstance(out, tuple):
                x, mask = out
            else:
                x = out
        logits = self.lm_head(self.ln_f(x))
        return (logits, mask) if kw.get("return_mask") else logits


# ---------------------------------------------------------------- data

def _mem(name):
    return np.memmap(os.path.join(TOK, name), dtype=np.uint16, mode="r")


def batch(step, seed=0, n=B):
    data = _mem("train.bin")
    g = torch.Generator().manual_seed(seed * 1_000_003 + step * 7919)
    span = data.size - (T + 1) * n - 1
    off = int(torch.randint(0, span, (1,), generator=g).item())
    blk = torch.from_numpy(np.asarray(data[off:off + (T + 1) * n],
                                      dtype=np.int64).reshape(n, T + 1))
    return blk[:, :-1].to(DEV), blk[:, 1:].to(DEV)


def eval_windows(k=8):
    data = _mem("val.bin")
    out = []
    for i in range(k):
        blk = torch.from_numpy(np.asarray(data[i * (T + 1) * B:(i + 1) * (T + 1) * B],
                                          dtype=np.int64).reshape(B, T + 1))
        out.append((blk[:, :-1].to(DEV), blk[:, 1:].to(DEV)))
    return out


def ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))


@torch.no_grad()
def evaluate(model, windows):
    model.eval()
    tot = sum(float(ce(model(x), y)) for x, y in windows)
    model.train()
    return tot / len(windows)


@torch.no_grad()
def expert_share(model, windows):
    """Share of routed slots per expert. Collapse shows up as one near 1.0."""
    caps, handles = [], []
    for blk in model.blocks:
        if isinstance(blk.layer, MoELayer):
            cap = []
            caps.append(cap)
            handles.append(blk.layer.router.register_forward_hook(
                lambda _m, _i, out, cap=cap: cap.append(out.topk(TOP_K, dim=-1).indices)))
    model.eval()
    for x, _y in windows:
        model(x)
    model.train()
    for h in handles:
        h.remove()
    idx = torch.cat([t for cap in caps for t in cap]).reshape(-1)
    counts = torch.bincount(idx, minlength=N_EXP).float()
    return [round(float(c) / max(1, idx.numel()), 4) for c in counts]


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    if not model.moe:
        return total, total
    all_experts = sum(p.numel() for blk in model.blocks
                      for p in blk.layer.experts.parameters())
    one_expert = sum(p.numel() for p in model.blocks[0].layer.experts[0].parameters())
    # per layer, not once overall: every layer drops (N_EXP - TOP_K) experts
    return total, total - all_experts + TOP_K * one_expert * len(model.blocks)


# ---------------------------------------------------------------- trainers

def train_full(model, steps, lr=3e-4, windows=None, eval_every=250):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    curve, t0 = [], time.perf_counter()
    for s in range(steps):
        x, y = batch(s)
        opt.zero_grad(set_to_none=True)
        ce(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if windows and (s + 1) % eval_every == 0:
            curve.append(round(evaluate(model, windows), 4))
    return curve, time.perf_counter() - t0


def trunk_params(model):
    """Everything that is not a routed expert.

    Attention, LayerNorms, the embedding, the (tied) readout, the router and the
    shared expert are all read by every token, so no single expert column can
    train them: a column only ever sees the tokens its router sent it. Leaving
    them out of *every* optimiser is the silent allow-list bug that cost 0.8 nats
    in MOSAIC (frozen random readout). They belong to the global step.
    """
    return [p for n, p in model.named_parameters() if ".experts." not in n]


def freeze_trunk(model, frozen: bool):
    for p in trunk_params(model):
        p.requires_grad_(not frozen)


def column_step(model, col, x, y, opt, local_steps, owns_router):
    """Train one expert column on the tokens its router sends it.

    ``owns_router`` is the slice-owned variant: this column also updates the
    router and shared expert from its own routed tokens — the only local signal
    it has, and a biased one (it never sees the tokens it did not get).
    """
    last = 0.0
    for _ in range(local_steps):
        opt.zero_grad(set_to_none=True)
        logits, mask = model(x, only=col, detach_router=not owns_router, return_mask=True)
        if not bool(mask.any()):
            return last
        loss = F.cross_entropy(logits[mask].reshape(-1, V), y[mask].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        last = float(loss.detach())
    return last


def train_columns(model, rounds, local_steps, *, outer_lr=0.15, momentum=0.9,
                  router_mode="anchor", anchor_every=1, windows=None, eval_every=50,
                  lr=1e-3, balance_weight=0.0):
    cfg = crowd_mosaic.MosaicConfig(slices=N_EXP, local_steps=local_steps,
                                    outer_lr=outer_lr, outer_momentum=momentum)
    state = None
    curve, t0 = [], time.perf_counter()

    # One persistent optimiser per column: a fresh AdamW at one step is sign-SGD.
    col_opts = [
        torch.optim.AdamW([p for blk in model.blocks if isinstance(blk.layer, MoELayer)
                           for p in blk.layer.experts[j].parameters()],
                          lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        for j in range(N_EXP)
    ]
    trunk = trunk_params(model)
    trunk_opt = torch.optim.AdamW(trunk, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    if router_mode == "slice":
        # column 0 also owns the trunk; it is the only optimiser allowed to step it
        col_opts[0] = torch.optim.AdamW(
            [p for blk in model.blocks for p in blk.layer.experts[0].parameters()] + trunk,
            lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"  trunk params {sum(p.numel() for p in trunk):,} of "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    for r in range(rounds):
        base = {n: p.detach().clone() for n, p in model.named_parameters()}

        for k in range(N_EXP):
            owns = (router_mode == "slice" and k == 0)
            freeze_trunk(model, not owns)
            x, y = batch(r * N_EXP + k)
            column_step(model, k, x, y, col_opts[k], local_steps, owns)
        freeze_trunk(model, False)

        changed = {n: (p.detach() - base[n]) for n, p in model.named_parameters()
                   if n in base and not torch.equal(p.detach(), base[n])}
        if changed:
            new, state = crowd_mosaic.aggregate(base, [changed], cfg=cfg, state=state)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in new:
                        p.copy_(new[n])

        if router_mode == "anchor" and anchor_every and (r + 1) % anchor_every == 0:
            ax, ay = batch(10_000_000 + r)
            caps, handles = [], []
            for blk in model.blocks:
                if isinstance(blk.layer, MoELayer):
                    handles.append(blk.layer.router.register_forward_hook(
                        lambda _m, _i, out, caps=caps: caps.append(out)))
            trunk_opt.zero_grad(set_to_none=True)
            loss = ce(model(ax), ay)
            if balance_weight > 0 and caps:
                aux = 0.0
                for logits_r in caps:
                    probs = torch.softmax(logits_r.float(), dim=-1)
                    _topv, topi = probs.topk(TOP_K, dim=-1)
                    f = torch.bincount(topi.reshape(-1), minlength=N_EXP).float() / topi.numel()
                    pbar = probs.reshape(-1, N_EXP).mean(0)
                    aux = aux + N_EXP * (f * pbar).sum()
                loss = loss + balance_weight * (aux / len(caps))
            loss.backward()
            for h in handles:
                h.remove()
            torch.nn.utils.clip_grad_norm_(trunk, 1.0)
            trunk_opt.step()

        if windows and (r + 1) % eval_every == 0:
            curve.append(round(evaluate(model, windows), 4))

    return curve, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--local-steps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--outer-lr", type=float, default=0.15)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--balance-weight", type=float, default=0.0)
    ap.add_argument("--arms", default="anchor,slice")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    windows = eval_windows()
    out = {"device": str(DEV), "vocab": V, "dim": D, "layers": L, "experts": N_EXP,
           "topK": TOP_K, "hExp": H_EXP, "hDense": H_DENSE}

    if a.skip_baselines:
        windows = windows
    if not a.skip_baselines:
        torch.manual_seed(0)
        dense = Net(moe=False).to(DEV)
        out["dense"] = dict(zip(("params", "active"), param_counts(dense)))
        print(f"dense  total={out['dense']['params']:,} active={out['dense']['active']:,}", flush=True)
        curve, secs = train_full(dense, a.steps, windows=windows, eval_every=max(1, a.steps // 4))
        out["dense"].update(curve=curve, final=curve[-1] if curve else None, seconds=round(secs, 1))
        print(f"dense  curve={curve} {secs:.1f}s", flush=True)

        torch.manual_seed(0)
        moe = Net(moe=True).to(DEV)
        out["moe"] = dict(zip(("params", "active"), param_counts(moe)))
        print(f"moe    total={out['moe']['params']:,} active={out['moe']['active']:,}", flush=True)
        curve, secs = train_full(moe, a.steps, windows=windows, eval_every=max(1, a.steps // 4))
        out["moe"].update(curve=curve, final=curve[-1] if curve else None,
                          seconds=round(secs, 1), share=expert_share(moe, windows))
        print(f"moe    curve={curve} {secs:.1f}s share={out['moe']['share']}", flush=True)

    for mode in [m.strip() for m in a.arms.split(",") if m.strip()]:
        torch.manual_seed(0)
        m = Net(moe=True).to(DEV)
        curve, secs = train_columns(m, a.rounds, a.local_steps, router_mode=mode,
                                    outer_lr=a.outer_lr, momentum=a.outer_momentum,
                                    balance_weight=a.balance_weight,
                                    windows=windows, eval_every=max(1, a.rounds // 4))
        key = f"columns_{mode}_olr{a.outer_lr}_bw{a.balance_weight}"
        out[key] = {"curve": curve, "final": curve[-1] if curve else None,
                    "seconds": round(secs, 1), "share": expert_share(m, windows),
                    "outerLr": a.outer_lr, "balanceWeight": a.balance_weight}
        print(f"columns/{mode:6s} olr={a.outer_lr} bw={a.balance_weight} curve={curve} "
              f"{secs:.1f}s share={out[key]['share']}", flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", a.json, flush=True)


if __name__ == "__main__":
    main()
