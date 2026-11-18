"""306 M-preset DiLoCo confirmation: does the tiny-model result survive at scale?

Sequential workers (one model resident at a time) so a 306 M model plus four
worker snapshots fits in 16 GB. Deviation from a true DiLoCo run, stated in the
report: each worker gets a fresh AdamW per round instead of carrying inner
optimizer state, because four sets of Adam moments for 306 M params do not fit
alongside the weights. Inner steps per round are what the tiny-model result
turned on, and that is preserved exactly.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

V, D, L, H, T, B = 4096, 1024, 24, 16, 512, 4
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
K = 4
TARGET = int(float(os.environ.get("DL_TARGET_M", "4")) * 1e6)
INNERS = [int(x) for x in os.environ.get("DL_INNER", "1,25,100").split(",")]
EVAL_EVERY = 500_000
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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def batch(seed, worker):
    g = torch.Generator().manual_seed(seed * 1_000_003 + worker * 7919)
    off = torch.randint(0, train.size - T - 1, (B,), generator=g).numpy()
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


def run(inner):
    torch.manual_seed(0)
    globe = Model().to(DEV)
    # GPT-2-style init on the tied embedding/readout. Without it the logits scale
    # as sqrt(D) and an untrained 306M model scores ~700 instead of ln(4096)=8.32,
    # so the run starts in a regime no optimizer can recover from in 4M tokens.
    with torch.no_grad():
        globe.wte.weight.normal_(0.0, 0.02)
    n = sum(p.numel() for p in globe.parameters())
    opt_o = torch.optim.SGD(globe.parameters(), lr=0.7, momentum=0.9, nesterov=True)
    rounds = max(1, TARGET // (inner * K * B * T))
    # bf16 snapshots: four 306M fp32 copies per round dominated the step time at inner=1.
    snap = [{k: v.detach().to('cpu', torch.bfloat16).clone() for k, v in globe.state_dict().items()}
            for _ in range(K)]
    # One inner optimizer per worker, created once. Rebuilding AdamW inside the round
    # loop reallocated ~2.4 GB of moments per worker per round (4x per round at inner=1)
    # and dropped throughput to ~1.5k tok/s. Keeping the state is also what DiLoCo does:
    # inner optimizers persist across outer steps.
    opts = [torch.optim.AdamW(globe.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
            for _ in range(K)]
    curve, seen, nxt = [evaluate(globe)], 0, EVAL_EVERY
    t0 = time.perf_counter()
    for r in range(rounds):
        for wi in range(K):
            globe.load_state_dict(snap[wi])
            opt = opts[wi]; opt.zero_grad(set_to_none=True)
            for j in range(inner):
                x, y = batch(r * inner + j, wi)
                opt.zero_grad(set_to_none=True)
                ce(globe(x), y).backward()
                torch.nn.utils.clip_grad_norm_(globe.parameters(), 1.0)
                opt.step()
            snap[wi] = {k: v.detach().cpu().clone() for k, v in globe.state_dict().items()}
        globe.load_state_dict(snap[0])              # restore base weights
        with torch.no_grad():
            for name, p in globe.named_parameters():
                acc = torch.zeros_like(p, dtype=torch.float32)
                for wi in range(K):
                    acc += snap[wi][name].to(DEV, torch.float32)
                p.grad = p.float() - acc / K       # DiLoCo: theta - mean(theta_k)
            opt_o.step()
        seen += inner * K * B * T
        if seen >= nxt:
            curve.append(evaluate(globe)); nxt += EVAL_EVERY
    dt = time.perf_counter() - t0
    print(f"inner={inner:4d} params={n/1e6:.1f}M rounds={rounds:4d} tokens={seen:,} "
          f"curve={['%.4f' % c for c in curve]} held-out {curve[-1]:.4f} "
          f"{seen/dt:9,.0f} tok/s {dt:6.1f}s uploads={rounds}", flush=True)


print(f"K={K} B={B} T={T} target={TARGET:,} dev={DEV}", flush=True)
for inner in INNERS:
    run(inner)
