"""Does the learning-rate optimum transfer across depth?

Sweeps lr at L=4 and L=12 under two parametrisations:
  std  : PyTorch default init, residual stream added at unit scale
  mup  : residual branches scaled by 1/sqrt(2L), the depth part of muP
If mup works, the best lr is the same at both depths; under std it moves.
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn

V, H, T, B = 4096, 4, 256, 8
D = 128
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TRAIN, VAL = "/tmp/v15data/tok/train.bin", "/tmp/v15data/tok/val.bin"
train = np.memmap(TRAIN, dtype=np.uint16, mode="r")
val = np.memmap(VAL, dtype=np.uint16, mode="r")


class Block(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
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
        x = x + self.scale * self.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + self.scale * self.down(torch.nn.functional.silu(self.up(self.ln2(x))))


class Tiny(nn.Module):
    def __init__(self, layers, param):
        super().__init__()
        scale = 1.0 / math.sqrt(2 * layers) if param == "mup" else 1.0
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(scale) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def batch(seed):
    g = torch.Generator().manual_seed(seed * 1_000_003 + 17)
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


def evaluate(m):
    m.eval(); tot = 0.0
    with torch.no_grad():
        for x, y in EVAL:
            tot += float(ce(m(x), y))
    m.train(); return tot / len(EVAL)


def run(layers, param, lr):
    torch.manual_seed(0)
    m = Tiny(layers, param).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for s in range(STEPS):
        x, y = batch(s)
        opt.zero_grad(set_to_none=True)
        ce(m(x), y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return evaluate(m), sum(p.numel() for p in m.parameters())


print(f"steps={STEPS} tokens/run={STEPS*B*T:,} device={DEV}", flush=True)
rows = []
for param in ("std", "mup"):
    for layers in (4, 12):
        for lr in (1e-3, 3e-3, 1e-2, 3e-2):
            t0 = time.perf_counter()
            loss, n = run(layers, param, lr)
            rows.append((param, layers, lr, loss))
            print(f"{param:4s} L={layers:2d} lr={lr:7.0e}  held-out {loss:8.4f}  "
                  f"params {n/1e6:5.2f}M  {time.perf_counter()-t0:5.1f}s", flush=True)

for param in ("std", "mup"):
    print(f"\n{param}:")
    for layers in (4, 12):
        sub = [r for r in rows if r[0] == param and r[1] == layers]
        best = min(sub, key=lambda r: r[3])
        print(f"  L={layers:2d} best lr {best[2]:.0e} -> {best[3]:.4f}   "
              + "  ".join(f"{lr:.0e}:{l:.3f}" for _, _, lr, l in sub))
