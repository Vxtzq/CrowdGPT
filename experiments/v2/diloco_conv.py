"""DiLoCo: how much upload bandwidth can we delete for how much loss?

K workers train independently for `inner_steps` AdamW steps, then the server
averages their weights and applies one outer Nesterov-SGD step. The only thing
that crosses the network is that average, once per round. So the currency is
"uploads per million tokens" and the question is what it costs in held-out loss.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
K = int(os.environ.get("DILOCO_WORKERS", 4))
TARGET_TOKENS = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 8_000_000
INNER = [int(x) for x in os.environ.get("DILOCO_INNER", "1,50,500").split(",")]
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


def run(inner_steps):
    # equal tokens for every arm: more inner steps means proportionally fewer uploads
    rounds = max(1, TARGET_TOKENS // (inner_steps * K * B * T))
    torch.manual_seed(0)
    globe = Tiny().to(DEV)
    opt_o = torch.optim.SGD(globe.parameters(), lr=0.7, momentum=0.9, nesterov=True)
    workers = [Tiny().to(DEV) for _ in range(K)]
    inners = [torch.optim.AdamW(w.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01) for w in workers]
    curve = [evaluate(globe)]
    t0 = time.perf_counter()
    for r in range(rounds):
        for w, opt, wk in zip(workers, inners, range(K)):
            w.load_state_dict(globe.state_dict())
            for _ in range(inner_steps):
                x, y = batch(r * inner_steps + _, wk)
                opt.zero_grad(set_to_none=True)
                ce(w(x), y).backward()
                torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            for name, p in globe.named_parameters():
                acc = torch.zeros_like(p)
                for w in workers:
                    acc += w.state_dict()[name].float()
                # DiLoCo pseudo-gradient: theta - mean(theta_k). SGD then moves the
                # global weights toward the average; the opposite sign extrapolates
                # away from it and diverges (first run hit held-out 1234).
                p.grad = p.float() - acc / K
            opt_o.step()
        if (r + 1) % max(1, rounds // 20) == 0:
            curve.append(evaluate(globe))
    dt = time.perf_counter() - t0
    toks = rounds * inner_steps * K * B * T
    print(f"inner={inner_steps:4d}  uploads={rounds:5d}  tokens={toks:,}  "
          f"tokens/upload={toks/rounds:10,.0f}  held-out {curve[-1]:.4f}  "
          f"{toks/dt:9,.0f} tok/s  {dt:6.1f}s", flush=True)
    return curve[-1], rounds, toks


print(f"K={K} target tokens={TARGET_TOKENS:,}  model {sum(p.numel() for p in Tiny().parameters())/1e6:.2f}M  dev={DEV}", flush=True)
for inner in INNER:
    run(inner)
