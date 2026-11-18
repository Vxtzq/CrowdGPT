"""Where does bf16 rounding flip an honest improvement into a 'regression'?

The cross-check margin collapses to its 1e-4 floor (repeated evals of identical
weights are identical, so no measurable noise). A delta that genuinely helps by
less than bf16 shipping error then looks like a regression -- the false positive
reported from v1.

Sweep delta size: 1..20 AdamW steps at 1e-5..1e-3, replay the SAME delta as fp32
and as bf16, and judge it with the old rule and the paired rule.
"""
import sys, math
sys.path.insert(0, "/app/crowdgpt/CrowdGPT-Server")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from solo_check import paired_margin, judge_paired

DEV = torch.device("cpu")
V, D, L, H, T, B, W = 4096, 128, 4, 4, 256, 8, 8
torch.manual_seed(0)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(),
                                 nn.Linear(4 * D, D, bias=False))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        a = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x))

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_device = DEV
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight
        with torch.no_grad():
            self.wte.weight.normal_(0.0, 0.02)
    def embed(self, idx, head_w=None):
        return self.wte(idx)
    def run_blocks(self, x, start=0, stop=None, use_chunked=True, use_ckpt=True, stream_dtype=None):
        for b in self.blocks[start:stop]:
            x = b(x)
        return x
    def head_weight(self, dtype=None):
        return self.lm_head.weight if dtype is None else self.lm_head.weight.to(dtype)

def batch(seed):
    rng = np.random.default_rng(seed)
    data = np.memmap("/tmp/v15data/tok/train.bin", dtype=np.uint16, mode="r")
    off = rng.integers(0, data.size - (T + 1) * B)
    blk = np.asarray(data[off:off + (T + 1) * B], dtype=np.int64).reshape(B, T + 1)
    return torch.from_numpy(blk[:, :-1]), torch.from_numpy(blk[:, 1:])

def val_windows():
    data = np.memmap("/tmp/v15data/tok/val.bin", dtype=np.uint16, mode="r")
    out = []
    for i in range(W):
        blk = np.asarray(data[i * (T + 1) * B:(i + 1) * (T + 1) * B], dtype=np.int64).reshape(B, T + 1)
        out.append((torch.from_numpy(blk[:, :-1]), torch.from_numpy(blk[:, 1:])))
    return out

WINDOWS = val_windows()
model = Tiny()

def per_window(m):
    out = []
    with torch.no_grad():
        for x, y in WINDOWS:
            feats = m.ln_f(m.run_blocks(m.embed(x)))
            logits = feats.reshape(-1, feats.shape[-1]) @ m.head_weight().T
            out.append(float(F.cross_entropy(logits, y.reshape(-1))))
    return out

base = [p.detach().clone() for p in model.parameters()]
base_losses = per_window(model)
base_mean = sum(base_losses) / len(base_losses)

def restore():
    with torch.no_grad():
        for p, b in zip(model.parameters(), base):
            p.copy_(b)

def train(n, lr):
    x, y = batch(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        feats = model.ln_f(model.run_blocks(model.embed(x)))
        logits = feats.reshape(-1, feats.shape[-1]) @ model.head_weight().T
        F.cross_entropy(logits, y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return [p.detach().clone() - b for p, b in zip(model.parameters(), base)]

def replay(delta, quant=False):
    with torch.no_grad():
        for p, b, d in zip(model.parameters(), base, delta):
            p.copy_(b + (d.to(torch.bfloat16).to(torch.float32) if quant else d))
    return per_window(model)

print(f"base mean {base_mean:.6f} over {len(WINDOWS)} windows")
print(f"{'delta':>16} {'fp32 gain':>11} {'bf16 gain':>11} {'band':>9} {'3sig':>9} "
      f"{'old->fp32':>10} {'old->bf16':>10} {'new->bf16':>10}")
for n, lr in ((1, 1e-5), (1, 1e-4), (2, 1e-4), (5, 1e-4), (20, 1e-3)):
    restore()
    delta = train(n, lr)
    fp32 = replay(delta)
    bf = replay(delta, quant=True)
    m32, g32 = paired_margin(base_losses, fp32)
    mbf, gbf = paired_margin(base_losses, bf, bf16_band=abs((sum(bf) - sum(fp32)) / len(bf)))
    ok_old32 = judge_paired(base_losses, fp32, sigmas=0.0, floor=1e-4)[0]
    ok_oldbf = judge_paired(base_losses, bf, sigmas=0.0, floor=1e-4)[0]
    ok_newbf = judge_paired(base_losses, bf, bf16_band=abs((sum(bf) - sum(fp32)) / len(bf)))[0]
    print(f"{n:2d} steps @ {lr:g}".rjust(16) + f" {g32:>+11.6f} {gbf:>+11.6f} "
          f"{abs(gbf - g32):>9.6f} {mbf:>9.6f} {str(ok_old32):>10} {str(ok_oldbf):>10} {str(ok_newbf):>10}")
