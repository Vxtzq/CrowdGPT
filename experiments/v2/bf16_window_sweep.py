"""The case that actually hurt v1: an honest peer whose delta is neutral-to-slightly-worse
on the server's held-out slice, judged with one/few windows and a fixed 1e-4 floor."""
import sys, math
sys.path.insert(0, "/app/crowdgpt/CrowdGPT-Server")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from solo_check import paired_margin, judge_paired

DEV = torch.device("cpu"); V, D, L, T, B = 4096, 128, 4, 256, 8
torch.manual_seed(0)

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(D, 3*D, bias=False); s.o = nn.Linear(D, D, bias=False)
        s.mlp = nn.Sequential(nn.Linear(D, 4*D, bias=False), nn.GELU(), nn.Linear(4*D, D, bias=False))
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(s, x):
        h = s.ln1(x); q, k, v = s.qkv(h).chunk(3, -1)
        a = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x = x + s.o(a); return x + s.mlp(s.ln2(x))

class Tiny(nn.Module):
    def __init__(s):
        super().__init__(); s.main_device = DEV; s.wte = nn.Embedding(V, D)
        s.blocks = nn.ModuleList([Block() for _ in range(L)]); s.ln_f = nn.LayerNorm(D)
        s.lm_head = nn.Linear(D, V, bias=False); s.lm_head.weight = s.wte.weight
        with torch.no_grad(): s.wte.weight.normal_(0.0, 0.02)
    def embed(s, idx, head_w=None): return s.wte(idx)
    def run_blocks(s, x, start=0, stop=None, use_chunked=True, use_ckpt=True, stream_dtype=None):
        for b in s.blocks[start:stop]: x = b(x)
        return x
    def head_weight(s, dtype=None): return s.lm_head.weight if dtype is None else s.lm_head.weight.to(dtype)

def win(path, off, k):
    d = np.memmap(path, dtype=np.uint16, mode="r")
    out = []
    for i in range(k):
        o = off + i*(T+1)*B
        blk = np.asarray(d[o:o+(T+1)*B], dtype=np.int64).reshape(B, T+1)
        out.append((torch.from_numpy(blk[:, :-1]), torch.from_numpy(blk[:, 1:])))
    return out

VAL = win("/tmp/v15data/tok/val.bin", 0, 8)
model = Tiny()
def losses(ws):
    out = []
    with torch.no_grad():
        for x, y in ws:
            f = model.ln_f(model.run_blocks(model.embed(x)))
            out.append(float(F.cross_entropy(f.reshape(-1, f.shape[-1]) @ model.head_weight().T, y.reshape(-1))))
    return out
base = [p.detach().clone() for p in model.parameters()]
base_l = losses(VAL)

def delta_from(ws, n, lr):
    with torch.no_grad():
        for p, b in zip(model.parameters(), base): p.copy_(b)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    for x, y in ws[:n]:
        opt.zero_grad(set_to_none=True)
        f = model.ln_f(model.run_blocks(model.embed(x)))
        F.cross_entropy(f.reshape(-1, f.shape[-1]) @ model.head_weight().T, y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return [p.detach().clone() - b for p, b in zip(model.parameters(), base)]

def replay(delta, quant, ws):
    with torch.no_grad():
        for p, b, d in zip(model.parameters(), base, delta):
            p.copy_(b + (d.to(torch.bfloat16).to(torch.float32) if quant else d))
    return losses(ws)

# an honest peer who trained on a *different* shard (train data), judged on val
print(f"{'case':>26} {'windows':>8} {'fp32 d':>10} {'bf16 d':>10} {'band':>9} {'3sig':>9} {'old':>6} {'new':>6}")
for label, train_ws, steps in (("peer trained 1 step 1e-4", win("/tmp/v15data/tok/train.bin", 100_000, 4), 1),
                               ("peer trained 4 steps 1e-4", win("/tmp/v15data/tok/train.bin", 100_000, 4), 4)):
    d = delta_from(train_ws, steps, 1e-4)
    for k in (1, 4, 8):
        ws = VAL[:k]; bl = base_l[:k]
        p32 = replay(d, False, ws); pbf = replay(d, True, ws)
        band = abs((sum(pbf) - sum(p32)) / len(pbf))
        g32 = (sum(p32) - sum(bl)) / len(bl); gbf = (sum(pbf) - sum(bl)) / len(bl)
        m3 = paired_margin(bl, p32)[0]; mnew = paired_margin(bl, pbf, bf16_band=band)[0]
        old = judge_paired(bl, pbf, sigmas=0.0, floor=1e-4)[0]
        new = judge_paired(bl, pbf, bf16_band=band)[0]
        print(f"{label:>26} {k:>8} {g32:>+10.6f} {gbf:>+10.6f} {band:>9.6f} {mnew:>9.6f} {str(old):>6} {str(new):>6}")
