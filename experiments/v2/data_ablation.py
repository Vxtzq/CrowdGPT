"""Data ablation: does deduplicating the token stream change what a token buys?

FineWeb-Edu is quality-filtered but not deduplicated. This measures the duplicate
rate of the corpus at 128-token granularity and then trains identical arms on the
raw stream versus the deduplicated stream at fixed compute.
"""
import hashlib, os, sys, time
import numpy as np
import torch
import torch.nn as nn

V, D, Lh, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CHUNK = 128
train = np.memmap("/tmp/v15data/tok/train.bin", dtype=np.uint16, mode="r")
val = np.memmap("/tmp/v15data/tok/val.bin", dtype=np.uint16, mode="r")

# --- dedup: keep the first occurrence of every 128-token chunk --------------
t0 = time.perf_counter()
n_chunks = train.size // CHUNK
raw_bytes = np.asarray(train[: n_chunks * CHUNK]).tobytes()
seen, keep = set(), []
for i in range(n_chunks):
    h = hashlib.blake2b(raw_bytes[i * CHUNK * 2:(i + 1) * CHUNK * 2], digest_size=8).digest()
    if h not in seen:
        seen.add(h); keep.append(i)
dup_rate = 1.0 - len(keep) / n_chunks
dedup = np.concatenate([np.asarray(train[i * CHUNK:(i + 1) * CHUNK]) for i in keep])
print(f"chunks {n_chunks:,}  unique {len(keep):,}  duplicate rate {dup_rate:.2%}  "
      f"tokens {n_chunks*CHUNK:,} -> {dedup.size:,}  ({time.perf_counter()-t0:.1f}s)", flush=True)


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
        self.blocks = nn.ModuleList([Block() for _ in range(Lh)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def get(stream, seed, step):
    g = torch.Generator().manual_seed(seed * 1_000_003 + step)
    off = torch.randint(0, stream.size - T - 1, (B,), generator=g).numpy()
    x = np.stack([stream[o:o + T] for o in off]).astype(np.int64)
    y = np.stack([stream[o + 1:o + T + 1] for o in off]).astype(np.int64)
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


def run(label, stream, seed):
    torch.manual_seed(seed)
    m = Tiny().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.01)
    t0 = time.perf_counter()
    curve = [evaluate(m)]
    for s in range(STEPS):
        x, y = get(stream, seed, s)
        opt.zero_grad(set_to_none=True)
        ce(m(x), y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (s + 1) % (STEPS // 4) == 0:
            curve.append(evaluate(m))
    toks = STEPS * B * T
    print(f"{label:6s} eval {['%.4f' % c for c in curve]}  final {curve[-1]:.4f}  "
          f"{toks/(time.perf_counter()-t0):9,.0f} tok/s  {time.perf_counter()-t0:5.1f}s", flush=True)
    return curve[-1]


print(f"L={Lh} d={D} V={V}  steps={STEPS}  tokens/arm={STEPS*B*T:,}", flush=True)
for seed in (0, 1):
    a = run(f"raw{seed}", train, seed)
    b = run(f"dedup{seed}", dedup, seed)
    print(f"  seed {seed}: dedup - raw = {b - a:+.4f}", flush=True)
