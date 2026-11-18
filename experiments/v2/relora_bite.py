"""ReLoRA bite: how many bytes does a low-rank delta need for the same loss?

DiLoCo already cut uploads 25x. This asks the next question: each upload is still a
full parameter tensor. ReLoRA's premise is that the *update* is low rank, so we
factor the transmitted pseudo-gradient per weight matrix and send the factors.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
K = 4
INNER = int(os.environ.get("RELORA_INNER", 25))
RANK = int(os.environ.get("RELORA_RANK", 0))     # 0 = full-rank delta
TARGET = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 8_000_000
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


def compress_delta(g, rank):
    """Return (reconstructed delta, bytes actually put on the wire)."""
    if rank == 0:
        return g, g.numel() * 2                     # bf16 full rank
    out = g.clone(); wire = 0
    # factorise per weight, in the same order the flat vector was built
    off = 0
    for name, p in globe.named_parameters():
        n = p.numel(); sl = g[off:off + n]; off += n
        if p.dim() != 2:
            wire += n * 2
            continue
        m = sl.reshape(p.shape)
        u, s, vh = torch.linalg.svd(m.float(), full_matrices=False)
        r = min(rank, s.numel())
        rec = (u[:, :r] * s[:r]) @ vh[:r, :]
        out[p.shape[0] * 0 + (off - n):off] = rec.reshape(-1).to(out.dtype)
        wire += 2 * r * (p.shape[0] + p.shape[1])   # bf16 factors
    return out, wire


rounds = max(1, TARGET // (INNER * K * B * T))
torch.manual_seed(0)
globe = Tiny().to(DEV)
opt_o = torch.optim.SGD(globe.parameters(), lr=0.7, momentum=0.9, nesterov=True)
workers = [Tiny().to(DEV) for _ in range(K)]
inners = [torch.optim.AdamW(w.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01) for w in workers]
wire = 0
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
    with torch.no_grad():
        for name, p in globe.named_parameters():
            acc = torch.zeros_like(p)
            for w in workers:
                acc += w.state_dict()[name].float()
            p.grad = p.float() - acc / K
        flat = torch.cat([p.grad.reshape(-1) for p in globe.parameters()]).detach()
        rec, wbytes = compress_delta(flat, RANK)
        wire += wbytes
        off = 0
        for p in globe.parameters():
            n = p.numel()
            p.grad = rec[off:off + n].reshape(p.shape).to(p.dtype)
            off += n
        opt_o.step()
tok = rounds * INNER * K * B * T
dt = time.perf_counter() - t0
loss = evaluate(globe)
model_bytes = sum(p.numel() for p in globe.parameters())
print(f"rank={RANK:3d} inner={INNER} uploads={rounds:4d} tokens={tok:,} "
      f"wire={wire/1e6:8.3f}MB ({wire/tok:8.4f} B/token) held-out {loss:.4f} "
      f"{tok/dt:9,.0f} tok/s  (full-rank bytes would be {rounds*model_bytes*2/1e6:.1f}MB)",
      flush=True)
