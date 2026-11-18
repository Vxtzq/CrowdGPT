"""How much of a real delta survives the shipped codecs?

Not a proxy: we train, take the actual weight delta, compress it exactly the way
the client does, decode it, apply it, and score held-out loss. The question is
how many bytes buy how much of the update.
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/crowd-v15")
from crowd_v15.codec import (CODECS, compression_ratio, compress_frame,
                            decode_dense, decompress_frame, encode_dense)

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device("cpu")
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


def batch(step):
    g = torch.Generator().manual_seed(step)
    off = torch.randint(0, train.size - T - 1, (B,), generator=g).numpy()
    x = np.stack([train[o:o + T] for o in off]).astype(np.int64)
    y = np.stack([train[o + 1:o + T + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


EVAL = []
for i in range(8):
    o = i * (T + 1)
    EVAL.append((torch.from_numpy(np.stack([val[o:o + T]]).astype(np.int64)),
                 torch.from_numpy(np.stack([val[o + 1:o + T + 1]]).astype(np.int64))))


def evaluate(model):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for x, y in EVAL:
            lg = model(x)[:, :-1]
            tot += float(torch.nn.functional.cross_entropy(lg.reshape(-1, V), y[:, 1:].reshape(-1)))
    model.train(); return tot / len(EVAL)


torch.manual_seed(0)
model = Tiny()
base = {k: v.detach().clone() for k, v in model.named_parameters()}
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
t0 = time.perf_counter()
for step in range(300):
    x, y = batch(step)
    opt.zero_grad(set_to_none=True)
    lg = model(x)[:, :-1]
    torch.nn.functional.cross_entropy(lg.reshape(-1, V), y[:, 1:].reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"trained 300 steps in {time.perf_counter()-t0:.1f}s, held-out after training {evaluate(model):.4f}")

delta = {k: (v.detach() - base[k]) for k, v in model.named_parameters()}
flat = torch.cat([d.reshape(-1) for _, d in sorted(delta.items())]).numpy().astype(np.float32)
print(f"delta {flat.size:,} values, {flat.nbytes/1e6:.2f} MB fp32; rms {float(np.sqrt((flat**2).mean())):.6f}")


def apply_and_score(vec):
    i = 0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            n = p.numel()
            p.copy_(torch.from_numpy(base[name].numpy().reshape(-1) + vec[i:i + n]).view_as(p))
            i += n
    return evaluate(model)


print(f"\n{'codec':8s} {'bytes':>10s} {'ratio':>7s} {'cosine':>8s} {'held-out':>9s}")
print(f"{'none':8s} {flat.nbytes:10,d} {1.0:7.2f} {1.0:8.4f} {apply_and_score(flat):9.4f}")
for codec in CODECS:
    residual = codec == "ngd1r"
    if codec == "dense":
        # dense is raw bf16, not the 1-bit sign path: compress_frame(residual=False)
        # is ngd1, which is why the first table had two identical rows.
        blob = encode_dense(flat)
        back = decode_dense(blob, flat.size).astype(np.float32)
    else:
        blob = compress_frame(flat, residual=residual)
        back = decompress_frame(blob).astype(np.float32)
    cos = float(np.dot(flat, back) / (np.linalg.norm(flat) * np.linalg.norm(back) + 1e-12))
    print(f"{codec:8s} {len(blob):10,d} {compression_ratio(flat.nbytes, blob):7.1f} "
          f"{cos:8.4f} {apply_and_score(back):9.4f}")
