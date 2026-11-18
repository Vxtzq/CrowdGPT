"""End-to-end smoke for the v2 paradigm selector: backprop / SOLO / ReLoRA."""
import os, sys, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/app/crowdgpt/CrowdGPT")
import crowd_paradigm as CP

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device(os.environ.get("PE_DEV", "cuda"))
STEPS = int(os.environ.get("PE_STEPS", 40))
TRAIN, VAL = "/tmp/v15data/tok/train.bin", "/tmp/v15data/tok/val.bin"
train = np.memmap(TRAIN, dtype=np.uint16, mode="r")
val = np.memmap(VAL, dtype=np.uint16, mode="r")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D); self.up = nn.Linear(D, 4 * D); self.down = nn.Linear(4 * D, D)

    def forward(self, x, cos=None, sin=None, use_chunked=True):
        b, t, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, -1)
        q = q.view(b, t, H, D // H).transpose(1, 2)
        k = k.view(b, t, H, D // H).transpose(1, 2)
        v = v.view(b, t, H, D // H).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, D))
        return x + self.down(torch.nn.functional.silu(self.up(self.ln2(x))))


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
        stop = len(self.blocks) if stop is None else stop
        for i in range(int(start), int(stop)):
            x = self.blocks[i](x)
        return x

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        return w if dtype is None else w.to(dtype)

    def forward(self, idx):
        return self.lm_head(self.ln_f(self.run_blocks(self.embed(idx))))


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    off = torch.randint(0, train.size - T - 1, (B,), generator=g).numpy()
    x = np.stack([train[o:o + T] for o in off]).astype(np.int64)
    y = np.stack([train[o + 1:o + T + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)


EVAL = []
for i in range(8):
    o = i * (T + 1)
    EVAL.append((torch.from_numpy(np.stack([val[o:o + T]]).astype(np.int64)).to(DEV),
                 torch.from_numpy(np.stack([val[o + 1:o + T + 1]]).astype(np.int64)).to(DEV)))


def held(model):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for x, y in EVAL:
            lo = model(x)
            tot += float(torch.nn.functional.cross_entropy(lo[:, :-1].float().reshape(-1, V), y[:, :-1].reshape(-1)))
    model.train(); return tot / len(EVAL)


for name in ("backprop", "solo", "relora"):
    torch.manual_seed(0)
    model = Tiny().to(DEV)
    par = CP.select(name, solo_modules=2, relora_rank=8)
    info = par.attach(model)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 3e-3 if name == "relora" else 1e-2
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    before = held(model)
    t0 = time.perf_counter()
    for step in range(STEPS):
        x, y = batch(step)
        opt.zero_grad(set_to_none=True)
        par.step(model, x, y, loss_chunk=256)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if name == "relora" and (step + 1) % 20 == 0:
            CP.merge_and_restart(model)
    after = held(model)
    dt = time.perf_counter() - t0
    print(f"{name:9s} trainable={sum(p.numel() for p in params)/1e3:7.1f}k  "
          f"held {before:.4f} -> {after:.4f}  ({before - after:+.4f})  "
          f"{STEPS*B*T/dt:7.0f} tok/s  {dt:5.1f}s  attach={info}", flush=True)
