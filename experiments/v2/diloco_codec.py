"""DiLoCo: how much upload bandwidth can we delete for how much loss?

K workers train independently for `inner_steps` AdamW steps, then the server
averages their weights and applies one outer Nesterov-SGD step. The only thing
that crosses the network is that average, once per round. So the currency is
"uploads per million tokens" and the question is what it costs in held-out loss.
"""
import os, sys, time
sys.path.insert(0, '/tmp/crowd-v15')
from crowd_v15.codec import compress_frame, decompress_frame, encode_dense, decode_dense
import numpy as np
import torch
import torch.nn as nn

V, D, L, H, T, B = 4096, 128, 4, 4, 256, 8
DEV = torch.device(os.environ.get("SOLO_DEV", "cuda"))
K = int(os.environ.get("DILOCO_WORKERS", 4))
TARGET_TOKENS = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 8_000_000
CODEC = os.environ.get('DILOCO_CODEC', 'none')
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
    rounds = max(1, TARGET_TOKENS // (inner_steps * K * B * T))
    stale = int(os.environ.get("DILOCO_STALE", "0"))
    torch.manual_seed(0)
    globe = Tiny().to(DEV)
    opt_o = torch.optim.SGD(globe.parameters(), lr=0.7, momentum=0.9, nesterov=True)
    workers = [Tiny().to(DEV) for _ in range(K)]
    inners = [torch.optim.AdamW(w.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
              for w in workers]
    history = []
    curve = [evaluate(globe)]
    wire = 0
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
        # Worker 0 becomes a straggler: its contribution comes from `stale` rounds ago.
        src = workers
        if stale > 0 and len(history) >= stale:
            src = [history[-stale]] + list(workers[1:])
        with torch.no_grad():
            for name, p_ in globe.named_parameters():
                acc = torch.zeros_like(p_)
                for w in src:
                    sd = w.state_dict() if hasattr(w, "state_dict") else w
                    acc += sd[name].float()
                g = (p_.float() - acc / K).reshape(-1)
                if CODEC == 'dense':
                    blob = encode_dense(g.cpu().numpy())
                    back = decode_dense(blob, g.numel())
                    g = torch.from_numpy(back).to(p_.device); wire += len(blob)
                elif CODEC != 'none':
                    arr = g.cpu().numpy()
                    blob = compress_frame(arr, residual=(CODEC == 'ngd1r'))
                    back = decompress_frame(blob).astype(np.float32)
                    g = torch.from_numpy(back).to(p_.device); wire += len(blob)
                p_.grad = g.view_as(p_)
            opt_o.step()
        if stale > 0:
            history.append({k: v.detach().clone() for k, v in workers[0].state_dict().items()})
        if (r + 1) % max(1, rounds // 20) == 0:
            curve.append(evaluate(globe))
    dt = time.perf_counter() - t0
    toks = rounds * inner_steps * K * B * T
    print(f"inner={inner_steps:4d} stale={stale:2d} codec={CODEC:6s} wire={wire/1e6:.1f}MB "
          f"({wire/toks:.4f} B/token) uploads={rounds:5d} tokens={toks:,} "
          f"held-out {curve[-1]:.4f}  {toks/dt:9,.0f} tok/s  {dt:6.1f}s", flush=True)
    return curve[-1], rounds, toks


print(f"K={K} target tokens={TARGET_TOKENS:,}  model {sum(p.numel() for p in Tiny().parameters())/1e6:.2f}M  dev={DEV}", flush=True)
for inner in INNER:
    run(inner)
