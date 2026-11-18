"""One real `client.py` step per paradigm, on the actual 1.01 B SotaGPT.

The unit tests use a fake model, which is exactly how the ReLoRA adapter device
bug slipped through before. This runs the production model class through
`_attach_paradigm` + `_train_step`, so the real surface (`embed`, `run_blocks`,
`ln_f`, `head_weight`, `main_device`) is what is being exercised.
"""
import os
import sys
import time

sys.path.insert(0, "/app/crowdgpt/CrowdGPT")
import torch  # noqa: E402

import client as C  # noqa: E402

DEV = torch.device(os.environ.get("SMOKE_DEV", "cuda" if torch.cuda.is_available() else "cpu"))
B, T = 2, 64
name = os.environ.get("CROWDGPT_PARADIGM", "")


class Ladder:
    loss_chunk = 64

    def step_down(self, emit=None):
        return False


def emit(kind, msg=None):
    print(f"   {kind}: {msg}")


torch.manual_seed(0)
model = C.SotaGPT(C.N_LAYERS, DEV, bs=B, sl=T)
n = sum(p.numel() for _, p in model.named_parameters())
par = C._attach_paradigm(model, emit)
print(f"paradigm={name or '(stock)'} params={n/1e6:.1f}M dev={DEV} attached={par is not None}")

x = torch.randint(0, C.VOCAB_SIZE, (B, T), device=DEV)
y = torch.randint(0, C.VOCAB_SIZE, (B, T), device=DEV)
t0 = time.perf_counter()
loss, cards = C._train_step(model, x, y, T, False, None, 1.0, False, Ladder(), emit)
torch.cuda.synchronize() if DEV.type == "cuda" else None
dt = time.perf_counter() - t0

grads = [(nm, p.grad) for nm, p in model.named_parameters() if p.grad is not None]
nz = sum(1 for _, g in grads if float(g.abs().sum()) > 0)
print(f"loss={float(loss.item()):.4f} finite={torch.isfinite(loss).item()} grads={len(grads)} nonzero={nz} "
      f"step={dt*1000:.0f}ms cards={len(cards)}")
tail = sorted({nm.split('.')[0] for nm, _ in grads})
print("grad roots:", tail[:8], "…" if len(tail) > 8 else "")
