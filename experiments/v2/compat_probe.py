"""Which of ReLoRA / SOLO / MOSAIC actually compose, and which silently do nothing?"""
import sys
sys.path.insert(0, "/app/crowdgpt/CrowdGPT")
sys.path.insert(0, "/app/crowdgpt/CrowdGPT/tests")

import torch
import test_client_delta_layout as T
import crowd_paradigm as CP


class Fake(T._Fake):
    """_Fake plus the forward the loss-based paradigms call."""

    def forward(self, idx):
        h = self.run_blocks(self.embed(idx))
        return self.ln_f(h) @ self.head_weight().t()


def batch():
    x = torch.randint(0, T.VOCAB, (2, 8))
    return x, x.clone()


def cfg(name, **kw):
    base = dict(solo_modules=2, relora_rank=4, mosaic_slices=2,
                mosaic_local_steps=2, mosaic_inner_lr=3e-3, dust_population=8)
    base.update(kw)
    return CP.select(name, **base)


def run(label, order):
    torch.manual_seed(0)
    model = Fake()
    ref = T._server_vector(model).clone()
    try:
        for name in order:
            cfg(name).attach(model)
        last = cfg(order[-1])
        loss = last.step(model, *batch())
        # MOSAIC applies its own outer step inside step(); the others only
        # accumulate gradients and expect the client's optimizer to run.
        if last.name in ("backprop", "solo", "relora", "dust"):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            opt.step()
        blob = T._write(model, ref)
        # _decode returns the *delta* the server would merge, not the weights.
        server_delta = T._decode(blob)
        nz = int((server_delta != 0).sum())
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"{label:16s} loss={loss:7.4f} |d|={float(server_delta.norm()):9.5f} "
              f"nz={nz:4d}/{server_delta.numel():4d} trainable={len(trainable):3d}")
    except Exception as exc:
        print(f"{label:16s} FAILED  {type(exc).__name__}: {exc}")


print(f"tiny model: DIM={T.DIM} VOCAB={T.VOCAB} layers={T.LAYERS}")
for label, order in [
    ("backprop",      ["backprop"]),
    ("solo",          ["solo"]),
    ("relora",        ["relora"]),
    ("mosaic",        ["mosaic"]),
    ("solo+mosaic",   ["solo", "mosaic"]),
    ("mosaic+solo",   ["mosaic", "solo"]),
    ("relora+mosaic", ["relora", "mosaic"]),
    ("mosaic+relora", ["mosaic", "relora"]),
    ("relora+solo",   ["relora", "solo"]),
    ("solo+relora",   ["solo", "relora"]),
]:
    run(label, order)
