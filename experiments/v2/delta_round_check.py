"""Does the delta writer survive a paradigm that adds parameters?

The server merges one flat vector of exactly ``model_size`` values in base-model
order. Two of the v2 paradigms add client-side tensors (ReLoRA adapters, the SOLO
and MOSAIC local heads), so this probe captures what ``_write_base_delta_bf16``
actually does once they are attached -- before anything is changed.
"""

import contextlib
import io
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/app/crowdgpt/CrowdGPT")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import client as C
import crowd_paradigm as CP

DIM, VOCAB, LAYERS = 16, 32, 4
DEV = torch.device("cpu")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x):
        return torch.tanh(self.lin(x))


class _Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_device = DEV
        self.accel_device = DEV
        self.wte = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([_Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight

    def embed(self, idx, head_w=None):
        return self.wte(idx)

    def run_blocks(self, x, start=0, stop=None, use_chunked=True, use_ckpt=True, stream_dtype=None):
        stop = LAYERS if stop is None else stop
        for i in range(max(0, int(start)), stop):
            x = self.blocks[i](x)
        return x

    def head_weight(self, dtype=None):
        return self.lm_head.weight


def base_vector(model):
    return torch.cat([p.detach().reshape(-1).float() for n, p in model.named_parameters()
                      if not n.startswith("engram.")])


class _Sink:
    def __init__(self):
        self.n = 0

    def write(self, mv):
        self.n += len(bytes(mv))


for paradigm in ("", "relora", "solo", "mosaic"):
    model = _Fake()
    ref = base_vector(model).clone()
    if paradigm:
        obj = CP.select(paradigm, solo_modules=2, relora_rank=4, mosaic_slices=2,
                        mosaic_local_steps=1, dust_population=2, dust_sigma=0.05)
        info = obj.attach(model)
    else:
        info = None
    names = [n for n, _ in model.named_parameters()]
    sink = _Sink()
    try:
        C._write_base_delta_bf16(sink, model, ref)
        result = f"OK bytes={sink.n} expected={ref.numel() * 2}"
    except Exception as exc:  # noqa: BLE001 - the point is to capture the failure
        result = f"{type(exc).__name__}: {exc}"
    print(f"{(paradigm or 'stock'):8s} params={len(names):2d} ref={ref.numel():5d} "
          f"base_name_kept={'blocks.0.lin.weight' in names} -> {result}")
    if paradigm == "relora":
        print("        relora param names:", [n for n in names if n.endswith(('.A', '.B'))][:2],
              "| base renamed to:", [n for n in names if ".base." in n][:2])
