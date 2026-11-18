"""A full round's delta must still be the flat vector the server already merges.

The five v2 paradigms are attached to the real model, and two of them add
client-side tensors (ReLoRA rank-r factors, SOLO/MOSAIC local heads). Before
this was fixed, a round under those paradigms died inside the writer with
"The size of tensor a (256) must match the size of tensor b (192)" (relora) and
"a (32) must match b (0)" (solo/mosaic) -- the extra tensors ran past the
reference vector and shifted every offset after them. These tests pin the
property that matters for a live round: whatever paradigm trained, the uploaded
delta is exactly ``model_size`` values in base-model order, and a ReLoRA update
arrives at the weight position it actually changes.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIM, VOCAB, LAYERS = 16, 32, 4
DEV = torch.device("cpu")


@contextlib.contextmanager
def _quiet():
    """Import/heavy calls must not touch the real stdio (the dashboard closes it)."""
    out, err = io.StringIO(), io.StringIO()
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved


with _quiet():
    import client as C

import crowd_paradigm as CP


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _Fake(nn.Module):
    """Only the slice of SotaGPT the delta writers and the paradigms touch."""

    def __init__(self) -> None:
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


def _server_vector(model) -> torch.Tensor:
    """The base-model flat vector, exactly as the server would hold it."""
    return torch.cat([p.detach().reshape(-1).float()
                      for n, p in model.named_parameters()
                      if not n.startswith("engram.")])


def _write(model, ref) -> bytes:
    sink = io.BytesIO()

    class _Gz:
        def write(self, mv):
            sink.write(bytes(mv))

    C._write_base_delta_bf16(_Gz(), model, ref)
    return sink.getvalue()


def _decode(blob: bytes) -> torch.Tensor:
    import numpy as np

    a = np.frombuffer(blob, dtype=np.int16).copy()
    return torch.from_numpy(a).view(torch.bfloat16).float()


@pytest.mark.parametrize("paradigm", ["", "relora", "solo", "mosaic"])
def test_delta_is_exactly_the_server_layout(paradigm):
    model = _Fake()
    ref = _server_vector(model).clone()
    if paradigm:
        CP.select(paradigm, solo_modules=2, relora_rank=4, mosaic_slices=2,
                  mosaic_local_steps=1).attach(model)
    blob = _write(model, ref)
    assert len(blob) == ref.numel() * 2, f"{paradigm or 'stock'} emitted {len(blob)} bytes"


def test_untouched_paradigms_emit_a_zero_delta():
    for paradigm in ("solo", "mosaic"):
        model = _Fake()
        ref = _server_vector(model).clone()
        CP.select(paradigm, solo_modules=2, mosaic_slices=2, mosaic_local_steps=1).attach(model)
        delta = _decode(_write(model, ref))
        assert torch.all(delta == 0), f"{paradigm} changed weights without training"


def test_relora_adapter_is_folded_into_its_weight_position():
    model = _Fake()
    ref = _server_vector(model).clone()
    obj = CP.select("relora", relora_rank=4)
    obj.attach(model)
    lora = C._lora_layers(model)
    assert lora, "no LoRALinear attached"
    # Make the adapter non-trivial, then take the exact fused delta.
    with torch.no_grad():
        for mod in lora.values():
            mod.A.normal_(0.0, 0.2)
            mod.B.normal_(0.0, 0.2)
    fused = {name: mod.delta_weight().detach().reshape(-1).float().clone()
             for name, mod in lora.items()}
    # The writer frees each parameter as it streams it (the client discards the
    # model after a round), so the layout has to be captured first.
    layout = [(n, p.numel()) for n, p in model.named_parameters() if C._is_server_param(n, lora)]

    delta = _decode(_write(model, ref))

    # Walk the base layout and check the wrapped modules' weight slices.
    o = 0
    checked = 0
    for n, size in layout:
        s = size
        if n.endswith(".base.weight"):
            path = n[: -len(".base.weight")]
            want = fused[path]
            got = delta[o:o + s]
            assert torch.allclose(got, want, atol=2e-2), f"{path}: adapter delta not in the wire vector"
            checked += 1
        o += s
    assert checked == len(lora)


def test_relora_round_survives_load_base_weights():
    """The next round's download must land on the base weights, not the adapters."""
    model = _Fake()
    obj = CP.select("relora", relora_rank=4)
    obj.attach(model)
    lora = C._lora_layers(model)
    before = {name: mod.A.detach().clone() for name, mod in lora.items()}
    target = torch.full((sum(p.numel() for n, p in model.named_parameters()
                              if C._is_server_param(n, lora)),), 0.5)
    C.SotaGPT.load_base_weights(model, target)
    for n, p in model.named_parameters():
        if not C._is_server_param(n, lora):
            continue
        assert torch.allclose(p.detach().reshape(-1).float(), torch.full((p.numel(),), 0.5))
    for name, mod in lora.items():
        assert torch.equal(mod.A.detach(), before[name]), f"{name} adapter was overwritten by the download"
