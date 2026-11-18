"""The round loop must keep its stock path unless a paradigm is selected.

`client.py` is the production loop, so the only safe way to add a training rule
is: nothing changes unless ``CROWDGPT_PARADIGM`` names one. These tests pin
exactly that — an unset flag leaves the model untouched, a named paradigm
attaches, an unknown one falls back instead of taking the round down, and the
dispatcher routes a step to the attached object.

`client.py` installs a terminal dashboard over ``sys.stdout`` at import and the
old dashboard does not always hand the stream back, so it is imported with
throwaway streams: a test must never let the client close pytest's capture file.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CLIENT = None


@contextlib.contextmanager
def _quiet():
    saved = (sys.stdout, sys.stderr)
    try:
        sys.stdout = sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = saved


def client():
    """Import ``client`` once, without letting it keep the real stdio streams."""
    global _CLIENT
    if _CLIENT is None:
        with _quiet():
            import client as module

        _CLIENT = module
    return _CLIENT


class _Ladder:
    """Minimal stand-in for DegradeLadder (the step never OOMs here)."""

    loss_chunk = 8

    def step_down(self, emit=None):
        return False


def _fake(modules: int = 4):
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wte = torch.nn.Embedding(32, 16)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(modules)])
            self.ln_f = torch.nn.LayerNorm(16)

        def embed(self, idx):
            return self.wte(idx)

        def run_blocks(self, x, start=0, stop=None, use_ckpt=True, **_kw):
            for block in self.blocks[start:stop]:
                x = block(x)
            return x

        def head_weight(self, dtype=None):
            return self.wte.weight

    model = M()
    model.main_device = torch.device("cpu")
    model.accel_device = torch.device("cpu")
    return model


def test_unset_paradigm_leaves_the_stock_path(monkeypatch):
    monkeypatch.delenv("CROWDGPT_PARADIGM", raising=False)
    model = _fake()
    assert client()._attach_paradigm(model) is None
    assert getattr(model, "paradigm_obj", None) is None
    assert client()._paradigm_name() == ""


def test_backprop_by_name_is_also_the_stock_path(monkeypatch):
    monkeypatch.setenv("CROWDGPT_PARADIGM", "backprop")
    model = _fake()
    assert client()._attach_paradigm(model) is None


def test_solo_attaches_modules_and_heads(monkeypatch):
    monkeypatch.setenv("CROWDGPT_PARADIGM", "solo")
    model = _fake(4)
    par = client()._attach_paradigm(model)
    assert par is not None and par.name == "solo"
    assert model.paradigm_obj is par
    assert model.solo_splits == [(0, 2), (2, 4)]
    assert len(model.solo_heads) == 1


def test_relora_attaches_adapters(monkeypatch):
    monkeypatch.setenv("CROWDGPT_PARADIGM", "relora")
    model = _fake(2)
    par = client()._attach_paradigm(model)
    assert par is not None and par.name == "relora"
    names = [n for n, _ in model.named_parameters()]
    assert sum(1 for n in names if ".A" in n or ".B" in n) == 4


def test_unknown_paradigm_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("CROWDGPT_PARADIGM", "nonsense")
    logs = []
    model = _fake()
    assert client()._attach_paradigm(model, lambda kind, msg: logs.append(msg)) is None
    assert getattr(model, "paradigm_obj", None) is None
    assert any("could not attach" in m for m in logs)


def test_train_step_dispatches_to_the_attached_paradigm():
    class _P:
        name = "dust"

        def step(self, model, x, y, *, loss_scale=1.0, loss_chunk=256):
            return 0.25

    model = _fake()
    model.paradigm_obj = _P()
    loss, cards = client()._train_step(model, None, None, 8, False, None, 1.0, False, _Ladder(), None)
    assert float(loss.item()) == pytest.approx(0.25)
    assert cards == []


def test_gradient_accumulation_scale_is_applied_for_self_assigning_paradigms():
    class _P:
        name = "dust"

        def step(self, model, x, y, *, loss_scale=1.0, loss_chunk=256):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            return 0.5

    model = _fake(1)
    model.paradigm_obj = _P()
    client()._train_step(model, None, None, 8, False, None, 0.25, False, _Ladder(), None)
    for name, p in model.named_parameters():
        if name.startswith("blocks."):
            assert torch.allclose(p.grad, torch.full_like(p.grad, 0.25)), name
