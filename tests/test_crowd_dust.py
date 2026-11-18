"""DUST must produce a usable gradient direction without calling backward."""
from __future__ import annotations

import sys
from pathlib import Path

import unittest

# The repo's CI job is deliberately standard-library only (no pytest, no torch),
# and runs `python -m unittest discover`. Skip this module there instead of
# erroring on import; the tests run under pytest locally and in any env with deps.
try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
    import torch  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_dust import DustConfig, DustError, derive_seed, dust_step, linear_layers  # noqa: E402

V, D = 64, 32


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, 64)
        self.fc2 = nn.Linear(64, V)

    def forward(self, idx):
        h = torch.tanh(self.fc1(idx))
        return self.fc2(h)          # logits over V, no time dim


def batch(n=4, t=6):
    torch.manual_seed(0)
    x = torch.randn(n, t, D)
    y = torch.randint(0, V, (n, t))
    return x, y


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))


def test_dust_fills_every_linear_grad():
    model = MLP()
    x, y = batch()
    loss = dust_step(model, x, y, cfg=DustConfig(population=8), loss_fn=loss_fn)
    assert loss > 0
    for name, layer in linear_layers(model).items():
        assert layer.weight.grad is not None, name
        assert torch.isfinite(layer.weight.grad).all(), name
        assert layer.weight.grad.abs().sum() > 0, name


def test_dust_direction_correlates_with_backprop():
    model = MLP()
    x, y = batch()
    dust_step(model, x, y, cfg=DustConfig(population=400, sigma=0.05, chunk=20), loss_fn=loss_fn)
    d = torch.cat([layer.weight.grad.reshape(-1) for layer in linear_layers(model).values()])

    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    g = torch.cat([layer.weight.grad.reshape(-1) for layer in linear_layers(model).values()])

    cos = float(F.cosine_similarity(d[None, :], g[None, :]))
    assert cos > 0.2, cos


def test_dust_is_deterministic_for_a_seed():
    x, y = batch()
    a, b = MLP(), MLP()
    b.load_state_dict(a.state_dict())
    dust_step(a, x, y, cfg=DustConfig(population=4, seed=7), loss_fn=loss_fn)
    dust_step(b, x, y, cfg=DustConfig(population=4, seed=7), loss_fn=loss_fn)
    for (n1, l1), (n2, l2) in zip(linear_layers(a).items(), linear_layers(b).items()):
        assert torch.equal(l1.weight.grad, l2.weight.grad), n1


def test_seed_derivation_changes_with_layer_and_draw():
    assert derive_seed(1, 2, 3, "fc1") != derive_seed(1, 2, 3, "fc2")
    assert derive_seed(1, 2, 3, "fc1") != derive_seed(1, 2, 4, "fc1")
    assert derive_seed(1, 2, 3, "fc1") == derive_seed(1, 2, 3, "fc1")


def test_bad_config_is_rejected():
    for kwargs in ({"population": 0}, {"sigma": 0.0}, {"chunk": 0}):
        with pytest.raises(DustError):
            DustConfig(**kwargs)


def test_model_without_linears_is_rejected():
    class Empty(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(DustError):
        dust_step(Empty(), torch.zeros(1, 2, D), torch.zeros(1, 2, dtype=torch.long))
