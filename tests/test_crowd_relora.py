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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crowd_relora import (  # noqa: E402
    LoRALinear,
    ReLoRASchedule,
    attach,
    merge,
    trainable_parameters,
)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def test_attach_is_output_preserving():
    m = _model()
    x = torch.randn(4, 16)
    before = m(x).clone()
    n = attach(m, rank=4)
    assert n == 2
    assert torch.allclose(m(x), before, atol=1e-6)


def test_only_adapters_are_trainable():
    m = _model()
    attach(m, rank=4)
    trainable = trainable_parameters(m)
    assert len(trainable) == 4  # two modules x (A, B)
    assert all(p.requires_grad for p in trainable)
    frozen = [p for n, p in m.named_parameters() if not (n.endswith(".A") or n.endswith(".B"))]
    assert all(not p.requires_grad for p in frozen)


def test_merge_is_output_preserving_and_zeroes_adapters():
    m = _model()
    attach(m, rank=4)
    # give the adapters a non-trivial update
    for p in trainable_parameters(m):
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(4, 16)
    before = m(x).clone()
    merged = merge(m)
    assert merged == 2
    assert torch.allclose(m(x), before, atol=1e-5)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            assert torch.count_nonzero(mod.A) == 0
            assert torch.count_nonzero(mod.B) == 0


def test_schedule_jagged_lr_and_restart():
    s = ReLoRASchedule(base_lr=3e-4, segment_lr=3e-3, inner_steps=10)
    assert s.lr_at(0) == pytest.approx(3e-3)
    assert s.lr_at(9) < s.lr_at(0)
    assert s.lr_at(10) == pytest.approx(3e-3)   # restart resets the rate
    assert not s.wants_restart(0)
    assert s.wants_restart(10) and s.wants_restart(20)
    with pytest.raises(ValueError):
        s.lr_at(-1)


def test_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), rank=0)


def test_merge_refuses_trainable_base():
    m = _model()
    attach(m, rank=2)
    lay = next(x for x in m.modules() if isinstance(x, LoRALinear))
    lay.base.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        merge(m)
