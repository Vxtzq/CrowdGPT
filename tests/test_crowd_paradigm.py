"""The runtime paradigm switch: selection, preparation, stepping."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_paradigm import (PARADIGMS, PENDING, Paradigm, ParadigmConfig,
                            ParadigmError, merge_and_restart, select)
from test_crowd_solo import _FakeModel, _batch


class _Forwarding(_FakeModel):
    """The fake model plus the forward backprop needs."""

    def forward(self, idx):
        return torch.nn.functional.linear(
            self.ln_f(self.run_blocks(self.embed(idx))), self.head_weight())


def test_defaults_to_backprop_and_keeps_every_parameter():
    p = select()
    model = _Forwarding()
    info = p.attach(model)
    assert p.name == "backprop" and info["trainable"] == "all"
    x, y = _batch(2, 4)
    assert isinstance(p.step(model, x, y), float)


def test_solo_prepares_heads_and_steps_through_them():
    p = select("solo", solo_modules=2)
    model = _Forwarding()
    info = p.attach(model)
    assert info["paradigm"] == "solo" and len(info["splits"]) == 2
    assert hasattr(model, "solo_heads")
    x, y = _batch(2, 4)
    loss = p.step(model, x, y)
    assert loss > 0
    assert model.solo_heads["0"].phi.weight.grad is not None


def test_relora_freezes_the_base_and_steps_only_the_adapters():
    p = select("relora", relora_rank=4)
    model = _Forwarding()
    info = p.attach(model)
    assert info["paradigm"] == "relora" and info["adapters"] > 0
    base = model.blocks[0].lin.base.weight.detach().clone()
    x, y = _batch(2, 4)
    p.step(model, x, y)
    assert torch.equal(model.blocks[0].lin.base.weight.detach(), base)
    assert model.blocks[0].lin.B.grad is not None
    # the upload filter is adapters-only
    f = p.upload_filter()
    assert f("blocks.0.lin.A") and not f("blocks.0.lin.base.weight")


def test_merge_and_restart_is_a_noop_for_non_relora_models():
    model = _Forwarding()
    assert merge_and_restart(model) == 0


def test_every_named_paradigm_is_selectable():
    # DUST (V2I-4) and MOSAIC (V2I-3) are both implemented now, so nothing is pending.
    for name in PARADIGMS:
        assert select(name).name == name


def test_unknown_and_invalid_configs_are_rejected():
    with pytest.raises(ParadigmError, match="unknown paradigm"):
        select("backprop2")
    with pytest.raises(ParadigmError, match="at least two modules"):
        select("solo", solo_modules=1)
    with pytest.raises(ParadigmError, match="rank must be positive"):
        select("relora", relora_rank=0)
    assert PARADIGMS == ("backprop", "solo", "relora", "dust", "mosaic")
    assert PENDING == ()


def test_mosaic_ships_the_measured_optimum_and_uses_it(monkeypatch):
    """The production defaults must be the configuration that was measured.

    The rates in this file were originally the harness presets (H=30, inner
    3e-3, outer 0.7), and at that configuration MOSAIC measured 6.4-7.2 at
    65 M. Tuning the outer rate alone is worth 0.28 nats (1.0 -> 0.15) and the
    local-step count matters by 0.46 (H=1 6.2730 vs H=5 5.8121), so shipping the
    old defaults would ship the untuned method.
    """
    import crowd_mosaic

    cfg = ParadigmConfig(name="mosaic")
    assert cfg.mosaic_local_steps == 5
    assert cfg.mosaic_inner_lr == 1e-3
    assert cfg.mosaic_outer_lr == 0.15
    assert cfg.mosaic_outer_momentum == 0.9
    assert cfg.mosaic_share_trunk is True

    seen = {}
    real_local = crowd_mosaic.local_delta
    real_agg = crowd_mosaic.aggregate

    def spy_local(*a, **kw):
        seen["share_trunk"] = kw.get("share_trunk")
        seen["lr"] = kw.get("lr")
        seen["steps"] = kw.get("steps")
        return real_local(*a, **kw)

    def spy_agg(*a, **kw):
        cfg_ = kw.get("cfg")
        seen["outer_lr"] = cfg_.outer_lr
        seen["outer_momentum"] = cfg_.outer_momentum
        return real_agg(*a, **kw)

    monkeypatch.setattr(crowd_mosaic, "local_delta", spy_local)
    monkeypatch.setattr(crowd_mosaic, "aggregate", spy_agg)

    model = _Forwarding()
    torch.manual_seed(0)
    para = Paradigm(ParadigmConfig(name="mosaic"))
    para.attach(model)
    x, y = _batch(0)
    para.step(model, x, y, loss_scale=1.0, loss_chunk=8)

    assert seen["share_trunk"] is True
    assert seen["lr"] == pytest.approx(1e-3)
    assert seen["steps"] == 5
    assert seen["outer_lr"] == pytest.approx(0.15)
    assert seen["outer_momentum"] == pytest.approx(0.9)
