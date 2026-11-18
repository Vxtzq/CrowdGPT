"""The runtime paradigm switch: selection, preparation, stepping."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_paradigm import (PARADIGMS, Paradigm, ParadigmConfig, ParadigmError,
                            merge_and_restart, select)
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


def test_pending_paradigms_refuse_with_the_reason():
    with pytest.raises(ParadigmError, match="V2I-3"):
        select("mosaic")
    with pytest.raises(ParadigmError, match="V2I-4"):
        select("dust")


def test_unknown_and_invalid_configs_are_rejected():
    with pytest.raises(ParadigmError, match="unknown paradigm"):
        select("backprop2")
    with pytest.raises(ParadigmError, match="at least two modules"):
        select("solo", solo_modules=1)
    with pytest.raises(ParadigmError, match="rank must be positive"):
        select("relora", relora_rank=0)
    assert PARADIGMS == ("backprop", "solo", "relora")
