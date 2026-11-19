"""Tests for the SOLO core (arXiv:2609.35440v1, Algorithm 1).

The properties that matter are structural, not numerical:

* every module keeps its own gradient and none crosses a module boundary;
* the shared readout is read through a stop-gradient, so a local loss cannot
  move it — only the terminal loss trains `W`;
* at `tau = 1` with a current copy the local gradient equals the backprop
  gradient through the same readout (Appendix B of the paper);
* the auxiliary heads are client-local, so the delta writers must skip them.

These run against a tiny fake model rather than `SotaGPT` so they stay fast and
do not need the 564 M-parameter config or a GPU.
"""

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

from crowd_solo import (  # noqa: E402
    DEFAULT_MODULES,
    SoloHead,
    build_solo_heads,
    chunked_solo_ce,
    is_uploaded_param,
    reference_local_gradient,
    solo_step,
    split_layers,
)

def _ce_ref(logits, target):
    """Cross-entropy from primitives, so a polluted ``torch.nn.functional``
    cannot decide the expected value in a suite-wide run."""
    logp = logits.float().log_softmax(-1)
    return -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()


DIM = 16
VOCAB = 32
LAYERS = 4


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _FakeModel(nn.Module):
    """The slice of `SotaGPT` that `solo_step` touches."""

    def __init__(self, modules: int = 2, layers: int = LAYERS, tie: bool = False) -> None:
        super().__init__()
        self.main_device = torch.device("cpu")
        self.wte = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([_Block(DIM) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)
        if tie:
            self.lm_head.weight = self.wte.weight
        self.solo_splits = split_layers(layers, modules)
        self.solo_heads = build_solo_heads(DIM, VOCAB, len(self.solo_splits))
        torch.manual_seed(0)
        for p in self.parameters():
            if p.dim() > 1:
                with torch.no_grad():
                    p.mul_(0.3)

    def embed(self, idx, head_w=None):  # noqa: D401 - mirrors SotaGPT.embed
        return self.wte(idx)

    def run_blocks(self, x, start=0, stop=None, use_chunked=True, use_ckpt=True,
                   stream_dtype=None):
        stop = len(self.blocks) if stop is None else stop
        for i in range(max(0, start), stop):
            x = self.blocks[i](x)
        return x

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        return w if dtype is None else w.to(dtype)


def _batch(batch: int = 2, seq: int = 8):
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (batch, seq))
    y = torch.randint(0, VOCAB, (batch, seq))
    return x, y


def _named_grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


def test_default_modules_is_the_founders_four_chunks():
    assert DEFAULT_MODULES == 4


@pytest.mark.parametrize("layers,modules", [(24, 4), (24, 2), (24, 3), (5, 4), (2, 2)])
def test_split_layers_is_contiguous_and_covers_every_layer(layers, modules):
    splits = split_layers(layers, modules)
    assert splits[0][0] == 0
    assert splits[-1][1] == layers
    for (lo, hi), (nxt_lo, _) in zip(splits, splits[1:]):
        assert hi == nxt_lo
        assert hi > lo
    assert sum(hi - lo for lo, hi in splits) == layers


def test_split_layers_clamps_and_balances():
    assert split_layers(3, 8) == [(0, 1), (1, 2), (2, 3)]
    assert [hi - lo for lo, hi in split_layers(24, 4)] == [6, 6, 6, 6]
    with pytest.raises(ValueError):
        split_layers(0, 2)
    with pytest.raises(ValueError):
        split_layers(4, 0)


def test_chunked_ce_matches_cross_entropy():
    torch.manual_seed(2)
    xs = torch.randn(2, 8, DIM)
    w = torch.randn(VOCAB, DIM) * 0.1
    ys = torch.randint(0, VOCAB, (2, 8))
    want = _ce_ref((xs @ w.T), ys)
    got = chunked_solo_ce(xs, w, ys, pos_step=3)
    assert torch.allclose(got, want, atol=1e-6)

    tau = torch.tensor(1.7)
    bias = torch.randn(VOCAB) * 0.05
    scaled = xs @ w.T * tau + bias
    want_scaled = _ce_ref(scaled, ys)
    assert torch.allclose(chunked_solo_ce(xs, w, ys, tau=tau, bias=bias, pos_step=3),
                          want_scaled, atol=1e-6)

    with pytest.raises(ValueError):
        chunked_solo_ce(torch.randn(2, 4), w, torch.zeros(2, 4, dtype=torch.long))


def test_local_gradient_equals_backprop_through_the_same_readout():
    """Appendix B: at tau = 1 and a current copy, g_hat equals the BP gradient."""
    torch.manual_seed(3)
    xs = torch.randn(2, 6, DIM)
    w = torch.randn(VOCAB, DIM) * 0.2
    ys = torch.randint(0, VOCAB, (2, 6))

    local = reference_local_gradient(xs, w, ys)
    live = xs.detach().clone().requires_grad_(True)
    _ce_ref((live @ w.T), ys).backward()
    assert torch.allclose(local, live.grad, atol=1e-6)


def test_solo_step_trains_every_module_and_every_head():
    model = _FakeModel(modules=2)
    x, y = _batch()
    loss, local = solo_step(model, x, y, splits=model.solo_splits)

    assert loss.dim() == 0 and torch.isfinite(loss)
    assert len(local) == len(model.solo_splits) - 1
    assert all(v > 0 for v in local)

    for name, p in model.named_parameters():
        # The optional skip path (DGL-style input view) is zero-initialised and
        # only fed when a caller supplies the module input, which solo_step does
        # not; every always-active head tensor must still receive a gradient.
        if name.startswith("solo_heads.") and ".skip." not in name:
            assert p.grad is not None, name
    for i, _ in enumerate(model.blocks):
        assert model.blocks[i].lin.weight.grad is not None, f"block {i} got no gradient"


def test_module_gradients_do_not_cross_the_boundary():
    """Module 0's gradient must be the one its own local loss produces alone."""
    model = _FakeModel(modules=2)
    x, y = _batch()
    solo_step(model, x, y, splits=model.solo_splits)
    got = model.blocks[0].lin.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    lo, hi = model.solo_splits[0]
    head = model.solo_heads["0"]
    h = model.run_blocks(model.embed(x), lo, hi)
    w_bar = model.head_weight().detach()
    chunked_solo_ce(head.project(h), w_bar, y, tau=head.tau, bias=head.bias).backward()
    assert torch.allclose(got, model.blocks[0].lin.weight.grad, atol=1e-7)


def test_a_local_loss_cannot_move_the_shared_readout():
    model = _FakeModel(modules=3)
    x, y = _batch()
    lo, hi = model.solo_splits[0]
    head = model.solo_heads["0"]
    h = model.run_blocks(model.embed(x), lo, hi)
    chunked_solo_ce(head.project(h), model.head_weight().detach(), y,
                    tau=head.tau, bias=head.bias).backward()
    assert model.lm_head.weight.grad is None
    # The embedding does collect a gradient, and should: it is module 0's own
    # input, so it belongs to module 0's parameters. Only the *shared readout*
    # is off limits to a local loss.
    assert model.wte.weight.grad is not None


def test_only_the_terminal_loss_writes_the_readout():
    model = _FakeModel(modules=2)
    x, y = _batch()
    solo_step(model, x, y, splits=model.solo_splits)
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) > 0


def test_weight_tied_readout_still_gets_a_gradient():
    model = _FakeModel(tie=True)
    x, y = _batch()
    solo_step(model, x, y, splits=model.solo_splits)
    assert model.wte.weight.grad is not None
    assert model.lm_head.weight is model.wte.weight


def test_solo_heads_are_client_local():
    assert is_uploaded_param("blocks.0.attn.qkv.weight")
    assert is_uploaded_param("lm_head.weight")
    assert not is_uploaded_param("solo_heads.0.phi.weight")
    assert not is_uploaded_param("solo_heads.1.tau")
    assert not is_uploaded_param("engram.table.weight")


def test_head_capacity_is_opt_in_and_off_by_default():
    """A wider head must be an A/B, not a silent change to every SOLO run.

    The single-linear head is the published SOLO form; a hidden layer is the
    DGL-inspired capacity experiment, so it has to be bit-identical when off.
    """
    plain = SoloHead(dim=8, vocab=16)
    assert plain.phi2 is None
    assert not any(n.startswith("phi2") for n, _ in plain.named_parameters())

    h = torch.randn(2, 5, 8)
    assert torch.equal(plain.project(h), plain.phi(plain.ln(h)))

    wide = SoloHead(dim=8, vocab=16, mlp=True)
    assert wide.phi2 is not None
    expected = wide.phi2(torch.nn.functional.gelu(wide.phi(wide.ln(h))))
    assert torch.allclose(wide.project(h), expected)
    assert not torch.allclose(wide.project(h), plain.project(h))


def test_build_solo_heads_forwards_the_capacity_flag():
    assert all(head.phi2 is None for head in build_solo_heads(8, 16, 3).values())
    assert all(head.phi2 is not None
               for head in build_solo_heads(8, 16, 3, mlp=True).values())


def test_solo_step_requires_heads_and_two_modules():
    model = _FakeModel(modules=2)
    x, y = _batch()
    with pytest.raises(ValueError):
        solo_step(model, x, y, splits=[(0, LAYERS)])
    bare = _FakeModel()
    del bare.solo_heads
    with pytest.raises(RuntimeError):
        solo_step(bare, x, y, splits=bare.solo_splits)


def test_loss_scale_scales_both_local_and_terminal_gradients():
    model = _FakeModel(modules=2)
    x, y = _batch()
    solo_step(model, x, y, splits=model.solo_splits, loss_scale=1.0)
    full = model.blocks[-1].lin.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    solo_step(model, x, y, splits=model.solo_splits, loss_scale=0.5)
    half = model.blocks[-1].lin.weight.grad
    assert torch.allclose(half, full * 0.5, atol=1e-7)


# --- module-parallel replicas -------------------------------------------------

def test_uploaded_for_module_keeps_only_its_own_blocks():
    from crowd_solo import uploaded_for_module

    splits = [(0, 6), (6, 12), (12, 18), (18, 24)]
    assert uploaded_for_module("blocks.3.mlp.down.weight", splits, 0)
    assert not uploaded_for_module("blocks.7.mlp.down.weight", splits, 0)
    assert uploaded_for_module("blocks.7.mlp.down.weight", splits, 1)
    assert not uploaded_for_module("blocks.23.attn.qkv.weight", splits, 2)


def test_uploaded_for_module_gives_the_readout_to_the_terminal_module():
    from crowd_solo import uploaded_for_module

    splits = [(0, 6), (6, 12)]
    for name in ("lm_head.weight", "wte.weight", "ln_f.weight"):
        assert uploaded_for_module(name, splits, 1), name
        assert not uploaded_for_module(name, splits, 0), name


def test_uploaded_for_module_never_ships_heads_or_engrams():
    from crowd_solo import uploaded_for_module

    splits = [(0, 6), (6, 12)]
    for module in (0, 1):
        assert not uploaded_for_module("solo_heads.0.phi.weight", splits, module)
        assert not uploaded_for_module("engram.table.weight", splits, module)


def test_uploaded_for_module_without_a_module_is_the_old_behaviour():
    from crowd_solo import uploaded_for_module

    assert uploaded_for_module("blocks.19.attn.qkv.weight", None, None)
    assert uploaded_for_module("lm_head.weight", None, None)


def test_solo_module_step_rejects_a_bad_module_index():
    import pytest

    from crowd_solo import solo_module_step

    model = _FakeModel(modules=2)
    x, y = _batch(2, 4)
    with pytest.raises(ValueError):
        solo_module_step(model, x, y, splits=model.solo_splits, module=2)
    with pytest.raises(ValueError):
        solo_module_step(model, x, y, splits=model.solo_splits, module=-1)


def test_solo_module_step_trains_one_module_and_skips_the_rest():
    from crowd_solo import solo_module_step

    model = _FakeModel(modules=2)
    x, y = _batch(2, 4)
    loss, local = solo_module_step(model, x, y, splits=model.solo_splits, module=1)
    assert loss.item() > 0 and len(local) == 1
    grads = {n: p.grad is not None for n, p in model.named_parameters()}
    assert grads["blocks.2.lin.weight"] and grads["blocks.3.lin.weight"]
    assert not grads["blocks.0.lin.weight"] and not grads["blocks.1.lin.weight"]
    assert grads["lm_head.weight"]          # terminal module owns the readout
    assert not any(k.startswith("solo_heads.") and v for k, v in grads.items())
