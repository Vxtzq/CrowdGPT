"""MOSAIC: slicing, median-clipped uploads, outer step, slice-local training."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_mosaic import (MosaicConfig, MosaicError, aggregate, attach, fit_head, local_delta,  # noqa: E402
                          median_clip, slice_of)
from test_crowd_solo import _FakeModel, _batch  # noqa: E402


class _Forwarding(_FakeModel):
    def forward(self, idx):
        return torch.nn.functional.linear(
            self.ln_f(self.run_blocks(self.embed(idx))), self.head_weight())


def test_attach_covers_every_block_and_builds_k_minus_one_heads():
    model = _Forwarding(modules=2)
    info = attach(model, MosaicConfig(slices=2, local_steps=4))
    assert info["slices"] == [(0, 2), (2, 4)]
    assert info["heads"] == 1
    assert set(model.solo_heads.keys()) == {"0"}
    assert sorted(i for lo, hi in info["slices"] for i in range(lo, hi)) == [0, 1, 2, 3]


def test_slice_of_maps_blocks_and_assigns_the_trunk_ends():
    """Every trained tensor needs a slice.

    Returning None for the trunk used to freeze ``wte`` (and the readout tied to
    it) plus ``ln_f`` for an entire run, which capped the loss near log(V).
    """
    splits = [(0, 2), (2, 4)]
    assert slice_of("blocks.0.lin.weight", splits) == 0
    assert slice_of("blocks.3.lin.weight", splits) == 1
    assert slice_of("wte.weight", splits) == 0
    assert slice_of("ln_f.weight", splits) == 1
    assert slice_of("lm_head.weight", splits) == 1
    assert slice_of("solo_heads.0.phi.weight", splits) is None
    assert slice_of("blocks.x.lin.weight", splits) is None


def test_median_clip_bounds_the_oversized_upload():
    a = {"w": torch.ones(4)}
    b = {"w": torch.ones(4)}
    big = {"w": torch.full((4,), 10.0)}
    clipped = median_clip([a, b, big])
    norms = [float(v["w"].norm()) for v in clipped]
    assert norms[2] == pytest.approx(norms[0]), norms          # big pulled down to the median
    assert norms[0] == pytest.approx(2.0) and norms[1] == pytest.approx(2.0)


def test_aggregate_takes_an_outer_step_from_the_mean():
    cur = {"w": torch.zeros(3)}
    d1, d2 = {"w": torch.full((3,), 2.0)}, {"w": torch.full((3,), 2.0)}
    out, state = aggregate(cur, [d1, d2], cfg=MosaicConfig(outer_lr=0.7, outer_momentum=0.0))
    # deltas are additive: 0 + 0.7 * mean(2.0, 2.0) = 1.4
    assert torch.allclose(out["w"], torch.full((3,), 1.4))
    assert "w" in state.prev


def test_aggregate_rejects_empty_but_accepts_disjoint_slices():
    with pytest.raises(MosaicError):
        aggregate({"w": torch.zeros(2)}, [])
    # Two volunteers on different slices: disjoint key sets are the normal case,
    # and each slice must be merged independently rather than rejected.
    current = {"w": torch.zeros(2), "v": torch.zeros(2)}
    out, _ = aggregate(current, [{"w": torch.full((2,), 2.0)}, {"v": torch.full((2,), -2.0)}],
                       cfg=MosaicConfig(outer_lr=1.0, outer_momentum=0.0))
    assert torch.allclose(out["w"], torch.full((2,), 2.0))
    assert torch.allclose(out["v"], torch.full((2,), -2.0))


def test_local_delta_touches_only_its_slice():
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=2))
    x, y = _batch(2, 4)
    delta = local_delta(model, slices=model.mosaic_splits, k=0, x=x, y=y, steps=2, lr=1e-2,
                        loss_chunk=4)
    assert delta and all(n.startswith(("blocks.0", "blocks.1", "wte.")) for n in delta)
    assert any(float(v.abs().sum()) > 0 for v in delta.values())
    # slice 0 owns the embedding (and with it the tied readout)
    assert any(n.startswith("wte.") for n in delta)
    assert not any(n.startswith(("ln_f.", "solo_heads.")) for n in delta)



def test_overlap_lets_two_slices_train_the_boundary_block():
    """Two neighbours both own the block at their cut, so the cut is not a cliff.

    A hard cut means the block below it and the block above it are trained by two
    different local objectives and neither sees the other's loss. With
    `overlap=1` each slice also runs the neighbour's boundary block, so the
    aggregate applies the mean of two proposals for it.
    """
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=1))
    x, y = _batch(2, 4)

    plain = local_delta(model, slices=model.mosaic_splits, k=0, x=x, y=y, steps=1,
                        lr=1e-3)
    assert not any(nm.startswith("blocks.2.") for nm in plain), \
        "overlap=0 must not train the neighbour's block"

    wide = local_delta(model, slices=model.mosaic_splits, k=0, x=x, y=y, steps=1,
                       lr=1e-3, overlap=1)
    assert any(nm.startswith("blocks.2.") for nm in wide), \
        f"overlap=1 should reach block 2, got {sorted(wide)[:6]}"
    assert not any(nm.startswith("blocks.3.") for nm in wide), \
        "overlap=1 must reach exactly one block past the cut"

    other = local_delta(model, slices=model.mosaic_splits, k=1, x=x, y=y, steps=1,
                        lr=1e-3, overlap=1)
    assert any(nm.startswith("blocks.1.") for nm in other), \
        f"the upper slice should also reach down to block 1, got {sorted(other)[:6]}"
    assert not any(nm.startswith("solo_heads.") for nm in wide), \
        "heads are local and must never be uploaded"

def test_fresh_local_batches_use_each_steps_own_batch():
    """H steps on one batch is H epochs; `xy_seq` gives each step fresh data.

    A one-entry sequence must be the single-batch behaviour bit for bit (that is
    what makes the flag a clean A/B), and a two-batch sequence must change the
    step -- otherwise the slice never sees the second batch and the flag is
    decorative.
    """
    import copy

    torch.manual_seed(0)
    base = _Forwarding(modules=2)
    attach(base, MosaicConfig(slices=2, local_steps=2))
    x1, y1 = _batch(1)
    x2, y2 = _batch(2)

    def run(**kw):
        m = copy.deepcopy(base)
        return local_delta(m, slices=m.mosaic_splits, k=0, x=x1, y=y1, steps=2,
                           lr=1e-2, loss_chunk=4, **kw)

    plain = run()
    one = run(xy_seq=[(x1, y1)])
    two = run(xy_seq=[(x1, y1), (x2, y2)])

    assert all(torch.equal(plain[n], one[n]) for n in plain), "one-entry sequence changed the step"
    assert any(not torch.allclose(plain[n], two[n]) for n in plain), "the second batch was ignored"


def test_share_trunk_lets_the_terminal_slice_train_the_tied_readout():
    """``lm_head.weight is wte.weight``: one tensor, two roles.

    Owning it from the first slice alone leaves the readout trained only by a
    local surrogate; owning it from the last slice alone leaves the embedding
    trained only by the output loss. ``share_trunk`` gives it to both, and
    ``aggregate`` averages the two deltas because they are ``theta_k - theta``.
    """
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=2))
    slices = model.mosaic_splits
    x, y = _batch(2, 4)

    alone = local_delta(model, slices=slices, k=1, x=x, y=y, steps=1, lr=1e-3, loss_chunk=4)
    assert not any(n.startswith("wte.") for n in alone), "terminal slice owns the readout only when asked"

    shared = local_delta(model, slices=slices, k=1, x=x, y=y, steps=1, lr=1e-3, loss_chunk=4,
                         share_trunk=True)
    assert "wte.weight" in shared
    assert any(n.startswith("ln_f.") for n in shared)
    assert not any(n.startswith(("solo_heads.", "blocks.0.")) for n in shared)


def test_local_delta_trains_the_solo_head_but_never_uploads_it():
    """The gradient for a slice's SOLO head must reach its own weights.

    ``slice_of`` classifies by the ``blocks.<i>.`` prefix, so a naive
    ``params = [p for n, p in model.named_parameters() if slice_of(n, ...) == k]``
    silently drops the head: the loss still backpropagates into it, and then the
    optimizer never applies that gradient. Every non-terminal slice then trains
    against a frozen random projection, which is what made MOSAIC plateau while
    backprop kept descending (measured at 306 M: 7.2270 vs 5.6862).
    """
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=2))
    head = model.solo_heads["0"]
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    x, y = _batch(2, 4)
    delta = local_delta(model, slices=model.mosaic_splits, k=0, x=x, y=y, steps=3, lr=1e-2,
                        loss_chunk=4)
    # The skip path is zero-initialised and only receives a gradient when the
    # slice input is handed to the head (`head_skip`, covered separately), so it
    # is excluded here: this test is about the head that is always active.
    moved = {n: float((p.detach() - before[n]).abs().max())
             for n, p in head.named_parameters() if not n.startswith("skip.")}
    assert all(v > 0 for v in moved.values()), f"head weights did not move: {moved}"
    # ... but the head is client-local: it must never appear in the uploaded delta.
    assert not any("solo_heads" in n for n in delta)


def test_config_and_slice_bounds_are_validated():
    with pytest.raises(MosaicError):
        MosaicConfig(slices=1)
    with pytest.raises(MosaicError):
        MosaicConfig(local_steps=0)
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=1))
    x, y = _batch(2, 4)
    with pytest.raises(MosaicError):
        local_delta(model, slices=model.mosaic_splits, k=5, x=x, y=y, steps=1, lr=1e-2)


def test_head_skip_lets_the_head_see_the_slice_input():
    """The auxiliary head is the slice's only teacher; DGL gives it the input too.

    Zero-initialised, so with the flag off the head is bit-identical to before
    and the skip weights never move (they receive no gradient at all).
    """
    torch.manual_seed(0)
    plain = _Forwarding(modules=2)
    attach(plain, MosaicConfig(slices=2, local_steps=1))
    torch.manual_seed(0)
    skipping = _Forwarding(modules=2)
    attach(skipping, MosaicConfig(slices=2, local_steps=1))

    x, y = _batch(2, 4)
    local_delta(plain, slices=plain.mosaic_splits, k=0, x=x, y=y, steps=2, lr=1e-2)
    local_delta(skipping, slices=skipping.mosaic_splits, k=0, x=x, y=y, steps=2,
                lr=1e-2, head_skip=True)

    def skip_mag(model):
        return float(model.solo_heads["0"].skip.weight.abs().max())

    assert skip_mag(plain) == 0.0, "the skip path must stay untouched when it is off"
    assert skip_mag(skipping) > 0.0, "head_skip must give the skip path a gradient"
    # The head is still client-local: no head tensor may reach the wire.
    delta = local_delta(skipping, slices=skipping.mosaic_splits, k=0, x=x, y=y,
                        steps=1, lr=1e-2, head_skip=True)
    assert not any("solo_heads" in n for n in delta)


def test_fit_head_trains_the_head_and_freezes_the_trunk():
    """Recalibration must move the teacher and leave the student alone."""
    model = _Forwarding(modules=2)
    attach(model, MosaicConfig(slices=2, local_steps=1))
    head = model.solo_heads["0"]
    before_head = {n: p.detach().clone() for n, p in head.named_parameters()}
    before_trunk = {n: p.detach().clone() for n, p in model.named_parameters()
                    if n.startswith("blocks.")}
    x, y = _batch(2, 4)
    loss = fit_head(model, slices=model.mosaic_splits, k=0, x=x, y=y, steps=5,
                    lr=1e-2, loss_chunk=4)
    assert loss == loss  # finite, not nan
    moved = {n: float((p.detach() - before_head[n]).abs().max())
             for n, p in head.named_parameters() if not n.startswith("skip.")}
    assert all(v > 0 for v in moved.values()), f"head did not move: {moved}"
    for n, p in model.named_parameters():
        if n.startswith("blocks."):
            assert torch.equal(p, before_trunk[n]), f"recalibration moved the trunk: {n}"
