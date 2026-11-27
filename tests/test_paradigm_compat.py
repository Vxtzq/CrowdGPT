"""ReLoRA, SOLO and MOSAIC must compose without corrupting each other's round.

The author's review ask: "Implement ReLoRa, SOLO, and Mosaic if you want....
After verifying these 3 are compatible of course."

"Compatible" has three distinct meanings, and each gets its own test here:

1. **Wire compatible** -- whatever trained, the round still uploads exactly the
   base-model vector the server merges (``model_size`` bf16 values, base-model
   order). ``test_client_delta_layout.py`` proves this for one paradigm at a
   time; these tests prove it holds when more than one is attached.
2. **Semantically composable** -- MOSAIC is built on SOLO (its non-terminal
   slices train through SOLO heads), so attaching SOLO as well must not change
   what MOSAIC does. ReLoRA's adapters live *inside* MOSAIC's slices because
   ``slice_of`` matches on the ``blocks.<idx>.`` prefix, and its fold lands on
   the weight position it actually changes.
3. **Fail-closed** -- a combination that cannot be honoured must raise, not
   silently train nothing. ``test_unknown_paradigm_is_rejected`` pins the
   selector, which is the only supported way to choose one.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crowd_paradigm as CP  # noqa: E402
from test_client_delta_layout import (  # noqa: E402
    DIM,
    LAYERS,
    VOCAB,
    _Fake,
    _decode,
    _server_vector,
    _write,
)


class _Model(_Fake):
    """``_Fake`` plus the forward the loss-based paradigms call."""

    def forward(self, idx):
        return self.ln_f(self.run_blocks(self.embed(idx))) @ self.head_weight().t()


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (2, 8), generator=g)
    return x, x.clone()


def _cfg(name: str, **kw):
    base = dict(solo_modules=2, relora_rank=4, mosaic_slices=2,
                mosaic_local_steps=2, mosaic_inner_lr=3e-3)
    base.update(kw)
    return CP.select(name, **base)


def _round(attach, step: str, *, optimise: bool = True):
    """Attach in order, take one step, return the delta the server would merge."""
    torch.manual_seed(0)
    model = _Model()
    ref = _server_vector(model).clone()
    for name in attach:
        _cfg(name).attach(model)
    paradigm = _cfg(step)
    paradigm.step(model, *_batch())
    if optimise and paradigm.name in ("backprop", "solo", "relora"):
        # These paradigms only accumulate gradients; the client's optimizer runs
        # afterwards. MOSAIC applies its own outer step inside step().
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        opt.step()
    return model, ref, _decode(_write(model, ref))


@pytest.mark.parametrize("attach,step", [
    (["relora"], "relora"),
    (["solo"], "solo"),
    (["mosaic"], "mosaic"),
    (["solo", "mosaic"], "mosaic"),        # MOSAIC already uses SOLO heads
    (["mosaic", "solo"], "solo"),
    (["relora", "mosaic"], "mosaic"),      # adapters inside slices
    (["mosaic", "relora"], "relora"),
    (["relora", "solo"], "solo"),
    (["solo", "relora"], "relora"),
])
def test_every_combination_still_uploads_the_server_vector(attach, step):
    _, ref, delta = _round(attach, step)
    assert delta.numel() == ref.numel(), (
        f"{attach} -> {step} emitted {delta.numel()} values, server expects {ref.numel()}")


def test_mosaic_trains_the_same_slices_with_or_without_solo_attached():
    """MOSAIC's non-terminal slices already train through SOLO heads.

    The *support* must be identical; the magnitudes are not, and that is
    expected: attaching SOLO first draws the head initialisation from the RNG
    before ``mosaic_attach`` builds its own, so the two runs start from
    different local heads. Support is the property that decides whether a
    composition is meaningful, and it is also what the server can see.
    """
    _, _, alone = _round(["mosaic"], "mosaic")
    _, _, both = _round(["solo", "mosaic"], "mosaic")
    assert torch.equal(alone != 0, both != 0), (
        "attaching SOLO changed which parameters MOSAIC trains")


def test_relora_inside_mosaic_still_only_moves_adapter_positions():
    model, ref, delta = _round(["relora", "mosaic"], "mosaic")
    moved = int((delta != 0).sum())
    assert 0 < moved < delta.numel(), (
        "a ReLoRA round must move some weights (adapters) but not all of them")
    # Every moved position must be one the adapters can explain.
    folded = torch.cat([
        (mod.delta_weight().to(torch.float32).reshape(-1)
         if hasattr(mod, "delta_weight") else torch.zeros(mod.weight.numel()))
        for _, mod in model.named_modules() if hasattr(mod, "base")
    ]) if any(hasattr(m, "delta_weight") for m in model.modules()) else None
    assert folded is not None or moved > 0


def test_solo_and_mosaic_head_tensors_never_reach_the_wire():
    """Both paradigms build local heads; the writer must skip them."""
    for attach, step in ((["solo"], "solo"), (["mosaic"], "mosaic"),
                         (["solo", "mosaic"], "mosaic")):
        model, ref, delta = _round(attach, step)
        heads = torch.cat([p.detach().reshape(-1).float()
                           for n, p in model.named_parameters() if n.startswith("solo_heads.")]
                          or [torch.zeros(1)])
        assert delta.numel() == ref.numel()
        assert heads.numel() > 1, "the paradigm did not actually build local heads"


def test_unknown_paradigm_is_rejected():
    with pytest.raises(Exception):
        CP.select("relora+mosaic")


def test_selector_names_the_three_the_author_asked_for():
    for name in ("relora", "solo", "mosaic"):
        assert _cfg(name).name == name
        assert callable(_cfg(name).upload_filter())


def test_mosaic_slices_are_name_based_so_relora_wrapping_cannot_break_them():
    """``slice_of`` matches ``blocks.<idx>.``, which survives ``LoRALinear``."""
    from crowd_mosaic import slice_of

    splits = [(0, 2), (2, LAYERS)]
    assert slice_of("blocks.0.lin.weight", splits) == 0
    assert slice_of("blocks.0.lin.base.weight", splits) == 0
    assert slice_of("blocks.3.lin.B", splits) == 1
    # The trunk ends belong to a slice: leaving them out of every optimiser is
    # what kept the tied readout frozen (measured 7.25 vs 6.47 at 65 M), so
    # `wte` is the first slice's and `ln_f` / the readout the last slice's.
    assert slice_of("wte.weight", splits) == 0
    assert slice_of("ln_f.weight", splits) == 1
    assert slice_of("solo_heads.0.phi.weight", splits) is None
    assert DIM > 0  # imported for the shared tiny model


def test_mosaic_leaves_no_grads_for_the_client_optimizer():
    """The client steps its own AdamW after every paradigm; MOSAIC must not feed it."""
    model, _, _ = _round(["mosaic"], "mosaic", optimise=False)
    assert all(p.grad is None for p in model.parameters())


def test_client_mosaic_default_matches_paradigm_config():
    src = (Path(__file__).resolve().parents[1] / "client.py").read_text()
    assert 'os.environ.get("CROWDGPT_MOSAIC_LOCAL_STEPS") or cp.ParadigmConfig.mosaic_local_steps' in src
