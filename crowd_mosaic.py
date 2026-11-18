"""MOSAIC: module-sharded local learning with infrequent, clipped uploads.

Ported from the handoff repo (`lab/methods/mosaic.py`) to the v2 client surface.
Four ingredients, in the order the coordinator sees them:

1. the layers are cut into K slices; each volunteer owns one slice and trains it
   with ordinary backprop, using a SOLO local head (a stop-gradiented copy of the
   readout) when it is not the terminal slice;
2. a volunteer recomputes the earlier slices locally, so nothing but weights ever
   moves between peers;
3. it takes H local steps and uploads only its slice's delta;
4. the coordinator clips each upload to the median norm for that slice, averages
   and takes one outer step — "no peer can push harder than the typical one".

That last rule is the security property: a single oversized upload cannot steer
the round, and without it an attacker's slice delta dominates the average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from crowd_solo import chunked_solo_ce, split_layers


class MosaicError(ValueError):
    """Raised for an unusable MOSAIC configuration or update set."""


@dataclass(frozen=True)
class MosaicConfig:
    slices: int = 2
    local_steps: int = 30
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    clip_median: bool = True

    def __post_init__(self) -> None:
        if self.slices < 2:
            raise MosaicError("MOSAIC needs at least two slices")
        if self.local_steps <= 0:
            raise MosaicError("local_steps must be positive")
        if self.outer_lr <= 0:
            raise MosaicError("outer_lr must be positive")


@dataclass
class MosaicState:
    """Outer-momentum state, kept per parameter name."""

    prev: dict[str, torch.Tensor] = field(default_factory=dict)


def slice_of(name: str, slices: list[tuple[int, int]]) -> int | None:
    """Which slice a parameter name belongs to (``None`` for the readout/trunk)."""
    if not name.startswith("blocks."):
        return None
    try:
        idx = int(name.split(".")[1])
    except (IndexError, ValueError):
        return None
    for k, (lo, hi) in enumerate(slices):
        if lo <= idx < hi:
            return k
    return None


def median_clip(deltas: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Clip each upload to the median norm of the uploads (per slice, in practice).

    With a single upload the median is its own norm, so nothing is clipped — that
    is the honest degenerate case, not a bug.
    """
    if not deltas:
        return []
    norms = [torch.sqrt(sum((v.float() ** 2).sum() for v in d.values())) for d in deltas]
    med = torch.stack(norms).median()
    out = []
    for d, n in zip(deltas, norms):
        scale = min(1.0, float(med) / (float(n) + 1e-12))
        out.append({k: v * scale for k, v in d.items()})
    return out


def aggregate(current: dict[str, torch.Tensor], deltas: list[dict[str, torch.Tensor]], *,
              cfg: MosaicConfig = MosaicConfig(), state: MosaicState | None = None
              ) -> tuple[dict[str, torch.Tensor], MosaicState]:
    """Clip, average and apply one outer step. Returns new weights and new state."""
    if not deltas:
        raise MosaicError("no uploads to aggregate")
    names = set(deltas[0])
    for d in deltas[1:]:
        if set(d) != names:
            raise MosaicError("uploads disagree about which parameters they carry")
    clipped = median_clip(deltas) if cfg.clip_median else deltas
    state = state or MosaicState()
    out = {}
    for name in names:
        mean = torch.stack([d[name].float() for d in clipped]).mean(dim=0)
        prev = state.prev.get(name)
        step = mean if prev is None else mean + cfg.outer_momentum * (mean - prev)
        out[name] = current[name] - cfg.outer_lr * step
        state.prev[name] = mean.clone()
    return out, state


def local_delta(model, *, slices: list[tuple[int, int]], k: int, x, y, steps: int,
                lr: float, loss_chunk: int = 256) -> dict[str, torch.Tensor]:
    """Train slice ``k`` for ``steps`` local AdamW steps; return its weight delta.

    A non-terminal slice learns through its SOLO head against a detached copy of the
    readout (no gradient crosses slices); the terminal slice uses the true loss.
    """
    if not 0 <= k < len(slices):
        raise MosaicError(f"slice {k} outside {len(slices)}")
    lo, hi = slices[k]
    terminal = k == len(slices) - 1
    params = [(n, p) for n, p in model.named_parameters() if slice_of(n, slices) == k]
    if not params:
        raise MosaicError(f"slice {k} has no parameters")
    if not terminal and (not hasattr(model, "solo_heads") or str(k) not in model.solo_heads):
        raise MosaicError(f"slice {k} needs a SOLO head; call attach() first")

    before = {n: p.detach().clone() for n, p in params}
    opt = torch.optim.AdamW([p for _, p in params], lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    head = None if terminal else model.solo_heads[str(k)]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        h = model.run_blocks(model.embed(x), 0, hi)
        if terminal:
            loss = chunked_solo_ce(model.ln_f(h), model.head_weight(), y, pos_step=loss_chunk)
        else:
            loss = chunked_solo_ce(head.project(h), model.head_weight().detach(), y,
                                   tau=head.tau, bias=head.bias, pos_step=loss_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        opt.step()
    return {n: (p.detach() - before[n]) for n, p in params}


def attach(model, cfg: MosaicConfig = MosaicConfig()) -> dict:
    """Attach SOLO heads for the non-terminal slices; returns a description."""
    from crowd_solo import build_solo_heads

    splits = split_layers(len(model.blocks), cfg.slices)
    dim = model.wte.weight.shape[1]
    vocab = model.wte.weight.shape[0]
    device = model.wte.weight.device
    model.mosaic_splits = splits
    model.solo_heads = build_solo_heads(dim, vocab, len(splits)).to(device)
    return {"paradigm": "mosaic", "slices": splits, "heads": len(splits) - 1,
            "localSteps": cfg.local_steps}
