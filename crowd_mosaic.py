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
    # Per-tensor relative step (LARS in spirit). The slices have wildly different
    # scales -- an embedding row, an attention matrix and a LayerNorm gain do not
    # share a useful absolute step -- so one global outer_lr over-steps some
    # tensors and under-steps others. With trust_ratio every tensor's step is
    # rescaled to the tensor's own RMS, which makes ``outer_lr`` a *relative*
    # step size (use ~1e-3, not ~1.0).
    trust_ratio: bool = False
    # Give each auxiliary head a hidden layer (DGL: aux-net capacity matters).
    head_mlp: bool = False

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
    """Which slice a parameter name belongs to (``None`` for local-only tensors).

    Non-block parameters have to belong to a slice as well. They used to return
    ``None``, which excluded ``wte`` and ``ln_f`` from every slice's optimiser --
    and since the readout is tied to ``wte``, the model trained a representation
    against a readout that stayed at its random initialisation for the whole run.
    A frozen random readout puts a floor under the loss (we measured ~7.45-7.57
    against 5.52-5.63 for SOLO/backprop at 65 M), and no amount of inner lr,
    outer momentum or head training could move it. The embedding (and therefore
    the tied readout) belongs to the first slice, the final norm and any untied
    readout to the last; both are uploaded like any other slice tensor.
    """
    if name.startswith("wte."):
        return 0
    if name.startswith("ln_f.") or name.startswith("lm_head."):
        return len(slices) - 1
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
    # Uploads are per-slice, so different slices carry disjoint parameter sets and
    # two volunteers on the same slice overlap entirely. Group by key-set: the
    # median clip is a per-slice rule, so it must not compare a slice against
    # another slice's norm, and the average is taken within a group.
    groups: dict[frozenset[str], list[dict[str, torch.Tensor]]] = {}
    for d in deltas:
        groups.setdefault(frozenset(d), []).append(d)
    clipped: list[dict[str, torch.Tensor]] = []
    for members in groups.values():
        clipped.extend(median_clip(members) if cfg.clip_median else members)
    seen: set[str] = set().union(*map(set, clipped)) if clipped else set()
    state = state or MosaicState()
    out = {}
    for name in seen:
        parts = [d[name].float() for d in clipped if name in d]
        mean = torch.stack(parts).mean(dim=0) if len(parts) > 1 else parts[0]
        prev = state.prev.get(name)
        step = mean if prev is None else mean + cfg.outer_momentum * (mean - prev)
        base = current.get(name)
        if cfg.trust_ratio and base is not None:
            rms_step = torch.sqrt((step.float() ** 2).mean())
            rms_base = torch.sqrt((base.float() ** 2).mean())
            if float(rms_step) > 0.0 and float(rms_base) > 0.0:
                step = step * (rms_base / rms_step)
        # MOSAIC uploads are deltas (theta_k - theta), so the merge ADDS them.
        # Subtracting would move the model away from every slice's training.
        out[name] = (base if base is not None else torch.zeros_like(mean)) + cfg.outer_lr * step
        state.prev[name] = mean.clone()
    return out, state


def local_delta(model, *, slices: list[tuple[int, int]], k: int, x, y, steps: int,
                lr: float, loss_chunk: int = 256,
                opt_state: dict | None = None,
                share_trunk: bool = False,
                head_lr_mult: float = 1.0,
                head_skip: bool = False,
                overlap: int = 0,
                xy_seq: list | None = None) -> dict[str, torch.Tensor]:
    """Train slice ``k`` for ``steps`` local AdamW steps; return its weight delta.

    A non-terminal slice learns through its SOLO head against a detached copy of the
    readout (no gradient crosses slices); the terminal slice uses the true loss.

    ``opt_state`` lets the slice's AdamW state survive between rounds. Without it a
    fresh optimiser is built per call, and at one local step that is not AdamW at
    all: the first step of Adam is ``lr * sign(grad)`` because the bias correction
    cancels the moments, so a one-step round degenerates to sign-SGD with no
    momentum or second-moment memory. A volunteer trains the same slice round after
    round, exactly like the production client (which builds its optimisers once),
    so pass a dict and it will be reused per slice ``k``.
    """
    if not 0 <= k < len(slices):
        raise MosaicError(f"slice {k} outside {len(slices)}")
    lo, hi = slices[k]
    terminal = k == len(slices) - 1
    params = [(n, p) for n, p in model.named_parameters() if slice_of(n, slices) == k]
    lo_f, hi_f = lo, hi
    if overlap:
        # A harder cut than the method needs: the blocks either side of a slice
        # boundary are trained by two different local objectives and neither sees
        # the other's loss. Giving both slices the boundary block lets its
        # gradient be the average of the two proposals (aggregate already does
        # that for any tensor present in two deltas).
        lo_f, hi_f = max(0, lo - overlap), min(len(model.blocks), hi + overlap)
        have = {n for n, _ in params}
        for idx in list(range(lo_f, lo)) + list(range(hi, hi_f)):
            for n, p in model.blocks[idx].named_parameters():
                full = f"blocks.{idx}.{n}"
                if full not in have:
                    params.append((full, p))
                    have.add(full)
    if k == len(slices) - 1 and share_trunk:
        # The readout is `lm_head.weight is wte.weight`. One tensor, two roles: the
        # first slice trains it as an embedding through its local head, the terminal
        # slice needs it as the output layer under the true loss. Owning it from one
        # side only leaves the other role untrained -- measured at 65 M as 7.25 (both
        # roles on slice 0) vs 6.47 (readout only) vs 5.63 for backprop, so the two
        # roles are worth ~0.8 nats. `aggregate` averages a tensor that appears in
        # two slices' deltas, so sharing costs no wire mechanism, only half a step
        # per role per round.
        shared = [(n, p) for n, p in model.named_parameters() if n.startswith("wte.")]
        params = params + [(n, p) for n, p in shared if n not in {m for m, _ in params}]
    if not params:
        raise MosaicError(f"slice {k} has no parameters")
    if not terminal and (not hasattr(model, "solo_heads") or str(k) not in model.solo_heads):
        raise MosaicError(f"slice {k} needs a SOLO head; call attach() first")

    head = None if terminal else model.solo_heads[str(k)]
    # The head is the slice's only teacher, so it has to learn *with* the slice.
    # `slice_of` classifies by `blocks.<i>.` prefix and therefore never matches
    # `solo_heads.<k>.*`: training only the slice left every non-terminal module
    # chasing a frozen random projection, which is what a MOSAIC run looks like
    # when it plateaus ~1.5 nats behind backprop. Heads are local, so they are
    # optimised but never returned in the delta.
    trained = list(params)
    if head is not None:
        trained += [(f"solo_heads.{k}.{n}", p) for n, p in head.named_parameters()]
    before = {n: p.detach().clone() for n, p in params}
    # The head is the teacher, so it has to keep up with the trunk it is teaching.
    # Measured at 65 M: a fresh decoder fit on MOSAIC's own frozen features
    # reaches 5.817, while MOSAIC's own head path outputs 6.018 from those same
    # features -- the online head is worse than a newly fit head of comparable
    # capacity, which is what a teacher lagging its moving student looks like.
    # A separate param group lets the head run hotter than the trunk.
    groups = [{"params": [p for _, p in params], "lr": lr}]
    if head is not None:
        head_params = [p for _, p in head.named_parameters()]
        if float(head_lr_mult) != 1.0:
            groups.append({"params": head_params, "lr": lr * float(head_lr_mult),
                           "is_head": True})
        else:
            groups[0]["params"].extend(head_params)
    if opt_state is None:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    else:
        opt = opt_state.get(k)
        if opt is None:
            opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
            opt_state[k] = opt
        else:
            for group in opt.param_groups:
                group["lr"] = lr * (float(head_lr_mult) if group.get("is_head") else 1.0)
    for _s in range(steps):
        # H steps on one batch is H epochs on 2,048 tokens; `xy_seq` gives each
        # step its own batch instead, which is what lets a round afford many
        # local steps without the representation overfitting a single batch.
        if xy_seq:
            x_s, y_s = xy_seq[_s % len(xy_seq)]
        else:
            x_s, y_s = x, y
        opt.zero_grad(set_to_none=True)
        # Earlier slices are input, not student: run them without a graph. This
        # is what slicing is for (no gradient crosses a module boundary) and it
        # is bit-identical for this slice's gradients, since none of its
        # parameters appear upstream of `lo`.
        with torch.no_grad():
            h0 = model.run_blocks(model.embed(x_s), 0, lo_f)
        h = model.run_blocks(h0, lo_f, hi_f)
        if terminal:
            loss = chunked_solo_ce(model.ln_f(h), model.head_weight(), y_s,
                                   pos_step=loss_chunk)
        else:
            # `h0` is a no_grad constant here (earlier slices are input, not
            # student), so the skip path can see it without reintroducing the
            # cross-module gradient the method exists to avoid.
            proj = head.project(h, z=h0) if head_skip else head.project(h)
            loss = chunked_solo_ce(proj, model.head_weight().detach(), y_s,
                                   tau=head.tau, bias=head.bias, pos_step=loss_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trained], 1.0)
        opt.step()
    return {n: (p.detach() - before[n]) for n, p in params}


def fit_head(model, *, slices, k: int, x, y, steps: int = 20, lr: float = 1e-2,
             loss_chunk: int = 256) -> float:
    """Re-align slice ``k``'s SOLO head to the trunk as it is *now*.

    The head is the only teacher a non-terminal slice has, and it is trained
    online while the aggregated trunk keeps moving, so it can lag the student it
    supervises. Measured at 65 M: a fresh decoder fit on MOSAIC's own frozen
    features reaches 5.82, while MOSAIC's own head path gives 6.02 on those same
    features -- the online head is worse than a head fitted to the trunk. Raising
    its lr made that worse (3x/10x/30x -> 6.14/6.21/6.18 vs 6.08), because a
    hotter AdamW teacher is noisier rather than better aligned.

    So: freeze the slice, fit the head to convergence, resume. Nothing crosses a
    module boundary and the head is still client-local, so this changes only how
    well the teacher tracks the student. Returns the final local loss.
    """
    lo, hi = slices[k]
    if not hasattr(model, "solo_heads") or str(k) not in model.solo_heads:
        raise MosaicError(f"slice {k} needs a SOLO head; call attach() first")
    head = model.solo_heads[str(k)]
    params = [p for p in head.parameters()]
    # Frozen trunk: earlier slices are input, and this slice must not move while
    # its teacher catches up.
    with torch.no_grad():
        h0 = model.run_blocks(model.embed(x), 0, lo)
        h = model.run_blocks(h0, lo, hi)
    opt = torch.optim.AdamW(params, betas=(0.9, 0.95), weight_decay=0.01)
    w_bar = model.head_weight().detach()
    loss = None
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = chunked_solo_ce(head.project(h), w_bar, y, tau=head.tau,
                               bias=head.bias, pos_step=loss_chunk)
        loss.backward()
        opt.step()
    return float(loss.detach()) if loss is not None else float("nan")


def attach(model, cfg: MosaicConfig = MosaicConfig()) -> dict:
    """Attach SOLO heads for the non-terminal slices; returns a description."""
    from crowd_solo import build_solo_heads

    splits = split_layers(len(model.blocks), cfg.slices)
    dim = model.wte.weight.shape[1]
    vocab = model.wte.weight.shape[0]
    device = model.wte.weight.device
    model.mosaic_splits = splits
    model.solo_heads = build_solo_heads(dim, vocab, len(splits),
                                        mlp=cfg.head_mlp).to(device)
    return {"paradigm": "mosaic", "slices": splits, "heads": len(splits) - 1,
            "localSteps": cfg.local_steps}
