"""One runtime switch for the v2 training paradigms.

The founder asked for MOSAIC + ReLoRA + backprop + SOLO behind one client. Rather
than scattering env flags (`CROWDGPT_SOLO`, `CROWDGPT_SOLO_MODULE`, ...), a run
picks a paradigm by name and gets back the three things a round needs: how to
prepare the model, how to take one training step, and which parameters may be
uploaded. DiLoCo stays a *round layer* underneath any of them (`crowd_diloco`),
because it changes when peers exchange, not what gradient a block gets.

All five are selectable. DUST is the one measured *badly*: at lr 1e-4/3e-4/1e-3 it
ended at 8.3250/8.3294/8.3411 against an 8.3236 init (research/V2_DUST_VERDICT.md),
and at 200-400x backprop's cost per step it needs a population of >=512 before the
numbers mean anything. It stays selectable because a peer with spare compute and a
large round is exactly the population that could make it work; it is never a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

PARADIGMS = ("backprop", "solo", "relora", "dust", "mosaic")
PENDING = ()


class ParadigmError(ValueError):
    """Unknown or unimplemented paradigm."""


@dataclass(frozen=True)
class ParadigmConfig:
    name: str = "backprop"
    solo_modules: int = 2
    relora_rank: int = 8
    dust_population: int = 16
    dust_sigma: float = 0.01
    dust_seed: int = 0
    mosaic_slices: int = 2
    # Measured optimum at 65 M / 2 M tokens / 488 rounds, in that order: H=1
    # 6.2730, H=5 5.8121, H=10 6.3883, H=20 8.0004 (it diverges), so five local
    # steps is an interior optimum and the defaults ship that value, not the
    # 30 steps this file was written with.
    mosaic_local_steps: int = 5
    # The inner rate is a weak lever (3x smaller moved nothing on either device),
    # but the outer rate is not: `aggregate` applies `current + outer_lr * step`
    # where `step` is already a full local AdamW displacement, so 1.0 re-applied
    # the local step in full. Swept on fixed code: 1.0 -> 6.0902, 0.5 -> 5.9625,
    # 0.3 -> 5.8446, 0.15 -> 5.8121, 0.1 -> 5.8825.
    mosaic_inner_lr: float = 1e-3
    mosaic_outer_lr: float = 0.15
    mosaic_outer_momentum: float = 0.9
    # The readout is `lm_head.weight is wte.weight`, so one tensor has two roles:
    # the first slice trains it as an embedding, the terminal slice needs it as
    # the output layer. With share_trunk the terminal slice trains it too, and
    # `aggregate` averages the two proposals. Measured at 65 M: 7.25 (both roles
    # on slice 0) vs 6.47 (readout only), so the ownership is worth ~0.8 nats;
    # the extra `share_trunk` step is worth 0.03-0.05 on top of that, i.e. inside
    # the 0.04 five-run spread, and is kept on because the tuned configuration
    # (5.8121) was measured with it.
    mosaic_share_trunk: bool = True
    relora_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.name in PENDING:
            raise ParadigmError(
                f"{self.name} is not implemented in this build: "
                + ("MOSAIC has no reference paper or repo (TODO V2I-3)"
                   if self.name == "mosaic" else "DUST is TODO V2I-4"))
        if self.name not in PARADIGMS:
            raise ParadigmError(f"unknown paradigm {self.name!r}; have {PARADIGMS}")
        if self.name == "solo" and self.solo_modules < 2:
            raise ParadigmError("solo needs at least two modules")
        if self.name == "relora" and self.relora_rank <= 0:
            raise ParadigmError("relora rank must be positive")


def select(name: str = "backprop", **kwargs) -> "Paradigm":
    """Return the paradigm object for ``name``."""
    return Paradigm(type=ParadigmConfig(name=name, **kwargs))


def _ce(logits, target):
    import torch

    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        target[:, :-1].reshape(-1))


@dataclass
class Paradigm:
    """Prepared model hooks plus a single training step."""

    type: ParadigmConfig
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.type.name

    # ── preparation ────────────────────────────────────────────────────────
    def attach(self, model) -> dict:
        """Prepare ``model`` in place; returns what was done."""
        if self.name == "backprop":
            return {"paradigm": "backprop", "trainable": "all"}

        if self.name == "solo":
            from crowd_solo import build_solo_heads, split_layers

            splits = split_layers(len(model.blocks), self.type.solo_modules)
            dim = model.wte.weight.shape[1]
            vocab = model.wte.weight.shape[0]
            model.solo_splits = splits
            model.solo_heads = build_solo_heads(dim, vocab, len(splits)).to(model.main_device)
            return {"paradigm": "solo", "splits": splits, "heads": len(splits) - 1}

        if self.name == "dust":
            from crowd_dust import linear_layers

            return {"paradigm": "dust", "layers": len(linear_layers(model)),
                    "population": self.type.dust_population, "sigma": self.type.dust_sigma}

        if self.name == "mosaic":
            from crowd_mosaic import MosaicConfig, attach as mosaic_attach

            return mosaic_attach(model, MosaicConfig(slices=self.type.mosaic_slices,
                                                    local_steps=self.type.mosaic_local_steps,
                                                    outer_lr=self.type.mosaic_outer_lr,
                                                    outer_momentum=self.type.mosaic_outer_momentum))

        from crowd_relora import attach

        n = attach(model, rank=self.type.relora_rank, alpha=self.type.relora_alpha)
        return {"paradigm": "relora", "adapters": n, "rank": self.type.relora_rank}

    # ── one step ───────────────────────────────────────────────────────────
    def step(self, model, x, y, *, loss_scale: float = 1.0,
             loss_chunk: int = 256) -> float:
        """Run one training step; gradients are accumulated, not applied."""
        if self.name == "backprop":
            loss = _ce(model(x), y) * loss_scale
            loss.backward()
            return float(loss.detach())

        if self.name == "solo":
            from crowd_solo import solo_step

            loss, _ = solo_step(model, x, y, splits=model.solo_splits,
                                loss_scale=loss_scale, loss_chunk=loss_chunk)
            return float(loss)

        if self.name == "mosaic":
            # One MOSAIC round: every slice takes H local steps from the current
            # weights, uploads only its own delta, and the coordinator clips and
            # averages them into one outer step. Slower per call than the other
            # paradigms (H x slices), which is the method's actual cost.
            from crowd_mosaic import MosaicConfig, aggregate, local_delta

            cfg = MosaicConfig(slices=self.type.mosaic_slices,
                               local_steps=self.type.mosaic_local_steps,
                               outer_lr=self.type.mosaic_outer_lr,
                               outer_momentum=self.type.mosaic_outer_momentum)
            splits = model.mosaic_splits
            current = {n: p.detach().clone() for n, p in model.named_parameters()}
            deltas = [local_delta(model, slices=splits, k=k, x=x, y=y,
                                  steps=cfg.local_steps, lr=self.type.mosaic_inner_lr,
                                  loss_chunk=loss_chunk,
                                  share_trunk=self.type.mosaic_share_trunk)
                      for k in range(len(splits))]
            new_w, state = aggregate(current, deltas, cfg=cfg,
                                     state=getattr(model, "_mosaic_state", None))
            with __import__("torch").no_grad():
                for n, p in model.named_parameters():
                    if n in new_w:
                        p.copy_(new_w[n].to(p.device, p.dtype))
            model._mosaic_state = state
            # MOSAIC has applied its own outer step; clear the slices' leftover local
            # grads so the client's outer optimizer does not step on them again.
            for p in model.parameters():
                p.grad = None
            with __import__("torch").no_grad():
                return float(_ce(model(x), y))

        if self.name == "dust":
            from crowd_dust import DustConfig, dust_step

            cfg = DustConfig(population=self.type.dust_population, sigma=self.type.dust_sigma,
                             seed=self.type.dust_seed)
            return float(dust_step(model, x, y, cfg=cfg, loss_fn=_ce, loss_scale=loss_scale))

        from crowd_relora import trainable_parameters

        loss = _ce(model(x), y) * loss_scale
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and not any(p is q for q in trainable_parameters(model)):
                p.grad = None
        return float(loss.detach())

    # ── what may leave the peer ────────────────────────────────────────────
    def upload_filter(self) -> Callable[[str], bool]:
        if self.name == "relora":
            return lambda name: ".A" in name or ".B" in name
        if self.name == "solo":
            from crowd_solo import is_uploaded_param

            return is_uploaded_param
        return lambda name: not name.startswith("engram.")


def merge_and_restart(model) -> int:
    """ReLoRA's merge step; a no-op for other paradigms."""
    from crowd_relora import merge

    return merge(model)
