"""One runtime switch for the v2 training paradigms.

The founder asked for MOSAIC + ReLoRA + backprop + SOLO behind one client. Rather
than scattering env flags (`CROWDGPT_SOLO`, `CROWDGPT_SOLO_MODULE`, ...), a run
picks a paradigm by name and gets back the three things a round needs: how to
prepare the model, how to take one training step, and which parameters may be
uploaded. DiLoCo stays a *round layer* underneath any of them (`crowd_diloco`),
because it changes when peers exchange, not what gradient a block gets.

DUST is deliberately not selectable here: it was measured at 90 tok/s and closed
as a core paradigm, and porting it is TODO V2I-4 pending the founder's word.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

PARADIGMS = ("backprop", "solo", "relora")
PENDING = ("dust", "mosaic")


class ParadigmError(ValueError):
    """Unknown or unimplemented paradigm."""


@dataclass(frozen=True)
class ParadigmConfig:
    name: str = "backprop"
    solo_modules: int = 2
    relora_rank: int = 8
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
