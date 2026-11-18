"""ReLoRA: high-rank training through low-rank updates (arXiv 2307.05695).

The paper's claim is that accumulating many low-rank updates and periodically
merging them into the frozen base recovers high-rank training. This module is the
client-side half of that: attach rank-r adapters to every 2-D weight, train only
the adapters, and merge-and-restart on a schedule (zeroing the adapters so the
next segment starts from a small update again).

Two properties are load-bearing and are covered by tests: attaching adapters does
not change the model's output (the B factor starts at zero), and merging is
output-preserving (it folds B@A*scale into the base and zeroes both factors).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen Linear plus trainable rank-r factors."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        for p in self.base.parameters():
            p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        # Adapters must be born on the base weight's device and dtype: attaching after
        # model.to('cuda') otherwise leaves A/B on CPU and every forward raises
        # "Expected all tensors to be on the same device" (found by the GPU e2e run).
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.A = nn.Parameter(torch.zeros(self.rank, in_f, **kw))
        self.B = nn.Parameter(torch.zeros(out_f, self.rank, **kw))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B starts at zero, so the wrapped module is exactly the base at attach time.

    def delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t()) @ self.B.t() * self.scale


def attach(model: nn.Module, rank: int = 8, alpha: float = 1.0) -> int:
    """Replace every Linear (except the tied readout) with a LoRALinear. Returns count."""
    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(model, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += attach(child, rank=rank, alpha=alpha)
    return replaced


def trainable_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]


@torch.no_grad()
def merge(model: nn.Module) -> int:
    """Fold every adapter into its base weight, then zero the factors. Returns count."""
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.base.weight.requires_grad:
                raise ValueError("base weight must stay frozen; refusing to merge into a trainable base")
            module.base.weight.add_(module.delta_weight().to(module.base.weight.dtype))
            module.A.zero_()
            module.B.zero_()
            merged += 1
    return merged


@dataclass
class ReLoRASchedule:
    """Jagged learning-rate schedule: a high rate per segment, reset at each restart."""

    base_lr: float = 3e-4
    segment_lr: float = 3e-3
    inner_steps: int = 100

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        if self.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        within = step % self.inner_steps
        # warm back up to segment_lr at the start of every segment, then decay to base_lr
        frac = within / self.inner_steps
        return self.segment_lr + (self.base_lr - self.segment_lr) * frac

    def wants_restart(self, step: int) -> bool:
        return self.inner_steps > 0 and step > 0 and step % self.inner_steps == 0
