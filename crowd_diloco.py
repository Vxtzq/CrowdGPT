"""DiLoCo round loop for the Crowd client.

The measured design (see `research/V2_DILOCO_306M.md`): each peer runs a long
block of inner AdamW steps between synchronisations, then the coordinator
averages the peers' weights and applies one outer Nesterov-SGD step to the
pseudo-gradient ``theta - mean(theta_k)``. Twenty-five inner steps beat one step
per round by 1.17-1.39 held-out at equal tokens on both GPUs we tested, while
cutting uploads 27x.

Transport is reused from :mod:`crowdgpt_ng.diloco` (fixed-budget top-k sparse
payloads and its outer optimiser). This module adds what that one does not have:
the round schedule, the staleness cap, and a checkpoint that survives a long
inner run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from crowdgpt_ng.diloco import (
    DEFAULT_BUDGET_BYTES,
    OuterOptimizer,
    SparsePayload,
    average,
    compress,
    decompress,
)

__all__ = [
    "DilocoConfig",
    "DilocoRoundError",
    "DilocoCoordinator",
    "RoundCheckpoint",
]


class DilocoRoundError(ValueError):
    """A round was driven out of order or with the wrong peers."""


@dataclass(frozen=True)
class DilocoConfig:
    """Round shape. Defaults are the measured v2 configuration."""

    inner_steps: int = 25
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True
    max_stale_steps: int = 2
    budget_bytes: int = DEFAULT_BUDGET_BYTES
    min_peers: int = 2

    def __post_init__(self) -> None:
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1")
        if not 0.0 < self.outer_lr <= 1.0:
            raise ValueError("outer_lr must be in (0, 1]")
        if not 0.0 <= self.outer_momentum < 1.0:
            raise ValueError("outer_momentum must be in [0, 1)")
        if self.max_stale_steps < 0:
            raise ValueError("max_stale_steps must be >= 0")
        if self.budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        if self.min_peers < 2:
            raise ValueError("a DiLoCo round needs at least two peers")


@dataclass
class RoundCheckpoint:
    """Everything needed to resume mid-run: outer momentum plus schedule state."""

    round_index: int = 0
    tokens_seen: int = 0
    inner_steps_done: int = 0
    outer_state: dict | None = None
    saved_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "round_index": int(self.round_index),
            "tokens_seen": int(self.tokens_seen),
            "inner_steps_done": int(self.inner_steps_done),
            "outer_state": self.outer_state,
            "saved_at": float(self.saved_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RoundCheckpoint":
        return cls(
            round_index=int(data.get("round_index", 0)),
            tokens_seen=int(data.get("tokens_seen", 0)),
            inner_steps_done=int(data.get("inner_steps_done", 0)),
            outer_state=data.get("outer_state"),
            saved_at=float(data.get("saved_at", 0.0)),
        )


class DilocoCoordinator:
    """Collect one round's peer payloads, average them, take one outer step.

    The coordinator never sees peer gradients, only compressed deltas, so the
    wire cost per round is ``config.budget_bytes`` per peer regardless of model
    size. A payload stamped more than ``max_stale_steps`` behind the current
    round is dropped rather than silently folded in: the measured cost of a peer
    two rounds behind is +0.35 held-out, which is about half of DiLoCo's gain.
    """

    def __init__(self, config: DilocoConfig, total_params: int) -> None:
        self.config = config
        self.total_params = int(total_params)
        self.outer = OuterOptimizer(
            lr=config.outer_lr,
            momentum=config.outer_momentum,
            nesterov=config.nesterov,
        )
        self.round_index = 0
        self._pending: dict[str, tuple[int, SparsePayload]] = {}
        self.dropped_stale = 0
        self.rounds_completed = 0
        self.bytes_received = 0

    # ---- ingest -----------------------------------------------------------
    def submit(self, peer_id: str, round_index: int, tensor: np.ndarray) -> None:
        """Record one peer's delta for ``round_index``, compressing it on the wire."""
        if round_index < self.round_index:
            raise DilocoRoundError(
                f"round {round_index} already closed (current {self.round_index})"
            )
        lag = round_index - self.round_index
        if lag > self.config.max_stale_steps:
            self.dropped_stale += 1
            return
        payload = compress(np.asarray(tensor, dtype=np.float32), self.config.budget_bytes)
        self.bytes_received += payload.nbytes
        self._pending[peer_id] = (round_index, payload)

    def peers_ready(self) -> int:
        return len(self._pending)

    def ready(self) -> bool:
        return len(self._pending) >= self.config.min_peers

    # ---- round ------------------------------------------------------------
    def finalize(self, theta: np.ndarray, *, require: bool = True):
        """Average the pending payloads and apply one outer step to ``theta``.

        Returns ``(new_theta, stats)``. ``theta`` is the current global weight
        vector as float32; the return value is a new array, never a view.
        """
        if require and not self.ready():
            raise DilocoRoundError(
                f"{len(self._pending)} peer(s) pending, need {self.config.min_peers}"
            )
        if not self._pending:
            raise DilocoRoundError("no peer payloads to finalize")
        payloads: Sequence[SparsePayload] = [p for _, p in self._pending.values()]
        mean = average(payloads, self.total_params).astype(np.float32)
        # DiLoCo pseudo-gradient: theta - mean(theta_k). SGD then moves the global
        # weights toward the average; the opposite sign extrapolates away from it
        # and diverges (measured held-out 1234 in the first harness attempt).
        grad = np.asarray(theta, dtype=np.float32) - mean
        new_theta = self.outer.step(np.asarray(theta, dtype=np.float32), grad)
        stats = {
            "round": self.round_index,
            "peers": len(payloads),
            "bytes": sum(p.nbytes for p in payloads),
            "droppedStale": self.dropped_stale,
            "innerSteps": self.config.inner_steps,
        }
        self.rounds_completed += 1
        self.round_index += 1
        self._pending.clear()
        return np.asarray(new_theta, dtype=np.float32), stats

    # ---- durability -------------------------------------------------------
    def checkpoint(self, *, tokens_seen: int = 0, inner_steps_done: int = 0) -> RoundCheckpoint:
        return RoundCheckpoint(
            round_index=self.round_index,
            tokens_seen=tokens_seen,
            inner_steps_done=inner_steps_done,
            outer_state=self.outer.state_dict(),
        )

    def resume(self, cp: RoundCheckpoint) -> None:
        self.round_index = int(cp.round_index)
        if cp.outer_state is not None:
            self.outer = OuterOptimizer.from_state_dict(cp.outer_state)

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "round_index": self.round_index,
            "rounds_completed": self.rounds_completed,
            "dropped_stale": self.dropped_stale,
            "bytes_received": self.bytes_received,
            "total_params": self.total_params,
            "outer": self.outer.state_dict(),
        }
