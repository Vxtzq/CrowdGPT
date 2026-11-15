"""DiLoCo-style outer loop: few exchanges, one compressed pseudo-gradient.

The current protocol ships a dense delta every round. In a low-communication
outer loop a client takes ``H`` local steps instead and reports the difference
from the weights it started with — a pseudo-gradient — which is then
quantised to a fixed byte budget, averaged, and applied with an outer Nesterov
momentum. The coordinator sees one small upload per ``H`` steps instead of
``H`` dense ones.

Everything here is plain numpy and framework-free so the arithmetic can be
tested without a model: the point of the module is the update rule and the
byte budget, not the training loop that produces the pseudo-gradient.

Not wired into ``client.py``: that needs a server exchange endpoint
(``POST /fl/pseudograd``) and a paired measurement, which this module does not
pretend to provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

INDEX_BYTES = 4
VALUE_BYTES = 2
DEFAULT_BUDGET_BYTES = 50 * 1024**2
BF16_MAX = 0x7F7F


class DilocoError(ValueError):
    """A pseudo-gradient or config that cannot be applied safely."""


def _as_float32(tensor: np.ndarray) -> np.ndarray:
    if not isinstance(tensor, np.ndarray):
        raise DilocoError("pseudo-gradients must be numpy arrays")
    if not np.all(np.isfinite(tensor)):
        raise DilocoError("pseudo-gradient contains non-finite values")
    return tensor.astype(np.float32, copy=False)


def to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bit pattern.

    Truncation would bias every value toward zero; the rounding term is the
    standard ``0x7FFF + lsb`` trick, so a payload's error is the bf16 quantum
    (2^-8 relative), not half a quantum.
    """

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class SparsePayload:
    """A quantised pseudo-gradient: ``indices`` into the flattened tensor."""

    shape: tuple[int, ...]
    indices: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.indices.nbytes + self.values.nbytes)

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "count": int(self.indices.size),
            "bytes": self.nbytes,
            "indices": self.indices.tolist(),
            "values": self.values.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SparsePayload":
        return cls(
            shape=tuple(int(x) for x in data["shape"]),
            indices=np.asarray(data["indices"], dtype=np.uint32),
            values=np.asarray(data["values"], dtype=np.uint16),
        )


def _budget_per_tensor(budget_bytes: int, sizes: Sequence[int]) -> list[int]:
    total = sum(sizes)
    if total <= 0 or budget_bytes <= 0:
        return [0] * len(sizes)
    per_elem = INDEX_BYTES + VALUE_BYTES
    capacity = max(1, budget_bytes // per_elem)
    counts = [max(1, int(round(capacity * size / total))) for size in sizes]
    # Never exceed the budget: shave the largest counts until it fits.
    while sum(counts) > capacity:
        counts[counts.index(max(counts))] -= 1
        if max(counts) <= 0:
            break
    return counts


def compress(tensor: np.ndarray, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> SparsePayload:
    """Keep the largest-magnitude entries that fit, quantise them to bf16."""

    if budget_bytes < INDEX_BYTES + VALUE_BYTES:
        raise DilocoError("budget must hold at least one entry")
    flat = _as_float32(np.asarray(tensor).reshape(-1))
    size = int(flat.size)
    if size == 0:
        return SparsePayload(shape=tuple(np.asarray(tensor).shape), indices=np.zeros(0, np.uint32),
                             values=np.zeros(0, np.uint16))
    keep = min(size, max(1, budget_bytes // (INDEX_BYTES + VALUE_BYTES)))
    order = np.argsort(-np.abs(flat), kind="stable")
    chosen = order[:keep]
    chosen = chosen[np.abs(flat[chosen]) > 0]
    chosen.sort()
    return SparsePayload(
        shape=tuple(np.asarray(tensor).shape),
        indices=chosen.astype(np.uint32),
        values=to_bf16_bits(flat[chosen]),
    )


def compress_many(tensors: Iterable[np.ndarray], budget_bytes: int = DEFAULT_BUDGET_BYTES) -> list[SparsePayload]:
    """Split one round's byte budget across all tensors in proportion to size."""

    items = list(tensors)
    entries = _budget_per_tensor(budget_bytes, [int(np.asarray(t).size) for t in items])
    per_entry = INDEX_BYTES + VALUE_BYTES
    return [compress(t, max(1, count) * per_entry) for t, count in zip(items, entries)]


def decompress(payload: SparsePayload, dtype=np.float32) -> np.ndarray:
    out = np.zeros(int(np.prod(payload.shape)), dtype=np.float32)
    out[payload.indices] = from_bf16_bits(payload.values)
    return out.reshape(payload.shape).astype(dtype, copy=False)


def average(payloads: Sequence[SparsePayload], total_params: int) -> np.ndarray:
    """Sum the payloads into one dense pseudo-gradient (index collision safe)."""

    if not payloads:
        raise DilocoError("no pseudo-gradients to average")
    dense = np.zeros(int(total_params), dtype=np.float32)
    for payload in payloads:
        if int(np.prod(payload.shape)) != dense.size:
            raise DilocoError("pseudo-gradients disagree on model size")
        dense[payload.indices] += from_bf16_bits(payload.values)
    return dense / float(len(payloads))


@dataclass
class OuterOptimizer:
    """Nesterov momentum on the outer step, as in DiLoCo.

    ``theta <- theta - lr * (grad + momentum * (grad - prev_grad))`` when
    ``nesterov`` is on, otherwise ``theta <- theta - lr * (momentum * prev_grad + grad)``.
    """

    lr: float = 0.7
    momentum: float = 0.9
    nesterov: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise DilocoError("lr must be a positive finite number")
        if not 0.0 <= self.momentum < 1.0:
            raise DilocoError("momentum must be in [0, 1)")
        self._prev: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        grad = _as_float32(np.asarray(grad, dtype=np.float32))
        if grad.shape != theta.shape:
            raise DilocoError("gradient shape must match the parameters")
        prev = np.zeros_like(grad) if self._prev is None else self._prev
        if self.nesterov:
            direction = grad + self.momentum * (grad - prev)
        else:
            direction = self.momentum * prev + grad
        self._prev = grad.copy()
        return theta - self.lr * direction

    def state_dict(self) -> dict:
        return {"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov,
                "prev": None if self._prev is None else self._prev.tolist()}

    @classmethod
    def from_state_dict(cls, data: dict) -> "OuterOptimizer":
        opt = cls(lr=float(data["lr"]), momentum=float(data["momentum"]), nesterov=bool(data["nesterov"]))
        if data.get("prev") is not None:
            opt._prev = np.asarray(data["prev"], dtype=np.float32)
        return opt
