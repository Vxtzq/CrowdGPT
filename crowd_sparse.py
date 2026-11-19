"""Sparse top-k deltas with error feedback — the transfer cut, client side.

Why this module exists: at 657.8 M server parameters a dense bf16 delta is
**1.3 GB per round per volunteer**. The 1-bit codec (`NGD1`) already cuts that to
~83 MB, but it keeps a nonzero value in every position, so it pays for
coordinates that barely moved. A volunteer's delta is empirically very sparse in
*relative* terms: most of what a round changes is a handful of directions, and the
rest is noise the server will average away anyway.

So: keep the largest entries, drop the rest, and — this is the part that makes it
work rather than merely smaller — carry the dropped mass forward as a residual and
add it to the next round's delta. That is error feedback (SparseLoCo, arXiv
2407.07852; the same contract as 1-bit SGD's residual). Without the residual the
method is biased: coordinates that are consistently small never get sent, and the
model develops a permanent lag along exactly those directions. With it, an entry
that was dropped twice is twice as large the third time and gets through.

The index is a position in the **flat server vector**, not in the tensor, so the
server can scatter entries into the weights it already holds with `index_add_`
and never materialises a dense delta. Client-local tensors (`engram.`,
`solo_heads.`, `mtp_projs.`) are not in the vector and are never encoded here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

# Wire contract with the server's decoder: magic, then (count, total) as uint32,
# then `count` entries of (uint32 index into the flat vector, bf16 value).
MAGIC = b"SPK1"
ENTRY_BYTES = 6  # 4-byte index + 2-byte bf16 value

# The header carries the block length, so the engram tail starts after the
# entries rather than after `model_size * 2` bytes. A dense delta leaves both
# fields unset and the server falls back to the old arithmetic.
META_CODEC = "baseCodec"
META_BYTES = "baseBytes"
META_COUNT = "baseCount"


@dataclass(frozen=True)
class SparseConfig:
    """How much of a delta to send, and whether to carry the rest forward.

    ``frac`` is a fraction of *coordinates*, not of mass: a delta whose energy is
    concentrated sends almost all of its mass in 2% of its entries, while a
    diffuse delta still gets the same budget. ``min_entries`` keeps tiny tensors
    (LayerNorm gains, biases) from rounding down to zero entries, which would
    freeze them permanently under error feedback.
    """

    frac: float = 0.02
    min_entries: int = 1
    error_feedback: bool = True

    def __post_init__(self):
        if not (0.0 < float(self.frac) <= 1.0):
            raise ValueError(f"sparse frac must be in (0, 1], got {self.frac!r}")
        if int(self.min_entries) < 1:
            raise ValueError(f"min_entries must be >= 1, got {self.min_entries!r}")


def config_from_env(env=None):
    """Read the codec settings; ``None`` when sparsity is not requested."""
    import os

    env = os.environ if env is None else env
    if str(env.get("CROWDGPT_DELTA_CODEC", "")).strip().lower() != "sparse":
        return None
    try:
        frac = float(env.get("CROWDGPT_SPARSE_FRAC", "0.02"))
    except (TypeError, ValueError):
        frac = 0.02
    try:
        min_entries = int(env.get("CROWDGPT_SPARSE_MIN_ENTRIES", "1"))
    except (TypeError, ValueError):
        min_entries = 1
    feedback = str(env.get("CROWDGPT_SPARSE_FEEDBACK", "1")).strip().lower() not in ("0", "false", "no", "off")
    try:
        return SparseConfig(frac=frac, min_entries=min_entries, error_feedback=feedback)
    except ValueError:
        return SparseConfig()


HEADER_BYTES = 12  # magic(4) + uint32 entry count + uint32 flat-vector length


def block_bytes(count: int) -> int:
    """Bytes the base block occupies for ``count`` entries (header included)."""
    return HEADER_BYTES + int(count) * ENTRY_BYTES


def entry_count(numel: int, cfg: SparseConfig) -> int:
    """How many entries a tensor of ``numel`` coordinates gets."""
    k = int(round(float(cfg.frac) * int(numel)))
    return max(int(cfg.min_entries), min(int(numel), k))


def select(delta: torch.Tensor, cfg: SparseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-|value| entries of a flat delta. Returns ``(indices, values)``.

    Sorted=False because the server scatters by index and does not care about
    order; skipping the sort is measurable on the largest tensors.
    """
    flat = delta.detach().reshape(-1).float()
    k = entry_count(flat.numel(), cfg)
    if k >= flat.numel():
        idx = torch.arange(flat.numel(), dtype=torch.int64, device=flat.device)
        return idx, flat
    _, idx = torch.topk(flat.abs(), k, sorted=False)
    return idx, flat[idx]


def encode_block(pairs, total: int) -> bytes:
    """Serialise ``pairs`` (an iterable of (index tensor, value tensor)) to bytes.

    Indices are **global** positions in the flat server vector, so the caller adds
    each tensor's offset before handing them over.
    """
    idx_parts, val_parts, count = [], [], 0
    for idx, val in pairs:
        idx = idx.detach().reshape(-1).to("cpu", torch.int64)
        val = val.detach().reshape(-1).to("cpu", torch.bfloat16)
        idx_parts.append(idx.numpy().astype(np.uint32))
        val_parts.append(val.view(torch.int16).numpy().astype(np.uint16))
        count += int(idx.numel())
    idx_all = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.uint32)
    val_all = np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.uint16)
    return MAGIC + struct.pack("<II", count, int(total)) + idx_all.tobytes() + val_all.tobytes()


def decode_block(raw: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Inverse of :func:`encode_block` → ``(indices, values_fp32, total)``.

    Kept here as well as on the server so the wire format has one reference
    implementation that both sides' tests point at.
    """
    if raw[:4] != MAGIC:
        raise ValueError(f"not a sparse delta block: {raw[:4]!r}")
    if len(raw) < 8:
        raise ValueError("short sparse delta header")
    count, total = struct.unpack("<II", raw[4:12])
    need = block_bytes(count)
    if len(raw) < need:
        raise ValueError(f"short sparse delta: {len(raw)} < {need}")
    idx = np.frombuffer(raw[12:12 + count * 4], dtype=np.uint32)
    vals = torch.from_numpy(np.frombuffer(raw[12 + count * 4:need], dtype=np.int16).copy())
    return idx, vals.view(torch.bfloat16).to(torch.float32).numpy(), int(total)


class ErrorFeedback:
    """Residual of what a round dropped, re-added to the next round's delta.

    Stored in bf16 deliberately: the residual is a *delta*, shipped at bf16
    precision anyway, so keeping it in fp32 would only buy precision that the very
    next upload discards — at twice the memory (2.6 GB vs 1.3 GB at 657.8 M).
    """

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        self.residual: dict[str, torch.Tensor] = {}

    def correct(self, name: str, delta: torch.Tensor) -> torch.Tensor:
        """Add the carried residual for ``name`` to ``delta`` (if any)."""
        res = self.residual.get(name)
        if res is None or res.numel() != delta.numel():
            return delta
        return delta + res.to(delta.device, torch.float32)

    def keep(self, name: str, dropped: torch.Tensor) -> None:
        """Store what was not sent this round."""
        if dropped.numel() == 0:
            self.residual.pop(name, None)
            return
        self.residual[name] = dropped.detach().to("cpu", self.dtype)

    def bytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.residual.values()))

    def clear(self) -> None:
        self.residual.clear()
