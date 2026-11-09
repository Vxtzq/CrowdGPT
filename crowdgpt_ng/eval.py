"""Held-out evaluation with hash-pinned data and a bootstrap interval.

The evaluator is deliberately model-agnostic: the caller supplies a
``logits_fn(input_ids) -> array[seq_len, vocab]``. That keeps the contract
testable without a GPU and lets the same code score a tiny test model, a CPU
checkpoint, or an orchestrator's validator.

Every report carries the manifest hash, the model identifier, the seed, and
the number of windows. A loss without those is not comparable to anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .protocol import (
    EvalResult,
    ProtocolError,
    canonical_bytes,
    decimal_str,
    digest_of,
)

EVAL_SCHEMA = "ng-eval-1"
_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}
_DTYPE_BYTES = {name: np.dtype(value).itemsize for name, value in _DTYPES.items()}


class EvalError(ValueError):
    """The manifest, the data, or the model output is not usable."""


@dataclass(frozen=True)
class Shard:
    path: str
    sha256: str
    tokens: int
    dtype: str = "uint32"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "tokens": self.tokens, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Shard":
        _strict(data, {field.name for field in fields(cls)}, "shard")
        return cls(
            path=_text(data, "path"),
            sha256=_hex64(data, "sha256"),
            tokens=_positive_int(data, "tokens"),
            dtype=_dtype(data, "dtype"),
        )

    def validate(self) -> None:
        _text({"path": self.path}, "path")
        _hex64({"sha256": self.sha256}, "sha256")
        _positive_int({"tokens": self.tokens}, "tokens")
        _dtype({"dtype": self.dtype}, "dtype")


@dataclass(frozen=True)
class EvalManifest:
    name: str
    version: str
    seq_len: int
    stride: int
    vocab_size: int
    tokenizer: str
    shards: tuple[Shard, ...]
    created_at: str
    schema: str = EVAL_SCHEMA

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "version": self.version,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer,
            "shards": [shard.to_dict() for shard in self.shards],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalManifest":
        _strict(data, {field.name for field in fields(cls)}, "manifest")
        shards = data["shards"]
        if not isinstance(shards, (list, tuple)) or not shards:
            raise EvalError("manifest needs at least one shard")
        return cls(
            schema=_text(data, "schema", max_len=16),
            name=_text(data, "name"),
            version=_text(data, "version"),
            seq_len=_positive_int(data, "seq_len"),
            stride=_positive_int(data, "stride"),
            vocab_size=_positive_int(data, "vocab_size"),
            tokenizer=_text(data, "tokenizer"),
            shards=tuple(Shard.from_dict(item) for item in shards),
            created_at=_text(data, "created_at"),
        )

    def validate(self) -> None:
        if self.schema != EVAL_SCHEMA:
            raise EvalError(f"unsupported eval schema: {self.schema!r}")
        _text({"name": self.name}, "name")
        _text({"version": self.version}, "version")
        if _positive_int({"seq_len": self.seq_len}, "seq_len") < 2:
            raise EvalError("seq_len must be at least 2")
        _positive_int({"stride": self.stride}, "stride")
        if _positive_int({"vocab_size": self.vocab_size}, "vocab_size") < 2:
            raise EvalError("vocab_size must be at least 2")
        _text({"tokenizer": self.tokenizer}, "tokenizer")
        _text({"created_at": self.created_at}, "created_at")
        if not isinstance(self.shards, tuple) or not self.shards:
            raise EvalError("manifest needs at least one shard")
        seen: set[str] = set()
        for shard in self.shards:
            shard.validate()
            if shard.path in seen:
                raise EvalError(f"duplicate shard path: {shard.path}")
            seen.add(shard.path)

    def sha256(self) -> str:
        return digest_of(self.to_dict())


@dataclass(frozen=True)
class Window:
    shard: int
    offset: int


@dataclass(frozen=True)
class EvalReport:
    manifest_name: str
    manifest_sha256: str
    model_id: str
    windows: int
    tokens: int
    loss: str
    ci_low: str
    ci_high: str
    seed: int
    code_commit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVAL_SCHEMA,
            "manifest_name": self.manifest_name,
            "manifest_sha256": self.manifest_sha256,
            "model_id": self.model_id,
            "windows": self.windows,
            "tokens": self.tokens,
            "loss": self.loss,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "seed": self.seed,
            "code_commit": self.code_commit,
        }

    def as_eval_result(self, *, model_sha256: str, attested_by: Sequence[str]) -> EvalResult:
        return EvalResult(
            name=f"{self.manifest_name}",
            dataset_sha256=self.manifest_sha256,
            tokens=self.tokens,
            windows=self.windows,
            loss=self.loss,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            model_sha256=model_sha256,
            code_commit=self.code_commit,
            attested_by=tuple(attested_by),
        )


def _strict(data: Mapping[str, Any], allowed: set[str], what: str) -> None:
    if not isinstance(data, Mapping):
        raise EvalError(f"{what} must be a mapping")
    unknown = set(data) - allowed
    if unknown:
        raise EvalError(f"{what} has unknown fields: {sorted(unknown)}")
    missing = allowed - set(data)
    if missing:
        raise EvalError(f"{what} is missing fields: {sorted(missing)}")


def _text(data: Mapping[str, Any], key: str, *, max_len: int = 512) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise EvalError(f"{key} must be a non-empty string")
    return value


def _hex64(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvalError(f"{key} must be 64 lowercase hex characters")
    return value


def _positive_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvalError(f"{key} must be a positive integer")
    return value


def _dtype(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value not in _DTYPES:
        raise EvalError(f"{key} must be one of {sorted(_DTYPES)}")
    return value


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    name: str,
    version: str,
    shards: Sequence[str | Path],
    seq_len: int,
    stride: int,
    vocab_size: int,
    tokenizer: str,
    created_at: str,
    dtype: str = "uint32",
    base_dir: str | Path | None = None,
) -> EvalManifest:
    """Hash the shards and return a manifest ready to commit."""

    if dtype not in _DTYPES:
        raise EvalError(f"dtype must be one of {sorted(_DTYPES)}")
    base = Path(base_dir) if base_dir is not None else None
    entries: list[Shard] = []
    for item in shards:
        path = Path(item)
        if not path.is_file():
            raise EvalError(f"missing shard: {path}")
        size = path.stat().st_size
        itemsize = _DTYPE_BYTES[dtype]
        if size % itemsize:
            raise EvalError(f"shard size is not a multiple of {itemsize}: {path}")
        relative = str(path.relative_to(base)) if base is not None else str(path)
        entries.append(Shard(path=relative, sha256=sha256_file(path), tokens=size // itemsize, dtype=dtype))
    return EvalManifest(
        name=name,
        version=version,
        seq_len=seq_len,
        stride=stride,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        shards=tuple(entries),
        created_at=created_at,
    )


def load_manifest(path: str | Path, *, expected_sha256: str | None = None) -> EvalManifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalError(f"missing manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid manifest JSON: {exc}") from exc
    manifest = EvalManifest.from_dict(data)
    if expected_sha256 is not None and manifest.sha256() != expected_sha256:
        raise EvalError("manifest hash does not match the pinned value")
    return manifest


def verify_shards(manifest: EvalManifest, root: str | Path | None = None) -> list[Path]:
    """Check every shard's bytes against the manifest."""

    base = Path(root) if root is not None else Path(".")
    resolved: list[Path] = []
    for shard in manifest.shards:
        path = base / shard.path
        if not path.is_file():
            raise EvalError(f"missing shard: {path}")
        expected_size = shard.tokens * _DTYPE_BYTES[shard.dtype]
        if path.stat().st_size != expected_size:
            raise EvalError(f"shard size mismatch: {path}")
        if sha256_file(path) != shard.sha256:
            raise EvalError(f"shard hash mismatch: {path}")
        resolved.append(path)
    return resolved


def read_window(path: str | Path, offset: int, length: int, dtype: str) -> np.ndarray:
    if dtype not in _DTYPES:
        raise EvalError(f"unsupported dtype: {dtype}")
    if offset < 0 or length < 1:
        raise EvalError("offset must be >= 0 and length >= 1")
    array = np.fromfile(Path(path), dtype=_DTYPES[dtype], count=length, offset=offset * _DTYPE_BYTES[dtype])
    if array.size != length:
        raise EvalError(f"short read from {path} at offset {offset}")
    return array.astype(np.int64, copy=False)


def _capacity(manifest: EvalManifest) -> int:
    return sum(max(0, (shard.tokens - manifest.seq_len - 1) // manifest.stride + 1) for shard in manifest.shards)


def sample_windows(manifest: EvalManifest, count: int, *, seed: int = 0) -> list[Window]:
    """Deterministically choose windows spread across the shards."""

    _positive_int({"count": count}, "count")
    candidates: list[Window] = []
    for index, shard in enumerate(manifest.shards):
        last = shard.tokens - manifest.seq_len - 1
        if last < 0:
            continue
        for offset in range(0, last + 1, manifest.stride):
            candidates.append(Window(index, offset))
    if not candidates:
        raise EvalError("no shard is long enough for one window")
    if count > len(candidates):
        raise EvalError(f"requested {count} windows but only {len(candidates)} are available")

    def rank(window: Window) -> str:
        digest = hashlib.sha256(f"{seed}:{window.shard}:{window.offset}".encode("ascii")).hexdigest()
        return digest

    return sorted(candidates, key=rank)[:count]


def _nll_rows(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise EvalError(f"logits must be 2-D [seq_len, vocab], got {logits.shape}")
    if logits.shape[0] != targets.size:
        raise EvalError(f"logits rows {logits.shape[0]} do not match targets {targets.size}")
    if targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise EvalError("target token is outside the vocabulary")
    row_max = logits.max(axis=1, keepdims=True)
    log_sum_exp = row_max[:, 0] + np.log(np.exp(logits - row_max).sum(axis=1))
    return log_sum_exp - logits[np.arange(targets.size), targets]


def bootstrap_ci(samples: Sequence[float], *, seed: int = 0, iterations: int = 1000) -> tuple[float, float]:
    if not samples:
        raise EvalError("cannot bootstrap an empty sample")
    if iterations < 1:
        raise EvalError("iterations must be positive")
    rng = random.Random(seed)
    values = list(samples)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(len(values)):
            total += values[rng.randrange(len(values))]
        means.append(total / len(values))
    means.sort()
    low = means[max(0, int(0.025 * len(means)) - 1)]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return low, high


def evaluate(
    manifest: EvalManifest,
    logits_fn: Callable[[Sequence[int]], Any],
    *,
    root: str | Path | None = None,
    count: int | None = None,
    seed: int = 0,
    bootstrap_iterations: int = 1000,
    model_id: str = "unknown",
    code_commit: str = "unknown",
) -> EvalReport:
    """Score a fixed sample of held-out windows.

    ``logits_fn`` receives the ``seq_len`` input ids and returns logits shaped
    ``[seq_len, vocab_size]`` (a leading batch dimension of 1 is accepted).
    """

    paths = verify_shards(manifest, root)
    available = _capacity(manifest)
    if available < 1:
        raise EvalError("no window fits the manifest's shards")
    chosen = sample_windows(manifest, min(count or 64, available), seed=seed)

    totals: list[float] = []
    tokens = 0
    for window in chosen:
        shard = manifest.shards[window.shard]
        sequence = read_window(paths[window.shard], window.offset, manifest.seq_len + 1, shard.dtype)
        inputs = sequence[:-1]
        targets = sequence[1:]
        logits = np.asarray(logits_fn(inputs), dtype=np.float64)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        losses = _nll_rows(logits, targets)
        if not np.all(np.isfinite(losses)):
            raise EvalError("model produced a non-finite loss")
        totals.append(float(losses.mean()))
        tokens += int(targets.size)

    low, high = bootstrap_ci(totals, seed=seed, iterations=bootstrap_iterations)
    mean = sum(totals) / len(totals)
    return EvalReport(
        manifest_name=manifest.name,
        manifest_sha256=manifest.sha256(),
        model_id=model_id,
        windows=len(totals),
        tokens=tokens,
        loss=decimal_str(mean),
        ci_low=decimal_str(low),
        ci_high=decimal_str(high),
        seed=seed,
        code_commit=code_commit,
    )


def uniform_logits(vocab_size: int) -> Callable[[Sequence[int]], np.ndarray]:
    """A reference model with a known loss (``ln(vocab_size)``)."""

    def logits(inputs: Sequence[int]) -> np.ndarray:
        return np.zeros((len(inputs), vocab_size), dtype=np.float64)

    return logits
