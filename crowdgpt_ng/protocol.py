"""Wire objects for the CrowdGPT-ng orchestrator protocol.

The protocol has one job: make a round of training independently checkable.
Every object here is a pure payload with a canonical byte representation, so
two implementations (client and orchestrator, Python and anything else) agree
on what was signed and what was published.

Design rules:

* Canonical JSON: sorted keys, compact separators, UTF-8, no NaN/Infinity.
* Losses are fixed-precision decimal **strings** (six places). Floats drift
  across platforms and would change a record's hash.
* Signatures are detached Ed25519 envelopes. There is no chain and no custom
  cryptography: Ed25519 + SHA-256 + a Merkle tree are enough.
* Unknown fields are rejected. A silent field drop would change the hash of a
  record without anyone noticing.

This module has no heavyweight imports. Ed25519 support is loaded lazily and
raises :class:`ProtocolError` when ``cryptography`` is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "ng-1"

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX128 = re.compile(r"^[0-9a-f]{128}$")
_NODE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LOSS = re.compile(r"^-?\d+\.\d{1,6}$")
_URL = re.compile(r"^https?://[^\s]+$")

_DECIMAL_PLACES = Decimal("0.000001")
_KNOWN_ROLES = frozenset({"leader", "witness", "validator", "scheduler", "storage"})
_NODE_STATUS = frozenset({"active", "draining", "offline"})


class ProtocolError(ValueError):
    """A payload does not satisfy the protocol."""


def utc_now() -> str:
    """Current time in the protocol's timestamp shape (UTC, second precision)."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decimal_str(value: str | int | float | Decimal) -> str:
    """Format a finite number as a fixed six-place decimal string."""

    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProtocolError(f"not a decimal number: {value!r}") from exc
    if not number.is_finite():
        raise ProtocolError(f"decimal must be finite: {value!r}")
    try:
        quantized = number.quantize(_DECIMAL_PLACES)
    except InvalidOperation as exc:
        raise ProtocolError(f"decimal is out of range: {value!r}") from exc
    return format(quantized, "f")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical JSON bytes for hashing or signing."""

    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"payload is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def digest_of(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form, as lowercase hex."""

    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


# --------------------------------------------------------------------------
# Merkle tree over sorted leaves
# --------------------------------------------------------------------------


def _require_hex64(value: str, what: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ProtocolError(f"{what} must be 64 lowercase hex characters")
    return value


def merkle_root(leaves: Sequence[str]) -> str:
    """Root over sorted leaf digests. Empty input has a defined root."""

    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    layer = sorted(_require_hex64(leaf, "merkle leaf") for leaf in leaves)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(bytes.fromhex(layer[i]) + bytes.fromhex(layer[i + 1])).hexdigest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


def merkle_proof(leaves: Sequence[str], leaf: str) -> list[dict[str, str]]:
    """Inclusion proof for one leaf. Raises when the leaf is absent."""

    _require_hex64(leaf, "merkle leaf")
    layer = sorted(_require_hex64(item, "merkle leaf") for item in leaves)
    try:
        index = layer.index(leaf)
    except ValueError as exc:
        raise ProtocolError("leaf is not part of the tree") from exc

    proof: list[dict[str, str]] = []
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        sibling = index ^ 1
        proof.append({"side": "left" if sibling < index else "right", "hash": layer[sibling]})
        layer = [
            hashlib.sha256(bytes.fromhex(layer[i]) + bytes.fromhex(layer[i + 1])).hexdigest()
            for i in range(0, len(layer), 2)
        ]
        index //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: Sequence[Mapping[str, str]], root: str) -> bool:
    """Check an inclusion proof against a root."""

    try:
        current = _require_hex64(leaf, "merkle leaf")
        _require_hex64(root, "merkle root")
        for step in proof:
            side = step["side"]
            sibling = _require_hex64(step["hash"], "merkle sibling")
            if side == "left":
                current = hashlib.sha256(bytes.fromhex(sibling) + bytes.fromhex(current)).hexdigest()
            elif side == "right":
                current = hashlib.sha256(bytes.fromhex(current) + bytes.fromhex(sibling)).hexdigest()
            else:
                return False
    except (KeyError, TypeError, ProtocolError):
        return False
    return current == root


# --------------------------------------------------------------------------
# Ed25519 envelopes
# --------------------------------------------------------------------------


def _ed25519():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProtocolError("Ed25519 requires the 'cryptography' package") from exc
    return Ed25519PrivateKey, Ed25519PublicKey


def generate_keypair() -> tuple[str, str]:
    """Return ``(private_hex, public_hex)`` for a new Ed25519 key."""

    private_cls, _ = _ed25519()
    private = private_cls.generate()
    from cryptography.hazmat.primitives import serialization

    private_hex = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    ).hex()
    public_hex = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    return private_hex, public_hex


def sign_payload(payload: Mapping[str, Any], private_key_hex: str, signer: str) -> dict[str, Any]:
    """Sign a canonical payload and return a transportable envelope."""

    private_cls, _ = _ed25519()
    if not isinstance(private_key_hex, str) or not _HEX64.fullmatch(private_key_hex):
        raise ProtocolError("private key must be 64 lowercase hex characters")
    private = private_cls.from_private_bytes(bytes.fromhex(private_key_hex))
    from cryptography.hazmat.primitives import serialization

    public_hex = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    signature = private.sign(canonical_bytes(payload)).hex()
    return {
        "payload": dict(payload),
        "alg": "ed25519",
        "signer": _node_id(signer, "signer"),
        "pubkey": public_hex,
        "sig": signature,
    }


def verify_payload(
    envelope: Mapping[str, Any],
    *,
    signer: str | None = None,
    pubkey: str | None = None,
) -> dict[str, Any]:
    """Verify an envelope and return its payload, or raise ProtocolError."""

    if not isinstance(envelope, Mapping):
        raise ProtocolError("envelope must be a mapping")
    if set(envelope) != {"payload", "alg", "signer", "pubkey", "sig"}:
        raise ProtocolError("envelope fields are not the protocol's")
    if envelope["alg"] != "ed25519":
        raise ProtocolError(f"unsupported signature algorithm: {envelope['alg']!r}")
    if not isinstance(envelope["payload"], Mapping):
        raise ProtocolError("envelope payload must be a mapping")

    envelope_signer = _node_id(envelope["signer"], "signer")
    if signer is not None and envelope_signer != signer:
        raise ProtocolError(f"unexpected signer: {envelope_signer!r}")
    envelope_pubkey = _require_hex64(envelope["pubkey"], "pubkey")
    if pubkey is not None and envelope_pubkey != pubkey:
        raise ProtocolError("unexpected public key")
    signature = envelope["sig"]
    if not isinstance(signature, str) or not _HEX128.fullmatch(signature):
        raise ProtocolError("signature must be 128 lowercase hex characters")

    _, public_cls = _ed25519()
    try:
        public = public_cls.from_public_bytes(bytes.fromhex(envelope_pubkey))
        public.verify(bytes.fromhex(signature), canonical_bytes(envelope["payload"]))
    except Exception as exc:  # cryptography raises InvalidSignature and friends
        raise ProtocolError("signature verification failed") from exc
    return dict(envelope["payload"])


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("payload must be a mapping")
    return value


def _strict(data: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ProtocolError(f"{what} has unknown fields: {sorted(unknown)}")
    missing = allowed - set(data)
    if missing:
        raise ProtocolError(f"{what} is missing fields: {sorted(missing)}")


def _str(data: Mapping[str, Any], key: str, *, allow_empty: bool = False, max_len: int = 512) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ProtocolError(f"{key} must be a string")
    if not allow_empty and not value:
        raise ProtocolError(f"{key} must not be empty")
    if len(value) > max_len:
        raise ProtocolError(f"{key} is too long")
    return value


def _int(data: Mapping[str, Any], key: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    if value < minimum:
        raise ProtocolError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{key} must be <= {maximum}")
    return value


def _hex(data: Mapping[str, Any], key: str, pattern: re.Pattern[str], what: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ProtocolError(f"{key} must be {what}")
    return value


def _node_id(value: Any, what: str = "node id") -> str:
    if not isinstance(value, str) or not _NODE_ID.fullmatch(value):
        raise ProtocolError(f"{what} must match {_NODE_ID.pattern}")
    return value


def _timestamp(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise ProtocolError(f"{key} must be a UTC timestamp like 2026-11-09T00:00:00Z")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ProtocolError(f"{key} is not a valid timestamp") from exc
    return value


def _decimal(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _LOSS.fullmatch(value):
        raise ProtocolError(f"{what} must be a decimal string with 1-6 places")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ProtocolError(f"{what} is not a decimal number") from exc
    if not number.is_finite():
        raise ProtocolError(f"{what} must be finite")
    return value


def _tuple_of_str(data: Mapping[str, Any], key: str, *, max_items: int = 128) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) > max_items:
        raise ProtocolError(f"{key} must be a sequence of at most {max_items} items")
    return tuple(_str({key: item}, key, max_len=256) for item in value)


def _device(data: Mapping[str, Any]) -> dict[str, str]:
    value = data.get("device")
    if not isinstance(value, Mapping):
        raise ProtocolError("device must be a mapping")
    _strict(value, {"gpu", "vram_gb", "backend"}, "device")
    return {
        "gpu": _str(value, "gpu", max_len=128),
        "vram_gb": _str(value, "vram_gb", max_len=16),
        "backend": _str(value, "backend", max_len=32),
    }


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ticket:
    """Permission to produce one delta for one round."""

    ticket_id: str
    round: int
    client_id: str
    base_weights: str
    issuer: str
    issued_at: str
    expires_at: str
    step_budget: int
    token_budget: int
    data_shard: str | None
    nonce: str
    attempt: int = 1
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ticket_id": self.ticket_id,
            "round": self.round,
            "client_id": self.client_id,
            "base_weights": self.base_weights,
            "issuer": self.issuer,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "step_budget": self.step_budget,
            "token_budget": self.token_budget,
            "data_shard": self.data_shard,
            "nonce": self.nonce,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ticket":
        data = _mapping(data)
        _strict(data, set(FIELDS[cls]), "ticket")
        return cls(
            schema=_str(data, "schema", max_len=16),
            ticket_id=_str(data, "ticket_id"),
            round=_int(data, "round"),
            client_id=_str(data, "client_id", max_len=128),
            base_weights=_str(data, "base_weights"),
            issuer=_str(data, "issuer", max_len=64),
            issued_at=_str(data, "issued_at"),
            expires_at=_str(data, "expires_at"),
            step_budget=_int(data, "step_budget", minimum=1, maximum=1_000_000),
            token_budget=_int(data, "token_budget", minimum=1),
            data_shard=data.get("data_shard"),
            nonce=_str(data, "nonce"),
            attempt=_int(data, "attempt", minimum=1),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        if not _HEX32.fullmatch(self.ticket_id):
            raise ProtocolError("ticket_id must be 32 lowercase hex characters")
        if isinstance(self.round, bool) or not isinstance(self.round, int) or self.round < 0:
            raise ProtocolError("round must be a non-negative integer")
        for name, value in (("client_id", self.client_id), ("issuer", self.issuer)):
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ProtocolError(f"{name} must be a non-empty string")
        _node_id(self.issuer, "issuer")
        for name, value in (("base_weights", self.base_weights), ("nonce", self.nonce)):
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise ProtocolError(f"{name} must be 64 lowercase hex characters")
        for name, value in (("issued_at", self.issued_at), ("expires_at", self.expires_at)):
            _timestamp({name: value}, name)
        if self.expires_at <= self.issued_at:
            raise ProtocolError("expires_at must be after issued_at")
        for name, value, maximum in (
            ("step_budget", self.step_budget, 1_000_000),
            ("token_budget", self.token_budget, None),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProtocolError(f"{name} must be a positive integer")
            if maximum is not None and value > maximum:
                raise ProtocolError(f"{name} is too large")
        if self.data_shard is not None and (
            not isinstance(self.data_shard, str) or not self.data_shard or len(self.data_shard) > 256
        ):
            raise ProtocolError("data_shard must be null or a short string")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ProtocolError("attempt must be a positive integer")


@dataclass(frozen=True)
class StepReceipt:
    """What a client claims it did, and the evidence it offers."""

    ticket_id: str
    round: int
    client_id: str
    base_weights: str
    tokens: int
    steps: int
    loss_samples: tuple[str, ...]
    loss_mean: str | None
    fingerprint_root: str | None
    code_version: str
    device: dict[str, str]
    started_at: str
    finished_at: str
    nonce: str
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ticket_id": self.ticket_id,
            "round": self.round,
            "client_id": self.client_id,
            "base_weights": self.base_weights,
            "tokens": self.tokens,
            "steps": self.steps,
            "loss_samples": list(self.loss_samples),
            "loss_mean": self.loss_mean,
            "fingerprint_root": self.fingerprint_root,
            "code_version": self.code_version,
            "device": dict(self.device),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "nonce": self.nonce,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepReceipt":
        data = _mapping(data)
        _strict(data, set(FIELDS[cls]), "receipt")
        return cls(
            schema=_str(data, "schema", max_len=16),
            ticket_id=_str(data, "ticket_id"),
            round=_int(data, "round"),
            client_id=_str(data, "client_id", max_len=128),
            base_weights=_str(data, "base_weights"),
            tokens=_int(data, "tokens"),
            steps=_int(data, "steps"),
            loss_samples=_tuple_of_str(data, "loss_samples", max_items=10_000),
            loss_mean=data.get("loss_mean"),
            fingerprint_root=data.get("fingerprint_root"),
            code_version=_str(data, "code_version", max_len=64),
            device=_device(data),
            started_at=_str(data, "started_at"),
            finished_at=_str(data, "finished_at"),
            nonce=_str(data, "nonce"),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        for name, value, pattern, what in (
            ("ticket_id", self.ticket_id, _HEX32, "32 lowercase hex characters"),
            ("base_weights", self.base_weights, _HEX64, "64 lowercase hex characters"),
            ("nonce", self.nonce, _HEX64, "64 lowercase hex characters"),
        ):
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise ProtocolError(f"{name} must be {what}")
        if isinstance(self.round, bool) or not isinstance(self.round, int) or self.round < 0:
            raise ProtocolError("round must be a non-negative integer")
        if not isinstance(self.client_id, str) or not self.client_id or len(self.client_id) > 128:
            raise ProtocolError("client_id must be a non-empty string")
        for name, value in (("tokens", self.tokens), ("steps", self.steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if not isinstance(self.loss_samples, tuple) or len(self.loss_samples) > 10_000:
            raise ProtocolError("loss_samples must be a tuple")
        for sample in self.loss_samples:
            _decimal(sample, "loss sample")
        if self.loss_mean is not None:
            _decimal(self.loss_mean, "loss_mean")
        if self.fingerprint_root is not None and (
            not isinstance(self.fingerprint_root, str) or not _HEX64.fullmatch(self.fingerprint_root)
        ):
            raise ProtocolError("fingerprint_root must be null or 64 lowercase hex characters")
        if not isinstance(self.code_version, str) or not self.code_version or len(self.code_version) > 64:
            raise ProtocolError("code_version must be a short non-empty string")
        _device({"device": self.device})
        for name, value in (("started_at", self.started_at), ("finished_at", self.finished_at)):
            _timestamp({name: value}, name)
        if self.finished_at < self.started_at:
            raise ProtocolError("finished_at must not precede started_at")


@dataclass(frozen=True)
class EvalResult:
    """A held-out measurement with provenance."""

    name: str
    dataset_sha256: str
    tokens: int
    windows: int
    loss: str
    ci_low: str
    ci_high: str
    model_sha256: str
    code_commit: str
    attested_by: tuple[str, ...]
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "dataset_sha256": self.dataset_sha256,
            "tokens": self.tokens,
            "windows": self.windows,
            "loss": self.loss,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "model_sha256": self.model_sha256,
            "code_commit": self.code_commit,
            "attested_by": list(self.attested_by),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalResult":
        data = _mapping(data)
        _strict(data, set(FIELDS[cls]), "eval")
        return cls(
            schema=_str(data, "schema", max_len=16),
            name=_str(data, "name", max_len=64),
            dataset_sha256=_str(data, "dataset_sha256"),
            tokens=_int(data, "tokens"),
            windows=_int(data, "windows"),
            loss=_str(data, "loss"),
            ci_low=_str(data, "ci_low"),
            ci_high=_str(data, "ci_high"),
            model_sha256=_str(data, "model_sha256"),
            code_commit=_str(data, "code_commit", max_len=64),
            attested_by=_tuple_of_str(data, "attested_by", max_items=64),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ProtocolError("name must be a short non-empty string")
        for name, value in (("dataset_sha256", self.dataset_sha256), ("model_sha256", self.model_sha256)):
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise ProtocolError(f"{name} must be 64 lowercase hex characters")
        for name, value in (("tokens", self.tokens), ("windows", self.windows)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProtocolError(f"{name} must be a positive integer")
        loss = Decimal(_decimal(self.loss, "loss"))
        low = Decimal(_decimal(self.ci_low, "ci_low"))
        high = Decimal(_decimal(self.ci_high, "ci_high"))
        if not low <= loss <= high:
            raise ProtocolError("loss must lie inside its confidence interval")
        if not isinstance(self.code_commit, str) or not self.code_commit or len(self.code_commit) > 64:
            raise ProtocolError("code_commit must be a short non-empty string")
        if not isinstance(self.attested_by, tuple) or len(self.attested_by) > 64:
            raise ProtocolError("attested_by must be a tuple")
        for node in self.attested_by:
            _node_id(node, "attested_by")


@dataclass(frozen=True)
class CheckpointRef:
    """A content-addressed set of weights that a round produced or consumed."""

    sha256: str
    uri: str
    size_bytes: int
    round: int
    step: int
    format: str
    created_at: str
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sha256": self.sha256,
            "uri": self.uri,
            "size_bytes": self.size_bytes,
            "round": self.round,
            "step": self.step,
            "format": self.format,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointRef":
        data = _mapping(data)
        _strict(data, set(FIELDS[cls]), "checkpoint")
        return cls(
            schema=_str(data, "schema", max_len=16),
            sha256=_str(data, "sha256"),
            uri=_str(data, "uri", max_len=2048),
            size_bytes=_int(data, "size_bytes", minimum=1),
            round=_int(data, "round"),
            step=_int(data, "step"),
            format=_str(data, "format", max_len=32),
            created_at=_str(data, "created_at"),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        if not isinstance(self.sha256, str) or not _HEX64.fullmatch(self.sha256):
            raise ProtocolError("sha256 must be 64 lowercase hex characters")
        if not isinstance(self.uri, str) or not _URL.fullmatch(self.uri) or len(self.uri) > 2048:
            raise ProtocolError("uri must be an http(s) URL")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 1:
            raise ProtocolError("size_bytes must be a positive integer")
        for name, value in (("round", self.round), ("step", self.step)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if not isinstance(self.format, str) or not self.format or len(self.format) > 32:
            raise ProtocolError("format must be a short non-empty string")
        _timestamp({"created_at": self.created_at}, "created_at")


@dataclass(frozen=True)
class NodeRecord:
    """An orchestrator's public identity and reachable endpoints."""

    node_id: str
    pubkey: str
    endpoints: tuple[str, ...]
    roles: tuple[str, ...]
    software_version: str
    started_at: str
    bond_rounds: int
    status: str
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "node_id": self.node_id,
            "pubkey": self.pubkey,
            "endpoints": list(self.endpoints),
            "roles": list(self.roles),
            "software_version": self.software_version,
            "started_at": self.started_at,
            "bond_rounds": self.bond_rounds,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeRecord":
        data = _mapping(data)
        _strict(data, set(FIELDS[cls]), "node")
        return cls(
            schema=_str(data, "schema", max_len=16),
            node_id=_str(data, "node_id"),
            pubkey=_str(data, "pubkey"),
            endpoints=_tuple_of_str(data, "endpoints", max_items=16),
            roles=_tuple_of_str(data, "roles", max_items=8),
            software_version=_str(data, "software_version", max_len=64),
            started_at=_str(data, "started_at"),
            bond_rounds=_int(data, "bond_rounds"),
            status=_str(data, "status", max_len=16),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        _node_id(self.node_id)
        if not isinstance(self.pubkey, str) or not _HEX64.fullmatch(self.pubkey):
            raise ProtocolError("pubkey must be 64 lowercase hex characters")
        if not isinstance(self.endpoints, tuple) or not 1 <= len(self.endpoints) <= 16:
            raise ProtocolError("endpoints must contain 1-16 URLs")
        for endpoint in self.endpoints:
            if not isinstance(endpoint, str) or not _URL.fullmatch(endpoint) or len(endpoint) > 2048:
                raise ProtocolError("endpoint must be an http(s) URL")
        if not isinstance(self.roles, tuple) or not self.roles:
            raise ProtocolError("roles must not be empty")
        for role in self.roles:
            if role not in _KNOWN_ROLES:
                raise ProtocolError(f"unknown role: {role!r}")
        if not isinstance(self.software_version, str) or not self.software_version or len(self.software_version) > 64:
            raise ProtocolError("software_version must be a short non-empty string")
        _timestamp({"started_at": self.started_at}, "started_at")
        if isinstance(self.bond_rounds, bool) or not isinstance(self.bond_rounds, int) or self.bond_rounds < 0:
            raise ProtocolError("bond_rounds must be a non-negative integer")
        if self.status not in _NODE_STATUS:
            raise ProtocolError(f"unknown status: {self.status!r}")


@dataclass(frozen=True)
class RoundRecord:
    """The public, signed account of one training round."""

    round: int
    opened_at: str
    closed_at: str
    base_weights: str
    aggregate_weights: str
    leader: str
    witnesses: tuple[str, ...]
    tickets_root: str
    submissions_root: str
    deltas_root: str
    prev_record_hash: str | None
    eval: dict[str, Any] | None
    bytes: dict[str, int]
    code_versions: tuple[str, ...]
    certificate: tuple[dict[str, str], ...] = ()
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def signing_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("certificate")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "round": self.round,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "base_weights": self.base_weights,
            "aggregate_weights": self.aggregate_weights,
            "leader": self.leader,
            "witnesses": list(self.witnesses),
            "tickets_root": self.tickets_root,
            "submissions_root": self.submissions_root,
            "deltas_root": self.deltas_root,
            "prev_record_hash": self.prev_record_hash,
            "eval": None if self.eval is None else dict(self.eval),
            "bytes": dict(self.bytes),
            "code_versions": list(self.code_versions),
            "certificate": [dict(item) for item in self.certificate],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoundRecord":
        data = dict(_mapping(data))
        # The certificate is appended after signing, so an unsigned record is
        # a valid protocol object.
        data.setdefault("certificate", ())
        _strict(data, set(FIELDS[cls]), "round record")
        certificate = data.get("certificate", ())
        if not isinstance(certificate, (list, tuple)):
            raise ProtocolError("certificate must be a sequence")
        return cls(
            schema=_str(data, "schema", max_len=16),
            round=_int(data, "round"),
            opened_at=_str(data, "opened_at"),
            closed_at=_str(data, "closed_at"),
            base_weights=_str(data, "base_weights"),
            aggregate_weights=_str(data, "aggregate_weights"),
            leader=_str(data, "leader"),
            witnesses=_tuple_of_str(data, "witnesses", max_items=64),
            tickets_root=_str(data, "tickets_root"),
            submissions_root=_str(data, "submissions_root"),
            deltas_root=_str(data, "deltas_root"),
            prev_record_hash=data.get("prev_record_hash"),
            eval=data.get("eval"),
            bytes=dict(data.get("bytes") or {}),
            code_versions=_tuple_of_str(data, "code_versions", max_items=64),
            certificate=tuple(dict(item) for item in certificate),
        )

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema: {self.schema!r}")
        if isinstance(self.round, bool) or not isinstance(self.round, int) or self.round < 0:
            raise ProtocolError("round must be a non-negative integer")
        for name, value in (("opened_at", self.opened_at), ("closed_at", self.closed_at)):
            _timestamp({name: value}, name)
        if self.closed_at < self.opened_at:
            raise ProtocolError("closed_at must not precede opened_at")
        for name, value in (
            ("base_weights", self.base_weights),
            ("aggregate_weights", self.aggregate_weights),
            ("tickets_root", self.tickets_root),
            ("submissions_root", self.submissions_root),
            ("deltas_root", self.deltas_root),
        ):
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                raise ProtocolError(f"{name} must be 64 lowercase hex characters")
        _node_id(self.leader, "leader")
        if not isinstance(self.witnesses, tuple) or len(self.witnesses) > 64:
            raise ProtocolError("witnesses must be a tuple")
        for witness in self.witnesses:
            _node_id(witness, "witness")
        if self.prev_record_hash is not None and (
            not isinstance(self.prev_record_hash, str) or not _HEX64.fullmatch(self.prev_record_hash)
        ):
            raise ProtocolError("prev_record_hash must be null or 64 lowercase hex characters")
        if self.eval is not None:
            if not isinstance(self.eval, Mapping):
                raise ProtocolError("eval must be null or a mapping")
            EvalResult.from_dict(self.eval)
        if not isinstance(self.bytes, Mapping) or set(self.bytes) != {"down", "up"}:
            raise ProtocolError("bytes must contain exactly 'down' and 'up'")
        for name, value in self.bytes.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"bytes.{name} must be a non-negative integer")
        if not isinstance(self.code_versions, tuple) or len(self.code_versions) > 64:
            raise ProtocolError("code_versions must be a tuple")
        for version in self.code_versions:
            if not isinstance(version, str) or not version or len(version) > 64:
                raise ProtocolError("code_versions entries must be short strings")
        if not isinstance(self.certificate, tuple) or len(self.certificate) > 64:
            raise ProtocolError("certificate must be a tuple")
        for item in self.certificate:
            if not isinstance(item, Mapping) or set(item) != {"node", "pubkey", "sig"}:
                raise ProtocolError("certificate entries need node, pubkey and sig")
            _node_id(item["node"], "certificate node")
            if not isinstance(item["pubkey"], str) or not _HEX64.fullmatch(item["pubkey"]):
                raise ProtocolError("certificate pubkey must be 64 lowercase hex characters")
            if not isinstance(item["sig"], str) or not _HEX128.fullmatch(item["sig"]):
                raise ProtocolError("certificate sig must be 128 lowercase hex characters")

    def hash(self) -> str:
        """Hash of the signed payload (the certificate is not covered)."""

        return digest_of(self.signing_payload())

    def with_certificate(self, signatures: Sequence[tuple[str, str, str]]) -> "RoundRecord":
        """Return a copy carrying ``(node, pubkey, sig)`` certificate entries."""

        return RoundRecord(
            **{
                **{field.name: getattr(self, field.name) for field in fields(self) if field.name != "certificate"},
                "certificate": tuple(
                    {"node": node, "pubkey": pubkey, "sig": sig} for node, pubkey, sig in signatures
                ),
            }
        )


FIELDS = {cls: [field.name for field in fields(cls)] for cls in (Ticket, StepReceipt, EvalResult, CheckpointRef, NodeRecord, RoundRecord)}


def new_ticket_id() -> str:
    """A random 128-bit ticket identifier."""

    return secrets.token_hex(16)


def new_nonce() -> str:
    """A random 256-bit nonce."""

    return secrets.token_hex(32)


def verify_round_certificate(record: Mapping[str, Any], certificate: Mapping[str, Any]) -> bool:
    """Verify one ``{node, pubkey, sig}`` certificate over a round record.

    The certificate is excluded from the signed payload, matching
    ``RoundRecord.signing_payload()`` and the server's ``sign_record``. The
    caller decides how many distinct certificates are required.
    """
    try:
        pubkey = _require_hex64(certificate["pubkey"], "pubkey")
        signature = certificate["sig"]
        if not isinstance(signature, str) or not _HEX128.fullmatch(signature):
            return False
        body = {k: v for k, v in record.items() if k != "certificate"}
        _, public_cls = _ed25519()
        public = public_cls.from_public_bytes(bytes.fromhex(pubkey))
        public.verify(bytes.fromhex(signature), canonical_bytes(body))
        return True
    except Exception:
        return False
