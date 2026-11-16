#!/usr/bin/env python3
"""Thin HiveMind evaluation adapter: config, a local-file loader, and a FastAPI router.

Why: `hivemind_loss_check_service.LossCheckCoordinator` already owns the job/receipt state
machine, while `hivemind_loss_check` owns receipt signing and verification. The missing
pieces for an evaluator sidecar were (a) configuration from the environment and (b) a
loader that can read base/candidate/held-out artifacts without a network. This module
fills only those gaps so the coordinator can run unchanged.

The loader contract is the coordinator's own:

    loader(job) -> (base, candidate, heldout)
`FileJobLoader` resolves each URL as a local path or ``file://`` and reads it. Tests and
deployments that fetch over HTTP inject their own callable; this adapter deliberately
does not grow an HTTP client.

Environment (all read only by ``ServiceConfig.from_env``):

    CROWDGPT_HIVEMIND_STATE_PATH       JSON state file (default .crowdgpt_hivemind/eval_state.json)
    CROWDGPT_HIVEMIND_EVALUATOR_KEYS   {"eval-a": "<hex key>", ...} or "eval-a:<hex>,eval-b:<hex>"
    CROWDGPT_HIVEMIND_SERVER_SECRET    "hex:<hex>" or a literal string (at least 16 bytes)
    CROWDGPT_HIVEMIND_MODEL_FACTORY    optional "module:attr" factory; defaults to hivemind_loss_check.CrowdV1
    CROWDGPT_HIVEMIND_ROUTE_PREFIX     optional route prefix (default /fl/loss_check)

Usage:
    python tools/hivemind_eval_service.py --print-config
    python tools/hivemind_eval_service.py --serve --host 127.0.0.1 --port 8099

The module is additive: it changes no client training path, no pool sizing, and no
existing HiveMind module. FastAPI is imported opportunistically and uvicorn is needed
only for ``--serve``.

It also answers the coordinator's own evaluation contract
(``POST {prefix}/eval`` with ``{"round", "globalStep", "checkpoint"}``), which is what
``CROWDGPT_EVAL_URL`` already points at. Pass an ``eval_fn`` to ``create_app`` to enable
it; without one the route returns 503 instead of inventing a loss.
"""

from __future__ import annotations

import argparse
import array
import importlib
import json
import math
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:  # copied next to the service module in the CrowdGPT repository
    import hivemind_loss_check as lc
    import hivemind_loss_check_service as lcs
except ModuleNotFoundError as exc:  # inside HiveMind the modules live under hivemind.swarm
    if exc.name not in ("hivemind_loss_check", "hivemind_loss_check_service"):
        raise
    from hivemind.swarm import crowdgpt_loss_check as lc  # type: ignore[no-redef]
    from hivemind.swarm import crowdgpt_loss_check_service as lcs  # type: ignore[no-redef]

try:  # the gate needs numpy (eval library); the adapter must import without it
    from crowdgpt_ng import loss_gate as lg
except ImportError:  # pragma: no cover - stdlib-only installs answer 503 from /gate
    lg = None  # type: ignore[assignment]

try:  # optional: the adapter must import with the standard library alone (CI)
    from fastapi import APIRouter, FastAPI, Request, Response
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - FastAPI is an optional dependency
    APIRouter = FastAPI = Request = Response = JSONResponse = None  # type: ignore[assignment,misc]

DEFAULT_STATE_PATH = Path(".crowdgpt_hivemind") / "eval_state.json"
DEFAULT_GATE_MANIFEST = Path("eval") / "ng-eval-v1" / "manifest.json"
MAX_GATE_BODY_BYTES = 8192
DEFAULT_ROUTE_PREFIX = "/fl/loss_check"
MIN_SECRET_BYTES = 16
MAX_EVAL_BODY_BYTES = 64 * 1024
MAX_CHECKPOINT_CHARS = 4096
JobLoader = Callable[[lcs.LossCheckJob], tuple[Any, Any, Any]]
EvalFn = Callable[[dict[str, Any]], Mapping[str, Any]]


# --------------------------------------------------------------------------------------------------- config
def _require_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _parse_hex(value: str, *, field_name: str) -> bytes:
    text = value.strip()
    if text.startswith("hex:"):
        text = text[4:]
    try:
        key = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hex-encoded: {exc}") from exc
    if not key:
        raise ValueError(f"{field_name} must not be empty")
    return key


def parse_evaluator_keys(raw: str) -> dict[str, bytes]:
    """Parse evaluator id -> HMAC key from a JSON object or a comma-separated ``id:hex`` list."""
    text = raw.strip()
    if not text:
        raise ValueError("evaluator keys must not be empty")
    pairs: list[tuple[str, str]]
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"evaluator keys are not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("evaluator keys JSON must be an object of id -> hex key")
        pairs = [(str(k), str(v)) for k, v in obj.items()]
    else:
        pairs = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            eid, sep, key = item.partition(":")
            if not sep:
                eid, sep, key = item.partition("=")
            if not sep:
                raise ValueError(f"evaluator key {item!r} must be id:hex")
            pairs.append((eid.strip(), key.strip()))
    out: dict[str, bytes] = {}
    for eid, key_text in pairs:
        if not eid:
            raise ValueError("evaluator id must not be empty")
        if eid in out:
            raise ValueError(f"duplicate evaluator id {eid!r}")
        out[eid] = _parse_hex(key_text, field_name=f"evaluator key for {eid!r}")
    if not out:
        raise ValueError("evaluator keys must not be empty")
    return out


def _looks_hex(text: str) -> bool:
    return bool(text) and len(text) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in text)


def parse_server_secret(raw: str) -> bytes:
    """``hex:<hex>`` or bare hex decode as bytes; any other string is UTF-8 encoded."""
    text = raw.strip()
    if not text:
        raise ValueError("server secret must not be empty")
    if text.startswith("hex:"):
        try:
            secret = bytes.fromhex(text[4:])
        except ValueError as exc:
            raise ValueError(f"server secret hex is invalid: {exc}") from exc
    elif _looks_hex(text):
        secret = bytes.fromhex(text)
    else:
        secret = text.encode("utf-8")
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(f"server secret must be at least {MIN_SECRET_BYTES} bytes")
    return secret


GATE_SIGNING_KEY_ENV = "CROWDGPT_GATE_SIGNING_KEY"


def parse_gate_signing_key(raw: str) -> bytes | None:
    """32-byte Ed25519 seed from hex (``hex:`` prefix tolerated), or ``None`` when unset.

    Unset means "answer unsigned", which keeps an unsigned deployment behaving exactly as
    it did before receipts existed. A malformed key raises at startup rather than at the
    first gate request, because a silently unsigned answer is the failure this path exists
    to remove.
    """
    text = (raw or "").strip()
    if not text:
        return None
    seed = _parse_hex(text, field_name=GATE_SIGNING_KEY_ENV)
    if len(seed) != 32:
        raise ValueError(f"{GATE_SIGNING_KEY_ENV} must be a 32-byte Ed25519 seed, got {len(seed)} bytes")
    return seed


def _gate_private_key(seed: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - the wheel ships with the eval extras
        raise RuntimeError("signing gate receipts needs the 'cryptography' package") from exc
    return Ed25519PrivateKey.from_private_bytes(seed)


def gate_public_key(seed: bytes) -> str:
    """64-hex public key the coordinator puts in its trusted-evaluator map."""
    from cryptography.hazmat.primitives import serialization

    return _gate_private_key(seed).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def canonical_gate_receipt(receipt: Mapping[str, Any]) -> bytes:
    """The bytes the coordinator verifies: sorted keys, no spaces, UTF-8."""
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_gate_receipt(body: Mapping[str, Any], payload: Any, *, now: float | None = None) -> dict[str, Any] | None:
    """The receipt a coordinator accepts, or ``None`` when the answer is not signable."""
    if not isinstance(body, Mapping):
        return None
    loss = body.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        return None
    if not math.isfinite(float(loss)):
        return None
    request = payload if isinstance(payload, Mapping) else {}
    candidate = request.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    try:
        step = int(body.get("step", request.get("step", 0)))
        rnd = int(body.get("round", request.get("round", 0)))
    except (TypeError, ValueError):
        return None
    return {
        "step": step,
        "round": rnd,
        "requestId": str(request.get("requestId") or ""),
        "modelSha256": str(candidate.get("modelSha256") or ""),
        "engramSha256": str(candidate.get("engramSha256") or ""),
        "loss": float(loss),
        "issuedAt": float(time.time() if now is None else now),
    }


def sign_gate_answer(body: Mapping[str, Any], payload: Any, seed: bytes | None, *,
                     now: float | None = None) -> dict[str, Any]:
    """Attach a signed receipt to a successful gate answer.

    The coordinator verifies the signature over the canonical receipt and then compares
    ``requestId``, both candidate digests and the loss against what it sent, so the receipt
    has to echo them verbatim -- including an empty digest when the candidate declared none.
    """
    answer = dict(body) if isinstance(body, Mapping) else {}
    if seed is None:
        return answer
    receipt = build_gate_receipt(answer, payload, now=now)
    if receipt is None:
        return answer
    private = _gate_private_key(seed)
    from cryptography.hazmat.primitives import serialization

    answer["receipt"] = receipt
    answer["pubkey"] = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    answer["signature"] = private.sign(canonical_gate_receipt(receipt)).hex()
    return answer


def _load_model_factory(spec: str) -> Callable[[], Any]:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("model factory must be module:attr")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import model factory module {module_name!r}: {exc}") from exc
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ValueError(f"{spec!r} is not callable")
    return factory


def _factory_name(factory: Callable[[], Any] | None) -> str | None:
    if factory is None:
        return None
    return f"{getattr(factory, '__module__', '?')}:{getattr(factory, '__qualname__', getattr(factory, '__name__', '?'))}"


def _load_eval_fn(spec: str) -> EvalFn:
    """Resolve ``CROWDGPT_HIVEMIND_EVAL_FN=module:attr`` into the /eval evaluator."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("eval function must be module:attr")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import eval module {module_name!r}: {exc}") from exc
    evaluator = getattr(module, attr, None)
    if not callable(evaluator):
        raise ValueError(f"{spec!r} is not callable")
    return evaluator  # type: ignore[return-value]


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the coordinator needs that should come from deployment configuration."""

    state_path: Path
    evaluator_keys: Mapping[str, bytes]
    server_secret: bytes = field(repr=False)
    model_factory: Callable[[], Any] | None = None
    route_prefix: str = DEFAULT_ROUTE_PREFIX
    eval_fn: EvalFn | None = None

    def __post_init__(self) -> None:
        if not self.evaluator_keys:
            raise ValueError("at least one evaluator key is required")
        if any(not key for key in self.evaluator_keys.values()):
            raise ValueError("evaluator keys must not be empty")
        if len(self.server_secret) < MIN_SECRET_BYTES:
            raise ValueError(f"server secret must be at least {MIN_SECRET_BYTES} bytes")
        if self.model_factory is not None and not callable(self.model_factory):
            raise ValueError("model_factory must be callable or None")
        if self.eval_fn is not None and not callable(self.eval_fn):
            raise ValueError("eval_fn must be callable or None")
        prefix = self.route_prefix.strip()
        if not prefix.startswith("/"):
            raise ValueError("route_prefix must start with '/'")
        object.__setattr__(self, "route_prefix", prefix.rstrip("/") or "/")
        object.__setattr__(self, "state_path", Path(self.state_path))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServiceConfig":
        env = os.environ if environ is None else environ
        keys = parse_evaluator_keys(_require_env(env, "CROWDGPT_HIVEMIND_EVALUATOR_KEYS"))
        secret = parse_server_secret(_require_env(env, "CROWDGPT_HIVEMIND_SERVER_SECRET"))
        factory_spec = env.get("CROWDGPT_HIVEMIND_MODEL_FACTORY", "").strip()
        factory = _load_model_factory(factory_spec) if factory_spec else None
        eval_spec = env.get("CROWDGPT_HIVEMIND_EVAL_FN", "").strip()
        eval_fn = _load_eval_fn(eval_spec) if eval_spec else None
        state_path = Path(env.get("CROWDGPT_HIVEMIND_STATE_PATH", "").strip() or DEFAULT_STATE_PATH)
        prefix = env.get("CROWDGPT_HIVEMIND_ROUTE_PREFIX", "").strip() or DEFAULT_ROUTE_PREFIX
        return cls(state_path=state_path, evaluator_keys=keys, server_secret=secret, model_factory=factory,
                   route_prefix=prefix, eval_fn=eval_fn)

    def safe_summary(self) -> dict[str, Any]:
        """Loggable view: no secret or key material."""
        return {
            "state_path": str(self.state_path),
            "evaluator_ids": sorted(self.evaluator_keys),
            "server_secret_bytes": len(self.server_secret),
            "model_factory": _factory_name(self.model_factory),
            "eval_fn": _factory_name(self.eval_fn),
            "route_prefix": self.route_prefix,
            "eval_endpoint": f"{self.route_prefix}/eval",
            "gate_endpoint": f"{self.route_prefix}/gate",
        }


# ------------------------------------------------------------------------------------------- loader
def resolve_local_path(url: str) -> Path:
    """Resolve a plain path or ``file://`` URL; reject network schemes so failures are explicit."""
    value = url.strip()
    if not value:
        raise ValueError("artifact URL must not be empty")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme in ("http", "https"):
        raise ValueError(f"{parsed.scheme} artifact URLs need an injected loader: {value!r}")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"file URL host must be empty or localhost: {value!r}")
        path = urllib.parse.unquote(parsed.path)
        if not path:
            raise ValueError(f"file URL has no path: {value!r}")
        return Path(path)
    if parsed.scheme and len(parsed.scheme) > 1:  # a one-letter scheme is a Windows drive
        raise ValueError(f"unsupported artifact URL scheme {parsed.scheme!r}: {value!r}")
    return Path(value).expanduser()


def _slice_byte_range(data: bytes, heldout_range: str) -> bytes:
    """Apply the job's inclusive ``start-end`` range."""
    spec = heldout_range.strip()
    if not spec:
        return data
    start_text, sep, end_text = spec.partition("-")
    if not sep:
        raise ValueError(f"heldout_range must be start-end, got {heldout_range!r}")
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError(f"heldout_range must be numeric, got {heldout_range!r}") from exc
    if start < 0 or end < start:
        raise ValueError(f"heldout_range is invalid: {heldout_range!r}")
    if end >= len(data):
        raise ValueError(f"heldout_range {heldout_range!r} exceeds {len(data)} bytes")
    return data[start:end + 1]


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("numpy is required to read .npy held-out tokens") from exc
    return np


def read_heldout_tokens(path: Path | str, heldout_range: str = "") -> Any:
    """Read held-out token ids from ``.npy``, ``.json``, or raw little-endian uint32 bytes.

    Ranges apply only to raw files; a structured file is already an exact token array.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"held-out token file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".json", ".npy") and heldout_range.strip():
        raise ValueError(f"heldout_range applies only to raw token files, not {suffix}")
    if suffix == ".json":
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"held-out JSON is invalid: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("held-out JSON must be a list of token ids")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in raw):
            raise ValueError("held-out JSON must contain only integers")
        if any(x < 0 or x > 0xFFFFFFFF for x in raw):
            raise ValueError("held-out token ids must fit in uint32")
        return raw
    if suffix == ".npy":
        np = _require_numpy()
        arr = np.load(p, allow_pickle=False).reshape(-1)
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("held-out .npy tokens must have an integer dtype")
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 0xFFFFFFFF):
            raise ValueError("held-out token ids must fit in uint32")
        return np.ascontiguousarray(arr, dtype=np.uint32)
    data = _slice_byte_range(p.read_bytes(), heldout_range)
    if len(data) % 4:
        raise ValueError(f"raw held-out file {p} has {len(data)} bytes, not a multiple of 4")
    tokens = array.array("I")
    tokens.frombytes(data)
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens


class FileJobLoader:
    """Read a job's three artifacts from local files; safe to use without a network.

    ``weights_loader`` is injectable so tests can supply tiny byte blobs instead of a
    658M-parameter checkpoint. Hash checks fail before any forward pass when a file does
    not match the job, which is the cheapest point to catch a bad deployment.
    """

    def __init__(self, *, weights_loader: Callable[[Path], Any] | None = None, verify_hashes: bool = True) -> None:
        self.weights_loader = weights_loader or lc.load_flat_safetensors
        self.verify_hashes = verify_hashes

    def __call__(self, job: lcs.LossCheckJob) -> tuple[Any, Any, Any]:
        base = self.weights_loader(resolve_local_path(job.base_url))
        candidate = self.weights_loader(resolve_local_path(job.candidate_url))
        heldout = read_heldout_tokens(resolve_local_path(job.heldout_url), job.heldout_range)
        if self.verify_hashes:
            if lc.weights_sha256(base) != job.base_sha256:
                raise ValueError("base weights do not match job.base_sha256")
            if lc.weights_sha256(candidate) != job.candidate_sha256:
                raise ValueError("candidate weights do not match job.candidate_sha256")
            if lc.tokens_sha256(heldout) != job.heldout_sha256:
                raise ValueError("held-out tokens do not match job.heldout_sha256")
        return base, candidate, heldout


# ------------------------------------------------------------------------------------------- wiring
def build_coordinator(config: ServiceConfig, loader: JobLoader | None = None, **overrides: Any) -> lcs.LossCheckCoordinator:
    """Create the unchanged coordinator with config-derived defaults; ``overrides`` tune verification."""
    kwargs: dict[str, Any] = {"model_factory": config.model_factory, **overrides}
    return lcs.LossCheckCoordinator(
        evaluator_keys=config.evaluator_keys,
        server_secret=config.server_secret,
        state_path=config.state_path,
        loader=loader or FileJobLoader(),
        **kwargs,
    )


@dataclass(frozen=True)
class GateConfig:
    """Where the loss-gate contract gets its checkpoints and held-out tokens."""

    manifest: Path = DEFAULT_GATE_MANIFEST
    data_root: Path = REPO
    count: int = 64
    seed: int = 0
    device: str = ""
    # Empty means "answer unsigned"; see parse_gate_signing_key.
    signing_key: bytes = b""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "GateConfig":
        env = os.environ if environ is None else environ
        manifest = Path(env.get("CROWDGPT_EVAL_MANIFEST", "").strip() or DEFAULT_GATE_MANIFEST)
        data_root = Path(env.get("CROWDGPT_EVAL_DATA_ROOT", "").strip() or REPO)
        count = _env_int(env, "CROWDGPT_EVAL_COUNT", 64, 1, 4096)
        seed = _env_int(env, "CROWDGPT_EVAL_SEED", 0, 0, 2**31 - 1)
        device = env.get("CROWDGPT_EVAL_DEVICE", "").strip()
        signing_key = parse_gate_signing_key(env.get(GATE_SIGNING_KEY_ENV, "")) or b""
        return cls(manifest=manifest, data_root=data_root, count=count, seed=seed, device=device,
                   signing_key=signing_key)

    def summary(self) -> dict[str, Any]:
        info = self._summary_body()
        if self.signing_key:
            try:
                info["receiptSigner"] = gate_public_key(self.signing_key)
            except Exception as exc:  # noqa: BLE001 - --print-config must never crash
                info["receiptSigner"] = f"unavailable ({type(exc).__name__})"
        return info

    def _summary_body(self) -> dict[str, Any]:
        if lg is None:
            return {"manifest": str(self.manifest), "missing": True,
                    "detail": "numpy is not installed; the gate answers 503"}
        if not self.manifest.is_file():
            return {"manifest": str(self.manifest), "missing": True,
                    "detail": "the gate answers 503 until a manifest exists"}
        try:
            return {**lg.gate_summary(self.manifest, self.data_root), "count": self.count, "seed": self.seed,
                    "device": self.device or "auto"}
        except Exception as exc:  # noqa: BLE001 - a broken manifest must not crash --print-config
            return {"manifest": str(self.manifest), "error": type(exc).__name__}


def _env_int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def build_gate_scorer(gate: GateConfig) -> Callable[[dict[str, Any]], Any] | None:
    """None when there is no manifest; the gate then refuses with a 503."""

    if lg is None or not gate.manifest.is_file():
        return None

    def logits_for(paths: Mapping[str, Path], manifest: Any) -> Callable[[Any], Any]:
        return lg.client_logits(paths["model"], paths.get("engram"), manifest.seq_len,
                                device=gate.device or None, repo_root=REPO)

    return lg.make_gate_scorer(manifest_path=gate.manifest, data_root=gate.data_root, count=gate.count,
                               seed=gate.seed, logits_for=logits_for)


def build_gate_signer(gate: GateConfig) -> Callable[[dict[str, Any], Any], dict[str, Any]] | None:
    """``None`` when no key is configured, so an unsigned deployment is unchanged."""
    seed = gate.signing_key
    if not seed:
        return None
    return lambda body, payload: sign_gate_answer(body, payload, seed)


def _require_fastapi() -> None:
    if APIRouter is None:
        raise RuntimeError("FastAPI is not installed; install fastapi to create the eval router")


# ------------------------------------------------------------------------------- coordinator /eval contract
# `server.py::_round_eval` posts to ``{CROWDGPT_EVAL_URL}/eval`` with
# ``{"round", "globalStep", "checkpoint"}`` and reads back ``{"loss", "ciLow",
# "ciHigh", "windows", "manifestSha256", "modelSha256"}``. The job/receipt routes
# above speak the HiveMind protocol; this route speaks the coordinator's, so one
# process can answer both and the env var stops 404ing.
def _eval_uint(body: Mapping[str, Any], name: str) -> int:
    value = body.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _eval_number(body: Mapping[str, Any], name: str) -> float | None:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def evaluate_round(payload: Any, eval_fn: EvalFn | None) -> tuple[int, dict[str, Any]]:
    """Answer the coordinator's ``POST /eval``: return ``(http_status, body)``.

    Kept as a pure function so the contract is testable with the standard library
    alone; the FastAPI route is a thin wrapper. A missing evaluator is ``503``
    rather than a fabricated loss, so a misconfigured sidecar cannot publish a
    number nobody computed.
    """
    if eval_fn is None:
        return 503, {"error": "evaluator not configured",
                     "detail": "pass eval_fn=... to create_app(); no loss was computed"}
    if not isinstance(payload, Mapping):
        return 400, {"error": "eval request must be a JSON object"}
    try:
        round_no = _eval_uint(payload, "round")
        step = _eval_uint(payload, "globalStep")
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")
        if len(checkpoint) > MAX_CHECKPOINT_CHARS:
            raise ValueError("checkpoint path is too long")
    except ValueError as exc:
        return 400, {"error": "invalid eval request", "detail": str(exc)}
    try:
        result = eval_fn({"round": round_no, "globalStep": step, "checkpoint": checkpoint})
    except Exception as exc:  # the type is useful; the traceback would leak paths
        return 502, {"error": "evaluator failed", "detail": type(exc).__name__}
    if not isinstance(result, Mapping):
        return 502, {"error": "evaluator returned a non-object"}
    try:
        loss = _eval_number(result, "loss")
        if loss is None:
            raise ValueError("loss is required")
        ci_low = _eval_number(result, "ciLow")
        ci_high = _eval_number(result, "ciHigh")
    except ValueError as exc:
        return 502, {"error": "evaluator returned an invalid result", "detail": str(exc)}
    if ci_low is not None and ci_high is not None and ci_low > ci_high:
        return 502, {"error": "evaluator returned an inverted confidence interval"}

    def _digest(name: str) -> str | None:
        value = result.get(name)
        return value if isinstance(value, str) and value else None

    return 200, {
        "round": round_no,
        "globalStep": step,
        "loss": loss,
        "ciLow": ci_low,
        "ciHigh": ci_high,
        "windows": result.get("windows"),
        "manifestSha256": _digest("manifestSha256"),
        "modelSha256": _digest("modelSha256"),
        "source": "hivemind-eval-service",
    }


def create_router(config: ServiceConfig, loader: JobLoader | None = None, *, coordinator: lcs.LossCheckCoordinator | None = None,
                  eval_fn: EvalFn | None = None, gate_scorer: Callable[[dict[str, Any]], Any] | None = None,
                  gate_signer: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
                  **overrides: Any) -> Any:
    """Return a FastAPI router exposing the job, receipt, and coordinator ``/eval`` handlers.

    The two handlers are wired here rather than through ``hivemind_loss_check_service.
    fastapi_router`` because that factory imports ``Request`` inside its own body; with
    ``from __future__ import annotations`` FastAPI later resolves the string annotation
    against module globals, does not find it, and turns the POST body into a required
    query parameter (422). Keeping FastAPI imports at module scope here avoids the trap
    without editing the shared module.
    """
    _require_fastapi()
    coord = coordinator or build_coordinator(config, loader, **overrides)
    evaluator = eval_fn if eval_fn is not None else config.eval_fn
    router = APIRouter(prefix=config.route_prefix)

    @router.get("/job")
    def get_job(evaluator_id: str) -> Response:
        status, body = coord.handle_get_job(evaluator_id)
        return Response(status_code=204) if body is None else JSONResponse(body, status_code=status)

    @router.post("/receipt")
    async def post_receipt(request: Request) -> Response:
        from starlette.concurrency import run_in_threadpool

        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > lcs.MAX_RECEIPT_BYTES:
            return JSONResponse({"error": "receipt too large"}, status_code=413)
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > lcs.MAX_RECEIPT_BYTES:
                return JSONResponse({"error": "receipt too large"}, status_code=413)
        # Verification loads weights and recomputes losses; keep it off the event loop.
        status, body = await run_in_threadpool(coord.handle_post_receipt, bytes(chunks))
        return JSONResponse(body, status_code=status)

    @router.post("/eval")
    async def post_eval(request: Request) -> Response:
        from starlette.concurrency import run_in_threadpool

        raw = await request.body()
        if len(raw) > MAX_EVAL_BODY_BYTES:
            return JSONResponse({"error": "eval request too large"}, status_code=413)
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        status, body = await run_in_threadpool(evaluate_round, payload, evaluator)
        return JSONResponse(body, status_code=status)

    @router.post("/gate")
    async def post_gate(request: Request) -> Response:
        from starlette.concurrency import run_in_threadpool

        raw = await request.body()
        if len(raw) > MAX_GATE_BODY_BYTES:
            return JSONResponse({"error": "gate request too large"}, status_code=413)
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        # Downloading a candidate and scoring 64 windows takes minutes; never on the loop.
        if lg is None:  # no numpy: refuse rather than import-fail at request time
            return JSONResponse({"error": "evaluator not configured",
                                 "detail": "numpy is not installed"}, status_code=503)
        status, body = await run_in_threadpool(lg.evaluate_gate, payload, gate_scorer)
        if gate_signer is not None and status == 200 and isinstance(body, Mapping):
            # A coordinator that requires receipts rejects an unsigned answer, so only
            # sign what it would accept: a 200 carrying a finite loss.
            body = gate_signer(body, payload)
        return JSONResponse(body, status_code=status)

    return router


def create_app(config: ServiceConfig, loader: JobLoader | None = None, *, coordinator: lcs.LossCheckCoordinator | None = None,
               eval_fn: EvalFn | None = None, gate_scorer: Callable[[dict[str, Any]], Any] | None = None,
               gate_signer: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
               **overrides: Any) -> Any:
    _require_fastapi()
    app = FastAPI(title="CrowdGPT HiveMind evaluation adapter")
    app.include_router(create_router(config, loader, coordinator=coordinator, eval_fn=eval_fn,
                                     gate_scorer=gate_scorer, gate_signer=gate_signer, **overrides))
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HiveMind evaluation adapter (job/receipt router)")
    parser.add_argument("--print-config", action="store_true", help="print the parsed environment config with secrets masked")
    parser.add_argument("--serve", action="store_true", help="run the router with uvicorn")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args(argv)
    if not args.print_config and not args.serve:
        parser.print_help()
        return 2
    config = ServiceConfig.from_env()
    gate = GateConfig.from_env()
    if args.print_config:
        summary = config.safe_summary()
        summary["gate"] = gate.summary()
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.serve:
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("uvicorn is not installed; install uvicorn to serve the adapter") from exc
        uvicorn.run(create_app(config, gate_scorer=build_gate_scorer(gate),
                              gate_signer=build_gate_signer(gate)),
                    host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
