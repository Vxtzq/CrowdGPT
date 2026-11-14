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
"""

from __future__ import annotations

import argparse
import array
import importlib
import json
import os
import sys
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

try:  # optional: the adapter must import with the standard library alone (CI)
    from fastapi import APIRouter, FastAPI, Request, Response
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - FastAPI is an optional dependency
    APIRouter = FastAPI = Request = Response = JSONResponse = None  # type: ignore[assignment,misc]

DEFAULT_STATE_PATH = Path(".crowdgpt_hivemind") / "eval_state.json"
DEFAULT_ROUTE_PREFIX = "/fl/loss_check"
MIN_SECRET_BYTES = 16
JobLoader = Callable[[lcs.LossCheckJob], tuple[Any, Any, Any]]


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


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the coordinator needs that should come from deployment configuration."""

    state_path: Path
    evaluator_keys: Mapping[str, bytes]
    server_secret: bytes = field(repr=False)
    model_factory: Callable[[], Any] | None = None
    route_prefix: str = DEFAULT_ROUTE_PREFIX

    def __post_init__(self) -> None:
        if not self.evaluator_keys:
            raise ValueError("at least one evaluator key is required")
        if any(not key for key in self.evaluator_keys.values()):
            raise ValueError("evaluator keys must not be empty")
        if len(self.server_secret) < MIN_SECRET_BYTES:
            raise ValueError(f"server secret must be at least {MIN_SECRET_BYTES} bytes")
        if self.model_factory is not None and not callable(self.model_factory):
            raise ValueError("model_factory must be callable or None")
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
        state_path = Path(env.get("CROWDGPT_HIVEMIND_STATE_PATH", "").strip() or DEFAULT_STATE_PATH)
        prefix = env.get("CROWDGPT_HIVEMIND_ROUTE_PREFIX", "").strip() or DEFAULT_ROUTE_PREFIX
        return cls(state_path=state_path, evaluator_keys=keys, server_secret=secret, model_factory=factory, route_prefix=prefix)

    def safe_summary(self) -> dict[str, Any]:
        """Loggable view: no secret or key material."""
        return {
            "state_path": str(self.state_path),
            "evaluator_ids": sorted(self.evaluator_keys),
            "server_secret_bytes": len(self.server_secret),
            "model_factory": _factory_name(self.model_factory),
            "route_prefix": self.route_prefix,
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


def _require_fastapi() -> None:
    if APIRouter is None:
        raise RuntimeError("FastAPI is not installed; install fastapi to create the eval router")


def create_router(config: ServiceConfig, loader: JobLoader | None = None, *, coordinator: lcs.LossCheckCoordinator | None = None,
                  **overrides: Any) -> Any:
    """Return a FastAPI router exposing the coordinator's job and receipt handlers.

    The two handlers are wired here rather than through ``hivemind_loss_check_service.
    fastapi_router`` because that factory imports ``Request`` inside its own body; with
    ``from __future__ import annotations`` FastAPI later resolves the string annotation
    against module globals, does not find it, and turns the POST body into a required
    query parameter (422). Keeping FastAPI imports at module scope here avoids the trap
    without editing the shared module.
    """
    _require_fastapi()
    coord = coordinator or build_coordinator(config, loader, **overrides)
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

    return router


def create_app(config: ServiceConfig, loader: JobLoader | None = None, *, coordinator: lcs.LossCheckCoordinator | None = None,
               **overrides: Any) -> Any:
    _require_fastapi()
    app = FastAPI(title="CrowdGPT HiveMind evaluation adapter")
    app.include_router(create_router(config, loader, coordinator=coordinator, **overrides))
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
    if args.print_config:
        print(json.dumps(config.safe_summary(), indent=2, sort_keys=True))
    if args.serve:
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("uvicorn is not installed; install uvicorn to serve the adapter") from exc
        uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
