"""The loss-gate contract: measure a staged candidate on the held-out set.

Two different evaluator contracts exist in this project and they are easy to
confuse:

* ``POST {prefix}/eval`` asks an evaluator to report the loss of the
  coordinator's *current* checkpoint. It is what publishes ``heldoutLoss``.
* ``POST {prefix}/gate`` is the ratchet: the coordinator stages the candidate
  weights it is about to accept and posts the URLs. The answer decides whether
  ``LOSS_CHECK_MODE=enforce`` rolls the round back.

This module implements the gate wire format and nothing else. It does not
decide accept or reject -- the coordinator owns the ratchet, the tolerance and
the rollback -- it only refuses to invent a number when it cannot measure one.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .eval import EvalError, EvalReport, evaluate, load_manifest, verify_shards

GATE_SOURCE = "loss-gate"
MAX_GATE_BODY_BYTES = 8192
MAX_URL_CHARS = 4096
DEFAULT_MAX_BYTES = 4 * 1024**3
DOWNLOAD_CHUNK = 1 << 20


class GateError(ValueError):
    """A request that must be answered with a 400."""


@dataclass(frozen=True)
class Candidate:
    model_url: str
    engram_url: str | None = None
    model_sha256: str | None = None
    engram_sha256: str | None = None


def _url(value: Any, name: str, *, required: bool) -> str | None:
    if value is None or value == "":
        if required:
            raise GateError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise GateError(f"{name} must be a string")
    if len(value) > MAX_URL_CHARS:
        raise GateError(f"{name} is longer than {MAX_URL_CHARS} characters")
    if not value.startswith(("http://", "https://", "file://")):
        raise GateError(f"{name} must be http(s):// or file://")
    return value


def _digest(value: Any, name: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise GateError(f"{name} must be a 64-character hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GateError(f"{name} must be hex") from exc
    return value.lower()


def _uint(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{name} must be a non-negative integer")
    return value


def parse_gate_request(payload: Any) -> dict[str, Any]:
    """Validate a coordinator gate request and return the normalised form."""

    if not isinstance(payload, Mapping):
        raise GateError("request body must be a JSON object")
    raw = payload.get("candidate")
    if not isinstance(raw, Mapping):
        raise GateError("candidate must be a JSON object")
    model_url = _url(raw.get("modelUrl") or raw.get("model_url"), "candidate.modelUrl", required=True)
    engram_url = _url(raw.get("engramUrl") or raw.get("engram_url"), "candidate.engramUrl", required=False)
    candidate = Candidate(
        model_url=str(model_url),
        engram_url=engram_url,
        model_sha256=_digest(raw.get("modelSha256") or raw.get("model_sha256"), "candidate.modelSha256"),
        engram_sha256=_digest(raw.get("engramSha256") or raw.get("engram_sha256"), "candidate.engramSha256"),
    )
    previous = payload.get("previousAcceptedLoss")
    if previous is not None:
        if isinstance(previous, bool) or not isinstance(previous, (int, float)) or not math.isfinite(float(previous)):
            raise GateError("previousAcceptedLoss must be a finite number or null")
        previous = float(previous)
    return {
        "step": _uint(payload.get("step"), "step"),
        "round": _uint(payload.get("round"), "round"),
        "previousAcceptedLoss": previous,
        "candidate": candidate,
    }


def _report_body(report: EvalReport) -> dict[str, Any]:
    loss = float(report.loss)
    return {
        "loss": loss,
        "ciLow": float(report.ci_low),
        "ciHigh": float(report.ci_high),
        "windows": int(report.windows),
        "tokens": int(report.tokens),
        "manifestSha256": report.manifest_sha256,
        "modelId": report.model_id,
        "seed": int(report.seed),
        "source": GATE_SOURCE,
    }


def evaluate_gate(payload: Any, scorer: Callable[[dict[str, Any]], Any] | None) -> tuple[int, dict[str, Any]]:
    """Answer one gate request.

    ``scorer`` receives the parsed request and returns an :class:`EvalReport`
    (or any mapping with a finite ``loss``). It is injected so the wire format
    can be tested without a checkpoint or a GPU.
    """

    try:
        request = parse_gate_request(payload)
    except GateError as exc:
        return 400, {"error": "invalid gate request", "detail": str(exc)}
    if scorer is None:
        return 503, {"error": "evaluator not configured", "detail": "no scorer is wired; no loss was measured"}
    try:
        result = scorer(request)
    except EvalError as exc:
        return 502, {"error": "evaluator failed", "detail": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - any backend failure is a 502, not a crash
        return 502, {"error": "evaluator failed", "detail": type(exc).__name__}
    if isinstance(result, EvalReport):
        body = _report_body(result)
    elif isinstance(result, Mapping):
        body = dict(result)
        if not isinstance(body.get("loss"), (int, float)) or isinstance(body.get("loss"), bool):
            return 502, {"error": "evaluator returned no numeric loss"}
        if not math.isfinite(float(body["loss"])):
            return 502, {"error": "evaluator returned a non-finite loss"}
        body["loss"] = float(body["loss"])
        body.setdefault("source", GATE_SOURCE)
    else:
        return 502, {"error": "evaluator returned an unsupported result"}
    body["step"] = request["step"]
    body["round"] = request["round"]
    if request["previousAcceptedLoss"] is not None:
        body["previousAcceptedLoss"] = request["previousAcceptedLoss"]
    return 200, body


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, *, opener: Callable[..., Any] = urllib.request.urlopen,
              max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    written = 0
    with opener(url, timeout=600) as response, open(dest, "wb") as handle:
        while True:
            block = response.read(DOWNLOAD_CHUNK)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise EvalError(f"candidate artifact exceeds {max_bytes} bytes")
            handle.write(block)
    if written == 0:
        raise EvalError(f"empty candidate artifact: {url}")
    return dest


def fetch_candidate(candidate: Candidate, dest_dir: Path, *,
                    opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Path]:
    """Download the staged candidate and verify any declared digests."""

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, url, expected in (("model", candidate.model_url, candidate.model_sha256),
                                ("engram", candidate.engram_url, candidate.engram_sha256)):
        if url is None:
            continue
        path = _download(url, dest_dir / f"{role}.bin", opener=opener)
        if expected is not None:
            actual = _sha256(path)
            if actual != expected:
                raise EvalError(f"{role} digest mismatch: coordinator declared {expected}, downloaded {actual}")
        paths[role] = path
    return paths


def make_gate_scorer(*, manifest_path: str | Path, data_root: str | Path, count: int | None,
                     seed: int, logits_for: Callable[[Mapping[str, Path], Any], Callable[[Any], Any]],
                     fetch: Callable[[Candidate, Path], dict[str, Path]] = fetch_candidate) -> Callable[[dict[str, Any]], EvalReport]:
    """Build a scorer over a fixed manifest.

    ``logits_for(paths)`` turns downloaded artifacts into a ``logits_fn`` that
    :func:`crowdgpt_ng.eval.evaluate` can call; the adapter supplies one that
    runs ``client.py``'s model so the gate scores the same forward pass the
    swarm trains.
    """

    manifest = load_manifest(manifest_path)
    verify_shards(manifest, root=data_root)

    def scorer(request: dict[str, Any]) -> EvalReport:
        with tempfile.TemporaryDirectory(prefix="loss-gate-") as tmp:
            paths = fetch(request["candidate"], Path(tmp))
            fn = logits_for(paths, manifest)
            return evaluate(manifest, fn, root=data_root, count=count, seed=seed,
                            model_id=f"sha256:{request['candidate'].model_sha256 or 'unverified'}")

    return scorer


def gate_summary(manifest_path: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Describe the scoring setup for ``--print-config`` without loading weights."""

    manifest = load_manifest(manifest_path)
    return {
        "manifest": str(manifest_path),
        "manifestSha256": manifest.sha256(),
        "name": manifest.name,
        "version": manifest.version,
        "seqLen": manifest.seq_len,
        "stride": manifest.stride,
        "vocabSize": manifest.vocab_size,
        "tokens": sum(shard.tokens for shard in manifest.shards),
        "shards": [shard.path for shard in manifest.shards],
    }


def flat_bf16_tensor(path: str | Path, count: int, *, label: str = "weights") -> Any:
    """Read the coordinator's candidate format: flat little-endian bf16."""

    import numpy as np
    import torch

    path = Path(path)
    expected = int(count) * 2
    size = path.stat().st_size
    if size != expected:
        raise EvalError(f"{label}: expected {expected} bytes for {count} params, got {size}")
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16)).view(torch.bfloat16)


def load_client_module(repo_root: str | Path | None = None) -> Any:
    """Import ``client.py`` once; the gate and the repro tool share one copy."""

    import importlib.util
    import sys

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    path = root / "client.py"
    if not path.is_file():
        raise EvalError(f"client.py not found at {path}")
    name = "crowdgpt_client_for_eval"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def client_logits_from_arrays(iw: Any, ie: Any, seq_len: int, *, device: str | None = None,
                              repo_root: str | Path | None = None) -> Callable[[Any], Any]:
    """Build ``client.py``'s model from flat weight tensors and score with it."""

    import numpy as np
    import torch

    client = load_client_module(repo_root)
    dev, backend = client.detect_training_backend(device)
    model = client._build_model(backend, dev, iw, ie, lambda *a, **k: None, bs=1, sl=seq_len)
    model.eval()
    torch.set_grad_enabled(False)

    def logits(inputs: Any) -> Any:
        idx = torch.as_tensor(np.asarray(inputs, dtype=np.int64), device=dev).unsqueeze(0)
        with torch.no_grad():
            return model(idx)[0].float().cpu().numpy()

    return logits


def client_logits(weights: str | Path, engram: str | Path | None, seq_len: int, *,
                  device: str | None = None, repo_root: str | Path | None = None) -> Callable[[Any], Any]:
    """Gate mode: the coordinator serves candidate weights as flat bf16."""

    client = load_client_module(repo_root)
    iw = flat_bf16_tensor(weights, client.EXPECTED_MODEL_SIZE, label="candidate model")
    ie = None
    if engram is not None and getattr(client, "ENGRAM_ACTIVE", False):
        ie = flat_bf16_tensor(engram, client.EXPECTED_ENGRAM_SIZE, label="candidate engram")
    return client_logits_from_arrays(iw, ie, seq_len, device=device, repo_root=repo_root)


def client_logits_from_safetensors(model_path: str | Path, engram_path: str | Path | None, seq_len: int, *,
                                   device: str | None = None,
                                   repo_root: str | Path | None = None) -> Callable[[Any], Any]:
    """Reproduction mode: the HF checkpoint is one flat safetensors tensor per role."""

    client = load_client_module(repo_root)
    iw = client._load_safetensors_flat(str(model_path), client.EXPECTED_MODEL_SIZE, "model weights")
    ie = None
    if engram_path is not None and getattr(client, "ENGRAM_ACTIVE", False):
        ie = client._load_safetensors_flat(str(engram_path), client.EXPECTED_ENGRAM_SIZE, "engram weights")
    return client_logits_from_arrays(iw, ie, seq_len, device=device, repo_root=repo_root)
