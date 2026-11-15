#!/usr/bin/env python3
"""Reproduce the published held-out loss from a fresh clone.

``ng-eval-v1`` pins every shard by SHA-256, so a third party only needs this
repository, the manifest and a checkpoint. Nothing is trained and no
coordinator is contacted.

    # prove the harness itself: uniform logits must give ln(vocab_size)
    python tools/reproduce_eval.py --model uniform

    # score the real checkpoint
    python tools/reproduce_eval.py --model client --count 64
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crowdgpt_ng.eval import (  # noqa: E402
    EvalError,
    evaluate,
    load_manifest,
    uniform_logits,
    verify_shards,
)

DEFAULT_MANIFEST = "eval/ng-eval-v1/manifest.json"


def _load_client():
    path = REPO_ROOT / "client.py"
    if not path.is_file():
        raise EvalError(f"client.py not found at {path}")
    spec = importlib.util.spec_from_file_location("crowdgpt_client_for_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def client_logits(manifest, *, device: str | None = None, weights: str | None = None, engram: str | None = None):
    """Logits from the real CrowdGPT checkpoint, in flat safetensors order."""

    import torch

    client = _load_client()
    if weights is None:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(client.HF_MODEL_REPO_ID, "model_bf16.safetensors", revision="main")
    if engram is None:
        from huggingface_hub import hf_hub_download

        engram = hf_hub_download(client.HF_MODEL_REPO_ID, "engram_bf16.safetensors", revision="main")

    backend, dev = client.detect_training_backend(device)
    iw = client._load_safetensors_flat(weights, client.EXPECTED_MODEL_SIZE, "model weights")
    ie = None
    if getattr(client, "ENGRAM_ACTIVE", False):
        ie = client._load_safetensors_flat(engram, client.EXPECTED_ENGRAM_SIZE, "engram weights")
    model = client._build_model(backend, dev, iw, ie, lambda *a, **k: None, bs=1, sl=manifest.seq_len)
    model.eval()
    torch.set_grad_enabled(False)

    def logits(inputs):
        idx = torch.as_tensor(np.asarray(inputs, dtype=np.int64), device=dev).unsqueeze(0)
        out = model(idx)
        return out[0].float().cpu().numpy()

    return logits


def run(
    manifest_path: Path,
    *,
    model: str,
    count: int | None,
    seed: int,
    data_root: Path,
    device: str | None,
    weights: str | None,
    engram: str | None,
) -> dict:
    manifest = load_manifest(manifest_path)
    verify_shards(manifest, root=data_root)
    if model == "uniform":
        fn = uniform_logits(manifest.vocab_size)
        model_id = "uniform"
    elif model == "client":
        fn = client_logits(manifest, device=device, weights=weights, engram=engram)
        version = getattr(_load_client(), "CLIENT_VERSION", "unknown")
        model_id = f"crowdgpt-client-{version}"
    else:
        raise EvalError(f"unknown model: {model}")

    report = evaluate(manifest, fn, root=data_root, count=count, seed=seed, model_id=model_id)
    result = report.to_dict()
    if model == "uniform":
        expected = math.log(manifest.vocab_size)
        result["expectedLoss"] = f"{expected:.6f}"
        result["matchesExpected"] = abs(float(report.loss) - expected) < 1e-3
        if not result["matchesExpected"]:
            raise EvalError(f"uniform harness produced {report.loss}, expected {expected:.6f}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--model", choices=("uniform", "client"), default="uniform")
    parser.add_argument("--data-root", default=str(REPO_ROOT))
    parser.add_argument("--count", type=int, default=None, help="windows to score (default 64)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="client mode: cpu / cuda / rocm")
    parser.add_argument("--weights", default=None, help="client mode: flat model safetensors")
    parser.add_argument("--engram", default=None, help="client mode: flat engram safetensors")
    args = parser.parse_args(argv)

    try:
        result = run(
            Path(args.manifest),
            model=args.model,
            count=args.count,
            seed=args.seed,
            data_root=Path(args.data_root),
            device=args.device,
            weights=args.weights,
            engram=args.engram,
        )
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
