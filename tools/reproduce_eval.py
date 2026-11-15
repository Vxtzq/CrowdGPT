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
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from crowdgpt_ng.eval import (  # noqa: E402
    EvalError,
    evaluate,
    load_manifest,
    uniform_logits,
    verify_shards,
)
from crowdgpt_ng.loss_gate import client_logits_from_safetensors, load_client_module  # noqa: E402

DEFAULT_MANIFEST = "eval/ng-eval-v1/manifest.json"
HF_REPO = "Vxtzq/Crowd-v1"


def checkpoint_paths(model_path: str | None, engram_path: str | None) -> tuple[str, str]:
    if model_path and engram_path:
        return model_path, engram_path
    from huggingface_hub import hf_hub_download

    if not model_path:
        model_path = hf_hub_download(HF_REPO, "model_bf16.safetensors", revision="main")
    if not engram_path:
        engram_path = hf_hub_download(HF_REPO, "engram_bf16.safetensors", revision="main")
    return model_path, engram_path


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
        model_path, engram_path = checkpoint_paths(weights, engram)
        fn = client_logits_from_safetensors(model_path, engram_path, manifest.seq_len, device=device,
                                            repo_root=REPO_ROOT)
        version = getattr(load_client_module(REPO_ROOT), "CLIENT_VERSION", "unknown")
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
