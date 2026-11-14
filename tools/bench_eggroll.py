#!/usr/bin/env python3
"""Benchmark the EGGROLL mechanism spike on a toy linear model.

This is a mechanism benchmark, not a 658M-model result. It reports convergence
and the size of a scalar-only upload versus a dense bf16 delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowdgpt_ng.eggroll import EggrollConfig, benchmark  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model-params", type=int, default=657_787_392)
    args = p.parse_args()

    cfg = EggrollConfig(
        population=args.population,
        rank=args.rank,
        sigma=args.sigma,
        lr=args.lr,
        seed=args.seed,
    )
    history = benchmark(steps=args.steps, config=cfg)
    first, last = history[0], history[-1]
    dense_bytes = args.model_params * 2
    scalar_bytes = 8 + 4 * args.population
    result = {
        "schema": "ng-eggroll-spike-1",
        "steps": args.steps,
        "population": args.population,
        "rank": args.rank,
        "first_loss": round(first["loss"], 8),
        "last_loss": round(last["loss"], 8),
        "loss_ratio": round(last["loss"] / max(1e-12, first["loss"]), 6),
        "toy_dense_bytes": last["dense_bytes"],
        "toy_scalar_bytes": last["scalar_bytes"],
        "toy_compression": round(last["compression"], 4),
        "projected_dense_bytes": dense_bytes,
        "projected_scalar_bytes": scalar_bytes,
        "projected_compression": round(dense_bytes / max(1, scalar_bytes), 2),
        "note": "toy linear model; projected bytes are arithmetic, not a production measurement",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
