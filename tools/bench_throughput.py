#!/usr/bin/env python3
"""One throughput metric for every efficiency change.

Tokens per GPU per day and round wall-clock are the numbers N2.1 (ROCm
tuning) and N2.3 (delta codec) have to report, otherwise a win in one place
looks like a loss in another. This benchmark drives the production training
step (``client.py``'s ``_fwl``) and prints the same JSON on CUDA, ROCm and CPU.

    python tools/bench_throughput.py --device cpu --seq 32 --layers 2 --steps 1
    python tools/bench_throughput.py --device rocm --steps 50

``--preset real`` uses the shipped model configuration (658M base + 350M
engram). ``--seq``/``--batch``/``--layers`` shrink the work for a local smoke
test; a shrunk run says nothing about tokens/GPU/day. ``--overheads``
additionally measures the checkpointing cost that N2.5 asks about.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def load_client_module(repo_root: str | Path = REPO) -> object:
    """Import ``client.py`` once.

    Deliberately local: this tool must stay mergeable without the evaluator
    work, and it is the only thing it needs from the rest of the tree.
    """

    import importlib.util

    path = Path(repo_root) / "client.py"
    if not path.is_file():
        raise SystemExit(f"client.py not found at {path}")
    name = "crowdgpt_client_for_bench"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


# The shipped configuration. `--seq`, `--batch` and `--layers` override it for
# quick local runs; a trimmed run is a smoke test, not a throughput number.
PRESETS = {"real": {"seq": 2048, "batch": 1}}


def _sync(device: torch.device) -> None:
    if device.type in ("cuda", "xpu"):
        torch.cuda.synchronize()


def build(preset: str, seq: int | None, batch: int | None, layers: int | None,
          device: str | None = None) -> tuple[object, torch.device, str, dict]:
    client = load_client_module(REPO)
    cfg = dict(PRESETS[preset])
    dev, backend = client.detect_training_backend(device)
    sl = seq or cfg["seq"]
    bs = batch or cfg["batch"]
    model = client.SotaGPT(backend, dev, gpu_layers=None, bs=bs, sl=sl, u8_override=False)
    trimmed = 0
    if layers and layers < len(model.blocks):
        trimmed = len(model.blocks) - layers
        model.blocks = model.blocks[:layers]
    model.train()
    return model, dev, backend, {"seq": sl, "batch": bs, "preset": preset, "layers": len(model.blocks),
                                 "trimmedLayers": trimmed}


def step(model, client, dev, backend, sl: int, bs: int, opt, autocast: tuple[bool, torch.dtype], loss_chunk: int):
    x = torch.randint(0, client.VOCAB_SIZE, (bs, sl), device=dev)
    y = torch.randint(0, client.VOCAB_SIZE, (bs, sl), device=dev)
    enabled, dtype = autocast
    opt.zero_grad(set_to_none=True)
    loss, _ = client._fwl(model, x, y, sl, enabled, dtype, 1.0, False, loss_chunk)
    loss.backward()
    opt.step()
    return float(loss.detach())


def run(model, client, dev, backend, sl: int, bs: int, steps: int, warmup: int, loss_chunk: int) -> dict:
    autocast = client._pick_autocast(backend)
    ob, oe = client._build_optimizers(model, 1e-4)
    opt = client._MultiOpt([ob, oe])
    for _ in range(warmup):
        step(model, client, dev, backend, sl, bs, opt, autocast, loss_chunk)
    _sync(dev)
    start = time.perf_counter()
    losses = [step(model, client, dev, backend, sl, bs, opt, autocast, loss_chunk) for _ in range(steps)]
    _sync(dev)
    seconds = time.perf_counter() - start
    tokens = steps * bs * sl
    peak = None
    if dev.type in ("cuda", "xpu"):
        peak = round(torch.cuda.max_memory_reserved(dev) / 1024**3, 2)
    return {
        "steps": steps,
        "batch": bs,
        "seq": sl,
        "seconds": round(seconds, 4),
        "tokens": tokens,
        "tokensPerSecond": round(tokens / seconds, 2),
        "tokensPerGpuDay": int(tokens / seconds * 86400),
        "peakReservedGb": peak,
        "firstLoss": round(losses[0], 6),
        "lastLoss": round(losses[-1], 6),
        "autocast": {"enabled": autocast[0], "dtype": str(autocast[1])},
        "lossChunk": loss_chunk,
    }


def overheads(model, client, dev, backend, sl: int, bs: int, steps: int, loss_chunk: int) -> dict:
    """N2.5: what checkpointing and streamed blocks cost per step."""

    original = bool(getattr(model, "use_ckpt", True))
    out: dict[str, object] = {}
    for label, flag in (("withCheckpointing", True), ("withoutCheckpointing", False)):
        model.use_ckpt = flag
        out[label] = run(model, client, dev, backend, sl, bs, steps, 0, loss_chunk)["tokensPerSecond"]
    model.use_ckpt = original
    if out["withoutCheckpointing"]:
        # Positive means checkpointing costs throughput (the expected sign).
        out["checkpointingCostPct"] = round(
            100.0 * (out["withoutCheckpointing"] - out["withCheckpointing"]) / out["withoutCheckpointing"], 1)
    return out


def project(throughput: dict, *, round_steps: int, round_seq: int, devices: int = 1) -> dict:
    seconds_per_round = round_steps * round_seq / throughput["tokensPerSecond"]
    return {
        "roundSteps": round_steps,
        "roundSeq": round_seq,
        "devices": devices,
        "roundHours": round(seconds_per_round / 3600, 2),
        "tokensPerGpuDay": throughput["tokensPerGpuDay"],
        "tokensPerDayAtThisSize": int(throughput["tokensPerSecond"] * 86400 * devices),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cpu / cuda / rocm (default: auto)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="real")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--loss-chunk", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None, help="trim blocks after build (smoke tests only)")
    parser.add_argument("--overheads", action="store_true", help="also measure checkpointing cost")
    parser.add_argument("--round-steps", type=int, default=500)
    parser.add_argument("--round-seq", type=int, default=2048)
    parser.add_argument("--devices", type=int, default=1, help="fleet size for the round projection")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model, dev, backend, cfg = build(args.preset, args.seq, args.batch, args.layers, args.device)
    client = load_client_module(REPO)
    loss_chunk = args.loss_chunk or client.LOSS_CHUNK
    throughput = run(model, client, dev, backend, cfg["seq"], cfg["batch"], args.steps, args.warmup, loss_chunk)
    result = {
        "schema": "ng-throughput-1",
        "device": str(dev),
        "backend": backend,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "config": cfg,
        "params": int(sum(p.numel() for p in model.parameters())),
        "throughput": throughput,
        "projection": project(throughput, round_steps=args.round_steps, round_seq=args.round_seq,
                              devices=args.devices),
    }
    if args.overheads:
        result["overheads"] = overheads(model, client, dev, backend, cfg["seq"], cfg["batch"], args.steps, loss_chunk)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if not math.isfinite(throughput["tokensPerSecond"]) or throughput["tokensPerSecond"] <= 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
