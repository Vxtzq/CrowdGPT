#!/usr/bin/env python3
"""Per-domain held-out loss of a Crowd-v1 checkpoint, optionally next to a reference model with the same tokenizer.

Why: one held-out number hides where the model is weak. On the HiveMind T4 (2026-11-14, Crowd-v1 rev 5ac3916 + engram,
8 windows of 2048 tokens per domain) the gap to Qwen3-0.6B was +0.36 nat on CrowdGPT's own chunks but +3.84 on Python
code, +2.44 on math, +1.80 on Wikipedia text, +1.50 on science and +1.04 on fineweb-edu. That points at the data mix.
Run this on each new checkpoint to see whether added data closes the gaps.

Usage (one plain-text file per domain; documents separated by a blank line, or one document per file):
    python tools/domain_eval.py --model-dir path/with/model_bf16+engram_bf16 \\
        --domain code=code.txt --domain math=math.txt --domain wiki=wiki.txt \\
        [--reference Qwen/Qwen3-0.6B] [--device cuda] [--windows 8]

Report: mean next-token CE (nats/token) per domain, and the gap to the reference when given. Compare gaps rather than
absolute losses: a strong reference has probably seen these sources, and a domain the swarm trained on is not held out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EOS = 151643  # <|endoftext|> in the Qwen3 tokenizer
WINDOW = 2048


def parse_domains(items: Sequence[str]) -> dict[str, Path]:
    """`name=path` pairs -> {name: path}. Names must be unique and non-empty."""
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        name = name.strip()
        if not sep or not name or not path.strip():
            raise ValueError(f"--domain expects name=path, got {item!r}")
        if name in out:
            raise ValueError(f"duplicate domain {name!r}")
        out[name] = Path(path.strip())
    return out


def split_documents(text: str) -> list[str]:
    """Documents are separated by blank lines; empty pieces are dropped."""
    return [d.strip() for d in text.split("\n\n") if d.strip()]


def pack_documents(token_lists: Iterable[Sequence[int]], eos: int = EOS) -> list[int]:
    """Concatenate tokenized documents with an EOS after each, as CrowdGPT chunks are packed continuously."""
    out: list[int] = []
    for ids in token_lists:
        out.extend(int(t) for t in ids)
        out.append(int(eos))
    return out


def gap_report(results: dict[str, dict[str, float]], model: str = "crowd", reference: str | None = None) -> str:
    """Plain-text table, largest gap first when a reference is present."""
    rows = list(results.items())
    if reference:
        rows.sort(key=lambda kv: -(kv[1][model] - kv[1][reference]))
    lines = [f"{'domain':28s} {model:>8s}" + (f" {reference[:12]:>12s} {'gap':>8s}" if reference else "")]
    for name, r in rows:
        line = f"{name:28s} {r[model]:8.3f}"
        if reference:
            line += f" {r[reference]:12.3f} {r[model] - r[reference]:+8.3f}"
        lines.append(line)
    return "\n".join(lines)


def _reference_losses(name: str, windows: list, device: str) -> list[float]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    m = m.to(device).eval()
    out = []
    with torch.inference_mode():
        for w in windows:
            t = torch.as_tensor(w, dtype=torch.long, device=device)[None]
            h = m.model(input_ids=t[:, :-1]).last_hidden_state
            tot = 0.0
            for i in range(0, h.shape[1], 256):  # chunked: never materialize 2048 x vocab logits at once
                lg = F.linear(h[:, i:i + 256], m.lm_head.weight).float()
                tot += F.cross_entropy(lg[0], t[0, i + 1:i + 257], reduction="sum").item()
            out.append(tot / h.shape[1])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="directory with model_bf16.safetensors and engram_bf16.safetensors")
    ap.add_argument("--domain", action="append", required=True, help="name=path.txt (repeatable)")
    ap.add_argument("--reference", default=None, help="HF model id with the same tokenizer, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    import hivemind_loss_check as lc

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    domains = parse_domains(a.domain)
    windows: dict[str, list] = {}
    for name, path in domains.items():
        ids = np.array(pack_documents(tok(d).input_ids for d in split_documents(path.read_text(encoding="utf-8"))), dtype=np.uint32)
        starts = lc.heldout_windows(len(ids), WINDOW + 1, a.windows, a.seed)
        windows[name] = [ids[s:s + WINDOW + 1] for s in starts]

    model = lc.CrowdV1(lc.CROWD_V1_ENGRAM_CONFIG)
    model.load_flat(lc.load_flat_safetensors(a.model_dir / "model_bf16.safetensors").float(),
                    engram=lc.load_flat_safetensors(a.model_dir / "engram_bf16.safetensors").float())
    dtype = torch.float16 if a.device.startswith("cuda") else torch.float32
    results: dict[str, dict[str, float]] = {}
    for name, ws in windows.items():
        arr = np.concatenate(ws)
        starts = [i * (WINDOW + 1) for i in range(len(ws))]
        results[name] = {"crowd": float(np.mean(lc.window_losses(model, arr, starts, WINDOW + 1, device=a.device, dtype=dtype)))}
    del model
    if a.reference:
        for name, ws in windows.items():
            results[name][a.reference] = float(np.mean(_reference_losses(a.reference, ws, a.device)))
    print(gap_report(results, "crowd", a.reference))
    if a.json:
        a.json.write_text(json.dumps({"windows": a.windows, "window": WINDOW, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
