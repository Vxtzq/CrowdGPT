#!/usr/bin/env python3
"""Build a hash-pinned held-out evaluation set from raw text.

The training corpus (``Vxtzq/CrowdGPT``) is pre-tokenized raw ``uint32``. A
held-out set has to be data the swarm cannot train on, so this builder reads
text that is still in the review queue (``Vxtzq/CrowdGPT-Pending``), freezes it
under a pinned revision, tokenizes it with the corpus tokenizer and writes
``uint32`` shards plus a ``ng-eval-1`` manifest.

    python tools/build_eval_manifest.py \
        --repo Vxtzq/CrowdGPT-Pending \
        --out eval/ng-eval-v1 \
        --target-tokens 5000000

The revision is resolved once and recorded in ``sources.json``; re-running the
builder against the same revision reproduces the same shards byte for byte.
Nothing here talks to the coordinator, and no model is required.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowdgpt_ng.eval import (  # noqa: E402
    EvalError,
    build_manifest,
    sha256_file,
    verify_shards,
    load_manifest,
)

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
CHARS_PER_TOKEN = 4.0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _list_sources(repo: str, revision: str | None, pattern: str) -> tuple[str, list[dict]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.endswith(pattern)]
    if not files:
        raise EvalError(f"no {pattern} files in {repo}@{info.sha}")
    entries = [{"path": f.rfilename, "bytes": int(f.size or 0)} for f in files]
    entries.sort(key=lambda e: e["path"])
    return info.sha, entries


def _select(entries: list[dict], target_tokens: int) -> list[dict]:
    """Spread the sample across the queue instead of taking one book."""

    estimated = sum(e["bytes"] for e in entries) / CHARS_PER_TOKEN
    step = max(1, round(estimated / max(1, target_tokens)))
    picked = entries[::step]
    # A single source is not a sample; widen the stride if we came up short.
    while len(picked) < 4 and step > 1:
        step = max(1, step // 2)
        picked = entries[::step]
    return picked


def build(
    repo: str,
    out_dir: Path,
    *,
    revision: str | None,
    target_tokens: int,
    shards: int,
    seq_len: int,
    stride: int,
    tokenizer_name: str,
    name: str,
    version: str,
    created_at: str,
    pattern: str = ".txt",
) -> dict:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    sha, entries = _list_sources(repo, revision, pattern)
    picked = _select(entries, target_tokens)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    per_shard = max(1, target_tokens // shards)
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab <= 0:
        raise EvalError("tokenizer has no vocab_size")

    out_dir.mkdir(parents=True, exist_ok=True)
    buffers: list[list[np.ndarray]] = [[] for _ in range(shards)]
    counts = [0] * shards
    used: list[dict] = []
    shard = 0
    total = 0
    for entry in picked:
        if shard >= shards:
            break
        path = hf_hub_download(repo_id=repo, filename=entry["path"], repo_type="dataset", revision=sha)
        tokens = tokenizer(Path(path).read_text(encoding="utf-8", errors="replace"), add_special_tokens=False)["input_ids"]
        if not tokens:
            continue
        array = np.asarray(tokens, dtype=np.uint32)
        room = per_shard - counts[shard]
        if array.size > room:
            array = array[:room]
        buffers[shard].append(array)
        counts[shard] += int(array.size)
        total += int(array.size)
        used.append({"path": entry["path"], "bytes": entry["bytes"], "tokens": int(array.size), "shard": shard})
        if counts[shard] >= per_shard:
            shard += 1
    if total == 0:
        raise EvalError("no tokens were produced")

    shard_paths: list[Path] = []
    for index, chunks in enumerate(buffers):
        if not chunks:
            continue
        path = out_dir / f"shard-{index:03d}.bin"
        with open(path, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk.tobytes())
        shard_paths.append(path)

    repo_root = Path(__file__).resolve().parents[1]
    shard_paths = [p.resolve() for p in shard_paths]
    out_dir = out_dir.resolve()
    manifest = build_manifest(
        name=name,
        version=version,
        shards=shard_paths,
        seq_len=seq_len,
        stride=stride,
        vocab_size=vocab,
        tokenizer=tokenizer_name,
        created_at=created_at,
        base_dir=repo_root,
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")

    sources = {
        "schema": "ng-eval-sources-1",
        "dataset": repo,
        "datasetRevision": sha,
        "pattern": pattern,
        "tokenizer": tokenizer_name,
        "targetTokens": target_tokens,
        "tokens": total,
        "manifests": [manifest_path.name],
        "files": used,
    }
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")

    reloaded = load_manifest(manifest_path)
    verify_shards(reloaded, root=repo_root)
    capacity = sum(max(0, (s.tokens - reloaded.seq_len) // reloaded.stride) for s in reloaded.shards)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": reloaded.sha256(),
        "shards": [str(p) for p in shard_paths],
        "bytes": sum(p.stat().st_size for p in shard_paths),
        "tokens": total,
        "windows": capacity,
        "sources": len(used),
        "dataset_revision": sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Vxtzq/CrowdGPT-Pending")
    parser.add_argument("--revision", default=None, help="pin the dataset revision (default: current main)")
    parser.add_argument("--out", default="eval/ng-eval-v1")
    parser.add_argument("--name", default="ng-eval-v1")
    parser.add_argument("--version", default="1")
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--created-at", default=datetime.date.today().isoformat())
    parser.add_argument("--pattern", default=".txt")
    args = parser.parse_args(argv)

    if args.target_tokens < args.seq_len + 1:
        parser.error("--target-tokens must exceed --seq-len")
    if args.shards < 1:
        parser.error("--shards must be >= 1")

    result = build(
        args.repo,
        Path(args.out),
        revision=args.revision,
        target_tokens=args.target_tokens,
        shards=args.shards,
        seq_len=args.seq_len,
        stride=args.stride,
        tokenizer_name=args.tokenizer,
        name=args.name,
        version=args.version,
        created_at=args.created_at,
        pattern=args.pattern,
    )
    print(
        f"{result['manifest']}: {result['tokens']} tokens in {len(result['shards'])} shards "
        f"({_human(result['bytes'])}), {result['windows']} windows, {result['sources']} sources"
    )
    print(f"manifest sha256 {result['manifest_sha256']}")
    print(f"dataset revision {result['dataset_revision']}")
    for path in result["shards"]:
        print(f"  {path}  {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
