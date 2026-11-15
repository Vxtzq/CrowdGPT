# Held-out evaluation set (ng-eval-v1)

`ng-eval-v1` is the fixed, hash-pinned set that the published held-out loss is
measured on. It exists so a stranger can take a fresh clone, one manifest and
one checkpoint, and get the same number the site shows.

- Manifest: `eval/ng-eval-v1/manifest.json` (schema `ng-eval-1`)
- Shards: `eval/ng-eval-v1/shard-000.bin` … `shard-002.bin`, raw `uint32`
- Sources: `eval/ng-eval-v1/sources.json` (dataset revision and every file used)
- Tokenizer: Qwen3 (`Qwen/Qwen3-0.6B`, vocab 151,669) — the corpus tokenizer
- Windows: `seq_len` 2048, `stride` 1024, 64 scored per report by default

## Why this data

The training corpus is `Vxtzq/CrowdGPT`, pre-tokenized `chunks/chunk_XXXX.bin`.
Any slice of it is trainable by construction: the coordinator hands out 10 MB
ranges from whatever chunk a pool is assigned. A held-out set therefore has to
come from outside the corpus, so `ng-eval-v1` is built from text that was still
in the review queue (`Vxtzq/CrowdGPT-Pending`) at dataset revision
`a187a481d38c5c305e9183ed1e84b458928e0d89`, sampled with a fixed stride across
the queue rather than taken from one book.

Contamination risk is explicit: if a pending file was later merged into the
corpus unchanged, that shard's tokens were trainable. The next revision
(`ng-eval-v2`) should instead reserve whole chunk indices server-side and
exclude them from `validChunkIndices`, which makes contamination impossible
rather than unlikely. Until then, treat `ng-eval-v1` as a fixed yardstick for
comparing rounds, not as a claim about generalisation.

## Build it

```bash
python tools/build_eval_manifest.py --out eval/ng-eval-v1 --target-tokens 5000000 --shards 4
```

The builder pins the dataset revision, tokenizes with the corpus tokenizer,
writes the shards, then re-loads and re-verifies the manifest before printing
the manifest hash. Re-running it against the same revision reproduces the same
shard bytes. `.bin` files are committed on purpose: a hash that only exists in
someone's cache proves nothing.

## Reproduce a published loss

```bash
# 1. the harness itself must reproduce a number known in advance
python tools/reproduce_eval.py --model uniform
#    → loss 11.929284 = ln(151669)

# 2. the real checkpoint (downloads model_bf16 + engram_bf16 from HF)
python tools/reproduce_eval.py --model client --count 64
```

`--model uniform` scores uniform logits, whose loss is exactly
`ln(vocab_size)`; if that does not match, the reader, the window sampler or the
bootstrap is wrong and no other number from the tool should be trusted.
`--model client` runs `client.py`'s own model and weights, so it measures the
same forward pass the swarm trains. Both verify every shard hash before a
single token is scored.

`--count` and `--seed` are recorded in the report (`windows`, `seed`,
`manifestSha256`). Two runs with the same manifest, seed and count must agree
exactly; a different count changes the sample, so compare like with like.

## What it does not cover

- Quality. This is a loss on one distribution, not a benchmark score. Passing
  the loss gate says the model did not get worse on this data.
- Instruction following, safety, or benchmark parity (see N5).
- Statistical power for tiny deltas: 64 windows gives a 95 % bootstrap
  interval of roughly ±0.01 nats on a converged model, so a claim smaller than
  that needs a larger `--count`.
