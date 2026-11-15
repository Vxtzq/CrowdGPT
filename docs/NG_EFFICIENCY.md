# Efficiency measurements

Every efficiency change has to move one metric, measured the same way:
**tokens per GPU per day** and **round wall-clock**. Otherwise a win in one
place silently becomes a loss in another — a faster delta codec that costs
0.05 nats makes more tokens cheaper and the model worse.

## The harness

```bash
# the shipped configuration on whatever accelerator is present
python tools/bench_throughput.py --steps 50

# a quick, heavily trimmed smoke run (CPU)
python tools/bench_throughput.py --device cpu --seq 32 --batch 2 --layers 2 --steps 2
```

It drives `client.py`'s production step (`_fwl` → chunked cross-entropy, the
same autocast pick, the same optimizers) rather than a synthetic matmul, so the
number includes the parts that actually cost time.

Output (`schema: ng-throughput-1`):

| Field | Meaning |
|---|---|
| `throughput.tokensPerSecond` | tokens of training per wall-clock second, one process |
| `throughput.tokensPerGpuDay` | `tokensPerSecond * 86400` — the headline metric |
| `throughput.peakReservedGb` | `torch.cuda.max_memory_reserved` (CUDA/ROCm only) |
| `throughput.autocast` | which dtype the step actually ran in |
| `projection.roundHours` | `roundSteps * roundSeq / tokensPerSecond / 3600` |
| `projection.tokensPerDayAtThisSize` | the same rate across `--devices` GPUs |
| `config.trimmedLayers` | non-zero means `--layers` cut the model: **smoke test only** |

`--overheads` additionally runs the step with checkpointing on and off
(`model.use_ckpt`), reporting `checkpointingCostPct` where positive means
checkpointing costs throughput.

## Measured

Radeon 890M (gfx1150, ROCm 7.2, torch 2.14.1+rocm7.2), full 658M+350M model,
bf16 autocast, `seq 512`, `bs 4`, `loss_chunk` 128 — measured with the
validation harnesses in PR #116, before this tool existed:

| Setting | tok/s | Round (500 × 2048) |
|---|---|---|
| bs 1 | 233.1 | 1.22 h |
| bs 4 (recommended, 11.4 GB) | 447.9 | 2.54 h |
| bs 8 | 487.9 | 4.66 h |
| bs 4, SDPA forced on (old default) | 427.7 | — |
| bs 4, chunked attention (new default) | 501.3 | — |

CPU, trimmed (2 of 24 blocks, `seq 64`, `bs 2`), torch 2.6.0+cu124 on 16 cores:

| Check | Value |
|---|---|
| tokens/s | 31.0 |
| checkpointing on | 30.6 tok/s |
| checkpointing off | 34.4 tok/s |
| `checkpointingCostPct` | **10.9 %** |

That CPU row is a methodology check: checkpointing costs real throughput, and
the harness measures it. It says nothing about accelerator throughput.

## What is not measured yet

- **GPU/ROCm overheads.** The local RTX 4060 Ti is occupied by another
  project's backfill, so `--overheads` has only run on CPU. Re-run it on the
  ROCm host when it is free; the command is the same without `--device cpu`.
- **Codec loss impact (N2.3).** Requires 10 paired rounds `bf16` vs `ngd1` on
  one seed; blocked on an idle GPU.
- **Fleet-level tokens/day.** `--devices N` is arithmetic, not a scheduler
  simulation: it assumes every GPU is busy for the whole day, which the
  0.1 % download-to-contribution funnel does not deliver.

## Rule

Do not quote a number from a `--layers`/`--seq`-trimmed run as throughput.
`config.trimmedLayers` and `config.seq` are in the JSON precisely so a reader
can tell the difference.
