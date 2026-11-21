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

### ROCm APU — full model, 16-cell sweep

Radeon 890M (`gfx1150`, 24 cores, 30.48 GB shared RAM, ROCm 7.2, torch
2.14.1+rocm7.2), full 658M+350M model (1,007,788,032 params), bf16 autocast,
`--device rocm`. Every cell is a fresh process via `tools/bench_throughput.py`;
`seq 512` unless stated. Raw JSON: `g1-sweep-1`, 16 cells.

| Cell | bs | seq | loss_chunk | s/step | tok/s | tok/GPU/day | peak GB |
|---|---|---|---|---|---|---|---|
| bs1 | 1 | 512 | — | 1.884 | 271.7 | 23,474,474 | 12.38 |
| bs2 | 2 | 512 | — | 2.531 | 404.6 | 34,953,053 | 12.07 |
| bs4 | 4 | 512 | — | 4.035 | 507.5 | 43,851,466 | 11.46 |
| bs8 | 8 | 512 | — | 6.769 | 605.1 | 52,284,390 | 10.26 |
| bs16 | 16 | 512 | — | 12.775 | 641.2 | 55,404,320 | 9.98 |
| bs32 | 32 | 512 | — | 24.291 | 674.5 | 58,276,517 | 13.17 |
| bs4 | 4 | 512 | 64 | 4.069 | 503.3 | 43,483,404 | 11.45 |
| bs4 | 4 | 512 | 128 | 4.058 | 504.7 | 43,607,583 | 11.46 |
| bs4 | 4 | 512 | 256 | 4.096 | 500.0 | 43,197,505 | 11.46 |
| bs4 | 4 | 512 | 512 | 4.034 | 507.6 | 43,859,162 | 11.91 |
| bs2 | 2 | 1024 | — | 4.868 | 420.7 | 36,352,345 | 11.76 |
| bs2 | 2 | 2048 | — | 11.267 | 363.5 | 31,409,394 | 11.46 |
| bs4, `CROWDGPT_ROCM_SDPA=1` | 4 | 512 | — | 4.541 | 451.0 | 38,963,049 | 11.39 |
| bs4, `CROWDGPT_ROCM_SDPA=0` | 4 | 512 | — | 4.030 | 508.1 | 43,902,394 | 11.46 |
| bs4, `CROWDGPT_ROCM_VRAM_FRACTION=1.0` | 4 | 512 | — | 4.095 | 500.1 | 43,209,066 | 11.46 |

Read it as:

- Throughput keeps rising with batch — 271.7 → 674.5 tok/s from bs1 to bs32.
- **There is no 1.3x win in this matrix.** The shipped recommendation on this
  box is bs16 (641.2 tok/s), and the best cell measured is bs32 at 674.5 tok/s,
  which is **+5.2 %**. A 1.3x gate is not reachable by tuning batch size on
  gfx1150; the ceiling is ~58 M tokens/GPU/day.
- `loss_chunk` 64/128/256/512 is flat within noise (503.3/504.7/500.0/507.6
  tok/s), so the ROCm 128 clamp costs nothing.
- Chunked attention beats SDPA by **+12.7 %** (508.1 vs 451.0 tok/s), which is
  the evidence behind the gfx115* default flipped in #116.
- Longer sequences cost more per token: bs2 at seq 512/1024/2048 is
  404.6/420.7/363.5 tok/s.
- Peak reserved memory is **not** monotonic in batch (11.46 / 10.26 / 9.98 /
  13.17 GB at bs 4/8/16/32); checkpointing and placement shifts dominate.
- `CROWDGPT_ROCM_VRAM_FRACTION=1.0` changes nothing at bs4, which already fits
  the default budget. It matters at the top end — see the note below.

### Superseded: pre-harness ROCm spot checks

Earlier ad-hoc runs of the same model, kept only so old numbers can be traced:
bs 1 233.1, bs 4 447.9, bs 8 487.9 tok/s; bs 4 with SDPA forced on 427.7 vs
chunked 501.3. The table above supersedes all of them.

### CPU, trimmed (methodology check only)

2 of 24 blocks, `seq 64`, `bs 2`, torch 2.6.0+cu124 on 16 cores:

| Check | Value |
|---|---|
| tokens/s | 31.0 |
| checkpointing on | 30.6 tok/s |
| checkpointing off | 34.4 tok/s |
| `checkpointingCostPct` | **10.9 %** |

This row exists to prove the harness measures checkpointing cost at all. It
says nothing about accelerator throughput.

## What is not measured yet

- **GPU `--overheads`.** The `bs4-seq512-overheads` cell returned
  `overheads: null` instead of a `checkpointingCostPct`, so the only measured
  checkpointing cost is still the CPU one above. Treat the GPU checkpointing
  cost as unmeasured, not as zero.
- **`torch.compile`.** The harness has no flag for it, so no cell covers it.
- **Fleet-level tokens/day.** `--devices N` is arithmetic, not a scheduler
  simulation: it assumes every GPU is busy for the whole day, which the
  0.1 % download-to-contribution funnel does not deliver.

Codec loss impact (N2.3) is measured separately — see `docs/NG_DELTA_CODEC.md`.

## Rule

Do not quote a number from a `--layers`/`--seq`-trimmed run as throughput.
`config.trimmedLayers` and `config.seq` are in the JSON precisely so a reader
can tell the difference.
