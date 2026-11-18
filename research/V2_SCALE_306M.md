# Scale check: backprop vs MOSAIC at 306 M

The tiny-model table in `V2_PARADIGMS.md` says MOSAIC (2 slices, H=5) beats
backprop at 40 rounds. That is a wiring result, not a scaling result, so this is
the first measurement at a size worth quoting: 306 M parameters
(V=4096, D=1024, L=24, H=16, T=512, B=4, tied readout, GPT-2 0.02 init), 8 fixed
`val.bin` windows, both arms to a 2 M-token budget.

Harness: `experiments/v2/scale_306m.py`.

## Result (CUDA, RTX 4060 Ti 16 GB)

| arm | init | ¼ | ½ | final | wall | cost |
|---|---|---|---|---|---|---|
| backprop | 8.5367 | 6.2236 | 5.9203 | **5.6862** | 478.0 s | 4,181 tok/s |
| MOSAIC K=2 H=30 | 8.5367 | 7.8544 | 7.7257 | **7.7166** | 331.6 s | see below |
| MOSAIC K=2 H=5 | 8.5367 | 7.4897 | 7.4125 | **7.4921** | 347.4 s | 5,719 tok/s |

`delta(MOSAIC - backprop) = +2.0303` at H=30 and `+1.8059` at H=5. MOSAIC is not
close on this budget — but read the next section before concluding anything.

Lowering H from 30 to 5 halves the gap and leaves the same pathology, which is
the signature of the *budget*, not of the outer rule: at `--budget inner` a
smaller H buys more rounds and more unique data (97 rounds, 397,312 unique
tokens at H=5, versus 16 rounds and 65,536 at H=30), so quality tracks data seen
rather than the local-step count.

## Why that number is not yet a verdict on MOSAIC

The two arms do not consume the same data, and the difference is not small.

`per_round = slices × local_steps × B × T = 2 × 30 × 4 × 512 = 122,880`
inner-step tokens, so a 2 M-token budget buys **16 outer rounds**. Each slice
spends those 30 steps on **one** batch, so the data actually seen is
`16 × 2 × 4 × 512 = 65,536` unique tokens — **30× less than backprop's 2 M** — for
960 gradient steps instead of 976.

The reported 5,930 tok/s is inner-step throughput; unique-token throughput is
65,536 / 331.6 = **198 tok/s**. The arm trained on 1/30 of the corpus and still
finished 1.4× faster in wall clock, which is exactly what a data-starved run looks
like.

So the measurement to trust so far is narrow: *at a fixed budget of inner-step
tokens, H=30 leaves too few aggregation rounds and too little data*. It does not
say MOSAIC's clipping/averaging rule is wrong.

## The harness now reports the distinction

`--budget` chooses how `--tokens` is spent:

- `unique` (default) — equal data: `rounds = tokens / (slices·B·T)`, so H local
  steps cost proportionally more compute (the method's price) and nothing is
  starved. This is the fair test.
- `inner` — equal inner-step tokens, i.e. the run above, kept only to reproduce.

Both arms now report `gradSteps`, `rounds` and `uniqueTokens` alongside `tokens`,
so an equal-data claim cannot be read off an equal-inner-token run.

At 2 M tokens, `unique` budgeting gives 488 rounds for every H:

| H | rounds | gradient steps | vs backprop (976) |
|---|---|---|---|
| 1 | 488 | 976 | 1× |
| 5 | 488 | 4,880 | 5× |
| 30 | 488 | 29,280 | 30× |

H=1 is the decisive cheap probe: identical data, identical step count, only the
outer rule (median clip, slice averaging, `outer_lr=0.7`, momentum 0.9) and the
SOLO head on slice 0 differ. If H=1 loses to backprop, the rule is the problem;
if it matches, the local-step data reuse is.

## Status

- CUDA H=30 (`--budget inner`): measured, above.
- CUDA H=5 (`--budget inner`): measured, above.
- ROCm 890M (`--budget inner`): running, same configuration.
- `--budget unique` H=1 and H=5 (CUDA): running — these are the fair comparison.
- No MOSAIC number here is a verdict on the outer rule yet; only `--budget
  unique` at H=1 compares the rule on identical data and identical step count.

Numbers land here as they finish; the table is updated rather than replaced so
the correction stays visible.
