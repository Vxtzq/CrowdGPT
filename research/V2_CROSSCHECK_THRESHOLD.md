# The cross-check threshold, and why v1 produced false positives

Author's note that started this: *"you also need to see for the cross check
threshold, because we had waaaay too much false positives with crowd v1,
especially with bf16"*.

## What the old margin actually computed

`estimate_margin()` took repeated evaluations of the **same** base weights and
used `3 * stderr`. On a fixed held-out slice in eval mode, identical weights
produce identical losses, so `var == 0` and the margin **always collapsed to the
1e-4 floor**. Every submission was then judged against a constant, which has no
relationship to how much a loss can move for a legitimate reason.

Measured on this repository's own protocol (tiny model V=4096 d=128 L=4 H=4, 8
fixed `val.bin` windows, base mean 8.3501):

| delta | fp32 gain | bf16 gain | bf16 band | paired 3-sigma |
|---|---|---|---|---|
| 1 AdamW step @ 1e-5 | −0.002459 | −0.002463 | 0.000003 | 0.000180 |
| 1 step @ 1e-4 | −0.023298 | −0.023328 | 0.000029 | 0.001823 |
| 2 steps @ 1e-4 | −0.042723 | −0.042729 | 0.000006 | 0.003855 |
| 5 steps @ 1e-4 | −0.083588 | −0.083588 | 0.000001 | 0.007844 |
| 20 steps @ 1e-3 | −0.572397 | −0.572393 | 0.000004 | 0.045481 |

Two things fall out. First, **bf16 rounding costs 1e-6 to 3.4e-5 of loss**, i.e.
below the old 1e-4 floor — so on this protocol bf16 alone cannot turn an honest
improvement into a rejection. Second, the **paired** margin is two orders of
magnitude looser than the floor at realistic window counts, so the floor was
rejecting honest variation of that size.

## Where the margin comes from (and when it degenerates)

| windows | paired 3-sigma |
|---|---|
| 1 | **0.000100** (the floor — no measurable spread) |
| 4 | 0.002054 |
| 8 | 0.003946 |

So a single evaluation window reproduces v1's bug exactly, by construction. The
margin only becomes meaningful from ~4 windows upward, and the code now reports
the number it used (`verdict.margin`, `advisory["windows"]`) rather than hiding
a constant.

## Honest reading of "especially with bf16"

The measurement does **not** support "bf16 rounding alone caused the
false positives" at this scale. What it does support:

- the old gate was **geometric** (Krum pairwise distances, closest-pair anchor);
  bf16 quantisation perturbs exactly that metric, and a threshold expressed as a
  *ratio* of distances flips for any client near the boundary. That gate has
  already been replaced (PR #26), so the bf16 amplifier is gone with it;
- the fixed 1e-4 floor with no window-noise term rejected honest peers whose
  delta was legitimately worse by more than 1e-4 on a small slice;
- judging on one window makes any statistical margin degenerate to that floor.

## What the code does now

`crowd-diloco`-grade replay, but the relevant parts live in `solo_check.py`:

- `paired_margin(base_windows, cand_windows, *, sigmas, floor, bf16_band)` —
  margin from the spread of **paired per-window deltas**, `max(floor,
  3*stderr, bf16_band)`.
- `judge_paired(...)` — accept iff the mean paired delta is within that margin;
  non-finite deltas rejected.
- `bf16_band_from_replay(loss_fp32, loss_bf16)` — turns "what does bf16 cost"
  into a measured number instead of a guessed threshold.
- `select_by_objective(..., base_windows=..., bf16_band=...)` and
  `server.set_replay_evaluator(..., base_windows=..., bf16_band=...)` — a replay
  evaluator that returns **per-window losses** opts the round into the paired
  rule; the verdict carries `advisory["paired"]`, `["windows"]`, `["bf16Band"]`.

## Reproduce

```
/app/crowdgpt/.venv/bin/python /tmp/opencode/v15/bf16_small.py       # delta-size sweep
/app/crowdgpt/.venv/bin/python /tmp/opencode/v15/bf16_crossshard.py  # window-count sweep
```
