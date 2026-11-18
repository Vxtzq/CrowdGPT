# Staleness corrections under DiLoCo — measured on ROCm

Both papers the author supplied address the same weakness this project measured:
asynchronous/low-communication training sends **stale pseudo-gradients**, and the
penalty grows with data heterogeneity (non-IID), which is the volunteer reality.

- `research/heloco_2606.00271.*` — HeLoCo. Notes that staleness "does not just add
  noise, it implicitly introduces a momentum-like bias", that MLA (Ajanthan et al.
  2025) corrects it "by extrapolating the negative momentum direction uniformly
  across the entire pseudo-gradient", and objects that "tensor blocks within the
  same stale update can have fundamentally different alignment with the outer
  trajectory", so a uniform correction is too coarse.
- `research/async_localsgd_2401.09135.*` — Asynchronous Local-SGD for LMs (DeepMind):
  naive asynchronous Local-SGD takes more iterations to converge than synchronous.

## Protocol

`experiments/v2/staleness_bite.py`, run **only on aptai2 (gfx1150, ROCm 7.2)** —
the local RTX 4060 Ti was in use for other work, so every number below is a
second-device number. Tiny model (V=4096, d=128, 4 layers, 4 heads, tied readout,
0.02 init so the run starts at 8.3236 ≈ uniform `ln(4096)` = 8.3178), K=4 workers,
inner=25, 9 rounds = **1,843,200 tokens per arm**, outer SGD lr 0.7 / momentum 0.9.
Worker 0 is held `stale` rounds behind. Held-out = 8 fixed `val.bin` windows.

Arms:
- `naive` — `theta - mean(theta_k)`.
- `mla` — uniform negative-momentum extrapolation of the whole pseudo-gradient,
  `g + 0.9 * (g - g_prev)` (the MLA form HeLoCo describes).
- `block` — per-tensor alignment-weighted extrapolation, `g + align * 0.9 * (g - g_prev)`
  with `align = max(0, <g, g_prev> / |g_prev|^2)`, i.e. a block only extrapolates to
  the extent its current direction agrees with the outer trajectory.

Regimes: IID (every worker reads the whole corpus) and non-IID (worker *k* is
restricted to a disjoint band of the corpus).

## Results

| arm | stale=0 IID | stale=0 non-IID | stale=2 IID | stale=2 non-IID |
|---|---|---|---|---|
| naive | 6.8232 | 6.8230 | 6.8332 | 6.9592 |
| mla (uniform) | 6.8223 | 6.7947 | 6.8189 | 6.8293 |
| block (per-tensor) | 7.1145 | 7.1534 | **6.7630** | **6.7637** |

## What this says

1. **The staleness cost depends on the token budget and on the data split.** At
   1.84 M tokens one lagging worker costs +0.010 on IID but **+0.136 on non-IID**
   (naive). At 8 M tokens the same lag cost **+0.35** on the shared stream, so the
   penalty grows with training — never quote it without its budget.
2. **The uniform MLA correction is safe and it recovers the non-IID gap.** Without
   staleness it is neutral (6.8223 / 6.7947 vs 6.8232 / 6.8230); with staleness it
   takes non-IID from 6.9592 to 6.8293, i.e. back to the fresh baseline.
3. **The per-tensor correction is a staleness repair, not a free accelerator.** It
   is the best arm when a worker actually lags (6.7630 / 6.7637 — better than the
   fresh baseline by ~0.06) and it is **worse than doing nothing when nothing lags**
   (7.1145 / 7.1534, with a visibly oscillating curve). It must therefore be gated
   on observed lag, not applied every round.

That is the same conclusion HeLoCo reaches from the other direction: a uniform
correction leaves accuracy on the table under heterogeneity, and a per-block
correction needs to know it is looking at a stale update.

## Limits

One seed; 1.84 M tokens; tiny model; the `block` arm is an alignment-weighted
extrapolation *inspired by* HeLoCo's objection, not their published update rule.
A faithful implementation of Algorithm A.3 in `heloco_2606.00271.md` is the obvious
next step, together with a longer run where the naive non-IID penalty is larger.

Reproduce: `SB_ARM=naive|mla|block SB_NONIID=0|1 SB_STALE=0|2 python experiments/v2/staleness_bite.py`
on aptai2 via `rrun.sh` (~41 s per arm at 44–45 k tok/s).
