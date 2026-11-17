# Crowd-v2 technology evaluation — measured, not argued

Hardware: RTX 4060 Ti 16 GB (CUDA 12.4) and Radeon 890M / gfx1150 (ROCm 7.2).
Model for every bite: tiny decoder, L=4, d=128, V=4096, tied readout, 1.32 M params,
seq 256, batch 8 (2,048 tokens/step), real FineWeb-Edu tokens from
`peva3/crowd-v15-tokens`, held-out = fixed non-overlapping `val.bin` windows,
identical window order across arms. Scripts: `experiments/v2/`.

## Why this regime exists

Crowd-v1's binding cost is not compute. A peer's backprop runs **1,276 tok/s** at
1.010 B on a 4060 Ti (**157 tok/s** on the 890M iGPU), while every peer uploads a
**whole-model delta of 1,927 MiB bf16** each round. Two training-rule experiments
had already been measured and did not fix that:

- **DUST** (gradient-free): 90 tok/s, held-out plateaued at 6.686 vs `ln(4096) = 8.318`
  after 24 h / 6.1 M tokens. Closed.
- **SOLO** (shared-readout local learning): 1.38-3.49x per peer, but +3-15 % held-out
  and **zero** memory or download saving — a peer still holds the full 1.9 GiB base.

So this regime tests *communication* methods against the cost that actually binds.

## Results (held-out loss at equal tokens unless stated)

| technology | 4060 Ti | 890M ROCm | verdict |
|---|---|---|---|
| DiLoCo, inner=1 (976 uploads) | 7.0367 | 6.9933 | baseline (= v1 behaviour) |
| **DiLoCo, inner=25 (39 uploads)** | **6.2640** | **6.2642** | **25x fewer uploads AND 0.73-0.77 better** |
| DiLoCo, inner=100 (9 uploads) | 6.8936 | not run | 108x fewer uploads, still better than inner=1 |
| bf16 on the wire (0.3299 B/token) | 6.2662 | 6.2665 | +0.002 — free |
| ngd1r 1-bit + residual (0.0382 B/token) | 6.3801 | 6.3776 | +0.11 for 8.7x fewer bytes than bf16 |
| one worker 2 rounds stale | 6.6198 | 6.6199 | +0.35 — bounded, no divergence |
| low-rank delta, rank 8 (12.4x smaller) | 6.5045 | 6.5032 | +0.24 — loses to the codec |
| low-rank delta, rank 32 (3.3x smaller) | 6.3717 | 6.3768 | +0.11 — still loses to the codec |
| muP depth LR transfer (4L -> 12L) | best lr 1e-2 at both depths under std init | not run | depth does not break LR transfer |
| exact dedup of the corpus | duplicate rate **0.00 %** (2 of 453,128 chunks) | corpus property | no lever |

Cross-device agreement is <=0.002 on every arm that ran on both.

## Recommendation for Crowd-v2

1. **DiLoCo inner steps (start at 25, target 100-500) with an outer Nesterov-SGD
   (lr 0.7, momentum 0.9).** This is the only lever that improves loss *and* cuts
   traffic: at v1's 1,927 MiB delta it turns ~1.9 TB per 8 M tokens per peer into
   ~74 GB, and at inner=500 into ~490 B/token on a 1.010 B model.
2. **bf16 on the wire.** Halving traffic is free (cos 1.0000, +0.002 held-out).
3. **ngd1r only for slow links.** 8.7x further reduction costs +0.11 — worth it when
   a peer's uplink, not its GPU, is the limit.
4. **Keep `max_stale_steps` <= 2.** A 2-round lag costs +0.35, roughly half of
   DiLoCo's gain. Decoupled/asynchronous aggregation must be measured against that
   budget, not assumed free.
5. **Do not build the wire format on low-rank deltas.** ReLoRA's rank-r factors are
   a *memory* technique here, not a bandwidth one: rank 8 saves 12.4x bytes but costs
   +0.24, where the codec saves 8.7x for +0.11.
6. **Do not spend v2 effort on exact dedup of FineWeb-Edu.** The corpus is already
   duplicate-free at 128-token granularity; the raw-vs-dedup deltas (+0.023, +0.102)
   sign-flip across seeds. Fuzzy dedup and quality/curriculum ordering were not
   measured and remain open.
7. **muP is not required for depth scaling** at this scale: the LR optimum is 1e-2 at
   both 4 and 12 layers under stock init. Its width-transfer claim was not tested.

## Limits of this evidence

Tiny 1.32 M-param model, 8 M tokens per arm, one seed per arm (two for the data
bite), fixed-token rather than fixed-wall-clock comparisons, and a synthetic
simulated-straggler rather than a real churning swarm. The directions are measured
and cross-device consistent; the magnitudes are not transferable to 1 B without a
direct run, which is the next escalation.

## Papers (full text, in `research/`)

`diloco_2311.08105`, `opendiloco_2407.07852`, `dilocox_2506.21263`,
`decoupled_diloco_2604.21428`, `cosine_gated_adam_decay_2605.09126`,
`relora_2307.05695`, `mup_2203.03466`, `solo_2609.35440v1`.
