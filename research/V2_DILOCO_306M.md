# DiLoCo at 306 M — confirmation of the tiny-model result

Harness: `experiments/v2/diloco_306m.py`. Model 306.5 M params (V=4096, d=1024,
24 layers, 16 heads, tied readout). K=4 simulated workers, B=4 × T=512 = 8,192
tokens per round, inner AdamW (lr 3e-4, betas 0.9/0.95, wd 0.01, clip 1.0),
outer Nesterov SGD (lr 0.7, momentum 0.9). Held-out = 8 fixed `val.bin` windows,
evaluated every 500 k tokens. Data: FineWeb-Edu `sample/10BT`, BPE-4096.

Hardware: RTX 4060 Ti 16 GB, peak 15.7 GB.

| arm | uploads | tokens | held-out curve | final | tok/s | wall |
|---|---|---|---|---|---|---|
| inner=1 (v1 behaviour) | 244 | 1,998,848 | 8.5367 → 9.6751 → 8.1622 → 8.7581 | **8.7581** | 1,580 | 1,265 s |
| inner=25 | **9** | 1,843,200 | 8.5367 → 6.8594 → 6.5810 → 6.4495 | **6.4495** | 3,930 | 469 s |

Uniform baseline `ln(4096)` = 8.3178. Every arm starts at 8.5367, which is the
harness self-check: an untrained model must score near uniform, and this one does
(an earlier build scored 706.96 because the tied readout was initialised at unit
scale, so its logits grew as `sqrt(d)`; that run was discarded, not reported).

## What this establishes

At equal tokens (≈2 M, 0.0065 tokens/param) **inner=25 is 2.31 better on
held-out while making 27× fewer uploads** (9 vs 244) and running 2.5× faster,
because one bf16 weight copy per round is amortised over 100 inner steps instead
of 4. At this budget inner=1 has not even beaten uniform (8.7581 > 8.3178), while
inner=25 is clearly learning. This is the same ordering as the 1.32 M-param runs
(6.2640 vs 7.0367), so the effect is not a small-model artefact.

At v1's 1,927 MiB whole-model delta, 244 uploads per 2 M tokens is ~460 GB per
peer; 9 uploads is ~17 GB.

## What this does not establish

- 2 M tokens is 0.0065 tokens/param — nothing here says where the curve ends.
- Workers run sequentially with bf16 CPU snapshots, not on separate machines.
- Each worker keeps its own AdamW across rounds, but the harness does not carry
  inner optimizer state across an outer step the way a true DiLoCo run would.
- One seed, one model size.

Reproduce: `DL_TARGET_M=2 DL_INNER=1,25 SOLO_DEV=cuda python experiments/v2/diloco_306m.py`
