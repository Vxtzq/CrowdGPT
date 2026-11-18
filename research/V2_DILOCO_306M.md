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

## Matched comparison: both devices, 1.1 M tokens per arm

Both machines ran the identical protocol at `DL_TARGET_M=1.1` (306.5 M params,
K=4, B=4 x T=512, outer Nesterov SGD 0.7/0.9), so the two devices are directly
comparable. Uniform baseline `ln(4096)` = 8.3178; each device starts at its own
uniform-adjacent value, which is the harness self-check.

| device | arm | uploads | tokens | final held-out | vs inner=1 | tok/s |
|---|---|---|---|---|---|---|
| RTX 4060 Ti (CUDA 12.4) | inner=1 | 134 | 1,097,728 | 8.0368 | — | 1,575 |
| RTX 4060 Ti (CUDA 12.4) | inner=25 | **5** | 1,024,000 | **6.6490** | **1.39 better** | 3,929 |
| Radeon 890M (ROCm 7.2) | inner=1 | 134 | 1,097,728 | 7.9649 | — | 355 |
| Radeon 890M (ROCm 7.2) | inner=25 | **5** | 1,024,000 | **6.7962** | **1.17 better** | 378 |

Curves: CUDA inner=1 `8.5367 -> 9.6594 -> 8.0368`, inner=25 `8.5367 -> 6.8582 ->
6.6490`; ROCm inner=1 `8.5240 -> 10.5626 -> 7.9649`, inner=25 `8.5240 -> 6.9191 ->
6.7962`.

Both devices show the same ordering and the same magnitude: **inner=25 beats
inner=1 by 1.17-1.39 held-out at equal tokens while making 27x fewer uploads**
(5 vs 134), and inner=1 never leaves the neighbourhood of uniform while inner=25
learns. The earlier longer CUDA run (~2 M tokens per arm: 8.7581 vs 6.4495) is
consistent with this.

An earlier single-round ROCm run (`DL_TARGET_M=0.35`, 204,800 tokens) printed
`8.5240` unchanged; that was a harness-soundness check (one round, 0.00067
tokens/param, nothing can learn there), not a comparison, and is superseded by
the matched table above.

## What this does not establish

- 2 M tokens is 0.0065 tokens/param — nothing here says where the curve ends.
- Workers run sequentially with bf16 CPU snapshots, not on separate machines.
- Each worker keeps its own AdamW across rounds, but the harness does not carry
  inner optimizer state across an outer step the way a true DiLoCo run would.
- One seed, one model size.

Reproduce: `DL_TARGET_M=2 DL_INNER=1,25 SOLO_DEV=cuda python experiments/v2/diloco_306m.py`

## Second device (ROCm / gfx1150)

The same inner=25 configuration was launched on aptai2 (890M iGPU, ROCm 7.2,
torch 2.14.1+rocm7.2) with `DL_TARGET_M=0.35 DL_INNER=25`. It completed **one
round** — 100 inner steps x 4 sequential worker passes = 400 steps, 204,800
tokens:

```
inner=25  params=306.5M  rounds=1  tokens=204,800  curve=['8.5240']  held-out 8.5240
          388 tok/s  527.6 s  uploads=1
```

Read this for what it is. One round is 0.00067 tokens/param, and at that budget
nothing can learn: the 4060 Ti needed 1.84 M tokens to reach 6.4495. What the arm
does establish is that the harness is sound on ROCm — it starts at 8.5240 against
a uniform baseline of `ln(4096)` = 8.3178 (the same self-check as the 4060 Ti's
8.5367), completes a full round with no divergence and no OOM, at 388 tok/s.

It does **not** confirm the inner-step ordering on this device at 306 M. That
would need at least ~5 rounds per arm (~45 min/arm at 388 tok/s), which the APU
cannot supply in a session. Cross-device agreement for the inner-step result
therefore still rests on the 1.32 M runs (inner=25: 6.2640 CUDA vs 6.2642 ROCm).
