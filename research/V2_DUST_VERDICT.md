# DUST at its own learning rate — both-machine sweep

DUST was measured once at the shared learning rate (`1e-2`) and came out flat, which
is not a fair test: the estimator builds its own gradients from finite differences,
so its optimal step size is far below backprop's. This sweep gives DUST its own
learning rate before any verdict is recorded.

## Protocol

- Tiny model: V=4096, d=128, 4 layers, 4 heads, tied readout, init 8.3236
  (`ln(4096)` = 8.3178 uniform).
- B=8, T=256 → **2,048 tokens per step**, **40 steps = 81,920 tokens** per arm.
- Held-out: 8 fixed non-overlapping windows from `val.bin`.
- DUST: population 32, sigma 0.02, forward-only (no backward anywhere); cost is
  `2 * population` forwards per step.
- Same harness (`experiments/v2/paradigm_e2e.py`), same data order, same seed.

## Results

RTX 4060 Ti (CUDA 12.4):

| arm | lr | held-out | delta | tok/s |
|---|---|---|---|---|
| DUST | 1e-4 | 8.3250 | −0.0014 | 259 |
| DUST | 3e-4 | 8.3294 | −0.0058 | 269 |
| DUST | 1e-3 | 8.3411 | −0.0176 | 275 |
| backprop (reference) | 1e-2 | **6.9809** | **+1.3427** | 107,563 |

Radeon 890M (gfx1150, ROCm 7.2):

| arm | lr | held-out | delta | tok/s |
|---|---|---|---|---|
| DUST | 1e-4 | 8.3250 | −0.0014 | 565 |
| DUST | 3e-4 | 8.3294 | −0.0058 | 566 |
| DUST | 1e-3 | 8.3411 | −0.0176 | 563 |

The two devices agree to four decimal places on **every** learning rate
(8.3250 / 8.3294 / 8.3411 on both), which is the same cross-device agreement the
rest of the v2 evidence shows — the flatness is a property of the estimator, not of
one vendor's kernels.

## Reading

1. **It is not a learning-rate miss.** Loss gets *worse* monotonically as the rate
   rises (−0.0014 → −0.0058 → −0.0176). If the step were merely too small, raising it
   would help before it hurt. This is the signature of a gradient estimate dominated
   by variance: the finite-difference signal is weaker than the noise it carries.
2. **The estimator's own accuracy test agrees.** `crowd_dust`'s correctness test needs
   **population 400** to reach cosine 0.2 against the backprop gradient. At population
   32 the estimate is roughly a factor of three noisier than the test's already weak bar.
3. **Throughput decides it anyway.** 259–565 evidence tok/s versus 30,000–107,563 for
   backprop is a **200–400x** deficit, and it is structural: `2 * population` forward
   passes buy one step, whereas backprop buys a step with one forward and one backward.
   Even a variance fix that made DUST learn as well per step would leave it 100x+ slower
   in wall clock.
4. **So DUST stays a selectable paradigm, not the v2 core.** It is wired behind
   `CROWDGPT_PARADIGM=dust`, its gradients are scaled correctly under accumulation, and
   it runs on both devices — but nothing in this evidence supports making it the default.

## What would change this verdict

- Population ≥ 512 (or an antithetic/variance-reduced estimator) measured at **equal
  tokens**, not equal steps, on both machines.
- A demonstration that DUST+ReLoRA's toy-scale result survives at a population we can
  afford, i.e. that the noise in the estimate is being corrected by the low-rank
  update rather than merely absorbed.

Until then: backprop, SOLO and MOSAIC are the paradigms with evidence behind them;
ReLoRA needs a longer run; DUST needs a population we have not yet paid for.

## Reproduce

```bash
PE_ARMS=dust PE_DEV=cuda PE_LR=3e-4 PE_DUST_POP=32 PE_STEPS=40 \
  python experiments/v2/paradigm_e2e.py
```

On aptai2 the harness and modules are base64-inlined into `/root/CrowdGPT` because the
remote clone diverged; data is reached through a `/tmp/v15data` symlink.
