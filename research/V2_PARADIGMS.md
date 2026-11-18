# v2 paradigms run end-to-end through one selector

`crowd_paradigm.py` exposes `select(name)` over `backprop`, `solo`, `relora`
(and raises for the not-yet-implemented `dust` / `mosaic`, naming the TODO id).
Harness: `experiments/v2/paradigm_e2e.py`. Tiny model (V=4096, d=128, 4 layers,
4 heads, tied readout, init 8.3236 ≈ uniform `ln(4096)` = 8.3178), B=8 × T=256,
40 steps, AdamW (1e-2 backprop/SOLO, 3e-3 ReLoRA), clip 1.0, 8 fixed held-out
windows.

| paradigm | 4060 Ti (CUDA) | Radeon 890M (ROCm 7.2) | trainable |
|---|---|---|---|
| backprop | 8.3236 → **6.9689** (+1.3547), 97,071 tok/s | 8.3236 → **6.8746** (+1.4489), 43,522 tok/s | 1,317.6k |
| SOLO (2 modules) | 8.3236 → **6.9779** (+1.3457), 94,642 tok/s | 8.3236 → **6.9756** (+1.3479), 30,693 tok/s | 1,338.5k |
| ReLoRA (rank 8) | 8.3236 → 8.1788 (+0.1448), 61,220 tok/s | 8.3236 → 8.1788 (+0.1448), 38,443 tok/s | 592.1k (16 adapters) |

All three attach, train and evaluate through the same call path on both devices,
and the ordering is identical on CUDA and ROCm. ReLoRA's small 40-step gain is
expected — only the adapters train, `B` starts at zero and the jagged schedule
uses a 3e-3 segment LR — so a fair ReLoRA comparison needs a much longer run;
this table verifies the wiring, not the convergence ranking.

One real defect was caught by this run and fixed (`bf1392f`): `crowd_relora`
created its `A`/`B` adapters without inheriting the base weight's device, so
attaching after `model.to('cuda')` left them on CPU and every forward raised
"Expected all tensors to be on the same device". CPU-only unit tests could not
see it.

Not yet run: DUST (source `crowd_v15/dust.py`, TODO V2I-4) and MOSAIC (TODO
V2I-3 — blocked: no reference paper or repo has been identified).

Reproduce: `PE_DEV=cuda PE_STEPS=40 python experiments/v2/paradigm_e2e.py`.
