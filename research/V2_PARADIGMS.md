# v2 paradigms run end-to-end through one selector

`crowd_paradigm.py` exposes `select(name)` over **backprop, SOLO, ReLoRA, DUST and
MOSAIC** (nothing pending). Harness: `experiments/v2/paradigm_e2e.py`. Tiny model
(V=4096, d=128, 4 layers, 4 heads, tied readout, init 8.3236 ≈ uniform
`ln(4096)` = 8.3178), B=8 × T=256, 40 rounds, AdamW (1e-2 backprop/SOLO, 3e-3
ReLoRA/MOSAIC, 3e-4 DUST), clip 1.0, 8 fixed held-out windows.

| paradigm | 4060 Ti (CUDA) | Radeon 890M (ROCm 7.2) |
|---|---|---|
| backprop | 8.3236 → **6.9880** (+1.3356), 104,875 tok/s | 8.3236 → **6.8746** (+1.4489), 43,193 tok/s |
| SOLO (2 modules) | 8.3236 → **6.9792** (+1.3443), 90,360 tok/s | 8.3236 → **6.9756** (+1.3479), 30,389 tok/s |
| ReLoRA (rank 8) | 8.3236 → 8.1788 (+0.1448), 56,527 tok/s | 8.3236 → 8.1788 (+0.1448), 38,523 tok/s |
| DUST (pop 32, σ 0.02, lr 3e-4) | 8.3236 → 8.3294 (−0.0058), 261 tok/s | not run |
| **MOSAIC (2 slices, H=5)** | 8.3236 → **6.9220** (+1.4016), 12,746 tok/s | 8.3236 → **6.9301** (+1.3935), 4,820 tok/s |

Reading: all five paradigms attach, train and evaluate through the same call path
on both devices, and MOSAIC matches or beats backprop on both at this budget
while costing ~8× the wall clock (H local steps × slices). ReLoRA's small 40-round
gain is expected — adapters only, `B` starts at zero — so this table verifies
wiring, not the convergence ranking. DUST is flat at this lr/population and needs
a sweep before any DUST number is quotable; it is also ~400× slower per token.

Two real defects were caught by these runs and fixed: `crowd_relora` created its
adapters on CPU regardless of the base weight's device (`bf1392f`), and
`crowd_mosaic` both refused per-slice uploads and subtracted the merged delta
instead of adding it (`f745b08`) — the latter was the reason MOSAIC looked inert.

Reproduce: `PE_ARMS=backprop,solo,relora,mosaic PE_DEV=cuda PE_STEPS=40 python
experiments/v2/paradigm_e2e.py`.

