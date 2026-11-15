# ROCm tuning and diagnostics

The ROCm path is still CUDA-shaped in several places. This document describes the
tuning hooks added for ROCm and what must be validated on real hardware before
relying on the optional paths.

## Automatic behavior

- ROCm batch selection is memory-aware instead of capped at 1. The planner uses
  85% of the detected budget by default (`CROWDGPT_ROCM_VRAM_FRACTION`), and the
  existing OOM ladder remains the safety net.
- `bitsandbytes` is self-tested with a tiny `AdamW8bit` step on the ROCm device.
  If the step does not change the parameter, the client reports `adamw32` and
  does not advertise 8-bit AdamW to the coordinator.
- Startup logs one profile line: `gfx`, whether the target appears in
  `torch.cuda.get_arch_list()`, bf16 support, and the bnb self-test result.
- On an APU the VRAM pool is system RAM, so `mem_get_info` reports the whole
  machine as free VRAM. The budget is capped by a share of physical RAM
  (`CROWDGPT_UMA_VRAM_SHARE`) instead of trusting that number.
- Attention picks the kernel per architecture: `gfx115*` uses the manual chunked
  kernel, everything else tries SDPA first. `CROWDGPT_ROCM_SDPA` overrides it.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CROWDGPT_ROCM_BATCH` | unset (auto) | Hard cap on the recommended batch; set to `1` to restore the old behavior. |
| `CROWDGPT_ROCM_VRAM_FRACTION` | `0.85` | Fraction of the detected budget used for batch selection. |
| `CROWDGPT_ROCM_LOSS_CHUNK` | `128` | Loss chunk size; raise to 256 only if memory allows. |
| `CROWDGPT_DISABLE_BNB` | unset | Set to `1` to force fp32 AdamW. |
| `CROWDGPT_ENGRAM_GPU` | unset | Set to `1` to keep the engram table on the GPU when spare VRAM exceeds 2 GiB. Validated on gfx1150 (see below); still opt-in. |
| `CROWDGPT_UMA_VRAM_SHARE` | `0.75` | On an APU, cap the budget at this share of physical RAM; clamped to 0.4-1.0. Ignored on discrete cards. |
| `CROWDGPT_ROCM_SDPA` | per-arch | `0` forces the manual chunked attention kernel, `1` forces SDPA. Unset uses chunked attention on `gfx115*` (no enabled AOTriton backend) and SDPA elsewhere. |

## Measured on gfx1150 (Strix Halo / Radeon 890M)

Hardware: 24-core Strix Halo, 30.48 GB shared RAM, system ROCm 7.2, torch 2.14.1+rocm7.2,
`gcnArchName gfx1150`, bf16 supported, `bitsandbytes` absent.

| Check | Result |
|---|---|
| `_rocm_profile()` | `gfx1150`, `arch_supported=True`, `bf16=True`, `bnb8=False` |
| bnb self-test | correctly reports unavailable, fp32 AdamW fallback trains (finite loss, 8.1 s/step at bs2) |
| bf16 matmul (2048^3) | 10.2 TFLOPS vs 1.05 fp32, so bf16 autocast is mandatory |
| Throughput, real `_fwl`, bs4 sl512 | 447.9 tok/s (4.57 s/step), peak 11.39 GB; bs8 487.9 tok/s |
| VRAM estimator vs reality | bs4 estimate 11.361 GB vs 11.393 GB peak reserved (+0.03 GB) |
| Attention op, real GQA shape | SDPA 16.8 ms vs chunked 6.3 ms; end-to-end 4.79 s to 4.09 s per step with `CROWDGPT_ROCM_SDPA=0` |
| Loss chunk 64 / 128 / 256 | 4.12 / 4.04 / 4.10 s per step, so the 128 clamp costs nothing |
| Engram on GPU + `SparseAdam` | table on `cuda:0`, sparse grads accepted, 261,888 rows updated in one step, 15.18 GB peak |
| `_clear_accel_cache` | reserved 0.371 GB to 0.117 GB |
| `UserActivityMonitor` | no spurious pause on an idle 24-core box |

The one defect this found: on the APU the budget was 30.34 GB out of 30.48 GB RAM (the whole
shared pool counted as VRAM) and the recommended batch was 16, leaving nothing for the
desktop. The cap above yields 21.9 GB and a batch that fits alongside the OS.

## Wheel selection

`installer.py` maps the detected ROCm version to the matching PyTorch index. PyTorch
publishes one index per ROCm minor (6.1, 6.2, 6.3, 7.0, 7.1, 7.2), so a 7.2 machine gets
`rocm7.2` and anything newer clamps to the newest published index. `gfx906` (MI50) still
uses `rocm6.0` via `install_old.sh`; `requirements_rocm.txt` targets the current 7.2 tree.

## Review requirements

The following were validated on gfx1150 and remain opt-in:

- the bnb self-test on the actual `gfx` target (absent here, so the fallback was exercised);
- GPU engram placement with `SparseAdam`;
- bf16 availability and the fp32 fallback;
- `torch.cuda.get_arch_list()` matching the installed wheel;
- the memory fraction on APUs that share system RAM.

Still unvalidated on hardware: cards other than gfx1150, Windows ROCm, and multi-GPU
placement. If the target is not in the wheel's arch list, the startup log says
`arch_supported=False`; install a matching ROCm PyTorch wheel or set an
appropriate `HSA_OVERRIDE_GFX_VERSION` rather than forcing the run.
