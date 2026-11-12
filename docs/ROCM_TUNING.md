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

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CROWDGPT_ROCM_BATCH` | unset (auto) | Hard cap on the recommended batch; set to `1` to restore the old behavior. |
| `CROWDGPT_ROCM_VRAM_FRACTION` | `0.85` | Fraction of the detected budget used for batch selection. |
| `CROWDGPT_ROCM_LOSS_CHUNK` | `128` | Loss chunk size; raise to 256 only if memory allows. |
| `CROWDGPT_DISABLE_BNB` | unset | Set to `1` to force fp32 AdamW. |
| `CROWDGPT_ENGRAM_GPU` | unset | Set to `1` to keep the engram table on the GPU when spare VRAM exceeds 2 GiB. This is opt-in until ROCm `SparseAdam` + GPU engram is validated. |

## Review requirements

The following need a real ROCm machine before merge or before being enabled by
default:

- the bnb self-test on the actual `gfx` target;
- GPU engram placement with `SparseAdam`;
- bf16 availability and the fp32 fallback;
- `torch.cuda.get_arch_list()` matching the installed wheel;
- the memory fraction on APUs that share system RAM.

If the target is not in the wheel's arch list, the startup log says
`arch_supported=False`; install a matching ROCm PyTorch wheel or set an
appropriate `HSA_OVERRIDE_GFX_VERSION` rather than forcing the run.
