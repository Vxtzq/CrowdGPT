# CrowdGPT × HiveMind Synergy: Coordinator, Account Multiplexing & CrossCheck Guide

This document defines the exact integration points between **[CrowdGPT](https://github.com/Vxtzq/CrowdGPT)** (`Vxtzq/Crowd-v1`, `server.crowdgpt.net`, `client.py` v3.1+) and **[HiveMind](https://github.com/xaviercallens/HiveMind)** (OpenRouter inference gateway & volunteer GPU swarm).

## 1. Repository Modules & Intended Integration Points

All three root modules in this repository are active coordinator/client bridge components exercised by `tests/test_hivemind_integration.py` in CI (`python -m unittest discover -s tests -v`):

| File in `Vxtzq/CrowdGPT` | Role | Coordinator / Client Integration Point |
|---|---|---|
| `hivemind_bridge.py` | Pure-stdlib v3.1 `X-Client-*` header builder, `crossCheck` cohort parser & `/fl/submit` validator, precision cohort classifier, gzipped delta stream packer/unpacker, Multi-Krum & Loss Ratchet | Coordinator `POST /fl/submit` pre-merge verification + CLI `python hivemind_bridge.py headers --handle <user> --device <gpu>` |
| `hivemind_loss_check.py` | `CrowdV1` (`657.8M` trunk + optional `227,865`-bucket `CrowdEngram` on HF Step 348+), HMAC-signed loss receipts & spot-check verifier | Shared library imported by `hivemind_loss_check_service.py` and GPU evaluator nodes |
| `hivemind_loss_check_service.py` | Framework-agnostic backend job & receipt coordinator (`handle_get_job`, `handle_post_receipt`, `fastapi_router`) | Coordinator `GET /fl/loss_check/job` & `POST /fl/loss_check/receipt` before promoting a merged checkpoint to Hugging Face |

## 2. Onboarding HiveMind Trainers Without Per-User Account Explosion

CrowdGPT's coordinator keys active tasks on `user:<uuid>:dev:<device>` (`client.py:337`). `hivemind_bridge.build_crowdgpt_client_hardware_headers(...)` sets deterministic `X-Client-Device: hm-<handle>-<gpu>` headers alongside all 12 required `X-Client-*` headers:

1. **Tier 1 — Shared Fleet Account (`hivemind-swarm`)**: Casual or night-window HiveMinders (`RTX 2070`, `GCP Spot T4`, volunteer GPUs) share a single `hivemind-swarm` token while passing unique `X-Client-Device: hm-<handle>-<gpu>` IDs so tasks never collide on `server.crowdgpt.net`.
2. **Tier 2 — Personal CrowdGPT Account (`--login`)**: Dedicated contributors link their own CrowdGPT token for individual leaderboard credit while also earning HiveMind OpenTrainer rewards.

## 3. Exact Cross-Client Verification (`crossCheck`) Compliance

When `server.crowdgpt.net` enables a cross-client verification cohort (`task["crossCheck"]["enabled"] == True`):
- `parse_crowdgpt_task_crosscheck(task)` enforces `effective_slot = 0` and extracts `targetSeed`, `targetLocalSteps`, `targetBatchSize`, `targetAccumSteps`, and `targetAutocast`.
- `build_crowdgpt_submit_metadata(...)` validates that `actual_steps == targetLocalSteps` and `actual_batch_size == targetBatchSize` before packing `crossCheckUsed`, `actualSeed`, `actualSteps`, `actualBatchSize`, `lastLoss`, `lossStd`, and `lossSamples` into the `/fl/submit` JSON header.

## 4. Testing Live `Vxtzq/Crowd-v1` on HiveMind OpenRouter

`Vxtzq/Crowd-v1` (including `CrowdEngram` `engram_buckets=227865`) is served live via OpenAI-compatible `/v1/chat/completions` as **`hivemind/crowd-v1`** (aliases `crowdgpt/crowd-v1`, `Vxtzq/Crowd-v1`), polling `https://huggingface.co/Vxtzq/Crowd-v1/resolve/main/status.json` to hot-reload new `globalStep` / `hfRevision` checkpoints automatically.
