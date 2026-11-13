# LLM-JEPA for Crowd-v1: proposal (opt-in, nothing changes for contributors)

Status: **proposal + standalone module**. `crowd_jepa.py` is not imported by `client.py`; shards, rounds, cohort checks
and inference are untouched. Wiring it in is a later decision for the coordinator, after the pilot below.

Source: Huang, LeCun, Balestriero, *LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures*,
arXiv:2509.14252 (reference code `rbalestr-lab/llm-jepa`, Apache-2.0). Full study (French) with the reasoning:
`xaviercallens/HiveMind` `docs/CROWDGPT_LLM_JEPA_STUDY.md`.

## 1. What it is

Training adds one term; the architecture and generation do not change:

    L = CE_next_token + lambda * (1 - cos(h_view1, h_view2))

`h_view` is the final hidden state (after `ln_f`) at the last token of each view. The two views are two renderings of the same
content, for example a description and its regex, a question and its SQL, or a docstring and its function. Both views are
packed into one sequence with a block-causal mask, so a single forward pass serves both terms.

Results reported in the paper (Llama-3.2-1B-Instruct fine-tuning; **not Crowd-v1 numbers**):

| Task | SFT accuracy | SFT + JEPA accuracy |
|---|---|---|
| NL-RX-SYNTH | 57.3 | 71.5 |
| GSM8K | 32.4 | 36.4 |
| Spider | 47.5 | 50.6 |
| LoRA r512 on NL-RX | 50.2 | 72.4 |

Gains are small on already-strong models and on unstructured tasks. Training cost is about 2× (less with loss dropout).

**What it is not.** The paper has no evidence that it replaces pretraining tokens on raw web text. Plain FineWeb-style chunks
have no second view. It fits SFT and domain adapters, and possibly the code part of the corpus (docstring ↔ body), as an
experiment.

## 2. What is in this PR

`crowd_jepa.py` (torch only, imported lazily, so the stdlib CI can import it):
- `pack_two_views(text_ids, code_ids, k_pred=0, pred_token_id=None)` returns the packed ids, a block-causal `attn_mask`, the
  two last-token indices, and next-token targets (`-100` at the view boundaries and on predictor slots).
  - With `k_pred = 0` no token is added to the vocabulary, so `vocabSize`, the flat-weight offsets and the loss check are
    unchanged. That is the recommended start.
- `jepa_term`, `combined_loss` (no stop-gradient, as in the reference code; `gamma > 0` is enforced, since the paper reports
  that `gamma = 0` collapses).
- `jepa_step(seed, step, alpha)`: loss dropout that is deterministic per `(seed, step)`, so every client of a cohort makes
  the same decision and the cross-check stays valid.

`tests/test_crowd_jepa.py`:
- The determinism test runs in CI.
- The tensor tests run when real PyTorch is present; they are skipped under the CI's torch stub. One of them checks that
  attention under the mask equals attention on each view alone.

**To wire it later (not done here)**:
- `GroupedQueryAttention` must accept an `attn_mask` (SDPA `attn_mask=` instead of `is_causal=True`), which may lose the
  fused kernel; the existing `_chunked_attn` ROCm path already uses a manual mask.
- Pair-view shards need view offsets.
- The round config publishes `lambda`, `k` and `alpha`.
- The loss check should stay **pure next-token CE**, because inference is unchanged.

## 3. Measurements that matter for CrowdGPT (HiveMind T4, 2026-11-13)

Held-out next-token loss of `Vxtzq/Crowd-v1` at revision `5ac3916` (step 366): 12 windows of 2048 tokens per source, fixed
seed, Qwen3 tokenizer for every model. Raw data is in HiveMind `artifacts/experiments/CROWDGPT_E0_heldout_2026-11-13.json`.

| Model | chunk 1026 @0 | chunk 1026 @200 MB | wikitext-2 test |
|---|---|---|---|
| Crowd-v1 with engram (fp16 = fp32 = bf16) | 3.030 | 3.194 | 4.855 |
| Crowd-v1 **without** engram | 3.376 | 3.570 | 5.055 |
| Qwen3-0.6B (reference) | 2.536 | 2.776 | 3.020 |

- **The engram matters (+0.35 nat without it).** The `hivemind_loss_check.py` in this repo already models it; HiveMind's own
  copy did not, and is fixed in xaviercallens/HiveMind#907.
- **No sign of memorization.**
  - The gap between "seen" and "far" windows is 0.16 nat, against 0.24 for Qwen3-0.6B on the same windows.
  - Cohort mode (`targetBatchSize` 2 × `targetLocalSteps` 500) reads 1,000 windows of a 1,280-window sub-slice, i.e. 0.78
    passes.
- **Where the gap to Qwen3-0.6B is.** About 0.45 nat on CrowdGPT's own chunks, but **1.8 nat on wikitext**. That points to a
  data-mix question (encyclopedic / reference text), which is likely the cheapest lever and independent of JEPA.

## 4. Proposed pilot (before any wiring)

The pilot runs on the HiveMind T4, at night, under a VRAM guard that never takes the inference services' headroom.

| Setting | Value |
|---|---|
| Base | Crowd-v1 `5ac3916` + engram (frozen) |
| Adapter | LoRA r64, fp16 |
| Data | NL-RX-SYNTH |
| Arms | SFT vs SFT + JEPA (lambda = 1, k = 0) |
| Seeds | 3 per arm |

Both criteria are written in advance:
- **Floor:** if SFT-only accuracy is under 5 %, the task is too hard for the model and the pilot is uninformative.
- **Success:** JEPA mean minus SFT mean must exceed 2× the pooled seed std, with no more than 0.05 nat of held-out loss
  regression.

Only a pilot that passes would justify wiring `crowd_jepa.py` into `client.py`, behind a round-config flag.
