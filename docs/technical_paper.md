**⚠️⚠️Disclaimer: This paper is somewhat AI slop as i don't have time to write one myself⚠️⚠️** 
Feel free to contribute to this technical description!

# CrowdGPT: A Technical Description

**Status:** v1 (pre-release research system) · **Client:** v3.4 · **Server:** `CrowdGPT-Server` (current `main`)
**Scope:** This document describes what the code actually does, including its weaknesses. Where a value is a default that the operator can override via environment variable, it is marked as such. Nothing here is a claim of benchmark quality: no held-out evaluation has been published yet (see §10).

---

## 1. What CrowdGPT is

CrowdGPT is a volunteer-compute pretraining run for a single language model. Contributors run a client (`client.py`, with a local web UI `client.html`) on their own GPU. A central coordinator (`server.py`, FastAPI) hands out training tasks, collects weight deltas, filters them, merges them into one global checkpoint, and publishes that checkpoint.

It is **not** peer-to-peer, **not** trustless, and **not** a blockchain. It is a federated-averaging-style system with one coordinator that is trusted by every participant. Section 9 spells out what that trust covers.

Artifacts:

| Item | Location |
|---|---|
| Code | `Vxtzq/CrowdGPT` (client, website, release manifest) and `CrowdGPT-Server` |
| Coordinator | `https://server.crowdgpt.net` |
| Model weights | Hugging Face `Vxtzq/Crowd-v1` (`model_bf16.safetensors`, `engram_bf16.safetensors`) |
| Dataset | Hugging Face `Vxtzq/CrowdGPT` (pre-tokenized shards) |

---

## 2. Model

The architecture is a decoder-only transformer ("SotaGPT") plus a hashed n-gram memory ("engram").

| Property | Value |
|---|---|
| Vocabulary | 151,669 |
| Hidden dim | 1536 |
| Layers | 24 |
| Attention | 16 query heads, 4 KV heads (GQA), head_dim 96, RoPE, chunked attention |
| MLP | SwiGLU, hidden 2560 |
| Max sequence length | 2048 (training samples are 2049 tokens: input + shifted target) |
| Embeddings | tied input/output |
| Transformer parameters | 657,787,392 |
| Engram memory | 227,865 hash buckets × 1536 = 350,000,640 parameters |

The engram is a hashed bigram lookup table: a (previous token, current token) pair is hashed to one of 227,865 buckets and the bucket's vector is added to the representation. Only the rows touched by a batch receive gradient, so engram deltas are **sparse**, which matters for aggregation (§5.3).

Total trainable parameters are roughly 1.0 B including the engram. Because 350 M of those are a lookup table, "1 B parameters" overstates the compute-relevant capacity. The dense transformer is about 0.66 B.

---

## 3. Data

The corpus is pre-tokenized into 5,659 chunks, each served in 10 MB sub-chunks, 4 bytes per token (`DATASET_CONFIG`). Samples are 2049 tokens. The coordinator assigns shards to clients (`SHUFFLE_CHUNKS` is on), with timeouts so an abandoned shard returns to the pool (`SHARD_TIMEOUT_S` 1800).

The server enforces sanity limits on claimed work: `MAX_TOKENS_PER_STEP`, `MAX_TOKENS_PER_SEC` (1M tok/s) and `MAX_LOCAL_STEPS`, plus `MIN_TRAIN_TIME_S` (900 s): a submission that claims more work than is physically plausible, or finishes implausibly fast, is rejected.

---

## 4. Training protocol

### 4.1 Rounds and pools

Training proceeds in **rounds** (default maximum 4 h, `MAX_ROUND_HOURS`). Within a round:

1. A client registers (`/register`), then polls for work. The server applies a **version gate**: clients below the minimum version get HTTP 426.
2. **Matchmaking** groups clients into **pools** of `POOL_SIZE` = 3 (the cohort). Clients are paired only with matching hardware signature (backend, VRAM bucket, autocast mode, optimizer, architecture class, card, compute capability), so that their numerical behaviour is comparable.
3. A pool that cannot fill dissolves after a timeout (`FORMING_TIMEOUT_S` 30 min; solo-pool dissolve 30/60 min for bf16/fp32).
4. **Power users** (high-throughput, operator-designated) may run in "power-solo" pools of one. These skip the cross-check (§6) and keep native precision.
5. **Same-actor guard:** two clients with the same user UUID, or the same IP, cannot share a pool. This stops one person from filling a cohort with their own devices and trivially passing the cross-check.

### 4.2 The task

A task gives the client: the checkpoint to start from (URL + `sha256`), a data shard, a target local step count (default 500), batch size (default 2), gradient-accumulation steps (default 32), and the hardware-harmonised settings (precision, optimizer). Rounds are time-based; a submission for an earlier round is rejected as `stale_round`.

### 4.3 Local training (client)

- Optimizer: **AdamW**, lr 1e-4, weight decay 0.01, **re-initialised at the start of every task** (no optimizer state is carried between tasks or between clients).
- Gradient accumulation to reach the target effective batch.
- bf16 autocast when the pool allows it (§6.3), otherwise fp32. Optional 8-bit AdamW.
- A **VRAM degradation ladder** lets smaller cards participate: it progressively enables offloading and streaming of weights/optimizer state. This trades speed for fit; it does not change the maths.
- AMD ROCm is supported as of 3.4. **This path has not been exercised on real hardware by the maintainers' test suite** (§10).
- `UserActivityMonitor` (3.4): auto-pauses training when the machine appears to be in interactive use (CPU > 85%, RAM pressure, or *growth* in VRAM used by other processes beyond a 1.5 GB margin over a baseline measured at start, debounced over 4 samples). It does not trigger on a machine that only runs CrowdGPT. `CROWDGPT_ACTIVITY_PAUSE=0` disables it.

### 4.4 Weights download

Clients fetch weights from Hugging Face into a **content-addressed cache**; the server publishes the expected `sha256` in the task metadata and the client verifies it. If HF is unavailable or the hash mismatches, the client falls back to streaming from the coordinator. Orchestrator failover is supported (`CROWDGPT_ORCHESTRATORS`).

### 4.5 Delta upload

The client computes `delta = final_weights − initial_weights`, casts to bf16, gzips it with a small JSON metadata header, and uploads either in one request (`/fl/submit`) or via chunked upload (`/fl/submit/init` → chunks → `/fl/submit/finalize`). Limits: 3 GiB total upload, 4 GiB decompressed, 512 chunks, 30 min upload TTL.

Since client v3.1 the upload runs in a **background thread**: training of the next task may proceed while the previous delta uploads, with at most **one** upload in flight. The client remembers the id of the task being uploaded and will not retrain it (the server re-serves a task until its delta lands).

An opt-in 1-bit delta codec (NGD1) exists in the client. It is **off by default and the server has no decoder for it**, so it must not be enabled.

---

## 5. Aggregation

### 5.1 Per-pool merge

After a pool passes verification (§6), its deltas are merged with a token-weighted average:

```
pool_delta = Σ_i (tokens_i · delta_i) / Σ_i tokens_i
weights   += lr(step) · pool_norm · pool_delta
```

- `SERVER_LR` = 1.0 (current), `WEIGHT_DECAY` = 0.0 (current setting; applied server-side per aggregation if non-zero).
- `lr(step)` follows a cosine schedule to a floor: `lr = base · (0.05 + 0.95 · ½(1 + cos(π · step / 50 000)))`.
- Token weighting is capped (`TOKEN_CAP_FACTOR` 3×) so no one client dominates by claiming huge token counts.
- Pools are merged **as soon as they pass verification**, not at the end of the round. Each merge is a new checkpoint step.

### 5.2 Norm safeguards

- Per-task maximum delta norm: `500 · √(steps/100)`, absolute cap 5000.
- Applied-delta clip: `MAX_DELTA_APPLY_NORM` = 1200.
- A **norm-ratio sanity check** reverts a merge if the post-merge weight norm changes outside 0.70–1.50× of the previous.

### 5.3 Engram merge

Because engram updates are sparse, rows are averaged **over the clients that actually touched them**, then scaled by `lr · pool_norm · trust`. Per-row norms are capped (`ENGRAM_ROW_NORM_MAX` 8.0), and a "full-trust" weight (0.2) limits how far a single contributor can move a row. *Known issue:* the engram scale includes an extra 1/n_pools factor relative to the dense path, which makes engram learning slower than the dense weights for the same lr. This is under review (§10).

---

## 6. Verification ("cross-check")

The only defence against a malicious or faulty contributor is that **three hardware-matched clients train on the same task from the same checkpoint** and their deltas must agree.

### 6.1 Distance test

Pairwise L2 distances between deltas are computed in a streaming way (`krum_stream.py`, memory-bounded at 512 MB) so that three ~1 GB deltas never need to be fully loaded. A delta is flagged if its distance to the others exceeds the median-based threshold by `DISTANCE_THRESHOLD_RATIO` = 5.0. A Krum-style selection (`KRUM_F` = 1) picks the consensus when three or more deltas disagree. Near-duplicate deltas (relative distance < 0.02) are flagged as copies.

### 6.2 Consequences

Repeated flags (`BAN_THRESHOLD` 3) lead to a ban of the client. Subnet blacklists and per-subnet round limits exist (`MAX_SUBMISSIONS_PER_SUBNET_ROUND` 500, default very permissive). Trusted-proxy handling (`CROWDGPT_TRUSTED_PROXIES`) controls which `X-Forwarded-For` headers are believed.

### 6.3 The precision problem

bf16 matmuls differ across GPU generations/kernels. Over hundreds of steps, honest clients on different hardware drift apart, and the distance test then flags **honest** members. The `CROWDGPT_POOL_BF16` setting addresses this:

| Mode | Behaviour |
|---|---|
| `off` | all cross-checked pools run fp32 |
| `arch` (default) | bf16 only within the same GPU architecture class *and* compute capability; otherwise fp32 |
| `strict` | bf16 only on the same card model |
| `on` | bf16 pools may mix generations (legacy; caused false flags) |

Power-solo pools are never cross-checked and keep native precision.

**Honest limitation:** the 5.0 threshold is empirical, not derived. It is permissive enough to accept benign cross-hardware noise, which also means a *subtle* poisoning attack (small, consistent bias across all three colluding clients) would not be detected. Colluding clients are prevented only by the same-UUID/same-IP guards, which are weak against a determined attacker with several machines and addresses.

---

## 7. Hardening summary

- Version gate (426), signed client updates (§8).
- Same-actor pool guard (UUID and IP).
- Bans, subnet blacklists, trusted-proxy parsing.
- Plausibility limits on steps, tokens per step, tokens per second, minimum training time.
- Optional task tickets (`CROWDGPT_REQUIRE_TASK_TICKET`, off by default).
- Upload limits, concurrency caps (2 concurrent), free-disk gates.
- Content filtering of contributor display names.

None of these are a substitute for the open problems in §9.

---

## 8. Signed auto-update

The client updates itself from the public repo. To prevent a compromised repo or CDN from pushing arbitrary code:

- `docs/version.txt` is a manifest: version, `sha256_clientpy`, `sha256_clienthtml`, `expires`.
- `docs/version.txt.sig` is a base64 **Ed25519** signature over the manifest bytes.
- The client carries the public key(s) (`UPDATE_PUBKEYS`). It downloads the manifest and signature, verifies the signature, checks expiry, downloads `client.py` / `client.html`, and checks their SHA-256 against the manifest. **Any mismatch silently rejects the update.**
- The private key is held offline by the maintainer. A leaked key would allow malicious updates; rotation requires shipping a client with a new public key.

Operational note: any edit to `client.py` after signing invalidates the manifest, so the release must be re-signed after the final commit. Hashes are computed over LF-normalised text.

---

## 9. Trust model and what is *not* guaranteed

1. **The coordinator is fully trusted.** It chooses tasks, merges deltas, and publishes weights. Contributors cannot independently verify that the published checkpoint is the honest merge of the submitted deltas.
2. **Optional audit layer.** The newer server supports an `orchestrator` package: signed per-round records (Merkle root of included deltas, hashes, signature), a node registry, and `/audit` pages. This makes the coordinator's behaviour *auditable after the fact*, not provably correct. Without the package, the server runs memory-only and publishes no signed records.
3. **Gradient privacy.** Deltas are derived from training data; the dataset is public, so this is not a privacy-sensitive setup, but no secure aggregation or differential privacy is used.
4. **Contributor security.** The client is signed; however, contributors still run Python code on their machines. They should read the source.

---

## 10. Limitations and open problems

Stated plainly:

- **No held-out evaluation.** `LOSS_CHECK_MODE` is off and held-out loss in the public log is empty. The only public curve is **training loss on assigned shards**, which is not a measure of generalisation. We have not published benchmark results (MMLU, HellaSwag, perplexity on a standard set). Until we do, claims about model quality should be treated as unsupported.
- **Training dynamics unproven at scale.** Pooled FedAvg of 500-step local runs with fresh AdamW state per task is an under-studied regime. Server lr 1.0 with weight decay 0 is a recent setting change, not a validated optimum. Effective-token-count concerns raised from weight decay (decay horizon ≈ 1/(lr·wd) aggregations) no longer apply at wd = 0 but are replaced by the open question of drift with no decay.
- **Poisoning resistance is modest** (§6).
- **Engram merge scale** differs from the dense path (§5.3).
- **Missing-engram reload** silently keeps the current engram instead of failing.
- **Untested paths:** ROCm on real hardware, the memory offload/streaming ladder on real GPUs, the real `orchestrator` package, and multi-hour rounds with HF upload end-to-end. The test suite (76 passing client tests) covers logic with stubbed torch/numpy, not GPU numerics.
- **Single coordinator** is a single point of failure and trust.
- **Some past security issues:** secrets were exposed in earlier revisions and have been/are being rotated.

## 11. Roadmap (intended, not promised)

1. Publish a held-out evaluation set and perplexity/benchmark results, and show them on the website.
2. Make the audit layer mandatory and publish round records.
3. Derive the cross-check threshold from measured honest-variance data rather than a fixed constant.
4. Ablate server lr / weight decay / local steps on a small model to justify the settings.
5. Real-hardware CI for ROCm and offload paths.
6. Fix the engram scale and strict-fail on missing engram.

---

## 12. Reproducing / inspecting

Everything described here is in the repositories. Useful starting points: `MODEL_CONFIG`, `FL_CONFIG`, `CROSS_CHECK`, `MATCHMAKING`, `HARDEN` and `_server_lr()` in `server.py`; `krum_stream.py`; the training loop, upload thread and `UserActivityMonitor` in `client.py`; and the release manifest in `docs/version.txt`. Corrections and issues are welcome.
