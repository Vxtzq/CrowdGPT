DRAFT — requires human review

# MOSAIC handoff

This folder collects the MOSAIC code, results and write-ups from the Crowd v1.5 research work
(November 2026) so the CrowdGPT team can read, run and judge it in one place. Everything here is
an experiment on small models on one Mac. Nothing here is proven at the 1B scale.

## What MOSAIC is, in plain English

MOSAIC is a way to train one language model across many home GPUs joined by slow internet
links. The name stands for **MO**dule-**S**harded local learning with a **S**parse,
**A**uditable, **I**nfrequent **C**ommunication outer loop. It combines four ideas:

1. **Cut the model into a few slices ("modules").** The layers are split into K slices (K = 2 to
   4 in our tests). Each volunteer trains one slice. Following SOLO (arXiv 2609.35440), every
   slice except the last gets a small "local head" that predicts the next token through a frozen
   copy of the model's output layer, so a slice learns with ordinary backpropagation without
   waiting for gradients from later slices.
2. **Recompute the earlier slices instead of shipping activations.** A volunteer that owns slice
   k downloads the weights of slices 0..k-1 once per round and runs them forward itself to get
   its inputs. Nothing travels between volunteers during a round. This is the main difference
   from SOLO-style pipeline "crews", which send activations every step.
3. **Talk rarely and send little.** Each volunteer takes H local AdamW steps (H = 30 in our
   runs), then uploads only the change to its slice, compressed with SparseLoCo (top 64 values of
   every 64x64 block, 2-bit, with error feedback; Covenant-72B, arXiv 2603.08163) or, with
   ReLoRA, as two thin low-rank factors.
4. **Merge robustly and make every piece checkable.** The coordinator clips each upload to the
   median size for its slice, averages, and takes an outer step. A volunteer's work depends only
   on public weights, its assigned data and its slice, so anyone can re-run a sampled part of it
   and compare. The network MVP in `network/` does exactly that.

Costs: extra forward compute for the earlier slices, memory to hold them, and a full weight
download when a volunteer rejoins after missing rounds.

## Folder map

| Path | What it is |
|---|---|
| `lab/harness.py` | One small GPT, real FineWeb-Edu tokens, a token or time budget, one results file per run. Every method is a class with `step(it) -> Cost`. |
| `lab/methods/mosaic.py` | MOSAIC (simulated volunteers in one process): slicing, local heads, recompute of earlier slices, SparseLoCo uploads, median-norm clipping, outer step, optional attackers, dropouts and held-out loss checks. |
| `lab/methods/mosaic_relora.py` | MOSAIC with ReLoRA inside each slice: low-rank adapters merged and restarted every round; uploads factors, products or factor averages. |
| `lab/methods/mosaic_dust.py` | MOSAIC where some or all volunteers use DUST (forward passes only) instead of backpropagation. |
| `lab/methods/solo.py`, `relora.py`, `sparseloco.py` | Building blocks: SOLO local heads and pipeline delay, LoRA adapters, SparseLoCo compression. |
| `lab/methods/hsync.py` | Synchronisation study: flat averaging, groups of 16, 2-peer gossip, and the median-length x mean-direction merge. |
| `lab/crowd.py`, `lab/zo.py` | Shared helpers (local AdamW, attacks, random generators). |
| `lab/bandwidth.py` | Download and upload per volunteer per round for full replicas, SOLO crews, MOSAIC and MOSAIC + ReLoRA, 10M to 1B parameters. |
| `lab/round1.sh`, `lab/arms/*.txt` | Run a list of experiments ("arms"), one per line, skipping ones already done. |
| `lab/reference-results/` | Our result files (JSON with loss curves) for every number below. New runs write to `lab/results/`. |
| `crowd_v15/` | The v1.5 model (`model.py`, `plain.py`) and DUST estimator (`dust.py`), copied unchanged. |
| `network/` | Network MVP: several orchestrators, signed hash-chained record, beacon-assigned MOSAIC cells, commit-then-sampled-replay audits, deterministic merge, dashboard. See `network/README.md`. |
| `tools/prepare_tokens.py` | Builds the token files from FineWeb-Edu. |
| `docs/` | `V2-DAY-REPORT.md` (today's tests), `REPORT.md` (original MOSAIC proposal), `plans/` (per-experiment plans and findings), `SYNTHESIS-round2-addendum.md` (research synthesis), `ROADMAP-v2.md`. Some docs mention `archived/` and `research/` paths from the source repository. |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. Token files (about 80 MB). Needs the Hugging Face CLI.
hf download HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet --repo-type dataset --local-dir /tmp/fw
python tools/prepare_tokens.py --src /tmp/fw/sample/10BT/000_00000.parquet --out data/tok \
    --tokens 40000000 --val-tokens 2000000
# Token files somewhere else? export CROWD_TOKENS=/path/to/tok

# 2. A one-minute MOSAIC run (2 slices, 4 simulated volunteers)
cd lab
python harness.py mosaic K=2 workers=4 H=2 --tokens 30000 --seconds 60 --device cpu --label try-mosaic

# 3. The main MOSAIC + ReLoRA arm (about 7 minutes on an M-series Mac)
python harness.py mosaic_relora K=2 workers=8 rank=32 warm=4 up=factors \
    --tokens 10000000 --seconds 100000 --device mps --label my-r32-fac

# 4. Bandwidth table
python bandwidth.py

# 5. Network MVP tests and a 10-round simulation
cd ../network && python -m pytest -q && python sim.py scenarios/baseline.json
```

`--device` takes `cpu`, `mps` or `cuda`. A whole arms list runs with `PY=python sh round1.sh arms/relora.txt`.

## What we measured

Lab numbers: 10M-parameter preset unless stated, sequence 128, batch 8 per volunteer, about 10M
training tokens, one seed, one Mac. Lower loss is better. Files are in `lab/reference-results/`.

### MOSAIC and MOSAIC + ReLoRA (2 slices, 8 volunteers, 30 local steps per round)

| Arm | Upload per volunteer per round | Final held-out loss |
|---|---|---|
| MOSAIC, SparseLoCo compression (`rl-mo-k2w8-base`) | 1.1 MB | 5.46 |
| MOSAIC, uncompressed fp32 (`rl-mo-k2w8-dense`) | 151 MB | 5.28 |
| MOSAIC + ReLoRA rank 32, send factors, average products (`rl-mo-k2w8-r32-fac`) | 9.0 MB | **5.15** |
| same, rank 128 (`rl-mo-k2w8-r128-fac`) | 35 MB | **5.12** |
| rank 32, average the two factors separately (`rl-mo-k2w8-r32-favg`) | 9.0 MB | 5.23 |
| rank 32, compress the product with SparseLoCo (`rl-mo-k2w8-r32-prod`) | 1.1 MB | 5.64 |
| One machine, ordinary AdamW (`rl-full`) | none | 4.93 |
| One machine, ReLoRA rank 32 (`rl-r32`) | none | 5.11 |

Rules that fell out: send the low-rank factors, multiply them at the coordinator and average the
products. Do not average the factors separately, and do not compress the product.

### Does slicing hurt? (SOLO local heads, fixed 4M tokens, sequence 256, batch 16)

| Model | K=1 (normal backprop) | K=2 | K=4 | K=6 |
|---|---|---|---|---|
| 10M | 5.26 | 5.21 | not run (K=3: 5.25) | 5.22 |
| 120M | 5.17 | 5.14 | 5.14 | 5.12 |
| 310M | 5.12 (5.45 at 2M tokens) | 5.44 at 2M tokens, then stopped | not run | not run |

A pipeline delay of up to 8 steps between slices cost about 0.01-0.02 at 10M tokens (`s-10m-k2-st*`), within noise.

### Bandwidth per volunteer per round (`lab/bandwidth.py`; 1B model, 2 slices, 30 steps, sequence 2048 x batch 8)

| Scheme | Download | Upload |
|---|---|---|
| Full replica, dense bf16 (DiLoCo) | 2,030 MB | 2,030 MB |
| Full replica, SparseLoCo | 231 MB | 29 MB |
| SOLO crew, raw activations every step | 2,083 MB | 2,031 MB |
| SOLO crew, 64x compressed boundaries (ResBM-style) | 101 MB | 49 MB |
| MOSAIC, compressed weight changes | 97 MB | 16 MB |
| MOSAIC + ReLoRA rank 32 | 97 MB | 22 MB |

A volunteer that rejoins after missing rounds downloads full bf16 weights once (about 1.6-2.0 GB at 1B).

### Synchronising many volunteers (`hsync`, about 10M tokens)

| Arm | Loss |
|---|---|
| 16 volunteers, everyone averages (flat) | 5.88 |
| 32 flat / groups of 16 + one partner group | 6.11 / 6.10 |
| 32, 2-peer gossip only | 7.03 |
| 16 flat: plain mean / median length x mean direction | 5.88 / 5.78 |
| ... 2 of 16 send -5x updates: mean / median x direction | 6.36 / 5.84 |
| ... 4 of 16 send near-zero updates: mean / median x direction | 5.93 / 5.83 |

Groups of 16 plus one partner group matched flat averaging; pure 2-peer gossip did not. The
median-length x mean-direction merge was better when clean and much better under attack.

### Network MVP (`network/runs/`)

| Scenario | Result |
|---|---|
| Baseline: 3 orchestrators, 8 honest volunteers, 10 rounds | loss 8.37 -> 7.01; 10/10 rounds final with 3/3 signatures; 80/80 honest cells accepted; CPU-vs-CPU replay exact, Apple-GPU-vs-CPU cosine 0.985-1.0 |
| Attacks: 2 honest + 7 attackers (flip, scale, noise, zero, copy, wrong data, stale) | every attacker caught and banned by round 4; no honest volunteer rejected |
| Chaos: orchestrators killed and restarted, volunteers leave and rejoin | loss 8.37 -> 6.95; 10/10 rounds final; restarted orchestrators caught up from the record |

## The two questions from the CrowdGPT author

**"SOLO needs more downloads (activations)."** Correct for SOLO run as a pipeline crew: at 1B the
raw activations a volunteer receives per round are about the size of the whole model. MOSAIC
avoids this by recomputing the earlier slices from weights downloaded once per round: about 20x
less download than raw SOLO crews and about 2x less than compressed ones (table above). The price
is extra forward compute and memory for the earlier slices.

**"Does it work with ReLoRA?"** Yes, in our small test. MOSAIC + ReLoRA rank 32 reached 5.15
against 5.28 for uncompressed MOSAIC, at about 1/17 of the upload. See the rules under the first
table.

## What is not proven

- Every lab result is a 10M-310M model, one seed, about 10M tokens, simulated volunteers in one
  process on one Mac. Real networks add latency, churn and mixed hardware.
- MOSAIC itself has not run at 300M or above; only SOLO slicing has.
- DUST inside MOSAIC (`mosaic_dust.py`, arms `md-stack-*.txt`) was still running when this folder
  was made, and is very slow (about 15 minutes per round). DUST alone stalled at loss 7.4-8.2
  against about 5.1 for backprop in earlier lab runs.
- The network MVP runs on one machine, reads shared local token files, and is not hardened.
  Known open issue: a restarted orchestrator can open a round from a different beacon than its
  peers (see `network/runs/BUILD-SUMMARY.md`).
- No external security review.

## Where this came from

Copied from the `bahpmw/Crowdgpt-research` repository: lab code from `research/lab/` (commit
de54a0b), network MVP from `v2-mvp/` (commit 32f1a09), v1.5 model code from
`archived/crowd-v1.5/`. Path changes for this folder: `lab/harness.py`, `lab/bandwidth.py` and
`network/crowd_v2/paths.py` find `crowd_v15/` here and read token files from `data/tok` or
`CROWD_TOKENS`; `lab/round1.sh` uses `$PY` (default `python`).
