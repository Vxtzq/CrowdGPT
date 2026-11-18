# Beyond DUST: 12 papers, 12 MVPs, one hybrid (MOSAIC)

Local research report (git-ignored). Machine: Apple M5 Max, 48 GB, PyTorch MPS.
Papers: `research/papers/` (PDF + extracted text + notes, index in `INDEX.md`).
Code: `research/lab/` (harness, one method per paper, arms files, JSON results).

## Question

DUST trains without backprop by perturbing activations and scoring each draw by
the loss it causes. Our earlier runs showed that it works, but slowly. At 10m params
on real FineWeb-Edu tokens, it plateaus near unigram loss (~7.2) while Adam passes
that loss in under 100k tokens. At pop 128 it also diverges with the scale_test
settings. The paper (qlabs, Oct 2026) needs populations of 1k-16k to match backprop.

So: which recent ideas either **replace** DUST with something that gives good
results on tens of thousands of remote consumer GPUs, or **complement** it by fixing
variance, stability, communication or trust? Then combine the ones that prove out.

## Setup (shared harness)

- Task: GPT preset `10m` (7.3M params, vocab 4096, 6 layers, d 256) on FineWeb-Edu
  tokens (`runs/data/tok`, 40M train / 2M val, pinned qlabs BPE-4096 tokenizer).
- Batch 8 x seq 128 per step, 120 s training budget per arm (eval time excluded),
  MPS, bf16 autocast where the method allows; same 64 validation windows for every arm.
- `harness.py` logs eval loss every 10 s and at token marks 1k-500M, plus counters:
  tokens, forward/backward FLOP-equivalents, upload/download bytes.
- `summarize.py` interpolates loss at fixed wall time and at fixed tokens.
  `crowd_scale.py` projects each arm onto a consumer crowd: step time x1.6
  (consumer GPU vs M5 Max), 20 Mb/s up, 200 Mb/s down, 10,000 workers.
- `probe.py` measures gradient-estimate cosine vs backprop at equal forwards.

## The 12 papers and what each MVP tests

| # | Paper | Role | MVP (methods/) | Proves out if |
|---|-------|------|----------------|---------------|
| 1 | SZO-κ, *An Educated Guess* (2611.08712, Nov 2026) | complement | `szo`: activation-space ZO with per-sample (szo1) and shared (ntwp) probe projections, optional activation awareness, vs node perturbation (np, DUST's family) | cosine >= 1.5-2x np/DUST at equal forwards and lower 120 s loss |
| 2 | WIS-MeZO (2611.16317, Nov 2026) | complement | `wismezo`: SPSA on the row-wise top-r Wanda-salient weights, mask refreshed every 100 steps; random-mask control | beats MeZO and random mask, mainly when fine-tuning |
| 3 | MobileSpec (2611.04797, Nov 2026) | complement | `mobilespec`: ES on block singular values only (~0.03% of params), from scratch and from a checkpoint | beats MeZO from the checkpoint at 128 B/step upload |
| 4 | EGGROLL Unrolled / LOO-ROLL (2609.10980, Sep 2026) | complement | `loo`: rank-1 ES with leave-one-out baseline vs mirrored pairs | LOO < mirror at equal forwards |
| 5 | SOMA (2609.37899, Sep 2026) | replacement | `soma`: N experts on k-means data clusters, each SPSA-trained with no communication | N=4 ensemble beats N=1 at equal compute |
| 6 | ZO-Muon (2602.17155, Feb 2026) | complement | `zomuon`: perturb in a random r-dim output subspace, step with msign (Newton-Schulz) of the subspace estimate; ortho=false control | msign beats plain and MeZO |
| 7 | ZO at the Edge of Stability (2604.14669, Apr 2026) | complement | `eos`: MeZO with lr set from a forward-only curvature (trace) estimate, beta 0 and 0.9 | matches the best fixed-lr MeZO with no sweep, never diverges |
| 8 | LoCA (2608.03020, Aug 2026) | complement | `loca`: periodic backprop calibration of low-rank maps from output error to block-local targets; block-local updates in between. `locaft`: the paper's adapter fine-tune | cosPre >= 0.2 and within 0.3 nats of Adam |
| 9 | SOLO (2609.35440, Sep 2026) | replacement | `solo`: K gradient-isolated modules, each with a local loss through a frozen copy of the shared output head | K=2 within ~5% of K=1 at matched tokens, cos >= 0.5 |
| 10 | Covenant-72B / SparseLoCo (2603.08163, Mar 2026) | replacement | `sparseloco`: W workers, H local AdamW steps, top-k chunk + 2-bit + error-feedback pseudo-gradients, Gauntlet loss filter, Byzantine workers | within 2-5% of dense DiLoCo at ~137x less upload; Gauntlet survives attackers |
| 11 | Decoupled DiLoCo (2604.21428, Apr 2026) | replacement | `ddiloco`: event-driven async learners, quorum + grace window, RDA norm-preserving merge, int4 | async reaches sync loss in less simulated time |
| 12 | Throttle (2609.39357, Sep 2026) | complement | `throttle`: async SGD with geometric down-weighting of repeat arrivals and anchored clipping; fast-flooding attackers | stays near clean loss under flooding while mean diverges |

Baselines in the same harness: `adam` (backprop, upper bar), `dust` (crowd_v15 DUST
estimator, pop 8 and 32), `mezo` (plain SPSA, k=1/4/16).

## Round 1 results

RESULTS_ROUND1

## What proved out, what failed

RESULTS_VERDICT

## Round 2: MOSAIC

RESULTS_ROUND2

## Proposed design for tens of thousands of consumer GPUs

MOSAIC: **MO**dule-**S**harded local learning with a **S**parse,
**A**uditable, **I**nfrequent **C**ommunication outer loop. It changes the unit of
work a volunteer does. A volunteer no longer does "one forward-only population draw
of the whole model" (DUST). It does "H backprop steps on one module of the model,
against a shared frozen readout". Every piece below comes from a paper that proved
out in round 1. The combination is the new part.

1. **Module sharding (SOLO).** Cut the model into K contiguous groups of blocks.
   Module k trains with a local loss: its output goes through a small local head
   (one block + RMSNorm + learned temperature τ_k) and then the *frozen, shared* copy
   W̄ of the global output matrix. The last module uses the real head. No gradient
   crosses a module boundary, so a volunteer only needs memory for its module plus
   activations. It never needs the full backward graph. Module k's input comes
   from a no-grad forward of modules 0..k-1 at the round's published weights. That
   costs inference only, so slow cards can still serve deep modules.
2. **Infrequent, sparse uploads (SparseLoCo / Covenant-72B).** Each volunteer runs H
   local AdamW steps, then uploads Δ = θ_round − θ_local for its module only. Δ is
   compressed to the top 64 of every 64×64 chunk (1.6%), with 2-bit values and an
   error-feedback buffer (β=0.95) kept on the volunteer. With module sharding on
   top, upload per volunteer is about (1/K) × 0.234 bits/param per round.
3. **Robust, budget-based aggregation (Decoupled DiLoCo + Throttle + Gauntlet).**
   - The coordinator closes a round on a time budget, not on the slowest worker.
     Each volunteer does as many of its H steps as fit (`het=budget`), and the
     coordinator weights each module's Δs by tokens.
   - Every Δ is first clipped to the lower median norm of its module's cohort.
     This is the Throttle/Covenant idea and caps any single upload's influence.
   - Optionally, a Gauntlet-style check applies each Δ at its share to a fresh
     held-out batch and keeps it only if the loss improves.
   - The outer step is plain averaging (α=1) or Nesterov.
4. **Audit.** Because module training is deterministic given (weights, data seed,
   H), the existing crowd_v15 audit primitive generalises. The coordinator re-runs a
   salt-picked subset of a volunteer's local steps on its own hardware and compares
   cosines, the same pattern as `crowd_v15/audit.py`. That is cheaper than DUST's
   audit per unit of learning. This is a design claim, not measured here. AdamW
   steps are not bit-reproducible across GPU vendors, so a re-run gives a high
   cosine, not equality, and bf16 needs a same-class auditor, just as we measured
   for DUST.

Where DUST stays useful: as a *probe*. A volunteer who can only run inference
(old card, browser) can still contribute forward-only population draws for the
module it serves. With SZO-style per-sample projections, those draws carry 4-5x
more gradient signal per forward than DUST's node perturbation (measured with
`probe.py`). But round 1 says forward-only learning is a minor contribution next to
local backprop, so MOSAIC treats it as optional.

## Limits of this study

RESULTS_LIMITS
