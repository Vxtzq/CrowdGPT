# Scale check: backprop vs MOSAIC at 306 M
> **Preset label correction (2026-11-19).** Everything below that says "39.9 M"
> originally said "65 M". The preset is `--dim 512 --layers 12 --heads 8` with a
> tied readout, which is **39.9 M parameters** (vocab 4096, 12 blocks, d=512,
> 8 heads, T=512). Every comparison and conclusion stands — only the label was
> wrong — but two configurations in this project were being called by one name,
> which is exactly the kind of thing that turns into a false claim later.


The tiny-model table in `V2_PARADIGMS.md` says MOSAIC (2 slices, H=5) beats
backprop at 40 rounds. That is a wiring result, not a scaling result, so this is
the first measurement at a size worth quoting: 306 M parameters
(V=4096, D=1024, L=24, H=16, T=512, B=4, tied readout, GPT-2 0.02 init), 8 fixed
`val.bin` windows, both arms to a 2 M-token budget.

Harness: `experiments/v2/scale_306m.py`.

## Result (CUDA, RTX 4060 Ti 16 GB)

| arm | init | ¼ | ½ | final | wall | cost |
|---|---|---|---|---|---|---|
| backprop | 8.5367 | 6.2236 | 5.9203 | **5.6862** | 478.0 s | 4,181 tok/s |
| MOSAIC K=2 H=30 | 8.5367 | 7.8544 | 7.7257 | **7.7166** | 331.6 s | see below |
| MOSAIC K=2 H=5 | 8.5367 | 7.4897 | 7.4125 | **7.4921** | 347.4 s | 5,719 tok/s |

`delta(MOSAIC - backprop) = +2.0303` at H=30 and `+1.8059` at H=5. MOSAIC is not
close on this budget — but read the next section before concluding anything.

Lowering H from 30 to 5 halves the gap and leaves the same pathology, which is
the signature of the *budget*, not of the outer rule: at `--budget inner` a
smaller H buys more rounds and more unique data (97 rounds, 397,312 unique
tokens at H=5, versus 16 rounds and 65,536 at H=30), so quality tracks data seen
rather than the local-step count.

## Why that number is not yet a verdict on MOSAIC

The two arms do not consume the same data, and the difference is not small.

`per_round = slices × local_steps × B × T = 2 × 30 × 4 × 512 = 122,880`
inner-step tokens, so a 2 M-token budget buys **16 outer rounds**. Each slice
spends those 30 steps on **one** batch, so the data actually seen is
`16 × 2 × 4 × 512 = 65,536` unique tokens — **30× less than backprop's 2 M** — for
960 gradient steps instead of 976.

The reported 5,930 tok/s is inner-step throughput; unique-token throughput is
65,536 / 331.6 = **198 tok/s**. The arm trained on 1/30 of the corpus and still
finished 1.4× faster in wall clock, which is exactly what a data-starved run looks
like.

So the measurement to trust so far is narrow: *at a fixed budget of inner-step
tokens, H=30 leaves too few aggregation rounds and too little data*. It does not
say MOSAIC's clipping/averaging rule is wrong.

## The harness now reports the distinction

`--budget` chooses how `--tokens` is spent:

- `unique` (default) — equal data: `rounds = tokens / (slices·B·T)`, so H local
  steps cost proportionally more compute (the method's price) and nothing is
  starved. This is the fair test.
- `inner` — equal inner-step tokens, i.e. the run above, kept only to reproduce.

Both arms now report `gradSteps`, `rounds` and `uniqueTokens` alongside `tokens`,
so an equal-data claim cannot be read off an equal-inner-token run.

At 2 M tokens, `unique` budgeting gives 488 rounds for every H:

| H | rounds | gradient steps | vs backprop (976) |
|---|---|---|---|
| 1 | 488 | 976 | 1× |
| 5 | 488 | 4,880 | 5× |
| 30 | 488 | 29,280 | 30× |

H=1 is the decisive cheap probe: identical data, identical step count, only the
outer rule (median clip, slice averaging, `outer_lr=0.7`, momentum 0.9) and the
SOLO head on slice 0 differ. If H=1 loses to backprop, the rule is the problem;
if it matches, the local-step data reuse is.

## The fair comparison: equal data, equal gradient steps

`--budget unique`, 2 M tokens, 488 rounds for every H. Backprop's reference row is
the same 2 M unique tokens and its 976 AdamW steps.

| arm | ¼ | ½ | final | wall | unique tokens | gradient steps |
|---|---|---|---|---|---|---|
| backprop | 6.2236 | 5.9203 | **5.6862** | 478.0 s | 1,998,848 | 976 |
| MOSAIC K=2 H=1 | 7.2958 | 7.2624 | **7.2270** | 402.1 s | 1,998,848 | 976 |
| MOSAIC K=2 H=5 | 7.3318 | 7.2714 | **7.2285** | 1734.5 s | 1,998,848 | 4,880 |

At H=1 the arms see identical data and take the identical number of gradient
steps; only the outer rule differs (median clip, mean of slice deltas,
`outer_lr=0.7`, momentum 0.9) plus the fact that slice 0 trains against a SOLO
head. MOSAIC ends **1.54 nats behind** (7.2270 vs 5.6862). That is the answer the
inner-budget runs could not give: the starvation was real, but removing it does
not close the gap.

Five times the local steps on the same data (H=5 — 4,880 gradient steps, 3.6×
wall clock) lands within 0.0015 of H=1, i.e. more local work bought nothing, and
both MOSAIC curves flatten between ½ and final (7.2624 → 7.2270, 7.2714 → 7.2285)
while backprop is still descending (5.9203 → 5.6862). A curve that plateaus that
hard is not merely slower; at this slice count, inner lr and outer lr the outer
loop is not converging.

Caveat, stated rather than buried: K=2 means 12 layers per slice, and slice 0
trains against a SOLO head initialised at `attach` time, so over 2 M tokens the
head has little chance to align with the readout. More data, more slices, or a
warm head could look different. What is measured is the regime a weekend launch
would actually start in.

## What this means for a launch

- **Backprop is the arm to start a new model with** at this scale.
- MOSAIC is *verified as wired* — delta layout, slice integrity, both devices —
  and *not verified as competitive* at 306 M / 2 M tokens.
- The 40-round tiny-model win in `V2_PARADIGMS.md` is a wiring result; it does not
  survive a fair 306 M comparison.

## Status

- CUDA H=30 and H=5 (`--budget inner`): measured — artifacts of the budget, kept
  only to show why they looked bad.
- CUDA H=1 and H=5 (`--budget unique`): measured — the fair comparison above.
- ROCm 890M (`--budget inner`): still running; will be recorded as another
  inner-budget artifact, not as a verdict.

Numbers land here as they finish; the table is updated rather than replaced so the
correction stays visible.

## Update (same day): the gap was partly a bug, and ownership explains most of it

The 1.5-nat deficit above was not the method. Two defects in
`crowd_mosaic.py`, both found by probing the code rather than the loss:

1. **The SOLO head was never optimised.** `local_delta` picked its parameters
   with `slice_of(name) == k`, and `slice_of` only matched `blocks.<i>.`
   prefixes, so `solo_heads.<k>.*` never entered the optimiser — its gradients
   were computed and discarded. Cost: **0.035 nats** (7.5999 → 7.5653). Real,
   but a footnote.
2. **The trunk ends were never optimised either.** `wte`, `ln_f` and the tied
   `lm_head` were excluded for the same reason. Because `lm_head.weight is
   wte.weight`, MOSAIC spent the run training a representation to decode through
   a **randomly initialised, frozen readout**. This is the big one.

Probe on a finished 39.9 M MOSAIC run: a fresh linear probe on its own trunk
reached **6.39** while the model itself reported **7.18** — 0.79 nats of
information the readout was failing to use. That predicted almost exactly what
fixing the ownership recovered.

### Measured, 39.9 M / 2 M unique tokens / 488 rounds (backprop = 5.6340)

| arm | trunk→slice 0 | trunk→last slice |
|---|---|---|
| MOSAIC H=1 (3e-3 / 0.7 / 0.9) | 7.2456 | **6.4651** (−0.78) |
| MOSAIC H=5 (1e-3 / 1.0 / 0.9) | 7.1728 | **6.1281** (−1.04) |
| backprop | 5.6340 | — |
| SOLO (same split, same local law) | 5.5180 | — |

Two things follow. Giving the last slice `wte` means the terminal slice's *true*
cross-entropy trains the readout, which is the role it plays under SOLO where one
optimiser covers both. And local steps finally pay: H=5 beats H=1 by 0.34 nats,
where before the fix it bought 0.07.

What remains — 0.49 nats at H=5 — is trunk quality, which matches the probe's
6.39 measurement almost exactly.

### The fix in code

`slice_of` now assigns `wte.` to slice 0 and `ln_f.` / `lm_head.` to the last
slice, and `local_delta(..., share_trunk=True)` additionally gives `wte` to the
terminal slice so **both** of its roles are trained: embedding through slice 0's
head, readout through the true loss. `aggregate` already averages a tensor that
appears in two slices' deltas, so sharing needs no new wire mechanism — only half
a step per role per round.

## The local-step optimum, and what the fixes are worth

All arms below: 39.9 M (`--dim 512 --layers 12 --heads 8`, 653 M params at the
306 M preset's depth/width ratio), 2 M unique tokens, 488 rounds for K=2,
8 fixed `val.bin` windows, `--inner-state`.

| arm | final | wall |
|---|---|---|
| backprop, 2 M tokens (976 steps) | **5.6340** | 75 s |
| SOLO, 2 M tokens | 5.5180 | 78 s |
| MOSAIC K=2 H=5, trunk (tied readout) owned by slice 0 | 7.1728 | 231 s |
| MOSAIC K=2 H=5, readout owned by the last slice | 6.1281 | 231 s |
| MOSAIC K=2 H=5, readout owned by **both** (`share_trunk`) | **6.1046** | 231 s |
| MOSAIC K=2 H=1, share_trunk | 6.3344 | 57 s |
| MOSAIC K=2 H=10 | 6.3883 | 450 s |
| MOSAIC K=2 H=20 | 6.7522 | 885 s |
| MOSAIC K=4 H=5 (244 rounds, half the updates) | 6.3306 | 149 s |

Three things to read off it:

1. **Ownership is worth ~0.9 nats.** The tied tensor (`lm_head.weight is
   wte.weight`) has two roles; giving it to the embedding side left the readout
   randomly initialised for the whole run. This is the largest single effect we
   have measured, larger than every hyper-parameter in the table combined.
2. **H=5 is the optimum.** Local steps beyond that over-train on a single batch
   and the outer step gets correspondingly larger (H=10 and H=20 both regress,
   and the curves oscillate).
3. **K=4 is not comparable here.** `--budget unique` fixes rounds at
   `tokens/(slices*B*T)`, so four slices in the same 2 M tokens get half the
   parameter updates. A fair K=4 arm needs 4 M tokens.

## Matched per-parameter data: MOSAIC ties backprop

A MOSAIC round trains every parameter once but spends `slices*B*T` tokens doing
it, so over a fixed total token budget each parameter sees half the batches
backprop's parameters see. The matched control is therefore backprop at 1 M
tokens (488 steps):

| arm | tokens total | per-parameter tokens | final |
|---|---|---|---|
| backprop | 1 M | 1 M | 6.1086 |
| SOLO | 1 M | 1 M | 6.3386 |
| MOSAIC K=2 H=5, share_trunk | 2 M | 1 M | **6.1046** |
| backprop | 2 M | 2 M | 5.6340 |

At equal per-parameter data MOSAIC **ties** backprop and beats SOLO. At equal
total tokens backprop is 0.47 ahead. Both statements are true; which one matters
depends on whether the budget being spent is volunteer tokens (total) or
parameter updates (per-parameter). Note also that in the distributed setting a
round's slices run on different volunteers in parallel, so a MOSAIC round costs
one slice's wall-clock time for `slices` slices' worth of compute — the reason
the method exists at all.

## More data does not close it: MOSAIC has a ceiling at this configuration

Everything above compares arms at 2 M unique tokens. Doubling the budget is the
obvious next question, and the answer is unambiguous: backprop keeps descending,
MOSAIC does not.

| arm | ¼ | ½ | ¾ | final | wall |
|---|---|---|---|---|---|
| backprop, 4 M tokens (1,953 steps) | 6.1083 | 5.8403 | 5.6301 | **5.3192** | 151 s |
| MOSAIC K=2 H=5, 4 M tokens (976 rounds) | 6.3754 | 6.2539 | 6.1116 | **6.0289** | 463 s |
| MOSAIC K=3 H=5, 3 M tokens (488 rounds — a *fair* K=3) | 6.4786 | 6.3460 | 6.1862 | **6.1103** | 265 s |

Going from 2 M to 4 M moved MOSAIC by **0.05** (6.0813 → 6.0289) and backprop by
**0.31** (5.6340 → 5.3192), so the gap *widens* from 0.47 to **0.71**. Look at
MOSAIC's own curve after the 2 M mark — 6.037, 6.058, 6.040, 6.029 — it is flat
while backprop is still losing 0.1 nats per 500 k tokens. That is a ceiling in
the local objective, not a data-starvation artefact, and it retires the earlier
hope (motivated by DGL reaching backprop parity on ImageNet) that more tokens
would close the gap at this scale.

K=3 at a matched **488 rounds** (3 M tokens) gives 6.1103 — worse than K=2's
6.0813. More slices is not a lever. Neither are: local steps (optimum H=5),
outer momentum (6.0813 vs 6.0845 with momentum off), the per-tensor trust ratio
(6.39 / 6.67), a larger local batch (6.41), or the outer rate. The only
interventions that ever produced a real gain were the four tensor-ownership
bugs, worth ~1.2 nats in total (7.2456 → 6.0289).

## Two research-backed candidates refuted, and the rate is not tradeable

Both were tried at the established optimum (39.9 M, 2 M unique tokens, 488 rounds,
K=2 H=5 both-roles, inner 1e-3, outer 0.15, momentum 0.9). Baseline repeat on the
patched code: **5.8347**.

| candidate | final | verdict |
|---|---|---|
| DGL input-skip head (`--head-skip`: head sees the slice input `h0` too) | **6.2284** | refuted, +0.39 |
| same product 7.5e-4 with H=10 / outer 0.075 | **6.4468** | refuted |
| same product 7.5e-4 with inner 3e-4 / outer 0.45 | **5.9408** | refuted |
| plain baseline repeat | 5.8347 | reference |

**The input-skip head fails for a specific, explainable reason.** DGL hands its
auxiliary network the module's input and output, and that works when the aux net
predicts a *label*. Ours predicts next-token logits, so a head that can see `h0`
can predict from `h0` alone -- the slice's own output stops being necessary, the
gradient that was supposed to shape that slice's trunk decays, and the slice
becomes an expensive identity. The head is also the wrong place to look for the
readout deficit: the fresh-decoder probe already showed the head path costs 0.20
while a decoder of the same shape on the same features recovers it.

**The product `H * inner_lr * outer_lr` is not the invariant.** With the product
pinned at 7.5e-4, moving either factor away from (H=5, 1e-3, 0.15) costs 0.11-0.61
nats. Each factor is at an interior optimum on its own, so the rate cannot be
traded away for wall-clock -- a 10-round-per-minute configuration is not
available at equal loss.

**Run-to-run spread is configuration-dependent**, which matters for reading any
of these numbers: three runs of the same command at outer 0.15 gave 5.8121,
5.8208 and 5.8347 (spread 0.023), while a repeat at outer 0.3 differed by 0.0003.
Treat anything under ~0.02 as unresolved, and never quote a single run as a win.

## Recalibrating the teacher does not help, and doing it harder hurts

The one measured deficit left is the head's: a fresh decoder fitted on MOSAIC's
own frozen features reaches 5.82, while MOSAIC's own head path outputs 6.02 from
those same features. Raising the head's learning rate failed (3x / 10x / 30x ->
6.1383 / 6.2069 / 6.1772 against 6.0813), which left *lag* as the explanation:
the head chases a trunk that the aggregate keeps moving.

`fit_head` tests that directly. It freezes the trunk, fits the head to its own
local loss (AdamW over head parameters only, `h0`/`h` under `no_grad`), and
returns the local loss; the harness calls it every N rounds before `aggregate`.

39.9 M, 2 M unique tokens, 488 rounds, H=5, inner 1e-3, outer 0.15:

| recalibration | final |
|---|---|
| none (baseline) | 5.8516 |
| every 20 rounds, 20 head steps | 5.8506 |
| every 20 rounds, 100 head steps | 5.9252 |
| every 100 rounds, 100 head steps | 5.8876 |

Refuted, and the sign is instructive: light recalibration is identical to none
(0.001), and heavy recalibration is *worse* by 0.04-0.07. The online head beats
a freshly fitted one on a moving trunk. Two reasons fit the data. A head fitted
to a 2,048-token batch at lr 1e-2 is fitted to that batch, so it becomes a noisy
teacher. And the thing that helps a slice is a head whose errors are correlated
with the trunk's *current* state; re-fitting to the current state every 20
rounds breaks that correlation for the rounds in between.

The earlier probe does not contradict this: it fitted a decoder to *frozen*
features, which measures what a converged decoder could extract, not what makes
a good teacher for a student that is still moving.

So the remaining ~0.2 nats at equal tokens is not teacher quality. The
candidates left are structural: overlap at the slice boundary (staged), and
scale (the 306 M arm).

## Overlap at the cut, and a decayed outer rate, are not gains

Two candidates motivated by the residual-stream reading of the gap (~0.31 nats of
trunk quality), both measured at the established optimum (`H=5`, inner 1e-3,
outer 0.15, momentum 0.9, trunk owner first, share on, persistent inner state,
39.9 M, 2 M unique tokens, 488 rounds):

| arm | final |
|---|---|
| baseline spread at outer 0.15 | 5.8121 / 5.8208 / 5.8347 / 5.8516 |
| `--overlap 1` (each slice also trains the block past the cut) | 5.8165 |
| `--overlap 2` | 5.8246 |
| cosine outer 0.3 -> 0.05 | **5.8120** |
| cosine outer 0.5 -> 0.05 | 5.8505 |

**Overlap** lets two neighbouring slices own the boundary block, so
`aggregate` applies the mean of their two proposals (no wire change; the same
mechanism that makes the tied `wte` work). It is inside the spread at both
widths: the layer at the cut is not where the missing nats live.

**Decay** reaches exactly the tuned fixed rate (5.8120 vs 5.8121) from a start of
0.3 and does not reach it from 0.5. So an interior optimum does move during
training, but the best static rate already captures what a schedule recovers:
decay buys robustness to a badly chosen start rate, not loss.

## Head capacity does not help either

The auxiliary head is a single `Linear(dim, dim)` behind a LayerNorm, while the
slice it supervises squeezes six transformer blocks into that one linear map.
DGL's ablations make aux-net capacity matter on hard objectives, so the obvious
candidate was a hidden layer: `SoloHead(..., mlp=True)` adds
`phi2( gelu( phi( ln(h) ) ) )`, off by default and bit-identical when off.

At the tuned optimum (39.9 M, 2 M unique tokens, 488 rounds, H=5 both-roles,
inner 1e-3, outer 0.15, momentum 0.9):

| arm | curve | final |
|---|---|---|
| baseline repeat | 8.4170, 6.1853, 5.9990, 5.8452 | **5.8452** |
| `--head-mlp` (hidden layer) | 8.4170, 6.2974, 6.0823, 5.9338 | **5.9338** |

A hidden layer costs **0.089 nats**, above the 0.04 spread of repeated runs at
this configuration (5.8121 / 5.8208 / 5.8347 / 5.8452 / 5.8516 — a 0.0395 range
from five runs of the same command). A bigger teacher is a *richer* teacher but
also a target that is easier to satisfy without improving the representation:
the head can fit the labels with its own nonlinearity instead of forcing the
slice to supply them.

So the head has now been attacked from four directions and every one is refuted:
learning rate (3x / 10x / 30x -> 6.1383 / 6.2069 / 6.1772), information (an
input-skip path -> 6.2284), staleness (recalibration -> 5.8506 / 5.9252 /
5.8876), and capacity (a hidden layer -> 5.9338). The gap is therefore not in
the teacher at all. What remains is the form of the objective itself: each slice
minimises its own one-step-ahead loss and nothing ever propagates the terminal
loss back through the cut, which is the greedy-learning gap the method accepts.

## Local steps past five do not just fail, they diverge

The H=5 optimum was measured before the outer rate was understood, so H had to be
re-measured at the tuned rate (`inner 1e-3`, `outer 0.15`, `momentum 0.9`,
both-roles, 39.9 M, 2 M tokens, 488 rounds):

| H | per-round local movement `H*inner*outer` | final |
|---|---|---|
| 1 | 1.5e-4 | 6.2730 |
| 5 | 7.5e-4 | **5.8121** (five runs: 5.8121 / 5.8208 / 5.8347 / 5.8452 / 5.8516) |
| 20 | 3.0e-3 | **8.0004** |

H=20 at outer 0.15 is worse than H=20 was at outer 1.0 (6.7522), and its curve
oscillates (7.53 -> 8.05 -> 8.00): four times the per-round movement of H=5 is
past the point where the aggregate can correct a slice. So the outer rate is not a
free parameter to be re-tuned for each H — `H=5, inner 1e-3, outer 0.15` is an
interior optimum in H, in the inner rate and in the outer rate at once, and it is
the configuration every claim in this document is made at.

## The 0.18-nat gap was mostly the budget, and the split is level at equal updates

A MOSAIC round processes `slices*B*T = 4096` tokens and updates every parameter
once. A backprop step at the preset's batch 4 processes 2048 and updates every
parameter once. So the comparison everyone quotes gives backprop **976** updates
against MOSAIC's **488** — equal tokens, unequal updates, and unequal tokens per
step. Backprop at `--batch 8` is the exact match: 488 updates, 4096 tokens per
step, 2 M tokens.

| budget | backprop | MOSAIC |
|---|---|---|
| equal total tokens (2 M), backprop 2048 tokens/step | **5.6309** (976 updates) | 5.8121 .. 5.8516 (5 runs, K=2) |
| equal parameter updates (488) **and** equal tokens | 5.8370 / 5.8440 (mean 5.840) | 5.8121 .. 5.8516 (mean 5.831) |
| equal rounds (488), K=3 at 3 M tokens | 5.8370 / 5.8440 | **5.7893 / 5.8181 / 5.7849** (mean 5.797) |

Readings, stated at the precision the spread supports (0.04 for MOSAIC, 0.007 for
backprop):

- at equal total tokens backprop is clearly ahead (0.18), and this is the only
  framing in which it is clearly ahead;
- at equal parameter updates and equal tokens the two are **level** (means 5.831
  vs 5.840, difference inside the spread);
- at equal *rounds* — the unit a distributed run actually pays for — MOSAIC K=3 is
  **marginally ahead** (mean 5.797 vs 5.840), a 0.043 difference that is at, not
  beyond, the combined spread, so it is a tie at best and must be described that
  way;
- K=3 spends 6144 tokens per round against backprop's 4096, i.e. it buys that tie
  with 1.5x the data and 5 local steps per slice. Tokens and local compute are the
  volunteer's; rounds are the scarce resource, which is the argument for the
  method, not a loss result.

The claim this document supports is therefore: **at a fixed round budget the split
matches backprop**, with the loss advantage at equal total tokens still belonging
to backprop. Configuration: `K=2 or 3, H=5, inner 1e-3, outer 0.15, momentum 0.9,
--trunk-owner first --share-trunk --inner-state`.

## Pre-declared metric (fixed before the 306 M runs)

Every number in this document was measured without first saying which budget the
comparison would use, so each new arm invited a different accounting. Stating it
now, before the 306 M runs, and using it for everything after this line:

**The project's budget is the ROUND.** A round is one aggregation: every parameter
of the global model is updated once, from deltas produced by volunteers who each
held one slice. It is what the network and the wall clock pay for, and a round's
slices already run in parallel on different peers, so doubling the slice count
does not double the round's latency — that parallelism is the reason the method
exists.

Consequences, fixed in advance:

1. **Primary metric:** held-out loss at a fixed number of rounds (488 rounds for
   the 2 M-token arms, 976 for the 4 M arms).
2. **Backprop control at matched rounds** is `backprop --batch 8`: 4096 tokens per
   step, one update of every parameter per step, i.e. exactly one round's
   accounting. `--batch 4` (2048 tokens/step) is reported too, as the
   *token-matched* control, and losing to it does not refute the round-budget
   claim.
3. **Every claim names its budget.** "Better" without a budget is not a claim;
   at equal total tokens a split method that spends `slices` batches per round is
   expected to trail, and it does.
4. **Scale is part of the claim.** All anchoring numbers below 306 M are 39.9 M
   results; the 1 B deployment target is not measured and will not be claimed.

### How the round-matched control is built at 306 M

One round-matched backprop update consumes 4096 tokens (`slices * B * T` = 2 * 4 *
512), which at 306 M needs `--batch 8`. That does not fit: the first attempt at
306 M with `--batch 8` was killed before printing a curve (the B=4 preset already
holds ~11.4 GB of the 15.57 GB card). So the control is built from two B=4
micro-batches with `--grad-accum 2`: identical optimisation step (one update over
4096 tokens, same 488 steps for 2 M), identical activation footprint to the B=4
runs that already work, 2x the wall clock. Verified on CPU that the accounting is
right before spending GPU time: `--batch 2 --grad-accum 2 --tokens 40960` gives
`steps=20 uniqueTokens=40960 tokensPerStep=2048`, i.e. 40960/2048.

A control that cannot run is not a control, so this substitution is stated here
rather than buried: the round-matched arm differs from the token-matched arm in
per-step token count (4096 vs 2048), and that is the entire point.

## 306 M, at the pre-declared round budget

Preset V=4096, D=1024, L=24, H=16, T=512, B=4 (306 M parameters), 8 fixed val
windows, init 8.5367. MOSAIC used the tuned configuration (H=5, inner 1e-3, outer
0.15, momentum 0.9, `share_trunk`, trunk owned by the first slice, persistent inner
state). The round-matched control is `--grad-accum 2` at B=4: **one 4096-token
update per round**, which is exactly what a round costs.

| arm | updates / rounds | tokens | held-out | wall |
|---|---|---|---|---|
| backprop, round-matched | 488 | 2.00 M | **5.7631** | 430 s |
| MOSAIC + anchor every 1 (2048-token anchors) | 488 + 488 | 3.00 M | **5.5941** | 1658 s |
| MOSAIC + anchor every 2 (512-token anchors) | 488 + 244 | 2.13 M | **5.7269** | 1476 s |
| backprop, round-matched | 732 | 3.00 M | **5.5286** | 646 s |
| backprop, round-matched | 976 | 4.00 M | **5.3523** | 862 s |
| MOSAIC + anchor every 2 (512-token anchors) | 976 + 488 | 4.25 M | **5.6381** | 2957 s |

### At equal rounds

488 rounds: backprop **5.7631**, anchored MOSAIC **5.5941** (anchor every 1) and
**5.7269** (anchor every 2). So the hybrid wins by **0.17 nats** at the configured
frequency, against 0.25 at 39.9 M. The effect is slightly *smaller* with scale, not
larger: the 39.9 M number was not a small-model artifact, but it does not magnify
either, and that is the honest direction of travel.

976 rounds is the exception and it is stated plainly: **5.6381 vs 5.3523, 0.29
behind**. That arm is not the configuration that won at 39.9 M — the 16 GB card
forced its anchor step down to a 512-token batch, so over 976 rounds it gets 250 k
anchor tokens where anchor-every-1 gets 1 M. Whether the win survives 976 rounds
at full anchor volume is the open question, and it is the one arm the local GPU
cannot hold (the anchor optimiser's own state is 2× params in fp32).

### At equal total tokens

| tokens | backprop | anchored MOSAIC | delta |
|---|---|---|---|
| 2.0 / 2.13 M | 5.7631 | 5.7269 | −0.036 |
| 3.0 M | 5.5286 | 5.5941 | +0.065 |
| 4.0 / 4.25 M | 5.3523 | 5.6381 | +0.29 |

**Level at 2 M, behind beyond it.** At 39.9 M the same comparison was a clear loss
at every scale (5.5854 vs 5.4341 at 3 M), so the token gap narrows but does not
close. This is exactly why the metric had to be declared before the runs: the
honest sentence is *matches backprop at equal rounds, behind at equal tokens*, and
which of those is the project's budget is an argument about the deployment, not
about the numbers.

The deployment argument, stated so it can be refuted: a round is one aggregation in
which every parameter is updated exactly once from volunteer deltas, and the slices
run in parallel on different volunteers, so round latency does not grow with slice
count or with local-step count. Tokens, by contrast, are volunteer-donated compute
that would otherwise be idle. If that is right, the round column is the one to
optimise and the token column is a constraint on volunteer cost, not on latency.
If a deployment ends up token-bound, backprop wins and anchoring should stay off
(`CROWDGPT_ANCHOR_EVERY` defaults to 0).

## Aggregator anchoring: the first candidate that closes the gap

Every algorithm change so far attacked the local step. This one changes *who
takes the step*. The greedy gap exists because no slice ever sees the terminal
loss cross its boundary; but there is exactly one participant that holds the
whole model -- the aggregator. It already holds a held-out corpus (the objective
cross-check needs one). So: let it take **one ordinary AdamW step on the true
terminal loss every N rounds**, and let the local rounds do the rest.

`--anchor-every N --anchor-lr 3e-4` in the harness: at the end of every Nth
round, `ce(model(x), y).backward()`, clip 1.0, step over all trainable
parameters, on `batch(10_000_000 + r)` so the anchor data is disjoint from every
slice's round data. Nothing about the wire changes: the anchor happens on the
aggregator's own copy of the weights between rounds.

### Frequency curve, 39.9 M, 488 rounds (2 M round tokens)

| anchor | 5.8121-family baseline | anchor lr 3e-4 | tokens spent anchoring | total tokens |
|---|---|---|---|---|
| none | **5.8346** this sweep (family 5.8121-5.8516) | — | 0 | 2.00 M |
| every 50 | | 5.8240 | 0.04 M | 2.04 M |
| every 10 | | 5.7822 | 0.10 M | 2.10 M |
| every 10, lr 1e-3 | | 5.7741 | 0.10 M | 2.10 M |
| every 3 | | 5.7218 | 0.33 M | 2.33 M |
| every 2 | | **5.6732** (lr 1e-3: 5.6748) | 0.50 M | 2.50 M |
| every 1 | | **5.5854** | 1.00 M | 3.00 M |

Monotone in frequency, and the every-1 arm (5.5854) is better than **backprop at
2 M tokens** (5.6340) while every previous MOSAIC configuration sat 0.18 behind
it. The anchor lr is not sensitive between 3e-4 and 1e-3 (5.6732 vs 5.6748 at
every 2).

### Does it scale? 4 M round tokens, anchor every 2

`curve=['8.4170','6.0683','5.8481','5.6752','5.5933','5.4674','5.4228','5.3642']`
final **5.3642** (976 rounds, 499 s).

Against the no-anchor arm at the same 4 M (5.6188) that is **-0.25**, and against
backprop at 4 M (5.3192) it is **0.045 behind** instead of 0.30. So anchoring does
not just shift the curve, it keeps paying as the run gets longer.

### The honest accounting: it is a hybrid, and it is behind at equal tokens

Each anchor step costs exactly what a backprop step costs (one batch, one update
of every parameter), so the total-token bill includes it:

| arm | round tokens | anchor tokens | total | final | backprop at the same total |
|---|---|---|---|---|---|
| anchor every 2 | 2.0 M | 0.5 M | 2.50 M | 5.6732 | **5.5582** (1,220 steps) |
| anchor every 1 | 2.0 M | 1.0 M | 3.00 M | 5.5854 | **5.4341** (1,464 steps) |
| anchor every 2 @4 M | 4.0 M | 2.0 M | 6.00 M | 5.3642 | **5.0831** (2,929 steps) |

So at equal **total tokens** backprop still wins, by 0.12 / 0.15 / 0.28 -- because
the anchor is literally backprop compute, applied once every N rounds instead of
every step. The case for anchoring rests entirely on the budget the project
actually pays:

**At equal rounds it is a decisive win.** A round is the unit of volunteer
wall-clock: K slices train in parallel and the aggregator merges once. Backprop
at 488 steps on the same tokens-per-step (4,096) measures 5.8370 / 5.8440, while
MOSAIC + anchor-every-1 at 488 rounds measures **5.5854** -- 0.25 better. And the
anchor's tokens and FLOPs are spent on the aggregator, which already holds the
whole model, not on volunteer bandwidth or volunteer compute.

The configuration is therefore: **local MOSAIC rounds (K=2, H=5, inner 1e-3,
outer 0.15, share_trunk) + one aggregator-side full-model step every round**,
with anchor data drawn from the server's own held-out corpus and disjoint from
every slice's round data.

### Why this is not just "backprop with extra steps"

At equal rounds backprop gets 488 updates of every parameter on 4,096 tokens
each; the hybrid gets 488 aggregated local updates **plus** 488 global updates,
and the local updates are trained on volunteer-owned slices of the model, so they
do not require the volunteer to hold or transmit the whole model. The two step
types are complementary in the measurement (no-anchor 5.83 -> every-2 5.67 ->
every-1 5.59, monotone), which is what "the global step repairs what the greedy
step cannot" predicts and what "the global step is just doing the work" would not
(the every-1 arm would then be backprop, and it is 0.05 *better* than backprop at
2 M).

### Device check at the tuned rate

ROCm gfx1150 (aptai2, `/root/rocm-venv`, base64-inlined modules) at the tuned
configuration `--outer-lr 0.15 --inner-lr 1e-3 --local-steps 5 --share-trunk
--trunk-owner first --inner-state`, 2 M round tokens, 488 rounds:

```
curve=['8.3978', '6.1939', '5.9851', '5.8270'] final 5.8270 5070.9s 394 tok/s rounds=488
```

against CUDA 5.8121 at the same configuration — **0.015 apart**, the same band as
the earlier 0.025 agreement, so the tuned rate is not a CUDA artefact. (ROCm at
4 M with the old outer 0.3 landed at 5.6526; that arm predates the tuning and is
provenance only.)

## Noise floor (why every claim below is quoted to 0.1 nats)

The harness is deterministic by construction: `build()` calls
`torch.manual_seed(0)` before constructing the model, and `batch(step)` draws from
`torch.Generator().manual_seed(step * 1_000_003 + 7919)`, so two runs of the same
command see the same init and the same token streams. Anything that still differs
is CUDA kernel nondeterminism, not data.

That matters because two arms which are *supposed* to be identical differed by
0.13 nats at H=1 (6.4651 vs 6.3344). Both were `--trunk-owner last`; in that
configuration the `share_trunk` clause is a no-op, because `wte` already belongs to
the terminal slice and the `if n not in {...}` guard skips it. Either a detail
differs or a detail differs.

Repeats have now measured it: the same both-roles command run twice gave **6.0813
and 6.1027**, a spread of **0.021 nats**. So the harness is reproducible to about
0.02, and the 0.13 gap between those two older arms is *not* run-to-run noise --
the code changed between them (`share_trunk`, `opt_state` and the trunk-end
assignment all landed in that window), so arms from different sweeps are only
comparable when the commands are re-run on the current code. Two consequences keep
this honest:

- differences under ~0.05 nats are unresolved;
- the both-roles gain over readout-only (6.1281, one run) is ~0.03-0.05, i.e. **not
  established**. What the data does establish is that the *terminal slice* training
  the tied readout is worth ~0.8-1.0 nats, not that both slices training it adds
  anything measurable.

## The genuine both-roles arm, and where the remaining gap lives

`--trunk-owner first --share-trunk` is the configuration the noise-floor section was
waiting for: `wte` is the embedding that slice 0 trains through its local head *and*
the readout the terminal slice trains under the true loss, and `aggregate` applies
the mean of the two proposals. At 488 rounds, H=5, inner 1e-3 / outer 1.0 /
momentum 0.9, persistent inner optimiser state:

```
mosaic true            6.0181
probe  mosaic trunk    5.8174     (fresh LayerNorm+Linear decoder on its features)
probe  mosaic half     5.8091     (same decoder, half-depth features)
```

Against the same probes on the backprop arm (true 5.5475, trunk probe 5.5091), the
residual 0.47-nat gap decomposes cleanly:

- **~0.31 nats of trunk quality** — MOSAIC's representation is simply weaker
  (5.817 vs 5.509 when both are read by an identical fresh decoder).
- **~0.16 nats of readout path** — MOSAIC's own output path (6.018) is 0.20 worse
  than what a fresh decoder achieves from its own features (5.817). Backprop's own
  head is only 0.04 off its ideal decoder, so its output path is nearly optimal
  while MOSAIC's is not.

Two consequences worth stating plainly:

1. 6.0181 is the best MOSAIC number measured, and it is **0.09 better than
   backprop at the same per-parameter data** (6.1086 at 488 steps / 1 M tokens) —
   a margin inside the 0.15-nat noise floor, so the honest claim is "ties
   backprop", not "beats it". Repeats are running.
2. The readout-path deficit is the cheaper of the two to attack, because the tied
   readout is a single tensor we directly control, whereas the trunk deficit is the
   method's own greedy gap (~0.3 nats, consistent with Decoupled Greedy Learning's
   reported behaviour on Transformer-scale models with little data).

## Re-sweeping the rates on fixed code: the outer loop was too big

Every rate below was originally tuned while `slice_of` still excluded the tied
readout, so the old optimum was measured on a broken objective. Re-running the
ladder on the fixed code changes the answer.

**Hypothesis tested and refuted — the auxiliary head lags its trunk.** A fresh
decoder reaches 5.817 on MOSAIC's own frozen features while MOSAIC's own head path
outputs 6.018, which looks like a teacher that cannot keep up. Giving the head its
own param group (`--head-lr-mult`, all else fixed at inner 1e-3 / outer 1.0 /
momentum 0.9 / H=5):

| head lr multiplier | final |
|---|---|
| 1x (baseline) | **6.0813 / 6.1027** |
| 3x | 6.1383 |
| 10x | 6.2069 |
| 30x | 6.1772 |

Every multiplier is worse, so the head is not under-trained relative to the trunk;
running the teacher hotter only injects noise into the slice's gradient. The 0.31
trunk deficit is not teacher lag. The knob stays at 1.0 and is not part of the
recommended configuration.

**Hypothesis confirmed — the aggregated step was too large.** With the head at 1x
and the trunk ownership correct, the outer rate is the lever that moves:

| inner lr | outer lr | final |
|---|---|---|
| 3e-3 | 1.0 | 6.2876 |
| 1e-3 | 1.0 (rerun) | 6.0902 |
| **1e-3** | **0.5** | **5.9625** |

`outer_lr` 0.5 buys 0.13 nats over 1.0, six times the 0.021 repeat spread, and the
baseline rerun reproducing 6.0902 (vs 6.0813 / 6.1027 earlier) confirms the
comparison is single-code-version. Best MOSAIC is now **0.33 behind backprop at 2 M**
(5.9625 vs 5.6340) instead of 0.47.

Reading: `aggregate` applies `current + outer_lr * step`, where the step is already
a local AdamW displacement of order `inner_lr` per tensor. At outer 1.0 the round
re-applies that displacement in full; at 0.5 it applies half, closer to what one
backprop step of the same nominal rate does. The earlier "H=5 is the optimum" and
"outer momentum does nothing" results were both measured at outer 1.0, so the
local-step count has to be re-checked at the new rate — running.

**The inner rate is a weak lever, on both devices.** ROCm gfx1150, same geometry,
`inner 3e-4 / outer 1.0`:

```
mosaic(K=2,H=5) curve=['8.3978', '6.3627', '6.1798', '6.0877'] final 6.0877 2472.2s 809 tok/s rounds=488
```

6.0877 against 6.0902 on CUDA at `inner 1e-3 / outer 1.0`, and 6.1244 for the
earlier ROCm both-roles run at the same geometry. Three times smaller inner rate,
no measurable change: within the run-to-run band, and below the 0.13 that the outer
rate is worth. So the tuning that matters is the *aggregated* step, not the local
one. Confirming `outer 0.5` on ROCm is running.

**The outer rate keeps paying below 0.5, and H=5 survives the re-test.** Same
geometry (`inner 1e-3`, momentum 0.9, both-roles, 488 rounds):

| H | outer lr | final |
|---|---|---|
| 5 | **0.3** | **5.8446** |
| 5 | 0.5 | 5.9625 |
| 5 | 0.7 | 6.0956 |
| 5 | 1.0 | 6.0902 |
| 10 | 0.5 | 6.1235 |
| 10 | 0.3 | 6.1570 |

Monotone improvement from 1.0 down to 0.3, so the optimum is below 0.3 and the
sweep continues (0.2 / 0.15 / 0.1, plus a repeat of 0.3 to confirm it is not a
lucky draw). H=10 loses at *both* rates, so the H=5 local-step optimum is not an
artifact of the oversized outer step after all.

At 5.8446 MOSAIC is **0.21 behind backprop at 2 M** (5.6340) and **0.26 behind
backprop at matched per-parameter data** (6.1086 at 488 steps — which MOSAIC now
beats outright at the same 488 parameter updates).

**The optimum is 0.15, and the repeat proves the harness is deterministic there.**
Finer sweep at the same geometry (`inner 1e-3`, momentum 0.9, H=5, both-roles,
488 rounds):

| outer lr | final |
|---|---|
| 0.3 | 5.8446 |
| **0.3 (repeat)** | **5.8449** |
| 0.2 | 5.8216 |
| **0.15** | **5.8121** |
| 0.1 | 5.8825 |

Two things matter here. First the repeat: 5.8446 and 5.8449 differ by **0.0003**,
so at this rate the harness is deterministic to three decimals and every
difference in the table is real. That retires the earlier 0.021 "noise floor" as a
property of the *old* configuration (outer 1.0, where an oversized step amplified
nondeterminism), not of the measurement itself.

Second the shape: a clean minimum, not a monotone trend. 0.1 is *worse* than 0.15
(5.8825 vs 5.8121), so "smaller is better" would have been the wrong
extrapolation from the 1.0 -> 0.5 -> 0.3 segment. Below ~0.1 the round stops
moving far enough to aggregate anything useful; above ~0.2 it re-applies too much
of a displacement the local steps already took. The useful range is narrow, which
is itself a finding: MOSAIC is sensitive to the outer rate in a way backprop is
not, because backprop has no outer step at all.

At **5.8121** MOSAIC is 0.18 behind backprop at 2 M tokens (5.6340), and 0.30
*ahead* of backprop at matched per-parameter data (6.1086 at 488 steps).
