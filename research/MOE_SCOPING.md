# Scoping: from the 1 B dense model to a 4 B / 7 B A1B MoE

Written 2026-11-19, after v1.5 feature freeze (client #133-#136, server #26-#28).
This is a scoping memo, not a result: it separates what we have measured from what
we are projecting, and it recommends an order of work.

## The question

Should the first MoE be 7 B total / 1 B active, or 4 B total / 1 B active?

## What the swarm actually constrains

Three budgets, and MoE only helps one of them.

| budget | 1 B (current 657.8 M) | 4 B A1B | 7 B A1B |
|---|---|---|---|
| resident weights, bf16 (volunteer) | 1.3 GB | 8 GB | **14 GB** |
| full AdamW over trained params | 5.3 GB | 32 GB | **56 GB** |
| delta on the wire, bf16 | 1.3 GB | 8 GB | **14 GB** |
| coordinator fp32 master | 2.6 GB | 16 GB | **28 GB** |
| + anchor AdamW state (2x params) | 5.3 GB | 32 GB | **56 GB** |
| + DiLoCo outer buffer (1x params) | 2.6 GB | 16 GB | 28 GB |

Read the last column twice. A 7 B MoE with anchoring *and* outer momentum wants
roughly **112 GB of coordinator RAM and 14 GB of delta per upload per round**.
Those are not tuning problems; they are re-architecture problems:

- `krum_stream._read_base(df)` reads the **whole** delta block into memory
  (`model_size * 2` bytes) for every submission. At 1.3 GB that is tolerable; at
  14 GB it is not. It has to scatter in chunks.
- Every volunteer currently uploads the **whole** model delta, zero-padded outside
  whatever it trained (MOSAIC slices included). At 7 B that is 14 GB per round per
  volunteer regardless of how little they touched.

So the honest statement is: **MoE buys sparse compute, not sparse transfer.** In a
swarm, sparse transfer is the budget that binds, and we have to build it
deliberately.

## The one thing that makes MoE worth it here

Expert-parallel training. A MoE is the only architecture where "the model is
bigger than any volunteer's memory" is the *design*, not a compromise: a volunteer
holds a subset of experts, and total parameters grow while per-volunteer cost stays
flat.

That is exactly the MOSAIC shape we already have (contiguous owned slices, local
objective, aggregator merges). Which means the real prerequisite for any MoE is not
size, it is **slice-scoped upload**: send only the tensors you own, not a
zero-padded full vector. Today the sparse container (`SPK1`: magic + count + total,
6 bytes per entry, server-side `index_add_` scatter) can already express that — the
missing piece is a *range* selector instead of top-|value| selection, driven by the
slice's span map. That same span map is V2I-14, so the two items are one item.

## The A1B arithmetic the vocab changes

`MODEL_CONFIG` is `vocabSize 151,669`, `dim 1536`, `nLayers 24`, `mlpHidden 2560`,
tied. The embedding is `151669 * 1536 = 233 M` parameters — **35% of the model**.

In an A1B MoE that 233 M is *shared and always active*. So "1 B active" at this
vocab leaves only ~0.75 B for attention + experts. Options, in order of preference:

1. **Shrink the output vocabulary** (32-64 k BPE) before the MoE. It is a tokenizer
   migration, but it triples the expert budget at the same active count and halves
   the delta's dominant tensor.
2. Keep 151 k and accept a 0.7 B-active MoE (call it A0.7B, not A1B).
3. Train a small adapter head over a frozen 151 k embedding — orthogonal, more
   moving parts.

Do not design a "1 B active" MoE on top of a 233 M shared embedding and then
discover the active budget is mostly embedding lookup.

## The decision: 4 B A1B first

**Recommendation: 4 B total / 1 B active first, then 7 B as a config change.**

1. **It fits a real volunteer.** 4 B in bf16 is 8 GB; a 16 GB card keeps 8 GB for
   activations, optimiser shards and the OS. 7 B in bf16 is 14 GB and does not fit
   with anything else — so 7 B *requires* expert partitioning to work before it can
   run at all. 4 B can run both ways, which gives the control.
2. **The failure modes are cheaper to find at 4 B.** Router collapse, aux-loss
   weighting, expert imbalance across non-IID volunteer shards, and
   capacity-factor clipping are all first-time-through bugs. Thirty-two experts and
   one node of coordinator RAM is a much better place to meet them than sixty-four
   and 28 GB.
3. **4 B -> 7 B is a config change, not a redesign.** Same code path, same wire
   format, different expert count and width. The lessons transfer completely; the
   reverse is not true (a 7 B bring-up that fails tells you little about which of
   five subsystems failed).
4. **It gives a controlled comparison we actually need**: 4 B-A1B vs our 1 B dense
   at matched *active* compute. That is the number that decides whether MoE belongs
   in this project at all, and it is cheaper to obtain at 4 B.

**The counter-case, honestly:** 7 B-A1B has a 7:1 total-to-active ratio, which is
the healthier sparsity regime (4:1 is close to dense and gets less specialisation
per expert). If expert-parallel slicing lands first and memory stops binding, going
straight to 7 B is defensible and saves a cycle. The condition is precise: *7 B only
after slice-scoped upload and chunked scatter exist and are measured*, otherwise
the arm cannot run.

## The precondition nobody can engineer around: tokens

MoE needs more tokens than a dense model of the same active compute to pay off —
each expert must see enough data to specialise. Practical floor is ~10x active
parameters, comfortable is 50-100x. At A1B that is **>=10 B tokens to not waste the
MoE, 50 B+ to benefit**.

If the corpus available at launch is ~10 B tokens, a well-trained 1 B dense beats a
4 B-A1B, and the MoE should wait. This is a data question, not a code question, and
it dominates the 4 B vs 7 B choice. **Check the corpus size before committing to
either.**

## Work items, in order

| # | item | why it is a prerequisite | size |
|---|---|---|---|
| 1 | chunked delta scatter in `krum_stream` | 8-14 GB per delta cannot be read whole | 1 day |
| 2 | slice-scoped upload (span map -> range selection in `SPK1`) | bytes must scale with owned tensors, not total; also closes V2I-14 | 2-3 days |
| 3 | coordinator weight dtype (bf16 master + fp32 per-round accumulation, or mmap) | fp32 master x 4 bytes/param is 28 GB at 7 B | 2-3 days |
| 4 | tiny-MoE harness arm (`scale_306m.py`: 4 experts, top-2, optional MOSAIC-by-expert) | proves the pipeline trains a MoE before we spend a month on 4 B | 2-3 days |
| 5 | expert-parallel MOSAIC + **router ownership** | the research risk (below) | 1-2 weeks |
| 6 | 4 B A1B bring-up (sharded init, checkpoint, resume) | first real MoE | 2-3 weeks |

## The research risk: the router is a shared tensor

Our three allow-list bugs in MOSAIC were all the same class: a tensor that belongs
to the model but to no slice's optimiser, silently frozen. The tied readout cost
**0.8 nats**. A MoE router is the same shape of problem and worse:

- the gate weights are touched by **every** token, so every slice's local objective
  produces router gradients;
- the router must be *one* tensor globally (routes are meaningless per-slice);
- if the router is trained only by the coordinator's anchor step, it never sees
  which experts actually help on volunteer data;
- if it is replicated per volunteer, replicas diverge and the merged model has one
  routing function nobody trained.

Priors we can start from: route-only-in-anchor (router learns from the anchor
batches, experts learn locally), or router-owner-slice (one slice owns it, the way
one slice owns the tied readout, and `share_trunk` averages the proposals). The
second reuses machinery we already have and is the first thing to test.

**Also measured and relevant:** anchoring is what closed MOSAIC's gap to backprop
(306 M, equal rounds: 5.5941 vs 5.7631, -0.17 nats). A MoE gives anchoring a second
job — it is the only place a global-loss step on the router can happen — so
`CROWDGPT_ANCHOR_EVERY` stops being optional in a MoE deployment.

## What we can test this week, cheaply

1. Tiny MoE at the 39.9 M preset (4 experts, top-2, ~10 M active) vs dense 39.9 M:
   does our harness train a MoE at all, and what does the loss curve look like?
2. The same with MOSAIC slices mapped to experts: does slicing by expert behave
   like slicing by layer (it should — the ownership problem is identical)?
3. Router under sharding: two slices, router owned by the anchor step vs by a
   slice, watch for routing collapse (max expert share > 50% is the tell).
4. MTP interacts well here: DeepSeek's V3 uses MTP with MoE, and our MTP heads are
   client-local, so a MoE client pays nothing on the wire for them.

None of these need new infrastructure — `scale_306m.py` already builds arbitrary
shapes via `--dim/--layers/--heads` and owns the round loop.

## What those four tests returned (measured, ROCm gfx1150)

`experiments/v2/moe_probe.py` (committed, 3.9 M-param model: 8 experts, top-2, one
shared expert, dense control matched on **active** MLP rows — 147,456 rows/layer
each). Real BPE-4096 tokens, 8 fixed val windows, `crowd_mosaic.aggregate` as the
merge rule.

| arm | data | final | wall | expert share (1/8 = 0.125) |
|---|---|---|---|---|
| dense, active-matched | 1,000 full steps (~1.02 M tokens) | **5.7647** | 51.9 s | — |
| MoE, full backprop | 1,000 full steps | **5.7371** | 125.0 s | 0.036–0.178, healthy |
| columns, trunk trained by **nobody** | 200 rounds | 7.9725 | 142.8 s | drifting |
| columns, trunk by anchor, no balance loss | 200 rounds | 6.6090 | 150.4 s | 0.020–0.196 |
| columns, trunk by anchor, balance 0.01 | 200 rounds | 6.5748 | 122.8 s | 0.013–0.245 |
| the same arm repeated | 200 rounds | **6.5748** | 122.8 s | identical |
| columns, **column 0 owns the trunk** | 200 rounds | 8.0611 | 167.9 s | oscillating 8.03→8.11 |
| columns under MOSAIC's tuned throttle (outer 0.15, mom 0.9) | 200 rounds | 6.7181 | 141.6 s | 0.042–0.241 |
| columns, 4x data | 800 rounds (per-expert tokens ≍ dense) | **6.1015** | 546.9 s | 0.076–0.226 |

Answers to the four questions:

1. **Does our loop train a MoE at all? Yes.** Full backprop lands 0.028 *ahead* of
   the active-matched dense control with healthy routing, so the architecture is not
   the problem. (One backprop step is slower — 8 experts evaluated per layer — which
   is the compute-sparsity trade showing up honestly.)
2. **Slicing by expert behaves like slicing by layer, with one difference that
   matters:** the slice is no longer contiguous in the residual stream, so a column
   *must* be given the trunk. The trunk (attention, LayerNorms, embedding, tied
   readout, router, shared expert) is **2,306,560 of 3,879,424 params = 59%**, and in
   the first column runs it sat in **no** optimiser — the same silent-allow-list bug
   as MOSAIC's tied readout, worth **~1.4 nats** here (7.97 → 6.61).
3. **Router ownership: the anchor step, decisively.** A column owning the trunk and
   router is fatal (8.06 and oscillating): it trains them on the ~25% of tokens its
   own router sends it, and the router bias feeds back into which tokens it gets.
   This resolves the open question above against the `share_trunk` analogy.
4. **MTP**: untested here; the synergy argument stands (client-local heads, no wire
   cost).

Two further readings that change the plan:

- **Expert columns are viable but not free.** At 4x data (per-expert masked tokens
  matched to the dense arm) the gap is **0.34 nats** (6.1015 vs 5.7647), down from
  0.81 at 1/4 data. It narrows with data at a decreasing rate — the greedy gap, the
  same shape MOSAIC shows. Repeatability is exact (6.5748 twice), so these are real
  differences, not noise.
- **The outer rate should be near 1.0 for expert columns, not MOSAIC's 0.15.**
  Throttling cost 0.14 nats. A column emits a genuine parameter delta for the
  experts it owns, not a full-model re-proposal, so there is nothing to discount —
  the opposite of the layer-sliced case where the aggregate re-applied a whole local
  displacement.

Contractual consequence for a deployment: a MoE client and coordinator need an
explicit **ownership table** — experts by column, trunk + router + shared expert by
the anchor — because "not in my slice" and "not in anyone's optimiser" look
identical in the wire format. That is the same span-map work as V2I-14, now with a
second reason to exist.

## Bottom line

- **4 B A1B first.** 7 B does not fit a 16 GB volunteer's memory even in bf16, so it
  presupposes the expert-parallel work; 4 B runs either way and gives the control.
- **Expert parallelism works, at a measured price.** 0.34 nats behind dense at
  matched per-expert tokens, closing with data, and stable (exact repeat). The
  aggregator must own the trunk and router (1.4 nats if it does not; 8.06 if a
  column does), and the outer rate there is ~1.0, not 0.15.
- The gating work is **transfer**, not architecture: chunked scatter + slice-scoped
  upload, both of which we have already half-built.
- The gating *question* is **corpus size**: below ~10 B tokens, MoE is the wrong
  move at any size.
- The gating *risk* is **router ownership**, and we have been bitten by exactly this
  bug class three times already.
