# Cutting the transfer: MTP and sparse deltas

Two separate things, because the author asked for both: a training change (MTP) and
a bandwidth change (sparse deltas).

## MTP — multi-token prediction

`crowd_mtp.py` attaches `k` client-local projections, each initialised to the
**identity**, and adds `weight * mean_d(depth d CE(y[:, d:], logits[:, :T-d]))` to the
training loss. Depth 1 is plain next-token prediction, so the objective strictly
contains the old one at `weight -> 0`.

Why a swarm wants it: a next-token-only gradient is a narrow signal to average. Two
volunteers at different data shards produce gradients that disagree in most
directions, and the disagreement is exactly what a round has to average away.
Predicting two, three, four tokens ahead forces features that survive that averaging,
because the feature must carry information about the future rather than about the
next token alone.

Design decisions that matter for the wire:

- the projections are **client-local**, added to `_LOCAL_ONLY_PREFIXES`
  (`engram.`, `solo_heads.`, `mtp_projs.`), so `_is_server_param` rejects them and the
  upload stays exactly the server's flat layout;
- they are attached **before** the optimiser is built, or they would be trained but
  never stepped;
- backprop path only. A MOSAIC slice has no readout of its own — that is the reason
  its objective is auxiliary-head based — so MTP has nothing to predict *from* there.
- `CROWDGPT_MTP_TOKENS` (default 4, max 8) and `CROWDGPT_MTP_WEIGHT` (default 0.2);
  unset, zero or garbage means off, so the default path is byte-for-byte unchanged.

Tests: `tests/test_crowd_mtp.py`, 11 passing. The shift semantics are pinned by
replacing `predict_logits` with per-depth fakes and comparing against the
hand-written sum, which is the only way to catch an off-by-one that would silently
train depth `d` on the wrong tokens.

## Sparse deltas with error feedback

At 657.8 M server parameters a dense bf16 delta is **1.31 GB per volunteer per
round**. `crowd_sparse.py` keeps the top-|value| entries and carries the dropped
mass into the next round (SparseLoCo, arXiv 2407.07852; the 1-bit-SGD residual
contract). Measured on a 16 M-coordinate heavy-tailed delta, scaled linearly to the
real vector (exact, because the encoding is per entry):

| frac | entries @16 M | payload | vs dense bf16 | payload @657.8 M | one-shot L2 error |
|---|---|---|---|---|---|
| 0.2% | 32,000 | 0.2 MB | 167x | 8 MB | 69% |
| 0.5% | 80,000 | 0.5 MB | 67x | 20 MB | 51% |
| 1% | 160,000 | 1.0 MB | 33x | 39 MB | 44% |
| 2% | 320,000 | 1.9 MB | 17x | 79 MB | 43% |
| 5% | 800,000 | 4.8 MB | 7x | 197 MB | 39% |

**The fidelity price is real and is not hidden here.** Dropping 99% of coordinates
removes 44% of the displacement's L2 norm on a delta whose mass is only mildly
concentrated. The aggregator's `lr` can absorb a pure scale error, but the residual
direction error is genuine, and that is why the default stays dense (`off`).

Error feedback is what keeps the codec unbiased rather than merely small — without it
a coordinate that is consistently small is never sent. Measured after 8 rounds, as a
lag in units of one round's delta norm (`rho` = correlation between consecutive
rounds):

| frac | rho | lag, no EF | lag, EF | gain |
|---|---|---|---|---|
| 1% | 0.9 | 5.35 | 4.92 | 1.1x |
| 1% | 0.5 | 14.97 | 11.73 | 1.3x |
| 5% | 0.5 | 10.59 | 6.57 | 1.6x |

So the gain from feedback is 1.1–1.6x at these sparsities, and it is largest exactly
where the theory says it should be: when consecutive rounds are correlated
(`rho` = 0.5, i.e. the client is still delivering a displacement). At `rho` = 0 the
stream is independent noise and no codec can track it — the lag is set by the entry
budget alone. **The honest summary: this is a bandwidth knob with a measured fidelity
cost, not free compression, and whether it pays depends on how concentrated a real
round's delta is — which has not been measured here.**

### Wire format

`SPK1` block: magic, `uint32 count`, `uint32 total`, then `count` entries of
`uint32 index` (position in the flat server vector) + `bf16 value`. Indices are
global, so the server `index_add_`s straight into the weights it already holds and
never materialises a dense 2.6 GB float32 receiver copy.

Because the block is variable length, the header declares `baseBytes` (and
`baseCodec`). Every server reader that used to assume `model_size * 2` now uses it —
`krum_stream._read_block` / `_parse_sparse`, both streaming adds, `read_engram_tail`,
and the engram-tail seek in `server.py`. A dense delta leaves the field unset and
keeps the old arithmetic exactly, which is why the default path is unchanged.

Enable with `CROWDGPT_DELTA_CODEC=sparse`, `CROWDGPT_SPARSE_FRAC` (default 0.02),
`CROWDGPT_SPARSE_FEEDBACK` (default on).

Tests: `tests/test_crowd_sparse.py` (17) client-side, including that the writer
sends the residual on the second call rather than the same entries again, and that
`mtp_projs.` / `solo_heads.` never appear in an index; `tests/test_sparse_delta.py`
(9) server-side, including that a reader using `model_size * 2` really would disagree
with the tail offset (so the test can fail).

## Is the delta low-rank? A measurement (arXiv:2410.17881)

AdaRankGrad claims layer-gradient rank decays toward one as training proceeds and
transports low-rank gradient factors instead of dense updates. That is a
*transport* claim we can test directly: if a round's delta is near rank one, a
low-rank codec beats top-k sparsity outright (no index bytes, and the error is
spread over every coordinate instead of zeroing 98% of them).

`experiments/v2/rank_probe.py` builds the 39.9 M preset (`D=512 L=12 H=8`), runs
30 MOSAIC rounds at the tuned configuration, and compares the last round's merged
delta against a backprop delta of five AdamW steps. Energy captured by the leading
`r` singular values, weighted by Frobenius mass over the 49 matrices:

| delta | rank 1 | rank 4 | rank 8 | rank 16 |
|---|---|---|---|---|
| MOSAIC round | 0.346 | 0.707 | 0.837 | 0.920 |
| backprop, 5 steps | 0.387 | 0.709 | 0.805 | 0.861 |

**The rank-1 asymptote does not hold here** — a third of the energy is in the top
singular value, not 90%+ — so AdaRankGrad's premise is about gradients accumulated
over far more steps than one round, not about our uploads.

But the *transport* choice it implies still wins, and the margin is large. At
**equal bytes** (the sparse-2% budget, 4.8 MB for this model's matrices):

| codec | bytes | relative Frobenius error |
|---|---|---|
| sparse, top 2% of entries | 4.8 MB | **0.936** |
| low-rank, `r ≈ 23` | 4.8 MB | **0.225** |

Sparse top-k is close to useless on these deltas for a structural reason: AdamW's
per-coordinate normalisation makes a round's delta **dense** (every coordinate
moves by roughly `lr`, so top-2% retains only ~12% of the squared norm). That is
also why the error-feedback numbers above needed a `rho` sweep to look respectable
at all. Low-rank is 4x better at the same bytes on this evidence.

Two honest caveats, both worth stating before anything is built on it:
- this is a 39.9 M model at 30 rounds; rank structure at 1 B and at 1 000 rounds is
  unmeasured, and AdaRankGrad's own claim is that rank *falls* with training, which
  our 30-round snapshot cannot see either way;
- error per round is not the metric that matters — error *after error feedback
  across rounds* is, and a low-rank codec's residual is itself low-rank, so the
  combination has to be measured rather than assumed.

## What was not done

- No end-to-end training comparison of sparse vs dense uploads. The payload and
  fidelity are measured; the *loss* effect of receiving 56% of the displacement is
  not. Doing it honestly needs the harness's aggregation path to speak the codec,
  which is a bigger change than this PR.
- MTP's loss benefit is unmeasured. It is implemented, wired, and its semantics are
  pinned by tests; the claim that multi-token prediction helps a swarm is a
  hypothesis from the literature, not a result from this repo.
