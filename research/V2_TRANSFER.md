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

## What was not done

- No end-to-end training comparison of sparse vs dense uploads. The payload and
  fidelity are measured; the *loss* effect of receiving 56% of the displacement is
  not. Doing it honestly needs the harness's aggregation path to speak the codec,
  which is a bigger change than this PR.
- MTP's loss benefit is unmeasured. It is implemented, wired, and its semantics are
  pinned by tests; the claim that multi-token prediction helps a swarm is a
  hypothesis from the literature, not a result from this repo.
