# Opt-in 1-bit delta codec (NGD1 / NGD2, ngd1r)

`CROWDGPT_DELTA_CODEC` selects how the client writes the base-weight delta. It accepts:

| value | container | bits per value | use |
|---|---|---|---|
| `off` (default) | gzip-compressed dense bf16 | 16 | everything, including engram tails |
| `ngd1` | `NGD2` / `NGD1` frames | 1 | smallest uploads, ~60% relative error on small values |
| `ngd1r` | `NGD3` / `NGD1R` frames | 2 | same container family with a residual plane, ~1.8× less error for ~2× the bytes |

It is opt-in because it only works against a coordinator that ships the matching
decoder (`orchestrator/compression.py`). A coordinator without it reads the dense bf16
stream and has no `deltaCodec` branch, so a coded upload is unreadable there. Leave the
flag unset unless you are running a server that decodes it.

## When it is used

The round loop picks the codec just before writing the delta:

```python
_codec = DELTA_CODEC if (DELTA_CODEC in DELTA_CODECS and not has_engram) else "off"
...
if _codec in DELTA_CODECS:
    residual = DELTA_CODEC_RESIDUAL.get(_codec, False)
    delta_size = _write_delta_file_ngd1(delta_path, struct.pack('<I', len(pl)) + pl, model, iw, residual=residual)
```

- A delta with engram tail rows always uses the dense path; the containers have no
  engram section.
- The chosen name is written to the in-file metadata and to the `meta` dict sent with
  the upload as `"deltaCodec"`, so the coordinator picks its reader from that field.
- Everything else about the upload is unchanged: the direct/chunked split, chunk
  retries, `_UPLOAD_ABORT`, the timing hand-off, and the 401 / 403 / 426 handling run on
  the same path with the same code. A coded body is **not** gzip, so the direct path
  omits `Content-Encoding: gzip` for it.

## Wire format

Each frame covers at most 4,000,000 values (the writer's `piece`).

Plain frame, `ngd1`:

```
frame = zlib( "NGD1" | u64 count | u64 1024
             | fp16 scale per 1024-value block
             | sign bits, little-endian within each byte )
```

Residual frame, `ngd1r` — the same header and body prefix, then a second plane:

```
frame = zlib( "NGD1R" | u64 count | u64 1024
             | fp16 scale per 1024-value block        (s1 = mean |x|)
             | fp16 residual scale per block          (s2 = mean |x - x1|)
             | residual sign bits
             | sign bits )
```

A set sign bit means `value >= 0`, so the decoder rebuilds `+scale` or `-scale` for that
value. The scale of a block is the mean absolute value over the block's **real** length:
the final block of a frame is zero-padded to 1024 for the bit stream, and dividing by
1024 instead of its real length shrinks that block's scale (fixed in `37d234c`).

The residual plane is built from `x1 = s1 * sign(b1)` and `x2 = s2 * sign(b2)` with
`x_hat = x1 + x2`. Because zero padding reconstructs to `±s1` rather than to zero, the
padding is cleared before `s2` is derived; otherwise the last block's residual scale is
inflated by `1024 / real_length`.

Both frame kinds share one container layout, with a different magic so the reader can
reject a mismatch:

```
file  = <u32 metadata length><metadata json>
      | "NGD2" | u64 total_values | u64 frame_count      (ngd1)
      | ( u32 frame_length | frame ) *
```

`ngd1r` writes `"NGD3"` in the same position. A decoder sniffs the magic and accepts
either, but a coordinator that has negotiated `deltaCodec` checks that the container
matches, and rejects a payload whose magic contradicts the declared codec.

## Measured size

Synthetic fixture, not a captured production delta: 4,000,000 float32 values drawn as
`np.random.default_rng(7).standard_normal(4_000_000) * 0.01`, encoded with
`client._ngd1_compress` (`residual=True` for the second row).

| measure | `ngd1` | `ngd1r` |
|---|---|---|
| dense float32 bytes | 16,000,000 | 16,000,000 |
| compressed bytes | 505,827 | 1,011,172 |
| ratio | 31.6x | 15.8x |
| bytes per value | 0.12646 | 0.25279 |
| mean absolute error | 0.004827 | 0.002682 |
| mean absolute value | 0.007979 | 0.007979 |
| projected for 657,787,392 base params | 83.2 MB | 166.4 MB |

The projection is arithmetic from the measured bytes-per-value, not a captured delta.
Reconstruction keeps one sign (and optionally one residual sign) plus block scales per
1024 values, so this is a bandwidth mechanism, not a lossless path: expect roughly 60%
relative error on small values with `ngd1` and roughly 36% with `ngd1r`, and use it only
where the coordinator already accepts lossy dense deltas.

## Tests

`tests/test_ng_codec.py` and `tests/test_ng_codec_residual.py` decode the client's bytes
with reference decoders written from the layouts above:

- the frame layout, header, block scales and bit order — including the `NGD1R` plane
  order and the `NGD3` container — run in the standard-library-only CI job, through the
  stubbed loader shared with `tests/test_coordinator_rejected.py`;
- the numeric encoders and the container round-trips need `torch` + `numpy` and are
  skipped when the training stack is not installed.

The residual tests assert only that the mean error falls (`< plain / 1.4`), that the
payload grows by 1.7–2.3×, and that residual decodes beat plain decodes. They never
assert an element-wise bound, because the second plane is a variance reduction, not an
error bound: an element whose true residual is smaller than `s2` overshoots, so an
individual value can land further from its target even though the mean error falls.

## Caveats

- The coordinator must ship the decoder. Without it the upload is unreadable, which is
  why the default stays `off`.
- `ngd1r` is not free: it doubles the payload for a ~1.8× reduction in mean error. Prefer
  `ngd1` unless the coordinator's loss ratchet is actually rejecting rounds on codec
  error.
- The fixture numbers above are synthetic; the real-model numbers are below.

## Measured on the real model (658M base, gfx1150)

Reference checkpoint `Vxtzq/Crowd-v1`, `CROWDGPT_ENGRAM=1`, three rounds of 20
steps at `bs 4`/`seq 512` (122,880 tokens per arm), same batch plan per arm. The
coded arm wrote with the real `_write_delta_file_ngd1` and decoded with the real
server `iter_frames`, so both sides of the wire format are the shipped code.

Dense bf16 per round: 1,315,574,784 B (1,315,574,784 / 325 frames).

| codec | round | bytes | ratio | MAE | relL2 | cosine | enc+dec |
|---|---|---|---|---|---|---|---|
| `ngd1` | 1 | 56,746,158 | 23.18x | 2.3751e-04 | 0.5612 | 0.6878 | 4.47 s |
| `ngd1` | 2 | 55,689,373 | 23.62x | 2.4991e-04 | 0.5377 | 0.7124 | 4.08 s |
| `ngd1` | 3 | 56,561,216 | 23.26x | 2.3138e-04 | 0.5666 | 0.6830 | 3.46 s |
| `ngd1r` | 1 | 118,813,924 | 11.07x | 1.3193e-04 | 0.3233 | 0.8707 | 13.82 s |
| `ngd1r` | 2 | 117,078,758 | 11.24x | 1.3655e-04 | 0.3032 | 0.8930 | 10.50 s |
| `ngd1r` | 3 | 118,373,954 | 11.11x | 1.2951e-04 | 0.3278 | 0.8614 | 10.35 s |

So on a real delta the residual plane buys **~1.8x lower mean absolute error for
~2x the bytes**, exactly as the synthetic fixture predicted, and the ratio is
about **23x** (`ngd1`) or **11x** (`ngd1r`) on this checkpoint rather than the
31.6x / 15.8x of the fixture.

Held-out loss (`ng-eval-v1`, 8 windows, `seed 0`) against the same dense arm:
`ngd1` **-0.180 nats**, `ngd1r` **-0.077 nats**, both negative (the coded round
handoff scored marginally better). **Neither is evidence that the codec is
harmless at the level the loss gate cares about:** each confidence interval is
about +-0.16 nats, eight times the 0.02-nat gate, so 8 windows cannot resolve a
0.02 difference in either direction. Read these as "no detectable codec damage"
and nothing stronger.

Two honest limits from these runs:

- Round-to-round delta size does not shrink monotonically, because each round
  here restarts from the freshly decoded weights rather than accumulating a
  fixed offset.
- `ngd1r` costs about 10 s of encode+decode per round versus 3-4 s for `ngd1`.
  On a 500-step round that is noise; on a tiny round it is not.
