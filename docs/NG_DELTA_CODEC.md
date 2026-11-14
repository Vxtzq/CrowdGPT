# Opt-in 1-bit delta codec (NGD1 / NGD2)

`CROWDGPT_DELTA_CODEC=ngd1` (default `off`) makes the client upload a framed 1-bit
container instead of the usual gzip-compressed bf16 delta. It is opt-in because it only
works against a coordinator that ships the matching decoder: `load_delta_from_disk` in
the current server reads the dense bf16 stream and has no `deltaCodec` branch, so an
`ngd1` upload is unreadable there. Leave the flag unset unless you are running a server
that decodes it.

## When it is used

The round loop picks the codec just before writing the delta:

```python
_codec = "ngd1" if (DELTA_CODEC == "ngd1" and not has_engram) else "off"
```

- A delta with engram tail rows always uses the dense path; the container has no engram
  section.
- The choice is written to the in-file metadata and to the `meta` dict sent with the
  upload as `"deltaCodec"`.
- Everything else about the upload is unchanged: the direct/chunked split, chunk
  retries, `_UPLOAD_ABORT`, the timing hand-off, and the 401 / 403 / 426 handling run on
  the same path with the same code.

## Wire format

Each frame covers at most 4,000,000 values (the writer's `piece`):

```
frame = zlib( "NGD1" | u64 count | u64 1024
             | fp16 scale per 1024-value block
             | sign bits, little-endian within each byte )

file  = <u32 metadata length><metadata json>
      | "NGD2" | u64 total_values | u64 frame_count
      | ( u32 frame_length | frame ) *
```

A set sign bit means `value >= 0`, so the decoder rebuilds `+scale` or `-scale` for that
value. The scale of a block is the mean absolute value over the block's **real** length:
the final block of a frame is zero-padded to 1024 for the bit stream, and dividing by
1024 instead of its real length shrinks that block's scale (fixed in `37d234c`).

## Measured size

Synthetic fixture, not a captured production delta: 4,000,000 float32 values drawn as
`np.random.default_rng(7).standard_normal(4_000_000) * 0.01`, encoded with
`client._ngd1_compress`.

| measure | value |
|---|---|
| dense float32 bytes | 16,000,000 |
| compressed bytes | 505,827 |
| ratio | 31.6x |
| bytes per value | 0.12646 |
| mean absolute error | 0.004827 |
| mean absolute value | 0.007979 |
| projected for 657,787,392 base params | 83.2 MB |

The projection is arithmetic from the measured bytes-per-value, not a captured delta.
Reconstruction keeps one sign and one block scale per 1024 values, so this is a bandwidth
mechanism, not a lossless path: expect roughly 60% relative error on small values and use
it only where the coordinator already accepts lossy dense deltas.

## Tests

`tests/test_ng_codec.py` decodes the client's bytes with a reference decoder written from
the layout above:

- the frame layout, header, block scales and bit order run in the standard-library-only
  CI job, through the stubbed loader shared with `tests/test_coordinator_rejected.py`;
- the numeric encoder and the `NGD2` container need `torch` + `numpy` and are skipped
  when the training stack is not installed.

## Caveats

- No server decoder is vendored in this repository. The flag exists so the client side can
  be tested and shipped ahead of the coordinator change.
- The direct-upload path still sends `Content-Encoding: gzip` for every delta, including
  an `NGD2` body that is not gzip. A proxy that honours that header cannot read the body;
  the coordinator has to select the reader from the metadata `deltaCodec` field.
- A production delta fixture is still unmeasured; the numbers above are synthetic.
