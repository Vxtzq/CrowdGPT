"""Tests for the opt-in two-plane ("ngd1r") delta codec.

``ngd1r`` is the same NGD1 frame with a second, residual bit-plane per block: a plain sign plane
(``s1 * sign``) plus a residual sign plane (``s2 * sign``) added back on decode. It costs roughly
twice the bytes of ``ngd1`` and, on real deltas, about 1.8x less mean absolute error. It is a
variance reduction, not an error bound: an element whose residual is smaller than ``s2`` gets
overshot, so the *mean* error falls while individual elements can grow.

The frame-layout tests run with the standard library only, by importing client.py through the
stubbed loader shared with ``tests/test_coordinator_rejected.py`` (torch and numpy replaced by
stubs) and decoding the bytes the client writes with a reference decoder written here. The numeric
encoder and container tests need the real stack and are skipped when it is absent -- exactly like
``tests/test_ng_codec.py``.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import numpy as _np
    import torch as _torch
except Exception:  # pragma: no cover - optional stack; the stub job has no wheels
    _np = _torch = None

FRAME_MAGIC = b"NGD1"
RESIDUAL_FRAME_MAGIC = b"NGD1R"
CONTAINER_MAGIC = b"NGD2"
RESIDUAL_CONTAINER_MAGIC = b"NGD3"
BLOCK = 1024


def _decode_ngd1r(blob):
    """Reference decoder for the two-plane layout ``_ngd1_frame`` writes.

    NGD1R | count u64 | block u64 | one fp16 scale per block | one fp16 residual scale per block
    | residual sign bits | sign bits, both little-endian within a byte, blocks back to back.
    A set bit is the positive sign; each scale is the mean absolute value over the block's real
    (unpadded) length -- the residual scale over the residual left by the plain plane.
    """
    raw = zlib.decompress(blob)
    if raw[:5] != RESIDUAL_FRAME_MAGIC:
        raise ValueError(f"bad frame magic {raw[:5]!r}")
    n, block = struct.unpack_from("<QI", raw, 5)
    if block != BLOCK:
        raise ValueError(f"unexpected block size {block}")
    nb = max(1, (n + BLOCK - 1) // BLOCK)
    plane = nb * BLOCK // 8
    scales = struct.unpack_from(f"<{nb}e", raw, 17)
    residual_scales = struct.unpack_from(f"<{nb}e", raw, 17 + 2 * nb)
    residual_bits = raw[17 + 4 * nb:17 + 4 * nb + plane]
    bits = raw[17 + 4 * nb + plane:]
    if len(residual_bits) != plane or len(bits) != plane:
        raise ValueError(f"sign planes are {len(residual_bits)}/{len(bits)} bytes, expected {plane}")
    out = []
    for i in range(n):
        k = i // block
        plain = scales[k] if (bits[i >> 3] >> (i & 7)) & 1 else -scales[k]
        residual = (residual_scales[k] if (residual_bits[i >> 3] >> (i & 7)) & 1
                    else -residual_scales[k])
        out.append(plain + residual)
    return out


def _decode_ngd1(blob):
    """Reference decoder for the plain one-plane frame, so A/B assertions can decode both."""
    raw = zlib.decompress(blob)
    if raw[:4] != FRAME_MAGIC:
        raise ValueError(f"bad frame magic {raw[:4]!r}")
    n, block = struct.unpack_from("<QI", raw, 4)
    if block != BLOCK:
        raise ValueError(f"unexpected block size {block}")
    nb = max(1, (n + BLOCK - 1) // BLOCK)
    scales = struct.unpack_from(f"<{nb}e", raw, 16)
    bits = raw[16 + 2 * nb:]
    if len(bits) != nb * BLOCK // 8:
        raise ValueError(f"sign stream is {len(bits)} bytes, expected {nb * BLOCK // 8}")
    return [scales[i // BLOCK] if (bits[i >> 3] >> (i & 7)) & 1 else -scales[i // BLOCK]
            for i in range(n)]


def _decode_ngd3(blob, header_len=0):
    """Reference decoder for the two-plane container: header | magic | total u64 | frames u64."""
    off = header_len
    if blob[off:off + 4] != RESIDUAL_CONTAINER_MAGIC:
        raise ValueError(f"bad container magic {blob[off:off + 4]!r}")
    total, nframes = struct.unpack_from("<QI", blob, off + 4)
    off += 16
    values = []
    for _ in range(nframes):
        length = struct.unpack_from("<I", blob, off)[0]
        off += 4
        values.extend(_decode_ngd1r(blob[off:off + length]))
        off += length
    if off != len(blob):
        raise ValueError(f"{len(blob) - off} trailing bytes after the last frame")
    if len(values) != total:
        raise ValueError(f"decoded {len(values)} values, container declares {total}")
    return values


def _pack_signs(signs):
    """Pack one bit per value, little-endian within a byte (the layout np.packbits writes)."""
    out = bytearray((len(signs) + 7) // 8)
    for i, s in enumerate(signs):
        if s:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


class _NeverClosedBytesIO(io.BytesIO):
    def close(self): pass


def _load_real_client():
    """Import client.py from a temp copy with the real torch/numpy, or None without them."""
    tmp = tempfile.mkdtemp(prefix="crowdgpt-codec-r-")
    shutil.copy(REPO / "client.py", tmp)
    saved_stdio, saved_cwd, saved_env = (sys.stdout, sys.stderr), os.getcwd(), set(os.environ)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    missing = object()
    saved_webview = sys.modules.get("webview", missing)
    sys.modules["webview"] = None
    spec = importlib.util.spec_from_file_location("crowdgpt_codec_residual_client",
                                                  Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        fakes = [io.TextIOWrapper(_NeverClosedBytesIO(), encoding="utf-8") for _ in range(2)]
        sys.stdout, sys.stderr = fakes
        os.chdir(tmp)
        spec.loader.exec_module(mod)
    except ImportError:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    finally:
        os.chdir(saved_cwd)
        sys.stdout, sys.stderr = saved_stdio
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for k in set(os.environ) - saved_env:
            del os.environ[k]
        if saved_webview is missing:
            sys.modules.pop("webview", None)
        else:
            sys.modules["webview"] = saved_webview
    return mod, tmp


_REAL = _load_real_client() if (_np is not None and _torch is not None) else None


def _base_loader():
    """The stubbed loader other NG tests share (torch/numpy/requests are mocks)."""
    try:
        import test_coordinator_rejected as base
    except ImportError:
        from tests import test_coordinator_rejected as base
    return base


class ResidualLayoutTests(unittest.TestCase):
    """The two-plane frame must follow the documented layout (stdlib only)."""

    @classmethod
    def setUpClass(cls):
        cls.client = _base_loader().client

    def test_codec_names_are_registered(self):
        self.assertIn("ngd1r", self.client.DELTA_CODECS)
        self.assertTrue(self.client.DELTA_CODEC_RESIDUAL["ngd1r"])
        self.assertFalse(self.client.DELTA_CODEC_RESIDUAL["ngd1"])

    def test_plain_frame_keeps_the_ngd1_magic(self):
        frame = self.client._ngd1_frame(1024, struct.pack("<e", 0.5), _pack_signs([True] * 1024))
        self.assertEqual(zlib.decompress(frame)[:4], FRAME_MAGIC)

    def test_residual_frame_header_and_planes(self):
        n = 1500  # one full block plus a 476-value partial block
        nb = 2
        signs = [True] * nb * BLOCK
        residual_signs = [False] * nb * BLOCK
        frame = self.client._ngd1_frame(
            n, struct.pack(f"<{nb}e", *[0.5] * nb), _pack_signs(signs),
            struct.pack(f"<{nb}e", *[0.25] * nb), _pack_signs(residual_signs))
        raw = zlib.decompress(frame)
        self.assertEqual(raw[:5], RESIDUAL_FRAME_MAGIC)
        self.assertEqual(struct.unpack_from("<QI", raw, 5), (n, BLOCK))
        plane = nb * BLOCK // 8
        self.assertEqual(len(raw), 17 + 4 * nb + 2 * plane)
        values = _decode_ngd1r(frame)
        self.assertEqual(len(values), n)
        self.assertEqual(values, [0.25] * n)  # 0.5 * +1 plus 0.25 * -1

    def test_residual_planes_cover_whole_blocks(self):
        for n in (1, 1023, 1024, 1025):
            with self.subTest(n=n):
                nb = max(1, (n + BLOCK - 1) // BLOCK)
                frame = self.client._ngd1_frame(
                    n, struct.pack(f"<{nb}e", *[0.5] * nb), _pack_signs([True] * (nb * BLOCK)),
                    struct.pack(f"<{nb}e", *[0.25] * nb), _pack_signs([True] * (nb * BLOCK)))
                raw = zlib.decompress(frame)
                self.assertEqual(len(raw), 17 + 4 * nb + 2 * (nb * BLOCK // 8))
                self.assertEqual(_decode_ngd1r(frame), [0.75] * n)


@unittest.skipIf(_REAL is None, "torch/numpy not installed")
class ResidualEncoderTests(unittest.TestCase):
    """_ngd1_compress(residual=True) must cut the mean error at roughly twice the bytes."""

    @classmethod
    def setUpClass(cls):
        cls.client = _REAL[0]

    def _values(self, count=4096, seed=7):
        return (_np.random.default_rng(seed).standard_normal(count) * 0.01).astype(_np.float32)

    @staticmethod
    def _mae(back, values):
        return float(_np.mean(_np.abs(_np.asarray(back, dtype=_np.float32) - values)))

    def test_round_trip_returns_the_original_count(self):
        values = self._values()
        back = _decode_ngd1r(self.client._ngd1_compress(values, residual=True))
        self.assertEqual(len(back), values.size)

    def test_residual_cuts_the_mean_error(self):
        values = self._values()
        plain = self._mae(_decode_ngd1(
            self.client._ngd1_compress(values, residual=False)), values)
        residual = self._mae(_decode_ngd1r(
            self.client._ngd1_compress(values, residual=True)), values)
        self.assertLess(residual, plain / 1.4)

    def test_residual_costs_about_twice_the_bytes(self):
        values = self._values()
        plain = len(self.client._ngd1_compress(values, residual=False))
        residual = len(self.client._ngd1_compress(values, residual=True))
        self.assertGreater(residual / plain, 1.7)
        self.assertLess(residual / plain, 2.3)

    def test_partial_block_is_not_swamped_by_zero_padding(self):
        # The padded positions reconstruct to +/-scale rather than to zero, so a naive residual
        # scale is inflated by 1024/476 and the tail decodes as garbage. Regression guard.
        values = _np.concatenate([
            (_np.random.default_rng(3).standard_normal(BLOCK) * 0.01).astype(_np.float32),
            _np.full(238, -0.25, dtype=_np.float32),
            _np.full(238, -0.05, dtype=_np.float32)])
        plain = self._mae(_decode_ngd1(self.client._ngd1_compress(values, residual=False)), values)
        residual = self._mae(_decode_ngd1r(self.client._ngd1_compress(values, residual=True)), values)
        self.assertLess(residual, plain)
        tail = slice(BLOCK, None)
        plain_tail = self._mae(_decode_ngd1(
            self.client._ngd1_compress(values, residual=False))[tail], values[tail])
        residual_tail = self._mae(_decode_ngd1r(
            self.client._ngd1_compress(values, residual=True))[tail], values[tail])
        self.assertLess(residual_tail, plain_tail)

    def test_residual_beats_dense_bf16_for_size(self):
        values = self._values(1_000_000)
        blob = self.client._ngd1_compress(values, residual=True)
        self.assertLess(len(blob) * 3, values.size * 2)  # >3x smaller than bf16


@unittest.skipIf(_REAL is None, "torch/numpy not installed")
class ResidualContainerTests(unittest.TestCase):
    """_write_delta_file_ngd1(residual=True) must emit header, NGD3 count, then frames."""

    @classmethod
    def setUpClass(cls):
        cls.client = _REAL[0]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crowdgpt-codec-rout-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    class _Tiny(_torch.nn.Module if _torch is not None else object):
        def __init__(self):
            super().__init__()
            self.a = _torch.nn.Parameter(_torch.full((1500,), 0.25))
            self.b = _torch.nn.Parameter(_torch.full((100,), -0.5))
            self.engram = _torch.nn.Module()
            self.engram.table = _torch.nn.Parameter(_torch.zeros(8))

    def _write(self, name, header, model, iw, residual):
        path = os.path.join(self.tmp, name)
        size = self.client._write_delta_file_ngd1(path, header, model, iw, residual=residual)
        with open(path, "rb") as f:
            return size, f.read()

    def test_residual_container_round_trips(self):
        header = b'{"taskId": "t", "deltaCodec": "ngd1r"}'
        iw = _torch.zeros(1600, dtype=_torch.bfloat16)
        size, blob = self._write("delta-ngd3", header, self._Tiny(), iw, True)
        self.assertEqual(size, len(blob))
        self.assertTrue(blob.startswith(header))
        self.assertEqual(blob[len(header):len(header) + 4], RESIDUAL_CONTAINER_MAGIC)
        self.assertEqual(_decode_ngd3(blob, len(header)), [0.25] * 1500 + [-0.5] * 100)

    def test_plain_container_still_uses_ngd2(self):
        header = b'{"taskId": "t"}'
        iw = _torch.zeros(1600, dtype=_torch.bfloat16)
        _size, blob = self._write("delta-ngd2", header, self._Tiny(), iw, False)
        self.assertEqual(blob[len(header):len(header) + 4], CONTAINER_MAGIC)

    def test_size_mismatch_is_rejected_for_residual_too(self):
        with self.assertRaises(RuntimeError):
            self.client._write_delta_file_ngd1(
                os.path.join(self.tmp, "delta-bad"), b"", self._Tiny(),
                _torch.zeros(1601, dtype=_torch.bfloat16), residual=True)


if __name__ == "__main__":
    unittest.main()
