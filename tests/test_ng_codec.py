"""Wire-format and round-trip tests for the opt-in 1-bit delta codec (CROWDGPT_DELTA_CODEC=ngd1).

The frame layout test runs in the standard-library-only CI job: it reuses the stubbed loader from
tests/test_coordinator_rejected.py (torch and numpy replaced by stubs) and decodes the bytes the
client emits with a reference decoder written here from the documented format.

The numeric encoder and the NGD2 container need the real training stack (torch + numpy); they are
skipped when it is not installed, exactly like tests/test_ng_hivemind.py.

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


def _decode_ngd1(blob):
    """Reference decoder for the layout ``_ngd1_frame`` writes; mirrors the coordinator's decoder.

    NGD1 | count u64 | block u64 | one fp16 scale per block | sign bits, little-endian within a
    byte, blocks back to back. A set bit is the positive sign; the block scale is the mean
    absolute value over the block's real (unpadded) length.
    """
    raw = zlib.decompress(blob)
    if raw[:4] != b"NGD1":
        raise ValueError(f"bad frame magic {raw[:4]!r}")
    n, block = struct.unpack_from("<QI", raw, 4)
    if block != 1024:
        raise ValueError(f"unexpected block size {block}")
    nb = max(1, (n + 1023) // 1024)
    scales = struct.unpack_from(f"<{nb}e", raw, 16)
    bits = raw[16 + 2 * nb:]
    if len(bits) != nb * block // 8:
        raise ValueError(f"sign stream is {len(bits)} bytes, expected {nb * block // 8}")
    return [scales[i // block] if (bits[i >> 3] >> (i & 7)) & 1 else -scales[i // block]
            for i in range(n)]


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
    """Import client.py from a temp copy with the real torch/numpy, or None without them.

    Only ``webview`` is stubbed (to force the "no GUI toolkit" branch); the module's own numpy
    and torch must be the real ones, since the numeric tests call into them.
    """
    tmp = tempfile.mkdtemp(prefix="crowdgpt-codec-")
    shutil.copy(REPO / "client.py", tmp)
    saved_stdio, saved_cwd, saved_env = (sys.stdout, sys.stderr), os.getcwd(), set(os.environ)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    missing = object()
    saved_webview = sys.modules.get("webview", missing)
    sys.modules["webview"] = None
    spec = importlib.util.spec_from_file_location("crowdgpt_codec_client", Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        # client.py rewraps sys.stdout/stderr at import time; hand it throwaway streams whose
        # buffer ignores close() so the real ones are never wrapped and later closed.
        fakes = [io.TextIOWrapper(_NeverClosedBytesIO(), encoding="utf-8") for _ in range(2)]
        sys.stdout, sys.stderr = fakes
        # Import migrates $CWD/checkpoints; run from the temp dir so a real profile is untouched.
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


class FrameLayoutTests(unittest.TestCase):
    """The bytes the client writes must decode with the documented layout (stdlib only)."""

    @classmethod
    def setUpClass(cls):
        cls.client = _base_loader().client

    @unittest.skipIf("CROWDGPT_DELTA_CODEC" in os.environ,
                     "environment overrides the opt-in default")
    def test_codec_is_off_by_default(self):
        self.assertEqual(self.client.DELTA_CODEC, "off")

    def test_frame_header_and_block_scales(self):
        n = 1500  # one full 1024 block plus a 476-value partial block (padded to 2048 bits)
        signs = [True] * 1024 + [i % 3 != 0 for i in range(476)]
        signs += [True] * (2 * 1024 - len(signs))  # sign stream covers whole blocks
        frame = self.client._ngd1_frame(n, struct.pack("<2e", 0.5, 0.125), _pack_signs(signs))
        raw = zlib.decompress(frame)
        self.assertEqual(raw[:4], b"NGD1")
        self.assertEqual(struct.unpack_from("<QI", raw, 4), (n, 1024))
        values = _decode_ngd1(frame)
        self.assertEqual(len(values), n)
        self.assertEqual(values[:1024], [0.5] * 1024)
        self.assertEqual(values[1024:], [0.125 if s else -0.125 for s in signs[1024:n]])

    def test_frame_sign_stream_covers_whole_blocks(self):
        for n in (1, 1023, 1024, 1025, 2048):
            with self.subTest(n=n):
                nb = max(1, (n + 1023) // 1024)
                frame = self.client._ngd1_frame(
                    n, struct.pack(f"<{nb}e", *[0.25] * nb), _pack_signs([True] * (nb * 1024)))
                raw = zlib.decompress(frame)
                self.assertEqual(len(raw), 16 + 2 * nb + nb * 1024 // 8)
                self.assertEqual(_decode_ngd1(frame), [0.25] * n)


@unittest.skipIf(_REAL is None, "torch/numpy not installed")
class EncoderTests(unittest.TestCase):
    """_ngd1_compress must quantize to signs plus per-block mean-abs scales."""

    @classmethod
    def setUpClass(cls):
        cls.client = _REAL[0]

    def test_round_trip_preserves_signs(self):
        values = (_np.random.default_rng(2).standard_normal(4096) * 0.01).astype(_np.float32)
        back = _decode_ngd1(self.client._ngd1_compress(values))
        self.assertEqual(len(back), values.size)
        self.assertTrue(_np.all(_np.sign(back) == _np.sign(values)))
        self.assertLess(_np.mean(_np.abs(_np.array(back) - values)), _np.mean(_np.abs(values)))

    def test_partial_block_scale_uses_real_length(self):
        values = _np.full(1500, 0.5, dtype=_np.float32)
        values[1024:] = -0.25  # 476 values, not 1024: padding must not dilute the last scale
        raw = zlib.decompress(self.client._ngd1_compress(values))
        self.assertEqual(struct.unpack_from("<2e", raw, 16), (0.5, 0.25))

    def test_size_target_for_a_million_params(self):
        blob = self.client._ngd1_compress(_np.zeros(1_000_000, dtype=_np.float32))
        self.assertLess(len(blob), 200_000)


@unittest.skipIf(_REAL is None, "torch/numpy not installed")
class ContainerTests(unittest.TestCase):
    """_write_delta_file_ngd1 must emit header, NGD2 count, then length-prefixed frames."""

    @classmethod
    def setUpClass(cls):
        cls.client = _REAL[0]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="crowdgpt-codec-out-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    class _Tiny(_torch.nn.Module if _torch is not None else object):
        def __init__(self):
            super().__init__()
            self.a = _torch.nn.Parameter(_torch.full((1500,), 0.25))
            self.b = _torch.nn.Parameter(_torch.full((100,), -0.5))
            self.engram = _torch.nn.Module()
            self.engram.table = _torch.nn.Parameter(_torch.zeros(8))

    def _write(self, header, model, iw):
        path = os.path.join(self.tmp, "delta-ngd2")
        size = self.client._write_delta_file_ngd1(path, header, model, iw)
        with open(path, "rb") as f:
            return size, f.read()

    def test_container_frames_decode_to_the_whole_delta(self):
        header = b'{"taskId": "t"}'
        iw = _torch.zeros(1600, dtype=_torch.bfloat16)
        size, blob = self._write(header, self._Tiny(), iw)
        self.assertEqual(size, len(blob))
        self.assertTrue(blob.startswith(header))
        off = len(header)
        self.assertEqual(blob[off:off + 4], b"NGD2")
        total, nframes = struct.unpack_from("<QI", blob, off + 4)
        self.assertEqual((total, nframes), (1600, 2))  # engram.table is not part of the delta
        off += 16
        values = []
        for _ in range(nframes):
            length = struct.unpack_from("<I", blob, off)[0]
            off += 4
            values.extend(_decode_ngd1(blob[off:off + length]))
            off += length
        self.assertEqual(off, len(blob))
        self.assertEqual(values, [0.25] * 1500 + [-0.5] * 100)

    def test_size_mismatch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.client._write_delta_file_ngd1(
                os.path.join(self.tmp, "delta-bad"), b"", self._Tiny(),
                _torch.zeros(1601, dtype=_torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
