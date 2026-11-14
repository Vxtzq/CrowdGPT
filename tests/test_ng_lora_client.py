"""Opt-in LoRA track and bounded engram delta selection.

Two layers, both standard-library ``unittest`` so ``python -m unittest
discover -s tests`` runs them in upstream CI:

* ``StubClientTests`` reuse the torch/numpy/requests stubs from
  ``test_coordinator_rejected``. They cover the opt-in flag, the train-mode
  headers/metadata, the engram update-mass arithmetic, and the LoRA payload
  wire format.
* ``TorchClientTests`` exercise the real torch wiring (freeze, adapters,
  payload offsets, forward + streamed functional_call, optimizer split). They
  skip when torch is not installed, which is the case in CI.

The wire format is ``gzip(<u32 header len><json header><NGL1 payload>
[<u32 engram rows><u32 indices><u16 values>])``. These tests prove the client
wiring and the selection bound on synthetic tensors; a real-model held-out loss
comparison is a separate unknown and is not claimed here.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import logging
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
_LORA_ENV = ("CROWDGPT_LORA", "CROWDGPT_LORA_RANK", "CROWDGPT_LORA_ALPHA")

try:  # CI installs nothing beyond the standard library.
    import torch
    _TORCH_ERROR = None
except Exception as _exc:  # pragma: no cover - the skip is the CI path
    torch = None
    _TORCH_ERROR = _exc

_TEMP_DIRS = []


def tearDownModule():
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    _TEMP_DIRS.clear()


def _load_stub_client(**env):
    """Reuse the coordinator-test loader, with the LoRA env vars made explicit."""
    try:
        import test_coordinator_rejected as base
    except ImportError:
        from tests import test_coordinator_rejected as base
    saved = {k: os.environ.get(k) for k in _LORA_ENV}
    for k in _LORA_ENV:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        mod, tmp = base._load_client()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    shutil.rmtree(tmp, ignore_errors=True)
    return mod


def _load_real_client(rank=4):
    """Import client.py with real torch, in a throwaway directory and env."""
    tmp = tempfile.mkdtemp(prefix="ng-lora-real-")
    _TEMP_DIRS.append(tmp)
    shutil.copy2(ROOT / "client.py", Path(tmp) / "client.py")
    saved_stdio, saved_cwd = (sys.stdout, sys.stderr), os.getcwd()
    saved_env = {k: os.environ.get(k) for k in _LORA_ENV}
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        os.environ["CROWDGPT_LORA"] = "1"
        os.environ["CROWDGPT_LORA_RANK"] = str(rank)
        os.environ.pop("CROWDGPT_LORA_ALPHA", None)
        # StringIO has no .buffer, so client.py leaves the streams alone and the
        # real stdout/stderr buffers are never wrapped or closed.
        sys.stdout = sys.stderr = io.StringIO()
        os.chdir(tmp)
        spec = importlib.util.spec_from_file_location("ng_lora_real_client", Path(tmp) / "client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(saved_cwd)
        sys.stdout, sys.stderr = saved_stdio
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


# ── fake torch-free objects for the payload writer ─────────────────────────────

class _Arr:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self._n = 1
        for d in self.shape:
            self._n *= int(d)

    def tobytes(self):
        return bytes(self._n * 2)  # float16 payload


class _Param:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self._n = 1
        for d in self.shape:
            self._n *= int(d)

    def numel(self):
        return self._n

    def detach(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def numpy(self):
        return _Arr(self.shape)


class _Adapter:
    def __init__(self, rows, cols, rank):
        self.A = _Param((rank, cols))
        self.B = _Param((rows, rank))


class _Owner:
    def __init__(self, **adapters):
        self.__dict__.update(adapters)


class _Block:
    def __init__(self):
        self.attn = _Owner()
        self.mlp = _Owner()


class _FakeModel:
    def __init__(self, blocks, params):
        self.blocks = blocks
        self._params = params
        self.is_lora = True
        self.lora_rank = 4

    def named_parameters(self):
        return iter(self._params)


class _Rows:
    """Stands in for a numpy row array: len() is the row count."""

    def __init__(self, count, payload):
        self._count = int(count)
        self._payload = payload

    def __len__(self):
        return self._count

    def tobytes(self):
        return self._payload


def _parse_headers(payload):
    header = struct.Struct("<HIIIH")
    count = struct.unpack_from("<I", payload, 4)[0]
    items, pos = [], 8
    for _ in range(count):
        nlen, offset, rows, cols, rank = header.unpack_from(payload, pos)
        pos += header.size
        name = payload[pos:pos + nlen].decode()
        pos += nlen
        a_bytes, b_bytes = rows * rank * 2, rank * cols * 2
        pos += a_bytes + b_bytes
        items.append({"name": name, "offset": offset, "rows": rows, "cols": cols, "rank": rank})
    return items, pos


class StubClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = _load_stub_client()
        cls.lora = _load_stub_client(CROWDGPT_LORA="true", CROWDGPT_LORA_RANK="7", CROWDGPT_LORA_ALPHA="64")
        cls.bad = _load_stub_client(CROWDGPT_LORA="1", CROWDGPT_LORA_RANK="oops", CROWDGPT_LORA_ALPHA="oops")

    # ── opt-in flag ────────────────────────────────────────────────────────────

    def test_lora_is_off_by_default(self):
        self.assertFalse(self.default.LORA_ACTIVE)
        self.assertEqual(self.default.LORA_RANK, 16)
        self.assertEqual(self.default.LORA_ALPHA, 32.0)

    def test_lora_opt_in_parses_env(self):
        self.assertTrue(self.lora.LORA_ACTIVE)
        self.assertEqual(self.lora.LORA_RANK, 7)
        self.assertEqual(self.lora.LORA_ALPHA, 64.0)

    def test_invalid_rank_and_alpha_fall_back(self):
        self.assertTrue(self.bad.LORA_ACTIVE)
        self.assertEqual(self.bad.LORA_RANK, 16)
        self.assertEqual(self.bad.LORA_ALPHA, 32.0)

    # ── headers and task metadata ─────────────────────────────────────────────

    def test_headers_declare_dense_mode_by_default(self):
        headers = self.default._hardware_headers()
        self.assertEqual(headers["X-Client-Train-Mode"], "dense")
        self.assertEqual(headers["X-Client-Lora-Rank"], "0")

    def test_headers_declare_lora_mode_and_rank(self):
        headers = self.lora._hardware_headers()
        self.assertEqual(headers["X-Client-Train-Mode"], "lora")
        self.assertEqual(headers["X-Client-Lora-Rank"], "7")

    def test_train_metadata_dense(self):
        self.assertEqual(self.default._train_mode_meta(True), {
            "loraRank": 0, "isLoRA": False, "trainMode": "dense",
            "trainedParams": ["base", "engram"],
        })
        self.assertEqual(self.default._train_mode_meta(False)["trainedParams"], ["base"])

    def test_train_metadata_lora(self):
        self.assertEqual(self.lora._train_mode_meta(True), {
            "loraRank": 7, "isLoRA": True, "trainMode": "lora",
            "trainedParams": ["adapters", "engram"],
        })
        self.assertEqual(self.lora._train_mode_meta(False)["trainedParams"], ["adapters"])

    # ── bounded engram selection arithmetic (pure Python, runs in CI) ─────────

    def test_mass_selection_drops_far_fewer_changed_rows_than_a_count_cap(self):
        masses = [10.0] * 100 + [1e-6] * 1900
        selected, covered, omitted = self.default._select_rows_by_mass(masses, max_frac=0.20, mass_frac=0.995)
        self.assertEqual(len(selected), 100)
        self.assertGreater(covered, 0.9999)
        self.assertLess(omitted, 1e-4)

    def test_mass_selection_cap_is_an_explicit_bound(self):
        selected, covered, omitted = self.default._select_rows_by_mass([1.0] * 2000, max_frac=0.05, mass_frac=0.995)
        self.assertEqual(len(selected), 100)
        self.assertAlmostEqual(covered, 0.05, places=9)
        self.assertAlmostEqual(omitted, 0.95, places=9)

    def test_mass_selection_zero_mass_produces_no_rows(self):
        self.assertEqual(self.default._select_rows_by_mass([]), ([], 0.0, 0.0))
        self.assertEqual(self.default._select_rows_by_mass([0.0] * 10), ([], 0.0, 0.0))
        self.assertEqual(self.default._select_rows_by_mass([1e-12] * 10), ([], 0.0, 0.0))

    def test_mass_selection_ignores_tiny_rows_below_threshold(self):
        selected, covered, _ = self.default._select_rows_by_mass([0.0, 1.0], max_frac=0.2, mass_frac=0.5)
        self.assertEqual(selected, [1])
        self.assertAlmostEqual(covered, 1.0, places=9)

    def test_mass_selection_clamps_fractions(self):
        # max_frac=0 still keeps the highest-mass row (the budget floor is 1),
        # and mass_frac is clamped into [0, 1].
        selected, _, _ = self.default._select_rows_by_mass([1.0, 2.0, 3.0], max_frac=0.0, mass_frac=2.0)
        self.assertEqual(selected, [2])

    # ── engram env contract (torch-free capture) ──────────────────────────────

    class _FakeTable:
        shape = (4, 2)

        def __getitem__(self, _key):
            return self

        def __sub__(self, _other):
            return self

        def abs(self):
            return self

        def __abs__(self):
            return self

        def float(self):
            return self

        def sum(self, dim=None):
            return 1.0

    def _capture_fractions(self, env, frac=None):
        calls = []

        def fake_select(_row_mass, max_frac=0.20, mass_frac=0.995):
            calls.append((max_frac, mass_frac))
            return [], 0.0, 0.0

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(self.default, "_engram_select_rows", fake_select):
            for key in ("CROWDGPT_ENGRAM_DELTA_MAX_FRAC", "CROWDGPT_ENGRAM_DELTA_MASS"):
                if key not in env:
                    os.environ.pop(key, None)
            self.default._engram_sparse_delta(self._FakeTable(), self._FakeTable(), frac=frac)
        return calls

    def test_engram_delta_defaults(self):
        self.assertEqual(self._capture_fractions({}), [(0.20, 0.995)])

    def test_engram_delta_reads_env(self):
        calls = self._capture_fractions({
            "CROWDGPT_ENGRAM_DELTA_MAX_FRAC": "0.15",
            "CROWDGPT_ENGRAM_DELTA_MASS": "0.9",
        })
        self.assertEqual(calls, [(0.15, 0.9)])

    def test_engram_delta_env_fallback_and_frac_override(self):
        calls = self._capture_fractions({
            "CROWDGPT_ENGRAM_DELTA_MAX_FRAC": "oops",
            "CROWDGPT_ENGRAM_DELTA_MASS": "oops",
        })
        self.assertEqual(calls, [(0.20, 0.995)])
        calls = self._capture_fractions({"CROWDGPT_ENGRAM_DELTA_MAX_FRAC": "0.5"}, frac=0.10)
        self.assertEqual(calls, [(0.10, 0.995)])

    # ── payload wire format (torch-free) ──────────────────────────────────────

    def test_payload_offsets_skip_lora_and_engram(self):
        block = _Block()
        block.attn.lora_wq = _Adapter(8, 8, 4)
        block.mlp.lora_w3 = _Adapter(8, 16, 4)
        params = [("wte.weight", 300), ("engram.table.weight", 500),
                  ("blocks.0.attn.wq.weight", 64), ("blocks.0.attn.wk.weight", 64),
                  ("blocks.0.attn.lora_wq.A", 32), ("blocks.0.attn.lora_wq.B", 32),
                  ("blocks.0.mlp.w3.weight", 128)]
        fake_params = [(name, _Param((numel,))) for name, numel in params]
        model = _FakeModel([block], fake_params)
        payload = self.default._lora_payload(model)

        self.assertTrue(payload.startswith(b"NGL1"))
        items, end = _parse_headers(payload)
        self.assertEqual(end, len(payload))
        self.assertEqual([item["name"] for item in items],
                         ["blocks.0.attn.wq.weight", "blocks.0.mlp.w3.weight"])
        # wte 300 -> wq at 300; wq 64 -> wk 64 -> w3 at 300 + 64 + 64 = 428.
        self.assertEqual([item["offset"] for item in items], [300, 428])
        self.assertEqual((items[0]["rows"], items[0]["cols"], items[0]["rank"]), (8, 8, 4))
        self.assertEqual((items[1]["rows"], items[1]["cols"], items[1]["rank"]), (8, 16, 4))

    def test_payload_without_adapters_is_an_empty_container(self):
        payload = self.default._lora_payload(_FakeModel([_Block()], []))
        self.assertEqual(payload, b"NGL1" + struct.pack("<I", 0))

    def test_lora_delta_file_appends_engram_tail(self):
        header = b'{"isLoRA":true,"loraRank":4}'
        payload = b"NGL1" + struct.pack("<I", 0)
        indices = _Rows(2, struct.pack("<II", 3, 7))
        values = _Rows(2, bytes(32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "delta.gz"
            size = self.default._write_lora_delta_file(
                path, struct.pack("<I", len(header)) + header, payload, (indices, values))
            self.assertEqual(size, path.stat().st_size)
            with gzip.open(path, "rb") as f:
                self.assertEqual(struct.unpack("<I", f.read(4))[0], len(header))
                self.assertEqual(f.read(len(header)), header)
                self.assertEqual(f.read(len(payload)), payload)
                self.assertEqual(struct.unpack("<I", f.read(4))[0], 2)
                self.assertEqual(f.read(8), indices.tobytes())
                self.assertEqual(f.read(32), values.tobytes())
                self.assertEqual(f.read(), b"")

    def test_lora_delta_file_without_engram_has_no_tail(self):
        header = b"{}"
        payload = b"NGL1" + struct.pack("<I", 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "delta.gz"
            self.default._write_lora_delta_file(path, header, payload)
            with gzip.open(path, "rb") as f:
                self.assertEqual(f.read(), header + payload)


@unittest.skipUnless(torch is not None, f"torch not installed ({_TORCH_ERROR})")
class TorchClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _load_real_client(rank=4)

    def _toy(self, with_engram=False):
        client = self.client
        nn = torch.nn

        class Lin(nn.Module):
            def __init__(self, out_features, in_features):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = Lin(8, 8); self.wk = Lin(8, 8); self.wv = Lin(8, 8); self.wo = Lin(8, 8)

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = Lin(16, 8); self.w2 = Lin(16, 8); self.w3 = Lin(8, 16)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn(); self.mlp = Mlp()

        roomy = type("Toy", (nn.Module,), {})
        model = roomy()
        model.blocks = nn.ModuleList([Block()])
        if with_engram:
            model.engram = nn.Embedding(4, 8, sparse=True)
        return model

    def test_apply_lora_freezes_base_and_trains_only_adapters(self):
        model = self._toy()
        self.client._apply_lora(model, 4)
        self.assertTrue(model.is_lora)
        self.assertEqual((model.lora_rank, model.lora_alpha), (4, 32.0))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 14)  # 7 adapters x 2 factors
        self.assertTrue(all(".lora_" in n for n in trainable))
        base = [p for n, p in model.named_parameters() if ".lora_" not in n]
        self.assertTrue(base and not any(p.requires_grad for p in base))

    def test_payload_offsets_match_base_weight_layout(self):
        model = self._toy()
        base_names = [n for n, _ in model.named_parameters()]
        numel = {n: p.numel() for n, p in model.named_parameters()}
        self.client._apply_lora(model, 4)
        payload = self.client._lora_payload(model)
        items, end = _parse_headers(payload)
        self.assertEqual(end, len(payload))
        self.assertEqual(len(items), 7)
        expected, running = [], 0
        for name in base_names:
            expected.append((name, running))
            running += numel[name]
        # named_parameters() interleaves adapters, so compare against the base
        # weights only, in their registered order.
        self.assertEqual([(item["name"], item["offset"]) for item in items],
                         [(name, off) for name, off in expected if "lora_" not in name])
        self.assertTrue(all(item["rank"] == 4 for item in items))

    def test_engram_mass_selection_meets_the_stated_bound(self):
        n = 2000
        mass = torch.zeros(n)
        mass[:100] = 10.0
        mass[100:200] = 1e-6
        selected, covered, omitted = self.client._engram_select_rows(mass, max_frac=0.20, mass_frac=0.995)
        self.assertEqual(int(selected.numel()), 100)
        self.assertGreater(covered, 0.9999)
        self.assertLess(omitted, 1e-4)

        q = torch.zeros(n); q[:100] = 0.95 / 100; q[100:200] = 0.05 / 100
        full = mass.clone()
        trunc = torch.zeros(n); trunc[selected] = full[selected]
        err = float((q * (full - trunc).abs()).sum().item())
        self.assertLessEqual(err, omitted * float(full.sum()) + 1e-9)

    def test_engram_selection_cap_and_empty(self):
        selected, covered, omitted = self.client._engram_select_rows(torch.ones(2000), max_frac=0.05)
        self.assertEqual(int(selected.numel()), 100)
        self.assertAlmostEqual(covered, 0.05, places=6)
        self.assertAlmostEqual(omitted, 0.95, places=6)
        empty, covered, omitted = self.client._engram_select_rows(torch.zeros(10))
        self.assertEqual(int(empty.numel()), 0)
        self.assertEqual((covered, omitted), (0.0, 0.0))

    def test_sparse_delta_wire_shape(self):
        import numpy as np

        n, dim = 1000, 8
        base = torch.zeros(n, dim)
        table = base.clone()
        table[:50] = 1.0
        table[50:500] = 1e-4
        ai, sv = self.client._engram_sparse_delta(table, base)
        self.assertEqual(len(ai), 50)
        self.assertEqual(tuple(sv.shape), (50, dim))
        self.assertEqual(ai.dtype, np.uint32)
        self.assertEqual(sv.dtype, np.uint16)
        self.assertLess(int(ai.max()), n)

    def test_optimizers_follow_requires_grad(self):
        model = self._toy(with_engram=True)
        self.client._apply_lora(model, 4)
        with mock.patch.object(self.client, "train_backend", "CPU"):
            base_opt, engram_opt = self.client._build_optimizers(model, lr=1e-4)
        self.assertEqual(sum(len(g["params"]) for g in base_opt.opts[0].param_groups), 14)
        self.assertEqual(engram_opt.opts, [])  # engram frozen by _apply_lora
        for p in model.engram.parameters():
            p.requires_grad_(True)
        with mock.patch.object(self.client, "train_backend", "CPU"):
            _, engram_opt = self.client._build_optimizers(model, lr=1e-4)
        self.assertIsInstance(engram_opt, torch.optim.SparseAdam)

    def test_attention_and_mlp_add_adapters_only_when_present(self):
        client = self.client
        with mock.patch.multiple(client, DIM=8, N_HEADS=2, N_KV_HEADS=1, HEAD_DIM=4, MLP_HIDDEN=16):
            block = client.Block()
            model = type("Wrap", (torch.nn.Module,), {})()
            model.blocks = torch.nn.ModuleList([block])
            client._apply_lora(model, 4)
            x = torch.randn(2, 5, 8)
            cos = torch.randn(1, 1, 5, 4); sin = torch.randn(1, 1, 5, 4)
            base_out = block(x, cos, sin)
            # B is zero-initialised, so the adapter contributes exactly nothing.
            self.assertTrue(torch.allclose(block(x, cos, sin), base_out))
            with torch.no_grad():
                for owner, name in (("attn", "wq"), ("attn", "wk"), ("attn", "wv"), ("attn", "wo"),
                                    ("mlp", "w1"), ("mlp", "w2"), ("mlp", "w3")):
                    getattr(getattr(block, owner), f"lora_{name}").B.fill_(0.1)
            self.assertFalse(torch.allclose(block(x, cos, sin), base_out))

            # Streamed (functional_call) path must accept the extra lora_* names
            # and send gradients only to the adapters.
            out = client._streamed_block(block, torch.device("cpu"), None, x, cos, sin, True)
            self.assertEqual(tuple(out.shape), tuple(base_out.shape))
            out.sum().backward()
            lora_grads = [n for n, p in block.named_parameters() if ".lora_" in n and p.grad is not None]
            self.assertEqual(len(lora_grads), 14)
            self.assertFalse(any(p.grad is not None for n, p in block.named_parameters() if ".lora_" not in n))


if __name__ == "__main__":
    unittest.main()
