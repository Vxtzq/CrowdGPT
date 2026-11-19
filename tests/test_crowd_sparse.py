"""Sparse top-k deltas with error feedback — encoder, wire format, and the residual.

The claim these tests defend is not "the payload is smaller" (arithmetic) but "the
codec is unbiased": what a round drops must come back, so a coordinate that is
consistently small is eventually sent rather than starved forever.
"""
import unittest

# The repo's CI job is deliberately standard-library only (no pytest, no torch),
# and runs `python -m unittest discover -s tests`. Skip this module there instead
# of erroring on import; the tests run under pytest locally and in any env with deps.
try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
    import torch  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import crowd_sparse as S


def _delta(values):
    return torch.tensor(values, dtype=torch.float32)


class ConfigTest(unittest.TestCase):
    def test_fraction_and_floor_are_validated(self):
        with self.assertRaises(ValueError):
            S.SparseConfig(frac=0.0)
        with self.assertRaises(ValueError):
            S.SparseConfig(frac=1.5)
        with self.assertRaises(ValueError):
            S.SparseConfig(min_entries=0)

    def test_entry_count_is_a_fraction_of_coordinates_with_a_floor(self):
        cfg = S.SparseConfig(frac=0.10, min_entries=1)
        self.assertEqual(S.entry_count(1000, cfg), 100)
        # A tiny tensor still sends one entry, or error feedback would freeze it.
        self.assertEqual(S.entry_count(3, cfg), 1)
        self.assertEqual(S.entry_count(4, S.SparseConfig(frac=1.0)), 4)

    def test_env_parsing_requires_the_codec_and_survives_garbage(self):
        self.assertIsNone(S.config_from_env({}))
        self.assertIsNone(S.config_from_env({"CROWDGPT_DELTA_CODEC": "off"}))
        cfg = S.config_from_env({"CROWDGPT_DELTA_CODEC": "sparse", "CROWDGPT_SPARSE_FRAC": "0.05"})
        self.assertAlmostEqual(cfg.frac, 0.05)
        self.assertTrue(cfg.error_feedback)
        off = S.config_from_env({"CROWDGPT_DELTA_CODEC": "sparse", "CROWDGPT_SPARSE_FEEDBACK": "0"})
        self.assertFalse(off.error_feedback)
        junk = S.config_from_env({"CROWDGPT_DELTA_CODEC": "sparse", "CROWDGPT_SPARSE_FRAC": "lots"})
        self.assertAlmostEqual(junk.frac, 0.02)


class SelectTest(unittest.TestCase):
    def test_it_keeps_the_largest_magnitudes_with_their_signs(self):
        d = _delta([0.01, -9.0, 0.2, 3.0, 0.0])
        idx, val = S.select(d, S.SparseConfig(frac=0.4))  # 2 of 5
        self.assertEqual(sorted(idx.tolist()), [1, 3])
        self.assertEqual({int(i): float(v) for i, v in zip(idx, val)}[1], -9.0)

    def test_a_tensor_smaller_than_the_budget_is_dense(self):
        d = _delta([1.0, -2.0])
        idx, val = S.select(d, S.SparseConfig(frac=1.0))
        self.assertEqual(sorted(idx.tolist()), [0, 1])

    def test_indices_are_unique(self):
        d = torch.randn(500)
        idx, _ = S.select(d, S.SparseConfig(frac=0.05))
        self.assertEqual(len(idx.tolist()), len(set(idx.tolist())))


class WireTest(unittest.TestCase):
    def test_block_length_matches_the_header(self):
        block = S.encode_block([(torch.tensor([1, 5]), torch.tensor([0.5, -0.25]))], total=8)
        self.assertEqual(len(block), S.block_bytes(2))
        self.assertEqual(len(S.encode_block([], total=8)), S.HEADER_BYTES)

    def test_round_trip_preserves_indices_values_and_total(self):
        idx = torch.tensor([0, 7, 1234])
        val = torch.tensor([1.5, -0.25, 3.0])
        block = S.encode_block([(idx, val)], total=4096)
        got_idx, got_val, total = S.decode_block(block)
        self.assertEqual(total, 4096)
        self.assertEqual(got_idx.tolist(), idx.tolist())
        self.assertEqual([round(float(v), 3) for v in got_val], [1.5, -0.25, 3.0])

    def test_multiple_tensors_concatenate(self):
        block = S.encode_block([(torch.tensor([1]), torch.tensor([1.0])),
                                (torch.tensor([9]), torch.tensor([-1.0]))], total=10)
        idx, val, _ = S.decode_block(block)
        self.assertEqual(idx.tolist(), [1, 9])
        self.assertEqual([float(v) for v in val], [1.0, -1.0])

    def test_a_foreign_or_truncated_block_is_rejected(self):
        with self.assertRaises(ValueError):
            S.decode_block(b"NOPE" + b"\x00" * 8)
        with self.assertRaises(ValueError):
            S.decode_block(S.MAGIC + b"\x01")


class ErrorFeedbackTest(unittest.TestCase):
    def test_correct_adds_the_carried_residual(self):
        fb = S.ErrorFeedback()
        d = _delta([1.0, 0.0, 0.0])
        fb.keep("t", _delta([0.0, 0.5, 0.0]))
        self.assertEqual(fb.correct("t", d).tolist(), [1.0, 0.5, 0.0])
        self.assertEqual(fb.correct("other", d).tolist(), [1.0, 0.0, 0.0])

    def test_a_shape_change_discards_the_stale_residual(self):
        fb = S.ErrorFeedback()
        fb.keep("t", _delta([1.0, 2.0]))
        self.assertEqual(fb.correct("t", _delta([0.0, 0.0, 0.0])).tolist(), [0.0, 0.0, 0.0])

    def test_the_second_round_sends_what_the_first_dropped(self):
        """The property the codec depends on: dropping is deferred, not forgotten."""
        cfg = S.SparseConfig(frac=0.5, min_entries=1)
        fb = S.ErrorFeedback()
        d = _delta([4.0, 3.0, 2.0, 1.0])
        idx1, _ = S.select(fb.correct("t", d), cfg)
        kept = torch.zeros(4); kept[idx1] = d[idx1]
        fb.keep("t", d - kept)
        self.assertEqual(set(idx1.tolist()), {0, 1})
        # Next round has nothing new to say (a converged coordinate): the residual
        # alone is what gets sent, at full strength -- not attenuated, not lost.
        idx2, val2 = S.select(fb.correct("t", torch.zeros(4)), cfg)
        self.assertEqual(set(idx2.tolist()), {2, 3})
        self.assertEqual(sorted(round(float(v), 3) for v in val2), [1.0, 2.0])

    def test_a_consistently_small_coordinate_is_not_starved(self):
        """Without feedback this coordinate is never sent; with it, it arrives."""
        cfg = S.SparseConfig(frac=0.5, min_entries=1)
        fb = S.ErrorFeedback()
        d = _delta([10.0, 9.0, 0.1, 0.1])
        sent = []
        for r in range(3):
            work = fb.correct("t", d if r == 0 else torch.zeros(4))
            idx, _ = S.select(work, cfg)
            kept = torch.zeros(4); kept[idx] = work[idx]
            fb.keep("t", work - kept)          # residual = should-have-sent minus sent
            sent.append(set(idx.tolist()))
        self.assertNotIn(2, sent[0])           # drowned out on the round it moved
        self.assertIn(2, sent[1])              # back on the next round, at full strength

    def test_memory_is_reported_and_cleared(self):
        fb = S.ErrorFeedback()
        fb.keep("t", torch.ones(8))
        self.assertEqual(fb.bytes(), 8 * 2)  # bf16 by design, not fp32
        fb.clear()
        self.assertEqual(fb.bytes(), 0)


class ClientIntegrationTest(unittest.TestCase):
    """`_sparse_base_block` is what actually walks the model, so wire it to a fake."""

    def _model(self):
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.main_device = torch.device("cpu")
                self.wte = nn.Embedding(8, 4)
                self.ln_f = nn.LayerNorm(4)
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.wte.weight
                self.solo_heads = nn.ModuleDict({"0": nn.Linear(4, 4)})

        m = Tiny()
        with torch.no_grad():
            m.wte.weight.zero_()      # a zero base makes the expected delta exact
            m.ln_f.weight.zero_()      # a zero base keeps the expected delta trivially 0.5
            m.ln_f.bias.zero_()
        return m

    def _client(self):
        """Import `client` without letting it close pytest's stdio."""
        import contextlib, io

        if "client" in sys.modules:
            return sys.modules["client"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import client as C
        return C

    def test_it_encodes_only_server_visible_tensors(self):
        C = self._client()
        m = self._model()
        numel = sum(p.numel() for n, p in m.named_parameters() if C._is_server_param(n, []))
        with torch.no_grad():
            m.wte.weight.add_(0.5)                    # a real delta against a zero base
            m.solo_heads["0"].weight.add_(9.0)        # client-local: must not travel
        iw = torch.zeros(numel)                       # "the weights the server holds"
        block, stats = C._sparse_base_block(m, iw, S.SparseConfig(frac=1.0), S.ErrorFeedback())
        idx, val, total = S.decode_block(block)
        self.assertEqual(total, numel)
        self.assertEqual(stats["coords"], numel)
        self.assertLess(int(idx.max()), numel)        # every index is inside the server vector
        # wte is +0.5 and the LayerNorm/head tensors are untouched -> values are 0.5 or 0.
        self.assertTrue(all(abs(abs(float(v)) - 0.5) < 1e-2 or abs(float(v)) < 1e-3 for v in val))

    def test_the_second_call_sends_the_residual_not_the_same_entries(self):
        """Proves error feedback is actually wired into the writer, not just available."""
        C = self._client()
        m = self._model()
        numel = sum(p.numel() for n, p in m.named_parameters() if C._is_server_param(n, []))
        iw = torch.zeros(numel)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if C._is_server_param(n, []):
                    p.add_(torch.full_like(p, 0.25))
        fb = S.ErrorFeedback()
        cfg = S.SparseConfig(frac=0.3, min_entries=1)
        first, _, _ = S.decode_block(C._sparse_base_block(m, iw, cfg, fb)[0])
        second, _, _ = S.decode_block(C._sparse_base_block(m, iw, cfg, fb)[0])
        self.assertFalse(set(first.tolist()) == set(second.tolist()))
