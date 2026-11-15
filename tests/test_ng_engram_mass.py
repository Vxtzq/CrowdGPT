"""Engram delta row selection: upload the rows that moved, not a fixed 10 %.

The old code always sent ``n * 10 %`` engram rows even when a handful had
changed, which is most of the per-round upload on a converged model. Selection
now follows the update mass: the smallest set of rows covering
``CROWDGPT_ENGRAM_DELTA_MASS`` (default 0.995) of the total, capped at
``CROWDGPT_ENGRAM_DELTA_MAX_FRAC`` (default 0.10, the old ceiling).

The pure helper runs in the stdlib-only CI job through the shared stub loader;
the tensor path needs torch and is skipped without it.
"""

from __future__ import annotations

import unittest

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _stub_client():
    try:
        import test_coordinator_rejected as base
    except ImportError:
        from tests import test_coordinator_rejected as base
    return base.client


def _real_client():
    try:
        from test_ng_codec import _REAL
    except ImportError:
        from tests.test_ng_codec import _REAL
    return _REAL[0] if _REAL else None


class SelectionTests(unittest.TestCase):
    """Post-hoc helper: no torch needed."""

    @classmethod
    def setUpClass(cls):
        cls.client = _stub_client()

    def test_concentrated_mass_selects_only_the_rows_that_moved(self):
        masses = [0.0] * 1000
        for i in (3, 500, 999):
            masses[i] = 1.0
        idx, covered, omitted = self.client._select_rows_by_mass(masses, max_frac=0.10, mass_frac=0.995)
        self.assertEqual(sorted(idx), [3, 500, 999])
        self.assertAlmostEqual(covered, 1.0)
        self.assertAlmostEqual(omitted, 0.0)

    def test_cap_bounds_the_upload(self):
        masses = [1.0] * 1000
        idx, covered, omitted = self.client._select_rows_by_mass(masses, max_frac=0.01, mass_frac=0.999)
        self.assertLessEqual(len(idx), 10)
        self.assertAlmostEqual(covered, len(idx) / 1000.0)
        self.assertAlmostEqual(omitted, 1.0 - covered)

    def test_mass_target_stops_early(self):
        masses = [10.0, 1.0, 1.0, 0.0]
        # target = 12 * 0.90 = 10.8, so the dominant row alone is not enough
        idx, _, _ = self.client._select_rows_by_mass(masses, max_frac=1.0, mass_frac=0.90)
        self.assertEqual(idx, [0, 1])
        # target = 12 * 0.80 = 9.6, which the dominant row covers alone
        idx, covered, _ = self.client._select_rows_by_mass(masses, max_frac=1.0, mass_frac=0.80)
        self.assertEqual(idx, [0])
        self.assertAlmostEqual(covered, 10 / 12)

    def test_equal_masses_keep_row_order(self):
        idx, _, _ = self.client._select_rows_by_mass([1.0, 1.0, 1.0, 1.0], max_frac=1.0, mass_frac=0.5)
        self.assertEqual(idx, [0, 1])

    def test_no_update_means_no_upload(self):
        self.assertEqual(self.client._select_rows_by_mass([0.0] * 16), ([], 0.0, 0.0))
        self.assertEqual(self.client._select_rows_by_mass([]), ([], 0.0, 0.0))

    def test_negative_and_nan_masses_never_select(self):
        masses = [float("nan"), -1.0, 0.0]
        self.assertEqual(self.client._select_rows_by_mass(masses), ([], 0.0, 0.0))


class TorchSparseDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _real_client()

    def setUp(self):
        if self.client is None:
            self.skipTest("torch is not installed")

    def _table(self, rows=64, width=4):
        import torch

        torch.manual_seed(0)
        return torch.zeros(rows, width, dtype=torch.float32), torch.zeros(rows, width, dtype=torch.bfloat16)

    def test_only_changed_rows_are_uploaded(self):
        table, before = self._table()
        table[[3, 7]] = 0.5
        idx, values = self.client._engram_sparse_delta(table, before)
        self.assertEqual(sorted(int(i) for i in idx), [3, 7])
        self.assertEqual(idx.dtype.name, "uint32")
        self.assertEqual(values.dtype.name, "uint16")
        self.assertEqual(values.shape, (2, 4))

    def test_unchanged_table_uploads_nothing(self):
        table, before = self._table()
        idx, values = self.client._engram_sparse_delta(table, before)
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(values), 0)

    def test_cap_is_configurable_and_shrinks_the_upload(self):
        import os

        table, before = self._table(rows=1000)
        table[:] = 0.25
        os.environ["CROWDGPT_ENGRAM_DELTA_MAX_FRAC"] = "0.005"
        try:
            idx, _ = self.client._engram_sparse_delta(table, before)
        finally:
            del os.environ["CROWDGPT_ENGRAM_DELTA_MAX_FRAC"]
        self.assertLessEqual(len(idx), 5)
        self.assertGreater(len(idx), 0)

    def test_values_match_the_row_delta(self):
        import torch

        table, before = self._table()
        table[5] = 0.125
        idx, values = self.client._engram_sparse_delta(table, before)
        self.assertEqual(int(idx[0]), 5)
        # the payload is raw bf16 (uint16 view), the framing the coordinator decodes
        decoded = torch.from_numpy(values).view(torch.bfloat16).float()
        self.assertEqual(decoded.shape, (1, 4))
        self.assertTrue(torch.allclose(decoded, torch.full((1, 4), 0.125), atol=1e-3))


    def test_mass_target_keeps_dominant_rows(self):
        table, before = self._table(rows=8)
        table[0] = 1.0
        table[1:] = 1e-6
        idx, _ = self.client._engram_sparse_delta(table, before)
        self.assertEqual(int(idx[0]), 0)


if __name__ == "__main__":
    unittest.main()
