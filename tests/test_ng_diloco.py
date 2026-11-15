"""Tests for the DiLoCo-style outer loop (``crowdgpt_ng/diloco.py``).

No model is involved: the module is the update rule and the byte budget, so
the tests are arithmetic. If the outer step cannot drive a quadratic to its
minimum, or the payload does not respect its budget, the design is wrong
regardless of what a 658M model would do.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
except ImportError:  # pragma: no cover - the stdlib-only CI job
    np = None

if np is not None:
    from crowdgpt_ng import diloco


@unittest.skipIf(np is None, "numpy is required for the outer-loop tests")
class Bf16Tests(unittest.TestCase):
    def test_rounding_is_not_truncation(self):
        values = np.array([1.0, 1.0 + 2**-9, -1.0 - 2**-9, 0.0], dtype=np.float32)
        back = diloco.from_bf16_bits(diloco.to_bf16_bits(values))
        # bf16 has 8 mantissa bits: 1 + 2^-9 rounds back to 1.0, not below it
        self.assertEqual(float(back[0]), 1.0)
        self.assertAlmostEqual(float(back[1]), 1.0)
        self.assertEqual(float(back[3]), 0.0)

    def test_relative_error_is_bounded_by_the_bf16_quantum(self):
        rng = np.random.default_rng(3)
        values = rng.standard_normal(4096).astype(np.float32)
        back = diloco.from_bf16_bits(diloco.to_bf16_bits(values))
        relative = np.abs(back - values) / np.maximum(np.abs(values), 1e-6)
        self.assertLess(float(relative.max()), 2**-8)


@unittest.skipIf(np is None, "numpy is required for the outer-loop tests")
class CompressTests(unittest.TestCase):
    def test_budget_is_respected(self):
        rng = np.random.default_rng(0)
        tensor = rng.standard_normal(1_000_000).astype(np.float32)
        payload = diloco.compress(tensor, budget_bytes=60_000)
        self.assertLessEqual(payload.nbytes, 60_000)
        self.assertGreater(payload.nbytes, 0)

    def test_largest_magnitudes_are_kept(self):
        tensor = np.zeros(1000, dtype=np.float32)
        tensor[[7, 500, 999]] = 5.0
        payload = diloco.compress(tensor, budget_bytes=18)
        self.assertEqual(sorted(int(i) for i in payload.indices), [7, 500, 999])
        dense = diloco.decompress(payload)
        self.assertAlmostEqual(float(np.abs(dense).sum()), 15.0, places=2)
        self.assertAlmostEqual(float(np.abs(dense - tensor).sum()), 0.0, places=2)

    def test_error_is_bounded_by_the_largest_omitted_value(self):
        rng = np.random.default_rng(1)
        tensor = rng.standard_normal(10_000).astype(np.float32)
        payload = diloco.compress(tensor, budget_bytes=600)
        dense = diloco.decompress(payload)
        kept = np.zeros_like(tensor, dtype=bool)
        kept[payload.indices] = True
        omitted = np.abs(tensor[~kept])
        if omitted.size:
            self.assertLessEqual(float(np.abs(dense - tensor).max()), float(omitted.max()) * 1.01)

    def test_zero_tensor_sends_nothing(self):
        payload = diloco.compress(np.zeros(64, dtype=np.float32), budget_bytes=600)
        self.assertEqual(payload.indices.size, 0)
        self.assertEqual(payload.nbytes, 0)

    def test_payloads_round_trip_through_json(self):
        tensor = np.arange(32, dtype=np.float32)
        payload = diloco.compress(tensor, budget_bytes=600)
        restored = diloco.SparsePayload.from_dict(payload.to_dict())
        np.testing.assert_allclose(diloco.decompress(restored), diloco.decompress(payload))

    def test_compress_many_stays_within_the_round_budget(self):
        rng = np.random.default_rng(2)
        tensors = [rng.standard_normal(n).astype(np.float32) for n in (5000, 20_000, 1)]
        payloads = diloco.compress_many(tensors, budget_bytes=6_000)
        self.assertEqual(len(payloads), 3)
        self.assertLessEqual(sum(p.nbytes for p in payloads), 6_000)
        self.assertTrue(all(p.indices.size >= 1 for p in payloads))

    def test_average_collapses_duplicate_indices(self):
        a = diloco.compress(np.array([1.0, 0.0], dtype=np.float32), budget_bytes=6)
        b = diloco.compress(np.array([3.0, 0.0], dtype=np.float32), budget_bytes=6)
        averaged = diloco.average([a, b], 2)
        self.assertAlmostEqual(float(averaged[0]), 2.0, places=2)


@unittest.skipIf(np is None, "numpy is required for the outer-loop tests")
class OuterOptimizerTests(unittest.TestCase):
    def test_nesterov_drives_a_quadratic_down(self):
        opt = diloco.OuterOptimizer(lr=0.2, momentum=0.9)
        theta = np.array([4.0, -3.0], dtype=np.float32)
        losses = []
        for _ in range(40):
            losses.append(float((theta**2).sum()))
            theta = opt.step(theta, 2.0 * theta)
        self.assertLess(losses[-1], losses[0] * 1e-3)

    def test_momentum_is_carried_between_steps(self):
        opt = diloco.OuterOptimizer(lr=0.1, momentum=0.5)
        theta = np.zeros(1, dtype=np.float32)
        first = opt.step(theta, np.array([1.0], dtype=np.float32))
        second = opt.step(first, np.array([1.0], dtype=np.float32))
        # step 1: prev is zero, so direction = g*(1+m) = 1.5
        self.assertAlmostEqual(float(first[0]), -0.1 * 1.5, places=5)
        # step 2: prev == g, so direction = 1 and only lr is applied
        self.assertAlmostEqual(float(second[0]), -0.15 - 0.1, places=5)

    def test_state_round_trips(self):
        opt = diloco.OuterOptimizer(lr=0.3, momentum=0.8)
        opt.step(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
        clone = diloco.OuterOptimizer.from_state_dict(opt.state_dict())
        np.testing.assert_allclose(
            clone.step(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
            opt.step(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
        )

    def test_rejects_bad_input(self):
        opt = diloco.OuterOptimizer()
        with self.assertRaises(diloco.DilocoError):
            opt.step(np.zeros(2, dtype=np.float32), np.ones(3, dtype=np.float32))
        with self.assertRaises(diloco.DilocoError):
            opt.step(np.zeros(2, dtype=np.float32), np.array([np.nan, 1.0], dtype=np.float32))
        with self.assertRaises(diloco.DilocoError):
            diloco.OuterOptimizer(lr=0.0)
        with self.assertRaises(diloco.DilocoError):
            diloco.OuterOptimizer(momentum=1.0)


if __name__ == "__main__":
    unittest.main()
