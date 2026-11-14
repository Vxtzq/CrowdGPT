"""Unit tests for the dependency-free EGGROLL mechanism spike."""

from __future__ import annotations

import random
import unittest

from crowdgpt_ng import eggroll


class EggrollTests(unittest.TestCase):
    def test_low_rank_shape_and_determinism(self):
        a = eggroll.sample_perturbation(random.Random(7), 6, 4, 2)
        b = eggroll.sample_perturbation(random.Random(7), 6, 4, 2)
        self.assertEqual(len(a.a), 6)
        self.assertEqual(len(a.b), 4)
        self.assertEqual(len(a.a[0]), 2)
        self.assertEqual(a.a, b.a)
        self.assertEqual(a.b, b.b)
        self.assertEqual(a.bias, b.bias)
        delta = eggroll._scaled_perturbation(a, 0.1)
        self.assertEqual(len(delta), 6)
        self.assertEqual(len(delta[0]), 4)

    def test_toy_model_converges(self):
        history = eggroll.benchmark(
            steps=80,
            config=eggroll.EggrollConfig(population=32, rank=2, sigma=0.1, lr=0.05, seed=1),
        )
        first, last = history[0]["loss"], history[-1]["loss"]
        self.assertGreater(first, last)
        self.assertLess(last, first * 0.25)

    def test_scalar_payload_is_smaller_than_dense(self):
        w = [[0.0 for _ in range(32)] for _ in range(32)]
        b = [0.0 for _ in range(32)]
        data = [([0.5] * 32, [1.0] * 32)]
        _, _, diag = eggroll.eggroll_step(
            w, b, data, eggroll.EggrollConfig(population=32, rank=4, sigma=0.1, lr=0.01, seed=2)
        )
        self.assertLess(diag["scalar_bytes"], diag["dense_bytes"])
        self.assertGreater(diag["compression"], 1.0)

    def test_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            eggroll.sample_perturbation(random.Random(1), 0, 4, 2)
        with self.assertRaises(ValueError):
            eggroll.sample_perturbation(random.Random(1), 4, 4, 0)


if __name__ == "__main__":
    unittest.main()
