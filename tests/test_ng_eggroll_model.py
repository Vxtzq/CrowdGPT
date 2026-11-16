"""Tests for tools/bench_eggroll_model.py (small Transformer EGGROLL benchmark).

The tool needs a real PyTorch. CI stubs torch, so every test here is skipped
cleanly when torch is absent or non-functional; the same suite runs locally on
CPU with tiny configurations that take well under a second each.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "bench_eggroll_model.py"


def _real_torch():
    try:
        import torch

        torch.zeros(2).sum().item()
        torch.nn.functional.layer_norm  # noqa: B018
        return torch
    except Exception:
        return None


TORCH = _real_torch()
BENCH = None
if TORCH is not None:
    _spec = importlib.util.spec_from_file_location("bench_eggroll_model", TOOL)
    BENCH = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = BENCH
    _spec.loader.exec_module(BENCH)


@unittest.skipIf(TORCH is None, "real torch not available (CI stubs torch)")
class EggrollModelTests(unittest.TestCase):
    def tiny_model(self):
        return BENCH.ModelConfig(dim=16, layers=1, heads=2, vocab=12, seq=8, batch=2)

    def tiny_options(self, **overrides):
        values = {"population": 8, "rank": 2, "sigma": 0.05, "lr": 0.1, "steps": 20, "seed": 1}
        values.update(overrides)
        return BENCH.EggrollOptions(**values)

    def test_tiny_cpu_run_reports_finite_metrics(self):
        model = self.tiny_model()
        options = self.tiny_options(steps=3)
        result = BENCH.run_benchmark(model, options, "cpu")

        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["tokens_processed"], options.steps * options.population * model.batch * model.seq)
        self.assertGreater(result["model"]["params"], 0)
        self.assertGreater(result["first_loss"], 0.0)
        self.assertGreater(result["last_loss"], 0.0)
        self.assertGreaterEqual(result["seconds"], 0.0)
        self.assertGreater(result["tokens_per_second"], 0.0)
        self.assertIsNone(result["peak_gpu_memory_mb"])
        self.assertIn("scaled-down", result["note"])

    def test_tiny_cpu_run_improves_loss(self):
        # Deterministic on CPU for a fixed seed, so this is a behaviour check,
        # not a statistical one.
        result = BENCH.run_benchmark(self.tiny_model(), self.tiny_options(), "cpu")
        self.assertLess(result["last_loss"], result["first_loss"])

    def test_run_is_deterministic_for_a_seed(self):
        first = BENCH.run_benchmark(self.tiny_model(), self.tiny_options(steps=4), "cpu")
        again = BENCH.run_benchmark(self.tiny_model(), self.tiny_options(steps=4), "cpu")
        other_seed = BENCH.run_benchmark(self.tiny_model(), self.tiny_options(steps=4, seed=2), "cpu")

        self.assertEqual(first["first_loss"], again["first_loss"])
        self.assertEqual(first["last_loss"], again["last_loss"])
        self.assertNotEqual(first["first_loss"], other_seed["first_loss"])

    def test_fused_low_rank_forward_matches_materialised_delta(self):
        model = self.tiny_model()
        params = BENCH.init_params(model, seed=3, device=TORCH.device("cpu"))
        gen = TORCH.Generator(device="cpu").manual_seed(4)
        pert = BENCH.sample_perturbation(params, rank=2, gen=gen)
        tokens = BENCH.make_batch(model, TORCH.Generator(device="cpu").manual_seed(5), TORCH.device("cpu"))
        sigma, rank = 0.05, 2

        with TORCH.inference_mode():
            fused = BENCH.forward(params, model, tokens, pert, sigma, rank)
            dense = {
                name: value + BENCH.materialize_delta(pert, name, value, sigma, rank)
                for name, value in params.items()
            }
            expected = BENCH.forward(dense, model, tokens)
        self.assertTrue(TORCH.allclose(fused, expected, atol=1e-5, rtol=1e-4))

    def test_payload_projection_is_a_dense_vs_scalar_comparison(self):
        model = self.tiny_model()
        params = BENCH.init_params(model, seed=3, device=TORCH.device("cpu"))
        payload = BENCH.payload_projection(params, population=8, rank=2)

        model_params = sum(value.numel() for value in params.values())
        self.assertEqual(payload["model_params"], float(model_params))
        self.assertEqual(payload["dense_bytes"], float(model_params * 2))
        self.assertEqual(payload["scalar_bytes"], float(8 * (4 + 8)))
        self.assertGreater(payload["scalar_compression"], 1.0)
        self.assertGreater(payload["factor_bytes"], 0.0)

    def test_cli_accepts_all_benchmark_flags(self):
        args = BENCH.parse_args(
            [
                "--device", "cpu",
                "--population", "4",
                "--rank", "2",
                "--sigma", "0.02",
                "--lr", "0.05",
                "--steps", "2",
                "--dim", "8",
                "--layers", "1",
                "--heads", "2",
                "--seq", "4",
                "--vocab", "8",
                "--seed", "7",
            ]
        )
        self.assertEqual((args.device, args.population, args.rank), ("cpu", 4, 2))
        self.assertEqual((args.dim, args.layers, args.heads), (8, 1, 2))
        self.assertEqual((args.seq, args.vocab, args.seed, args.steps), (4, 8, 7, 2))

    def test_member_seeds_are_deterministic_and_distinct(self):
        seeds = BENCH.member_seeds(11, 0, 6)
        self.assertEqual(seeds, BENCH.member_seeds(11, 0, 6))
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)
        self.assertNotEqual(BENCH.member_seed(11, 0, 0), BENCH.member_seed(11, 1, 0))
        self.assertNotEqual(BENCH.member_seed(11, 0, 0), BENCH.member_seed(11, 0, 1))
        self.assertNotEqual(BENCH.member_seed(11, 0, 0), BENCH.member_seed(12, 0, 0))

    def test_member_perturbations_do_not_depend_on_draw_order(self):
        model = self.tiny_model()
        params = BENCH.init_params(model, seed=5, device=TORCH.device("cpu"))
        seeds = BENCH.member_seeds(5, 0, 4)
        forward = [BENCH.sample_member(params, 2, TORCH.device("cpu"), s) for s in seeds]
        reverse = [BENCH.sample_member(params, 2, TORCH.device("cpu"), s) for s in reversed(seeds)]
        reverse.reverse()
        for first, second in zip(forward, reverse):
            self.assertTrue(BENCH._perturbations_equal(first, second))
        self.assertFalse(BENCH._perturbations_equal(forward[0], forward[1]))

    def test_check_seed_regen_passes(self):
        model = self.tiny_model()
        params = BENCH.init_params(model, seed=2, device=TORCH.device("cpu"))
        options = self.tiny_options(population=4)
        regen = BENCH.check_seed_regen(params, model, options, TORCH.device("cpu"), 3)
        self.assertTrue(regen["reproducible"])
        self.assertTrue(regen["distinct"])
        self.assertTrue(regen["changesWithStep"])
        self.assertTrue(regen["changesWithMember"])
        self.assertTrue(regen["passes"])
        self.assertEqual(regen["step"], 3)
        self.assertEqual(len(regen["seeds"]), 4)

    def test_eggroll_step_accepts_seeds_without_a_generator(self):
        model = self.tiny_model()
        options = self.tiny_options(population=4, steps=1)
        params = BENCH.init_params(model, seed=6, device=TORCH.device("cpu"))
        tokens = BENCH.make_batch(
            model, TORCH.Generator(device="cpu").manual_seed(0), TORCH.device("cpu")
        )
        seeds = BENCH.member_seeds(options.seed, 0, options.population)
        losses, updated = BENCH.eggroll_step(params, model, options, None, tokens, seeds=seeds)
        self.assertEqual(len(losses), 4)
        self.assertEqual(set(updated), set(params))
        again, _ = BENCH.eggroll_step(params, model, options, None, tokens, seeds=seeds)
        self.assertEqual(losses, again)

    def test_eggroll_step_rejects_a_bad_seed_count(self):
        model = self.tiny_model()
        options = self.tiny_options(population=4)
        params = BENCH.init_params(model, seed=6, device=TORCH.device("cpu"))
        tokens = BENCH.make_batch(
            model, TORCH.Generator(device="cpu").manual_seed(0), TORCH.device("cpu")
        )
        with self.assertRaises(ValueError):
            BENCH.eggroll_step(params, model, options, None, tokens, seeds=[1, 2])

    def test_run_benchmark_records_the_regeneration_check(self):
        result = BENCH.run_benchmark(
            self.tiny_model(), self.tiny_options(population=4, steps=2), "cpu", verify_regen=True
        )
        self.assertTrue(result["seeds"]["perMember"])
        self.assertTrue(result["seeds"]["regeneration"]["passes"])
        self.assertEqual(result["seeds"]["regeneration"]["population"], 4)

    def test_legacy_run_keeps_the_single_generator(self):
        result = BENCH.run_benchmark(
            self.tiny_model(), self.tiny_options(population=4, steps=1), "cpu"
        )
        self.assertFalse(result["seeds"]["perMember"])
        self.assertIsNone(result["seeds"]["regeneration"])

    def test_cli_exposes_the_regen_flags(self):
        args = BENCH.parse_args(["--device", "cpu", "--verify-regen", "--per-member-seeds"])
        self.assertTrue(args.verify_regen)
        self.assertTrue(args.per_member_seeds)
        args = BENCH.parse_args(["--device", "cpu"])
        self.assertFalse(args.verify_regen)
        self.assertFalse(args.per_member_seeds)


if __name__ == "__main__":
    unittest.main()
