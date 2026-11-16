"""Tests for the A1/A3 client changes (PLAN.md §3.1, §3.3).

The client is a single 290 k-line module so the tests import it and drive the pure helpers
directly; nothing here touches a GPU, a network or a checkpoint.

Run:  python3 -m unittest discover -s tests -v
"""
import contextlib
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client  # noqa: E402


@contextlib.contextmanager
def patched(**globals_):
    """Temporarily override client.py module globals (its tunables are read at import time)."""
    saved = {k: getattr(client, k) for k in globals_}
    try:
        for k, v in globals_.items():
            setattr(client, k, v)
        yield client
    finally:
        for k, v in saved.items():
            setattr(client, k, v)


class AccumStepsTest(unittest.TestCase):
    """A1: the server's optimizer-step budget, not bs*sl, decides the accum count."""

    def test_target_optimizer_steps_sets_accum(self):
        a = client._accum_steps_for_round
        self.assertEqual(a(True, 500, 2, 32, 2, 2048), 250)   # 2 steps of ~1.02 M tokens
        self.assertEqual(a(True, 500, 1, 32, 2, 2048), 500)   # 1 step of ~2.05 M tokens
        self.assertEqual(a(True, 500, 8, 32, 2, 2048), 63)    # ceil(500/8)
        self.assertEqual(a(True, 500, 16, 32, 2, 2048), 32)   # ceil(500/16) = v1's 32
        self.assertEqual(a(True, 500, 3, 32, 2, 2048), 167)   # non-divisor rounds up

    def test_tokens_per_round_are_unchanged(self):
        for h in (1, 2, 4, 8, 16):
            accum = client._accum_steps_for_round(True, 500, h, 32, 2, 2048)
            # every micro-step is bs*sl tokens and the round always spends 500 of them
            self.assertLessEqual(accum * h, 500 + accum)      # ceil() overshoot < one accum
            self.assertEqual(accum * (h - 1) < 500, True)

    def test_optimizer_steps_never_exceed_micro_steps(self):
        self.assertEqual(client._accum_steps_for_round(True, 4, 99, 32, 2, 2048), 1)

    def test_legacy_paths(self):
        self.assertEqual(client._accum_steps_for_round(True, 500, 0, 32, 2, 2048), 32)
        self.assertEqual(client._accum_steps_for_round(True, 500, 0, None, 2, 2048), 32)
        self.assertEqual(client._accum_steps_for_round(False, None, 0, None, 2, 2048), 32)
        self.assertEqual(client._accum_steps_for_round(False, None, 0, None, 8, 2048), 8)
        self.assertEqual(client._accum_steps_for_round(True, None, 2, None, 2, 2048), 32)


class InnerOptimizerTest(unittest.TestCase):
    """A1: at H <= 2 a cold AdamW is a sign() machine, so the default becomes SGD-Nesterov."""

    def test_auto_choice(self):
        c = client
        self.assertEqual(c._resolve_inner_optim(1), "sgd_nesterov")
        self.assertEqual(c._resolve_inner_optim(2), "sgd_nesterov")
        self.assertEqual(c._resolve_inner_optim(4), "adamw")
        self.assertEqual(c._resolve_inner_optim(8), "adamw")
        self.assertEqual(c._resolve_inner_optim(16), "adamw")
        self.assertEqual(c._resolve_inner_optim(0), "adamw")     # legacy fleet

    def test_env_override(self):
        with patched(INNER_OPTIM="adamw") as c:
            self.assertEqual(c._resolve_inner_optim(2), "adamw")
        with patched(INNER_OPTIM="sgd") as c:
            self.assertEqual(c._resolve_inner_optim(16), "sgd_nesterov")
        with patched(INNER_OPTIM="auto"):
            self.assertEqual(client._resolve_inner_optim(2), "sgd_nesterov")

    def test_lr_rules(self):
        c = client
        self.assertAlmostEqual(c._resolve_inner_lr(2), 8e-4)             # linear x8
        self.assertAlmostEqual(c._resolve_inner_lr(16), 1e-4)            # reference
        self.assertAlmostEqual(c._resolve_inner_lr(0), 1e-4)             # legacy
        with patched(INNER_LR_RULE="sqrt") as c:
            self.assertAlmostEqual(c._resolve_inner_lr(2), 1e-4 * 2.8284271247461903, places=9)
        with patched(INNER_LR_RULE="none") as c:
            self.assertAlmostEqual(c._resolve_inner_lr(2), 1e-4, places=12)
        with patched(INNER_LR_OVERRIDE=3e-3) as c:
            self.assertAlmostEqual(c._resolve_inner_lr(2), 3e-3, places=12)

    def test_sgd_builder_is_nesterov_and_engram_has_its_own_lr(self):
        model = _TinyModel()
        ob, oe = client._build_optimizers(model, lr=8e-4, inner="sgd_nesterov", engram_lr=3e-3)
        sgd = ob.opts[0]
        self.assertIsInstance(sgd, torch.optim.SGD)
        self.assertEqual(sgd.param_groups[0]["momentum"], 0.9)
        self.assertTrue(sgd.param_groups[0]["nesterov"])
        self.assertAlmostEqual(sgd.param_groups[0]["lr"], 8e-4)
        self.assertAlmostEqual(oe.param_groups[0]["lr"], 3e-3)

    def test_adamw_builder_keeps_the_old_hyperparameters(self):
        model = _TinyModel()
        ob, oe = client._build_optimizers(model, lr=1e-4, force_mode="adamw32", inner="adamw",
                                          engram_lr=1e-4)
        opt = ob.opts[0]
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual(opt.param_groups[0]["betas"], (0.9, 0.95))
        self.assertAlmostEqual(opt.param_groups[0]["weight_decay"], 0.01)

    def test_vram_estimate_for_sgd_state(self):
        """SGD carries one momentum tensor (4 B/param): between 8-bit and fp32 AdamW."""
        e8 = client.estimate_vram_bytes(2, 2048, True)
        esgd = client.estimate_vram_bytes(2, 2048, "sgd")
        e32 = client.estimate_vram_bytes(2, 2048, False)
        self.assertLess(e8, esgd)
        self.assertLess(esgd, e32)
        per_param = client._opt_state_bytes_per_param
        self.assertEqual((per_param(True), per_param("sgd"), per_param(False)), (2, 4, 8))


class SubmitMetadataTest(unittest.TestCase):
    """A3: the delta must carry the optimizer-step count, the accum, and the real optimizer."""

    def test_cohort_delta_reports_optimizer_steps(self):
        extra = client._build_submit_extra(
            cohort_mode=True, cc_seed=7, micro_step=500, bs=2, opt_steps_taken=2, accum=250,
            inner_opt="sgd_nesterov", inner_lr=8e-4, last_step_loss=3.25, loss_std=0.1,
            loss_samples=500)
        self.assertEqual(extra["optimizerSteps"], 2)
        self.assertEqual(extra["accum"], 250)
        self.assertEqual(extra["innerOptimizer"], "sgd_nesterov")
        self.assertAlmostEqual(extra["innerLr"], 8e-4)
        self.assertEqual(extra["actualSteps"], 500)        # micro steps stay for compatibility
        self.assertEqual(extra["actualBatchSize"], 2)
        self.assertEqual(extra["actualSeed"], 7)
        self.assertTrue(extra["crossCheckUsed"])

    def test_non_cohort_delta_has_no_seed(self):
        extra = client._build_submit_extra(
            cohort_mode=False, cc_seed=None, micro_step=10, bs=1, opt_steps_taken=3, accum=4,
            inner_opt="adamw", inner_lr=1e-4, last_step_loss=None, loss_std=0.0,
            loss_samples=0)
        self.assertIsNone(extra["actualSeed"])
        self.assertIsNone(extra["lastLoss"])
        self.assertEqual(extra["optimizerSteps"], 3)


class TaskMetaTest(unittest.TestCase):
    """The client must pick up `crossCheck.targetOptimizerSteps` and default to 0."""

    def _meta(self, cc):
        return {"taskId": "t1", "globalStep": 0, "crossCheck": cc}

    def test_parses_target_optimizer_steps(self):
        logs = []
        cc = {"enabled": True, "targetSeed": 1, "targetLocalSteps": 500, "targetBatchSize": 2,
              "targetAccumSteps": 32, "targetAutocast": "bf16", "targetOptimizer": "any",
              "targetOptimizerSteps": 2}
        got = client._parse_task_meta(self._meta(cc), lambda k, m: logs.append(m))
        self.assertEqual(got[3], True)
        self.assertEqual(got[4], 1)         # targetSeed
        self.assertEqual(got[5], 500)       # targetLocalSteps
        self.assertEqual(got[9], "any")
        self.assertEqual(got[10], 2)
        self.assertTrue(any("optSteps=2" in m for m in logs))

    def test_missing_field_is_legacy(self):
        cc = {"enabled": True, "targetSeed": 1, "targetLocalSteps": 500, "targetBatchSize": 2,
              "targetAccumSteps": 32, "targetAutocast": "bf16"}
        got = client._parse_task_meta(self._meta(cc), lambda k, m: None)
        self.assertEqual(got[10], 0)

    def test_incomplete_cohort_metadata_disables_cohort(self):
        cc = {"enabled": True, "targetLocalSteps": 500, "targetOptimizerSteps": 2}
        got = client._parse_task_meta(self._meta(cc), lambda k, m: None)
        self.assertFalse(got[3])
        self.assertEqual(got[10], 0)

    def test_bad_value_is_legacy(self):
        cc = {"enabled": True, "targetSeed": 1, "targetLocalSteps": 500, "targetBatchSize": 2,
              "targetAccumSteps": 32, "targetAutocast": "bf16",
              "targetOptimizerSteps": "not-a-number"}
        got = client._parse_task_meta(self._meta(cc), lambda k, m: None)
        self.assertEqual(got[10], 0)


    def test_planner_reserves_momentum_for_sgd(self):
        """`_plan_placement` must not plan SGD as if it were 8-bit Adam (2 B/param)."""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cpu":
            self.skipTest("placement planner returns early on CPU")
        budget = 40.0
        _, n32, _ = client._plan_placement(dev, "CUDA", budget, bs=2, sl=2048, u8=False)
        _, nsgd, _ = client._plan_placement(dev, "CUDA", budget, bs=2, sl=2048, u8="sgd")
        _, n8, _ = client._plan_placement(dev, "CUDA", budget, bs=2, sl=2048, u8=True)
        self.assertLessEqual(nsgd, n32)      # 4 B/param plans no better than 8 B/param
        self.assertGreaterEqual(nsgd, n8)    # and no better than 2 B/param

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA/ROCm device")
    def test_sgd_and_sparse_engram_step_on_the_device(self):
        """The real builders must run a step on the device, incl. SparseAdam on the engram."""
        dev = torch.device("cuda")
        model = _TinyModel().to(dev)
        ob, oe = client._build_optimizers(model, lr=8e-4, inner="sgd_nesterov", engram_lr=1e-3)
        ids = torch.randint(0, 4, (1, 3), device=dev)
        out = model.wte(ids).sum() + model.blocks[0](model.wte(ids)).sum()
        out.backward()
        ob.step(); ob.zero_grad(set_to_none=True)
        oe.step(); oe.zero_grad(set_to_none=True)
        self.assertTrue(torch.isfinite(model.wte.weight).all())


class _TinyModel(torch.nn.Module):
    """Just enough module surface for `_build_optimizers`."""

    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(4, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

        class _Eng(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Embedding(4, 2, sparse=True)

        self.engram = _Eng()


if __name__ == "__main__":
    unittest.main()
