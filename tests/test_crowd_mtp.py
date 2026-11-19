import unittest

# The repo's CI job is deliberately standard-library only (no pytest, no torch)
# and runs `python -m unittest discover`. Skip this module there instead of
# erroring on import; the tests run under pytest locally and with deps installed.
try:  # pragma: no cover - depends on the environment, not on the code
    import pytest  # noqa: F401
    import torch

    import torch.nn.functional as F
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest and torch: {_missing}")

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crowd_mtp as M


class _Tiny(torch.nn.Module):
    """Just enough trunk for the MTP surface: embed -> blocks -> ln_f -> tied head."""

    def __init__(self, dim=4, vocab=4, layers=1):
        super().__init__()
        self.main_device = torch.device("cpu")
        self.dim, self.vocab = dim, vocab
        self.wte = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim, bias=False) for _ in range(layers)]
        )
        self.ln_f = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.wte.weight
        with torch.no_grad():
            self.wte.weight.copy_(torch.eye(max(dim, vocab))[:vocab, :dim])
            for b in self.blocks:
                b.weight.copy_(torch.eye(dim))

    def embed(self, idx, head_w=None):
        return self.wte(idx)

    def run_blocks(self, x, start=0, stop=None, use_chunked=True, use_ckpt=True,
                   stream_dtype=None):
        for b in self.blocks[start:stop]:
            x = b(x)
        return x

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        return w if dtype is None else w.to(dtype)


def _batch(seq=6, n=2, vocab=4):
    x = torch.randint(0, vocab, (n, seq))
    y = torch.randint(0, vocab, (n, seq))
    return x, y


class AttachTest(unittest.TestCase):
    def test_attach_builds_one_identity_projection_per_extra_depth(self):
        m = _Tiny(dim=4, vocab=4)
        desc = M.attach(m, M.MTPConfig(tokens=4))
        self.assertEqual(desc["depths"], 4)
        self.assertEqual(desc["params"], 4 * 4 * 4)
        self.assertEqual(len(m.mtp_projs), 4)
        self.assertEqual(sorted(m.mtp_projs.keys()), ["1", "2", "3", "4"])
        for k in ("1", "2", "3", "4"):
            self.assertTrue(torch.equal(m.mtp_projs[k].weight, torch.eye(4)))
            self.assertIsNone(m.mtp_projs[k].bias)
        self.assertTrue(M.is_attached(m))
        self.assertEqual(len(M.parameters(m)), 4)

    def test_detached_model_reports_no_mtp_parameters(self):
        m = _Tiny()
        self.assertFalse(M.is_attached(m))
        self.assertEqual(M.parameters(m), ())

    def test_config_rejects_impossible_depths_and_weights(self):
        with self.assertRaises(ValueError):
            M.MTPConfig(tokens=0)
        with self.assertRaises(ValueError):
            M.MTPConfig(tokens=M.MAX_TOKENS + 1)
        with self.assertRaises(ValueError):
            M.MTPConfig(weight=11.0)

    def test_env_switch_is_off_by_default_and_parses_both_knobs(self):
        self.assertIsNone(M.config_from_env({}))
        self.assertIsNone(M.config_from_env({"CROWDGPT_MTP_TOKENS": "0"}))
        self.assertIsNone(M.config_from_env({"CROWDGPT_MTP_TOKENS": "many"}))
        cfg = M.config_from_env({"CROWDGPT_MTP_TOKENS": "4"})
        self.assertEqual((cfg.tokens, cfg.weight), (4, M.DEFAULT_WEIGHT))
        cfg = M.config_from_env({"CROWDGPT_MTP_TOKENS": "2", "CROWDGPT_MTP_WEIGHT": "0.5"})
        self.assertEqual((cfg.tokens, cfg.weight), (2, 0.5))
        # an out-of-range token count falls back to "disabled", never to a crash
        self.assertIsNone(M.config_from_env({"CROWDGPT_MTP_TOKENS": "99"}))


class LossTest(unittest.TestCase):
    def test_no_projection_attached_is_a_zero_that_cannot_be_backwarded_into(self):
        m = _Tiny()
        x, y = _batch()
        loss = M.mtp_loss(m, x, y)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_each_depth_is_supervised_against_the_token_that_far_ahead(self, monkeypatch=None):
        """The shift is the bug-prone part: depth d must read ``y[:, d:]`` only."""
        m = _Tiny(dim=4, vocab=4)
        M.attach(m, M.MTPConfig(tokens=2, weight=1.0))
        x, y = _batch(seq=6, n=2, vocab=4)
        torch.manual_seed(0)
        fakes = {d: torch.randn(2, 6, 4, requires_grad=True) for d in (1, 2)}
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(M, "predict_logits", lambda model, h, d: fakes[d])
            got = M.mtp_loss(m, x, y, M.MTPConfig(tokens=2, weight=1.0),
                             hidden=torch.zeros(2, 6, 4))
        finally:
            monkeypatch.undo()
        expected = sum(
            F.cross_entropy(fakes[d][:, : 6 - d, :].reshape(-1, 4), y[:, d:].reshape(-1))
            for d in (1, 2)
        ) / 2.0
        self.assertAlmostEqual(float(got), float(expected), places=6)

    def test_the_loss_weight_scales_the_auxiliary_term(self):
        m = _Tiny(dim=4, vocab=4)
        M.attach(m, M.MTPConfig(tokens=1))
        x, y = _batch()
        hidden = m.ln_f(m.run_blocks(m.embed(x)))
        a = float(M.mtp_loss(m, x, y, M.MTPConfig(tokens=1, weight=0.2), hidden=hidden))
        b = float(M.mtp_loss(m, x, y, M.MTPConfig(tokens=1, weight=0.4), hidden=hidden))
        self.assertAlmostEqual(b, 2.0 * a, places=6)

    def test_a_batch_shorter_than_the_deepest_shift_yields_a_finite_loss(self):
        m = _Tiny(dim=4, vocab=4)
        M.attach(m, M.MTPConfig(tokens=4))
        x, y = _batch(seq=2, n=1, vocab=4)
        loss = M.mtp_loss(m, x, y, M.MTPConfig(tokens=4))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_gradients_reach_the_projections_and_the_trunk(self):
        m = _Tiny(dim=4, vocab=4)
        M.attach(m, M.MTPConfig(tokens=2, weight=1.0))
        x, y = _batch(seq=5)
        M.mtp_loss(m, x, y, M.MTPConfig(tokens=2, weight=1.0)).backward()
        for k in ("1", "2"):
            self.assertIsNotNone(m.mtp_projs[k].weight.grad)
            self.assertGreater(float(m.mtp_projs[k].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(m.wte.weight.grad)


class TrainingTest(unittest.TestCase):
    def test_training_on_the_auxiliary_loss_lowers_it(self):
        """A non-vacuous end-to-end check: the head learns the offset task."""
        torch.manual_seed(0)
        m = _Tiny(dim=8, vocab=8, layers=1)
        cfg = M.MTPConfig(tokens=4, weight=1.0)
        M.attach(m, cfg)
        # A token stream with a learnable rule: y[t] = (t % 8), i.e. the offset
        # task is (target = position + depth) and the trunk can encode it.
        seq = torch.arange(32).remainder(8).unsqueeze(0)
        x, y = seq[:, :-1], seq[:, 1:]
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-2)
        before = float(M.mtp_loss(m, x, y, cfg))
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            loss = M.mtp_loss(m, x, y, cfg)
            loss.backward()
            opt.step()
        after = float(M.mtp_loss(m, x, y, cfg))
        # A uniform head over 8 tokens is log(8) = 2.08 per depth; 1.6 means the
        # offset task is actually being learned rather than staying at chance.
        self.assertLess(after, 1.6, f"no learning: {before:.3f} -> {after:.3f}")
        self.assertLess(after, before)


class WireTest(unittest.TestCase):
    def test_the_projections_are_local_only(self):
        """Same contract as the SOLO heads: trained, never uploaded."""
        # client.py closes stdio on import under pytest, so import it with
        # throwaway streams (same trick as tests/test_client_delta_layout.py).
        import io
        out, err = io.StringIO(), io.StringIO()
        saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            import client as C
        finally:
            sys.stdout, sys.stderr = saved

        self.assertIn("mtp_projs.", C._LOCAL_ONLY_PREFIXES)
        m = _Tiny()
        M.attach(m, M.MTPConfig(tokens=2))
        names = [n for n, _ in m.named_parameters()]
        self.assertTrue(any(n.startswith("mtp_projs.") for n in names))
        kept = [n for n in names if C._is_server_param(n, {})]
        self.assertFalse(any(n.startswith("mtp_projs.") for n in kept))
