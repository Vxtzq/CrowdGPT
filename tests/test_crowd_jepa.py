"""Tests for crowd_jepa.py (opt-in LLM-JEPA loss). The determinism test is stdlib-only; the tensor tests run only with a
real PyTorch (the CI stubs torch, so they are skipped there and run locally)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import crowd_jepa as cj  # noqa: E402


def _real_torch():
    try:
        import torch

        torch.zeros(2).sum().item()
        torch.nn.functional.scaled_dot_product_attention  # noqa: B018
        return torch
    except Exception:
        return None


TORCH = _real_torch()


class TestLossDropoutStdlib(unittest.TestCase):
    def test_deterministic_per_seed_and_step(self) -> None:
        a = [cj.jepa_step(7, s, 0.5) for s in range(200)]
        self.assertEqual(a, [cj.jepa_step(7, s, 0.5) for s in range(200)])  # same on every client of a cohort
        self.assertNotEqual(a, [cj.jepa_step(8, s, 0.5) for s in range(200)])

    def test_rate_matches_alpha(self) -> None:
        n = 4000
        for alpha in (0.0, 0.25, 0.75, 1.0):
            kept = sum(cj.jepa_step(1, s, alpha) for s in range(n)) / n
            self.assertAlmostEqual(kept, 1.0 - alpha, delta=0.03)

    def test_rejects_bad_alpha(self) -> None:
        with self.assertRaises(ValueError):
            cj.jepa_step(0, 0, 1.5)


@unittest.skipIf(TORCH is None, "real PyTorch not available (CI stubs torch)")
class TestPackingAndLoss(unittest.TestCase):
    def test_pack_layout_targets_and_indices(self) -> None:
        ids, mask, (it, ic), tgt = cj.pack_two_views([10, 11, 12], [20, 21], k_pred=2, pred_token_id=99)
        self.assertEqual(ids.tolist(), [[10, 11, 12, 99, 99, 20, 21]])
        self.assertEqual((it, ic), (4, 6))  # last predictor token, last code token
        I = cj.IGNORE
        self.assertEqual(tgt.tolist(), [[11, 12, I, I, I, 21, I]])
        m = mask[0, 0]
        self.assertTrue(bool(m[6, 5]) and not bool(m[5, 6]))  # causal inside view 2
        self.assertFalse(bool(m[5:, :5].any()) or bool(m[:5, 5:].any()))  # no attention across views
        self.assertTrue(bool(m.diagonal().all()))

    def test_block_causal_attention_equals_running_each_view_alone(self) -> None:
        torch = TORCH
        torch.manual_seed(0)
        n1, n2, h, d = 5, 4, 2, 8
        _, mask, _, _ = cj.pack_two_views(list(range(n1)), list(range(n2)))
        q, k, v = (torch.randn(1, h, n1 + n2, d) for _ in range(3))
        sdpa = torch.nn.functional.scaled_dot_product_attention
        packed = sdpa(q, k, v, attn_mask=mask)
        alone1 = sdpa(q[:, :, :n1], k[:, :, :n1], v[:, :, :n1], is_causal=True)
        alone2 = sdpa(q[:, :, n1:], k[:, :, n1:], v[:, :, n1:], is_causal=True)
        self.assertTrue(torch.allclose(packed[:, :, :n1], alone1, atol=1e-6))
        self.assertTrue(torch.allclose(packed[:, :, n1:], alone2, atol=1e-6))

    def test_jepa_term_range_and_gradient_reaches_both_views(self) -> None:
        torch = TORCH
        hid = torch.randn(3, 9, 16, requires_grad=True)
        j = cj.jepa_term(hid, 4, 8)
        self.assertTrue(0.0 <= float(j.detach()) <= 2.0)
        j.backward()
        self.assertGreater(float(hid.grad[:, 4].abs().sum()), 0.0)
        self.assertGreater(float(hid.grad[:, 8].abs().sum()), 0.0)  # no stop-gradient, as in the reference code
        same = torch.ones(1, 3, 4)
        self.assertAlmostEqual(float(cj.jepa_term(same, 0, 2)), 0.0, places=6)

    def test_stopgrad_blocks_the_target_view(self) -> None:
        torch = TORCH
        hid = torch.randn(3, 9, 16, requires_grad=True)
        cj.jepa_term(hid, 4, 8, stopgrad=True).backward()
        self.assertGreater(float(hid.grad[:, 4].abs().sum()), 0.0)
        self.assertEqual(float(hid.grad[:, 8].abs().sum()), 0.0)

    def test_center_defeats_the_shared_direction_shortcut(self) -> None:
        torch = TORCH
        torch.manual_seed(1)
        shared = torch.full((4, 1, 16), 10.0)
        hid = torch.cat([shared + 0.1 * torch.randn(4, 1, 16), shared + 0.1 * torch.randn(4, 1, 16)], dim=1)
        self.assertLess(float(cj.jepa_term(hid, 0, 1)), 0.01)  # plain cosine: "solved" by the common offset
        self.assertGreater(float(cj.jepa_term(hid, 0, 1, center=True)), 0.5)  # centered: the offset no longer helps

    def test_per_row_indices(self) -> None:
        torch = TORCH
        hid = torch.randn(2, 7, 8)
        j = cj.jepa_term(hid, torch.tensor([1, 2]), torch.tensor([5, 6]))
        a = 1.0 - torch.nn.functional.cosine_similarity(hid[[0, 1], [1, 2]], hid[[0, 1], [5, 6]], dim=-1).mean()
        self.assertAlmostEqual(float(j), float(a), places=6)

    def test_combined_loss_and_dropout(self) -> None:
        torch = TORCH
        ce = torch.tensor(2.0)
        hid = torch.randn(1, 6, 8)
        loss, j = cj.combined_loss(ce, hid, 2, 5, lam=0.5)
        self.assertAlmostEqual(float(loss), 2.0 + 0.5 * float(j), places=5)
        loss, j = cj.combined_loss(ce, hid, 2, 5, lam=0.5, apply_jepa=False)
        self.assertEqual((float(loss), j), (2.0, None))
        with self.assertRaises(ValueError):
            cj.combined_loss(ce, hid, 2, 5, gamma=0.0)

    def test_pack_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            cj.pack_two_views([], [1])
        with self.assertRaises(ValueError):
            cj.pack_two_views([1], [2], k_pred=1)
        with self.assertRaises(ValueError):
            cj.pack_two_views([1, 2, 3], [4, 5], max_len=4)


if __name__ == "__main__":
    unittest.main()
