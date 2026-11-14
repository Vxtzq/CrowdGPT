"""Smoke tests for the retained HiveMind modules.

They are stdlib-compatible and skip when the optional scientific stack is not
installed, so the upstream checks workflow still imports the module.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    import torch

    import hivemind_bridge
    import hivemind_loss_check
except Exception:  # pragma: no cover - optional stack
    np = None
    torch = None
    hivemind_bridge = None
    hivemind_loss_check = None


@unittest.skipIf(torch is None or hivemind_bridge is None, "torch/numpy not installed")
class HiveMindSmokeTests(unittest.TestCase):
    def test_delta_roundtrip(self):
        delta = torch.zeros(4, dtype=torch.bfloat16)
        delta[0] = 1.5
        blob = hivemind_bridge.pack_crowdgpt_delta_stream(
            {"taskId": "t1", "localSteps": 3}, delta
        )
        payload = hivemind_bridge.unpack_crowdgpt_delta_stream(blob, engram_row_dim=1536)
        self.assertEqual(payload.metadata["taskId"], "t1")
        self.assertEqual(payload.base_delta_bf16.shape, (4,))

    def test_receipt_signature_roundtrip(self):
        receipt = hivemind_loss_check.LossCheckReceipt(
            evaluator_id="eval-a",
            base_sha256="a" * 64,
            candidate_sha256="b" * 64,
            heldout_sha256="c" * 64,
            window=8,
            seed=1,
            starts=(0,),
            losses_before=(1.0,),
            losses_after=(0.9,),
            compute_dtype="float32",
            device="cpu",
            round_id="r1",
        )
        signed = receipt.signed(b"test-key")
        self.assertTrue(signed.verify_signature(b"test-key"))
        self.assertFalse(signed.verify_signature(b"wrong-key"))
        self.assertEqual(hivemind_loss_check.LossCheckReceipt.from_json(signed.to_json()), signed)


if __name__ == "__main__":
    unittest.main()
