"""Standard-library CI tests for CrowdGPT x HiveMind coordinator & client bridge modules.

Exercises `hivemind_bridge.py`, `hivemind_loss_check.py`, and `hivemind_loss_check_service.py`
using only Python's standard library (matching `.github/workflows/checks.yml`: no GPU, no network,
no external packages required), plus PyTorch tensor roundtrips when PyTorch is available locally.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hivemind_bridge as hb  # noqa: E402
import hivemind_loss_check as lc  # noqa: E402
import hivemind_loss_check_service as lcs  # noqa: E402


class TestHiveMindBridgeStdlib(unittest.TestCase):
    def test_device_multiplexing_headers_prevent_account_collision(self) -> None:
        h_rtx = hb.build_crowdgpt_client_hardware_headers(
            hiveminder_handle="xav_rtx2070",
            device_suffix="rtx2070",
            vram_gb=8.0,
            gpu_name="NVIDIA GeForce RTX 2070",
            autocast_dtype="fp32",
            compute_capability=(7, 5),
        )
        h_t4 = hb.build_crowdgpt_client_hardware_headers(
            hiveminder_handle="hermes_t4",
            device_suffix="t4spot",
            vram_gb=15.0,
            gpu_name="NVIDIA Tesla T4",
            autocast_dtype="fp32",
            compute_capability=(7, 5),
        )
        self.assertEqual(h_rtx["X-Client-Device"], "hm-xav_rtx2070-rtx2070")
        self.assertEqual(h_t4["X-Client-Device"], "hm-hermes_t4-t4spot")
        self.assertNotEqual(h_rtx["X-Client-Device"], h_t4["X-Client-Device"])
        self.assertEqual(h_rtx["X-Client-Engram"], "1")
        self.assertEqual(len(h_rtx), 12)

    def test_precision_cohort_stratification(self) -> None:
        turing = hb.classify_gpu_precision_cohort("nvidia", (7, 5), requested_dtype="auto")
        self.assertEqual(turing.effective_dtype, "fp32")
        self.assertTrue(turing.pre_ampere_fp32_enforced)

        ampere = hb.classify_gpu_precision_cohort("nvidia", (8, 6), requested_dtype="auto")
        self.assertEqual(ampere.effective_dtype, "bf16")
        self.assertFalse(ampere.pre_ampere_fp32_enforced)

    def test_crosscheck_cohort_parsing_and_submit_validation(self) -> None:
        spec = hb.parse_crowdgpt_task_crosscheck(
            {
                "taskId": "task-round-349",
                "globalStep": 348,
                "shardConfig": {"slot": 5},
                "crossCheck": {
                    "enabled": True,
                    "targetSeed": 987654,
                    "targetLocalSteps": 8,
                    "targetBatchSize": 2,
                    "targetAccumSteps": 4,
                    "targetAutocast": "fp32",
                },
            }
        )
        self.assertTrue(spec.cohort_mode)
        self.assertEqual(spec.effective_slot, 0)
        self.assertEqual(spec.target_seed, 987654)

        with self.assertRaises(ValueError):
            hb.build_crowdgpt_submit_metadata(
                spec=spec,
                mean_loss=3.05,
                last_step_loss=3.01,
                loss_std=0.04,
                loss_samples=8,
                actual_steps=7,  # wrong step count must be rejected before upload
                actual_batch_size=2,
                tokens_processed=32768,
            )

        submit_meta = hb.build_crowdgpt_submit_metadata(
            spec=spec,
            mean_loss=3.05,
            last_step_loss=3.01,
            loss_std=0.04,
            loss_samples=8,
            actual_steps=8,
            actual_batch_size=2,
            tokens_processed=32768,
        )
        self.assertTrue(submit_meta["crossCheckUsed"])
        self.assertEqual(submit_meta["actualSeed"], 987654)
        self.assertEqual(submit_meta["actualSteps"], 8)
        self.assertEqual(submit_meta["actualBatchSize"], 2)


class TestLossCheckServiceStdlib(unittest.TestCase):
    def test_loss_check_receipt_and_coordinator_end_to_end_without_gpu(self) -> None:
        heldout = [i % 97 for i in range(400)]
        base_bytes = b"\x01\x02" * 64
        cand_bytes = b"\x03\x04" * 64
        eval_key = b"t4-evaluator-secret"
        srv_secret = b"coordinator-backend-secret"

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "loss_check_state.json"
            coord = lcs.LossCheckCoordinator(
                evaluator_keys={"hivemind-t4": eval_key},
                server_secret=srv_secret,
                state_path=state_file,
                loader=lambda job: (base_bytes, cand_bytes, heldout),
                k=0,  # structural + HMAC + ratchet verification without PyTorch forward pass
                tol=2e-3,
                max_regression_tolerance=0.01,
                min_windows=8,
            )
            job = coord.open_job(
                base_url="https://huggingface.co/Vxtzq/Crowd-v1/resolve/step347/model.safetensors",
                base_sha256=lc.weights_sha256(base_bytes),
                candidate_url="https://huggingface.co/Vxtzq/Crowd-v1/resolve/step348/model.safetensors",
                candidate_sha256=lc.weights_sha256(cand_bytes),
                heldout_url="https://server.crowdgpt.net/heldout/val_0.bin",
                heldout_sha256=lc.tokens_sha256(heldout),
                window=16,
                n=8,
                job_id="round-348",
            )
            code, wire_job = coord.handle_get_job("hivemind-t4")
            self.assertEqual(code, 200)
            self.assertEqual(wire_job["job_id"], "round-348")

            starts = lc.heldout_windows(len(heldout), 16, 8, job.seed)
            receipt = lc.LossCheckReceipt(
                evaluator_id="hivemind-t4",
                base_sha256=job.base_sha256,
                candidate_sha256=job.candidate_sha256,
                heldout_sha256=job.heldout_sha256,
                window=16,
                seed=job.seed,
                starts=tuple(starts),
                losses_before=tuple(3.08 for _ in starts),
                losses_after=tuple(3.05 for _ in starts),
                compute_dtype="float32",
                device="cuda:0",
                round_id="round-348",
            ).signed(eval_key)

            status, body = coord.handle_post_receipt(receipt.to_json().encode("utf-8"))
            self.assertEqual(status, 200)
            self.assertTrue(body["recorded"])
            self.assertTrue(body["decision"]["accepted"])


if __name__ == "__main__":
    unittest.main()
