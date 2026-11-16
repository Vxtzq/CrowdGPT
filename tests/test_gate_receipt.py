"""Tests for the signed gate receipt the evaluator adapter returns.

The coordinator side (``CrowdGPT-Server`` PR #20) refuses a held-out loss that arrives
without a verified receipt, so the adapter has to sign what it answers. These tests pin
the three things that can silently drift apart:

* the canonical bytes (sorted keys, no separators, UTF-8),
* the exact set of fields the coordinator re-checks (``requestId``, both candidate
  digests, the loss, ``step``/``round``, ``issuedAt``),
* the "absent key means unsigned" rule, so an unsigned deployment keeps working.

The verifier below is a faithful local mirror of the server's ``_loss_receipt_problem``;
the server repo is a separate checkout, so it cannot be imported from here.

    python -m unittest tests.test_gate_receipt -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "hivemind_eval_service.py"

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.exceptions import InvalidSignature
except Exception:  # pragma: no cover - the eval extras carry the wheel
    Ed25519PrivateKey = None


def _load_module():
    """Import the adapter from its path; dataclass needs the module registered first."""
    spec = importlib.util.spec_from_file_location("hivemind_eval_service_receipts", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(receipt):
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


CANDIDATE = {
    "modelUrl": "http://coordinator/api/admin/candidate/model",
    "modelSha256": "a" * 64,
    "engramUrl": "http://coordinator/api/admin/candidate/engram",
    "engramSha256": "b" * 64,
}
PAYLOAD = {"step": 41, "round": 12, "requestId": "00112233445566778899aabbccddeeff", "candidate": CANDIDATE}
ANSWER = {"loss": 5.125, "ciLow": 4.98, "ciHigh": 5.27, "windows": 8, "step": 41, "round": 12}
# 32-byte deterministic seed; a real deployment passes it as CROWDGPT_GATE_SIGNING_KEY.
SEED = bytes(range(32))


def _verify(receipt, signature, pubkey, *, step, rnd, request_id, cand, max_age=None, now=None):
    """Mirror of the coordinator's checks; returns the rejection reason or None."""
    if not isinstance(receipt, dict) or not receipt:
        return "receipt_missing"
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey or ""))
        key.verify(bytes.fromhex(signature or ""), _canonical(receipt))
    except (InvalidSignature, ValueError, TypeError):
        return "receipt_signature_invalid"
    if int(receipt.get("step", -1)) != int(step):
        return "receipt_step_mismatch"
    if int(receipt.get("round", -1)) != int(rnd):
        return "receipt_round_mismatch"
    if str(receipt.get("requestId") or "") != str(request_id):
        return "receipt_request_mismatch"
    for field in ("modelSha256", "engramSha256"):
        if str(receipt.get(field) or "") != str(cand.get(field) or ""):
            return "receipt_candidate_mismatch"
    loss = receipt.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        return "receipt_loss_invalid"
    if max_age:
        if abs(float(now if now is not None else receipt["issuedAt"]) - float(receipt["issuedAt"])) > max_age:
            return "receipt_stale"
    return None


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography is not installed")
class SigningKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_absent_key_means_unsigned(self):
        self.assertIsNone(self.mod.parse_gate_signing_key(""))
        self.assertIsNone(self.mod.parse_gate_signing_key("   "))

    def test_hex_prefix_is_accepted(self):
        self.assertEqual(self.mod.parse_gate_signing_key("hex:" + SEED.hex()), SEED)

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.mod.parse_gate_signing_key("00" * 16)

    def test_non_hex_is_rejected(self):
        with self.assertRaises(ValueError):
            self.mod.parse_gate_signing_key("z" * 64)

    def test_public_key_is_the_ed25519_public_key(self):
        expected = Ed25519PrivateKey.from_private_bytes(SEED).public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        self.assertEqual(self.mod.gate_public_key(SEED), expected)
        self.assertEqual(len(expected), 64)


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography is not installed")
class ReceiptBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_receipt_echoes_what_the_coordinator_sent(self):
        receipt = self.mod.build_gate_receipt(ANSWER, PAYLOAD, now=1700000000.0)
        self.assertEqual(receipt["step"], 41)
        self.assertEqual(receipt["round"], 12)
        self.assertEqual(receipt["requestId"], PAYLOAD["requestId"])
        self.assertEqual(receipt["modelSha256"], CANDIDATE["modelSha256"])
        self.assertEqual(receipt["engramSha256"], CANDIDATE["engramSha256"])
        self.assertEqual(receipt["loss"], 5.125)
        self.assertIsInstance(receipt["loss"], float)
        self.assertEqual(receipt["issuedAt"], 1700000000.0)

    def test_missing_candidate_digests_are_empty_strings(self):
        payload = {"step": 1, "round": 2, "requestId": "r", "candidate": {"modelUrl": "http://x/m"}}
        receipt = self.mod.build_gate_receipt({"loss": 1.0}, payload, now=1.0)
        self.assertEqual(receipt["modelSha256"], "")
        self.assertEqual(receipt["engramSha256"], "")

    def test_unsigned_shapes_are_not_signed(self):
        for body in ({"loss": "5.1"}, {"loss": True}, {"loss": float("nan")}, {"ciLow": 1.0}, None):
            with self.subTest(body=body):
                self.assertIsNone(self.mod.build_gate_receipt(body, PAYLOAD, now=1.0))

    def test_absent_key_leaves_the_answer_untouched(self):
        answer = self.mod.sign_gate_answer(ANSWER, PAYLOAD, None)
        self.assertEqual(answer, ANSWER)
        for key in ("receipt", "pubkey", "signature"):
            self.assertNotIn(key, answer)


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography is not installed")
class SignedAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def _signed(self, body=ANSWER, payload=PAYLOAD):
        return self.mod.sign_gate_answer(body, payload, SEED, now=1700000000.0)

    def test_a_signed_answer_verifies(self):
        answer = self._signed()
        self.assertEqual(len(answer["pubkey"]), 64)
        self.assertEqual(
            _verify(answer["receipt"], answer["signature"], answer["pubkey"],
                    step=41, rnd=12, request_id=PAYLOAD["requestId"], cand=CANDIDATE),
            None,
        )

    def test_the_receipt_loss_is_the_reported_loss(self):
        answer = self._signed()
        self.assertEqual(answer["receipt"]["loss"], float(answer["loss"]))

    def test_tampering_with_the_reported_loss_breaks_the_signature(self):
        answer = self._signed()
        answer["receipt"]["loss"] = 4.0
        self.assertEqual(
            _verify(answer["receipt"], answer["signature"], answer["pubkey"],
                    step=41, rnd=12, request_id=PAYLOAD["requestId"], cand=CANDIDATE),
            "receipt_signature_invalid",
        )

    def test_tampering_with_the_body_does_not_change_the_signed_receipt(self):
        answer = self._signed()
        answer["loss"] = 0.0  # a lying body, but the receipt still says 5.125
        self.assertEqual(answer["receipt"]["loss"], 5.125)
        self.assertNotEqual(answer["receipt"]["loss"], answer["loss"])

    def test_a_correctly_signed_receipt_for_another_request_is_rejected(self):
        other = dict(PAYLOAD, requestId="ff" * 16)
        answer = self.mod.sign_gate_answer(ANSWER, other, SEED, now=1700000000.0)
        self.assertEqual(
            _verify(answer["receipt"], answer["signature"], answer["pubkey"],
                    step=41, rnd=12, request_id=PAYLOAD["requestId"], cand=CANDIDATE),
            "receipt_request_mismatch",
        )

    def test_a_correctly_signed_receipt_for_other_weights_is_rejected(self):
        other = dict(PAYLOAD, candidate=dict(CANDIDATE, modelSha256="c" * 64))
        answer = self.mod.sign_gate_answer(ANSWER, other, SEED, now=1700000000.0)
        self.assertEqual(
            _verify(answer["receipt"], answer["signature"], answer["pubkey"],
                    step=41, rnd=12, request_id=PAYLOAD["requestId"], cand=CANDIDATE),
            "receipt_candidate_mismatch",
        )

    def test_an_expired_receipt_is_rejected(self):
        answer = self._signed()
        self.assertEqual(
            _verify(answer["receipt"], answer["signature"], answer["pubkey"],
                    step=41, rnd=12, request_id=PAYLOAD["requestId"], cand=CANDIDATE,
                    max_age=3600, now=1700003601.0),
            "receipt_stale",
        )


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography is not installed")
class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_no_key_builds_no_signer(self):
        gate = self.mod.GateConfig(manifest=Path("/nonexistent"), data_root=REPO)
        self.assertIsNone(self.mod.build_gate_signer(gate))

    def test_a_key_builds_a_signer_that_signs(self):
        gate = self.mod.GateConfig(manifest=Path("/nonexistent"), data_root=REPO, signing_key=SEED)
        signer = self.mod.build_gate_signer(gate)
        self.assertIsNotNone(signer)
        answer = signer(ANSWER, PAYLOAD)
        self.assertIn("receipt", answer)
        self.assertEqual(answer["pubkey"], self.mod.gate_public_key(SEED))

    def test_from_env_reads_the_key_and_the_summary_names_the_signer(self):
        gate = self.mod.GateConfig.from_env({"CROWDGPT_GATE_SIGNING_KEY": SEED.hex()})
        self.assertEqual(gate.signing_key, SEED)
        self.assertEqual(gate.summary()["receiptSigner"], self.mod.gate_public_key(SEED))

    def test_from_env_without_a_key_stays_unsigned(self):
        gate = self.mod.GateConfig.from_env({})
        self.assertEqual(gate.signing_key, b"")
        self.assertNotIn("receiptSigner", gate.summary())

    def test_the_gate_route_only_signs_what_a_coordinator_would_accept(self):
        source = TOOL.read_text()
        self.assertIn("if gate_signer is not None and status == 200", source)
        self.assertIn("body = gate_signer(body, payload)", source)
        self.assertIn("gate_signer=gate_signer", source)  # create_app forwards it


if __name__ == "__main__":
    unittest.main()
