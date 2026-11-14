"""Round-certificate verification in the client protocol."""

from __future__ import annotations

import unittest

try:
    import cryptography  # noqa: F401
except Exception:  # pragma: no cover
    cryptography = None

from crowdgpt_ng import protocol


@unittest.skipIf(cryptography is None, "cryptography not installed")
class CertificateTests(unittest.TestCase):
    def test_round_certificate_verifies_and_tamper_fails(self):
        private, public = protocol.generate_keypair()
        record = {"round": 7, "leader": "node-a", "aggregate_weights": "a" * 64}
        envelope = protocol.sign_payload(record, private, "node-a")
        cert = {"node": "node-a", "pubkey": public, "sig": envelope["sig"]}
        self.assertTrue(protocol.verify_round_certificate(record, cert))
        tampered = {**record, "aggregate_weights": "b" * 64}
        self.assertFalse(protocol.verify_round_certificate(tampered, cert))

    def test_malformed_certificate_is_rejected(self):
        self.assertFalse(protocol.verify_round_certificate({"round": 1}, {"pubkey": "short", "sig": "short"}))


if __name__ == "__main__":
    unittest.main()
