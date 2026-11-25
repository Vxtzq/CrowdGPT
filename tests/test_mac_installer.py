"""Mac install and first-run checks: Rosetta detection, Command Line Tools, TLS-proxy hint.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

try:
    import test_coordinator_rejected as base
except ImportError:
    from tests import test_coordinator_rejected as base

import installer

client = base.client
INSTALL_SH = (Path(__file__).resolve().parents[1] / "install.sh").read_text()


def _mac(arch):
    api = object.__new__(installer.InstallerAPI)
    api.os_name, api.arch, api.backend_override = "Darwin", arch, ""
    api.backend = api.torch_index = None
    api.wheel_note = ""
    return api


class RosettaTest(unittest.TestCase):
    def test_x86_python_on_apple_silicon_stops_with_reason(self):
        api = _mac("x86_64")
        with mock.patch.object(installer, "run_capture", return_value=(0, "1\n")):
            with self.assertRaisesRegex(RuntimeError, "Rosetta"):
                api.detect_backend()
        self.assertIsNone(api.backend)

    def test_intel_mac_still_gets_cpu(self):
        api = _mac("x86_64")
        with mock.patch.object(installer, "run_capture", return_value=(1, "unknown oid")):
            api.detect_backend()
        self.assertEqual(api.backend, "cpu")

    def test_native_apple_silicon_gets_mps(self):
        api = _mac("arm64")
        with mock.patch.object(installer, "run_capture", return_value=(0, "1\n")):
            api.detect_backend()
        self.assertEqual(api.backend, "apple")


class InstallScriptTest(unittest.TestCase):
    def test_no_crlf_self_repair(self):
        # bash cannot parse a CRLF copy far enough to reach a repair block, and BSD sed -i needs a suffix.
        self.assertNotIn("sed -i 's/\\r$//'", INSTALL_SH)

    def test_command_line_tools_checked_before_git(self):
        self.assertLess(INSTALL_SH.index("xcode-select -p"), INSTALL_SH.index("command -v git"))


class DoctorTlsTest(unittest.TestCase):
    def test_certificate_failure_is_marked_tls_with_hint(self):
        err = client.requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")
        with mock.patch.object(client.requests, "get", side_effect=err, create=True):
            probe = client._doctor_http_probe("https://huggingface.co/api/models/x")
        self.assertTrue(probe["tls"])
        self.assertIn("REQUESTS_CA_BUNDLE", probe["error"])

    def test_other_failures_are_not_tls(self):
        with mock.patch.object(client.requests, "get", side_effect=OSError("down"), create=True):
            probe = client._doctor_http_probe("https://huggingface.co/api/models/x")
        self.assertNotIn("tls", probe)


if __name__ == "__main__":
    unittest.main()
