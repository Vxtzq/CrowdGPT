"""Client/server contract details: version header on uploads, chunk lookup errors, shared downloads.

Run:  python3 -m unittest discover -s tests -v
"""
import inspect
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

try:  # pragma: no cover - depends on the environment, not on the code
    import requests
    import torch  # noqa: F401  (client imports it)
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs requests and torch: {_missing}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client  # noqa: E402


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}


class VersionHeaderTest(unittest.TestCase):
    def test_round_headers_carry_client_version(self):
        src = inspect.getsource(client.run_single_round_wrapper)
        self.assertIn('h["X-Client-Version"] = CLIENT_VERSION', src)


class ChunkDiscoveryTest(unittest.TestCase):
    def _discover(self, head):
        ds = object.__new__(client.StreamingShardDataset)
        ds.repo_id = "org/data"
        with mock.patch.object(client.requests, "head", side_effect=head), \
             mock.patch.object(client.time, "sleep"):
            return ds._discover(7)

    def test_error_names_http_status(self):
        with self.assertLogs(client.log, level="WARNING"):
            with self.assertRaises(RuntimeError) as cm:
                self._discover(lambda *a, **k: _Resp(403))
        self.assertIn("No chunk 7", str(cm.exception))
        self.assertIn("HTTP 403", str(cm.exception))

    def test_error_names_network_failure(self):
        def boom(*a, **k):
            raise requests.exceptions.SSLError("certificate verify failed")
        with self.assertLogs(client.log, level="WARNING"):
            with self.assertRaises(RuntimeError) as cm:
                self._discover(boom)
        self.assertIn("certificate verify failed", str(cm.exception))

    def test_found_chunk_returns_size(self):
        self.assertEqual(self._discover(lambda *a, **k: _Resp(200, {"content-length": "1234"})), 1234)


class SharedDownloadTest(unittest.TestCase):
    def test_same_blob_downloads_one_at_a_time(self):
        """Two threads asking for one blob must not write the .part file together."""
        inside, overlap = [0], []
        gate = threading.Lock()

        class _Body:
            status_code = 200
            headers = {"content-length": "4"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                with gate:
                    inside[0] += 1
                    overlap.append(inside[0])
                yield b"ab"
                threading.Event().wait(0.05)
                yield b"cd"
                with gate:
                    inside[0] -= 1

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "blob.safetensors"
            stop = threading.Event()
            with mock.patch.object(client.requests, "get", side_effect=lambda *a, **k: _Body()):
                threads = [threading.Thread(target=client._download_file_with_progress,
                                            args=("http://x/blob", dest, lambda *a, **k: None, stop),
                                            kwargs={"expected_size": 4}) for _ in range(2)]
                for t in threads: t.start()
                for t in threads: t.join()
            self.assertEqual(dest.read_bytes(), b"abcd")
            self.assertFalse(dest.with_suffix(".safetensors.part").exists())
        self.assertEqual(max(overlap), 1)

    def test_cleanup_skips_partial_in_flight(self):
        src = inspect.getsource(client.fetch_weights_from_hf)
        self.assertIn("busy.locked()", src)


if __name__ == "__main__":
    unittest.main()
