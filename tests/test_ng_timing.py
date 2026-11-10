"""Tests for the opt-in client timing log (P0.5)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the stdlib-only client loader from the coordinator tests: it
        # stubs torch/numpy/requests, so this module runs in upstream CI.
        try:
            import test_coordinator_rejected as base
        except ImportError:
            from tests import test_coordinator_rejected as base
        cls.client = base.client

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="timing-out-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _rows(self, path: Path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_unset_is_inert(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CROWDGPT_TIMING_LOG", None)
            recorder = self.client.TimingRecorder.from_env()
            recorder.set_round(7)
            recorder.phase("queue", 1.25, bytes=10, tokens=20)
            recorder.finish(ok=True)
        self.assertEqual(list(self.tmp.iterdir()), [])
        self.assertFalse(recorder.enabled)

    def test_set_writes_one_row_per_phase(self):
        path = self.tmp / "nested" / "timing.jsonl"
        with mock.patch.dict(os.environ, {"CROWDGPT_TIMING_LOG": str(path)}):
            recorder = self.client.TimingRecorder.from_env()
            recorder.set_round(338)
            recorder.phase("queue", 0.5)
            recorder.phase("weights", 2.0, bytes=2_015_576_064)
            recorder.phase("train", 3.0, tokens=4_400_000)
            recorder.phase("upload", 4.0, bytes=1_386_000_000, tokens=4_400_000, ok=False)
            recorder.finish(ok=False, error="TimeoutError")
            recorder.finish(ok=True)

        rows = self._rows(path)
        self.assertEqual([row["phase"] for row in rows], ["queue", "weights", "train", "upload", "total"])
        self.assertTrue(all(row["schema"] == "ng-timing-1" for row in rows))
        self.assertTrue(all(row["round"] == 338 for row in rows))
        self.assertEqual(rows[1]["bytes"], 2_015_576_064)
        self.assertEqual(rows[2]["tokens"], 4_400_000)
        self.assertFalse(rows[3]["ok"])
        self.assertFalse(rows[4]["ok"])
        self.assertEqual(rows[4]["error"], "TimeoutError")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_rows_contain_no_identity_fields(self):
        path = self.tmp / "timing.jsonl"
        with mock.patch.dict(os.environ, {"CROWDGPT_TIMING_LOG": str(path)}):
            recorder = self.client.TimingRecorder.from_env()
            recorder.set_round(1)
            recorder.phase("queue", 0.1, bytes=1, tokens=2, ok=True)
            recorder.finish(ok=True)
        text = path.read_text(encoding="utf-8")
        for forbidden in ("10.0.0.", "Bearer", "Authorization", "secret", "http://", "https://"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(
            set(self._rows(path)[0]),
            {"schema", "ts", "round", "phase", "seconds", "bytes", "tokens", "ok"},
        )

    def test_write_failure_does_not_raise(self):
        blocker = self.tmp / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        recorder = self.client.TimingRecorder(path=blocker / "timing.jsonl", round_id=1)
        recorder.phase("queue", 0.1)
        recorder.finish(ok=False, error="OverflowError")
        self.assertTrue(recorder.enabled)


if __name__ == "__main__":
    unittest.main()
