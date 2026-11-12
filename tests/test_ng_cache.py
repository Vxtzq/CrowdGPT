"""Cache/resume helpers added by the NG client slice."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import test_coordinator_rejected as base
        except ImportError:
            from tests import test_coordinator_rejected as base
        cls.client = base.client

    def test_weight_cache_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.client, "CHECKPOINT_DIR", Path(tmp)):
                a = self.client._weight_blob_path("model", "a" * 64, 100, "rev1")
                b = self.client._weight_blob_path("model", "a" * 64, 100, "rev2")
                c = self.client._weight_blob_path("model", "b" * 64, 100, "rev1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_unit_checkpoint_is_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CROWDGPT_UNIT_CHECKPOINT_EVERY", None)
            cp = self.client.UnitCheckpoint.from_env("task/unsafe id")
        self.assertFalse(cp.enabled)
        self.assertNotIn("/", cp.path.name)

    def test_unit_checkpoint_path_is_task_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(self.client, "CHECKPOINT_DIR", Path(tmp)):
                cp = self.client.UnitCheckpoint("task-1", every=25)
        self.assertTrue(str(cp.path).endswith("task-1.pt"))
        self.assertEqual(cp.every, 25)


if __name__ == "__main__":
    unittest.main()
