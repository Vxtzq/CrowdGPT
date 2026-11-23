"""Apple Silicon memory handling: budget, hard cap, cache release and heap trim.

Run:  python3 -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

try:  # pragma: no cover - depends on the environment, not on the code
    import torch
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs torch: {_missing}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import client  # noqa: E402

GIB = 1024**3


class MpsBudgetTest(unittest.TestCase):
    def _budget(self, total_gib, recommended_gib, env=None):
        with mock.patch.object(client, "train_backend", "MPS"), \
             mock.patch.object(client.subprocess, "check_output", return_value=str(total_gib * GIB).encode()), \
             mock.patch.object(torch.mps, "recommended_max_memory", return_value=int(recommended_gib * GIB)), \
             mock.patch.dict(client.os.environ, env or {}, clear=False), \
             mock.patch.dict(client.memory_config, {}, clear=False):
            client.auto_detect_vram_budget()
            return client.memory_config["ram_gb"]

    def test_budget_capped_by_metal_working_set(self):
        # 48 GB Mac: 75% share is 36 GB; Metal's working set (30 GB here) is lower and wins.
        self.assertAlmostEqual(self._budget(48, 30), round(30 - client.SAFETY_MARGIN_GB, 2))

    def test_budget_honours_uma_share(self):
        got = self._budget(48, 40, {"CROWDGPT_UMA_VRAM_SHARE": "0.5"})
        self.assertAlmostEqual(got, round(24 - client.SAFETY_MARGIN_GB, 2))


class MpsCapTest(unittest.TestCase):
    def test_cap_sets_fraction_to_working_set(self):
        with mock.patch.object(torch.mps, "set_per_process_memory_fraction") as setf, \
             mock.patch.object(torch.mps, "recommended_max_memory", return_value=30 * GIB):
            cap = client.install_hard_vram_cap(torch.device("mps"))
        setf.assert_called_once_with(1.0)
        self.assertEqual(cap, 30.0)

    def test_cpu_has_no_cap(self):
        self.assertIsNone(client.install_hard_vram_cap(torch.device("cpu")))


class CacheReleaseTest(unittest.TestCase):
    def test_clear_cache_empties_mps_pool(self):
        with mock.patch.object(torch.mps, "empty_cache") as empty:
            client._clear_accel_cache(torch.device("mps"))
        empty.assert_called_once_with()

    def test_trim_heap_does_not_raise(self):
        client._trim_heap()


if __name__ == "__main__":
    unittest.main()
