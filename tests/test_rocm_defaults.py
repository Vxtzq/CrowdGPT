"""ROCm budget/attention defaults and the installer's wheel-index mapping.

Standard library only. The client is imported through the torch/numpy/requests stubs from
test_coordinator_rejected, so no GPU is needed; the values below are the ones measured on a
Radeon 890M (gfx1150, 24 cores, 30.48 GB shared RAM, ROCm 7.2, torch 2.14.1+rocm7.2).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import test_coordinator_rejected as base
except ImportError:
    from tests import test_coordinator_rejected as base

client = base.client
GiB = 1024 ** 3


def _fake_psutil(total_gb):
    mod = types.ModuleType("psutil")
    mod.virtual_memory = lambda: SimpleNamespace(total=int(total_gb * GiB), available=int(total_gb * GiB))
    return mod


def _cuda(total_gb, free_gb, integrated=False, arch="gfx1150"):
    return SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda idx=0: (int(free_gb * GiB), int(total_gb * GiB)),
        get_device_properties=lambda idx=0: SimpleNamespace(
            total_memory=int(total_gb * GiB), gcnArchName=arch, is_integrated=integrated),
    )


class UmaBudgetTests(unittest.TestCase):
    """An APU reports its whole shared pool as VRAM; the budget must leave the OS room."""

    def setUp(self):
        self._saved = (client.train_backend, client.train_device, dict(client.memory_config),
                       getattr(client.torch, "cuda", None))
        client.train_device = None
        self._env = mock.patch.dict("os.environ", {}, clear=False)
        self._env.start()
        mock.patch.dict("os.environ", {}, clear=False).start()

    def tearDown(self):
        self._env.stop()
        backend, device, cfg, cuda = self._saved
        client.train_backend = backend
        client.train_device = device
        client.memory_config.clear()
        client.memory_config.update(cfg)
        client.torch.cuda = cuda

    def _budget(self, total_gb, free_gb, ram_gb, integrated, arch="gfx1150"):
        client.train_backend = "ROCM"
        client.torch.cuda = _cuda(total_gb, free_gb, integrated=integrated, arch=arch)
        with mock.patch.dict(sys.modules, {"psutil": _fake_psutil(ram_gb)}):
            client.auto_detect_vram_budget()
        return client.memory_config["ram_gb"], client.memory_config.get("unified")

    def test_discrete_card_keeps_mem_get_info_budget(self):
        budget, unified = self._budget(24.0, 23.5, 64.0, integrated=False, arch="gfx1100")
        self.assertFalse(unified)
        self.assertAlmostEqual(budget, 23.5 - client.SAFETY_MARGIN_GB - client.ALLOCATOR_RESERVE_GB, places=2)

    def test_apu_budget_is_capped_by_system_ram(self):
        # Measured shape: 32 GB pool, 31.84 GB free, 30.48 GB RAM.
        budget, unified = self._budget(32.0, 31.84, 30.48, integrated=True)
        self.assertTrue(unified)
        ceiling = 30.48 * client.UMA_VRAM_SHARE_DEFAULT - client.SAFETY_MARGIN_GB
        self.assertAlmostEqual(budget, ceiling, places=2)
        self.assertLess(budget, 30.34)  # the uncapped value that produced batch 16

    def test_apu_share_is_configurable_and_clamped(self):
        with mock.patch.dict("os.environ", {"CROWDGPT_UMA_VRAM_SHARE": "0.5"}):
            budget, _ = self._budget(32.0, 31.84, 30.48, integrated=True)
        self.assertAlmostEqual(budget, 30.48 * 0.5 - client.SAFETY_MARGIN_GB, places=2)
        with mock.patch.dict("os.environ", {"CROWDGPT_UMA_VRAM_SHARE": "0.1"}):
            clamped, _ = self._budget(32.0, 31.84, 30.48, integrated=True)
        self.assertAlmostEqual(clamped, 30.48 * 0.4 - client.SAFETY_MARGIN_GB, places=2)

    def test_integrated_flag_is_optional(self):
        """Older torch builds lack is_integrated; a gfx115x arch still counts as unified memory."""
        client.train_backend = "ROCM"
        client.torch.cuda = _cuda(32.0, 31.84, integrated=False, arch="gfx1150")
        with mock.patch.dict(sys.modules, {"psutil": _fake_psutil(30.48)}):
            ceiling = client._uma_ram_ceiling_gb(0, int(32.0 * GiB))
        self.assertIsNotNone(ceiling)
        self.assertAlmostEqual(ceiling, 30.48 * client.UMA_VRAM_SHARE_DEFAULT, places=2)

    def test_gfx115_apu_without_the_integrated_flag_is_still_capped(self):
        """gfx1150/gfx1151 are APUs even when torch omits is_integrated."""
        budget, unified = self._budget(32.0, 31.84, 30.48, integrated=False, arch="gfx1151")
        self.assertTrue(unified)
        self.assertAlmostEqual(
            budget, 30.48 * client.UMA_VRAM_SHARE_DEFAULT - client.SAFETY_MARGIN_GB, places=2
        )

    def test_no_psutil_means_no_ceiling(self):
        with mock.patch.dict(sys.modules, {"psutil": None}):
            client.torch.cuda = _cuda(32.0, 31.84, integrated=True)
            self.assertIsNone(client._uma_ram_ceiling_gb(0, int(32.0 * GiB)))


class RocmSdpaDefaultTests(unittest.TestCase):
    """gfx115x has no enabled flash/mem-efficient backend, and chunked attention is faster."""

    def setUp(self):
        self._saved = (client._ROCM_SDPA_DEFAULT, getattr(client.torch, "cuda", None))

    def tearDown(self):
        client._ROCM_SDPA_DEFAULT, client.torch.cuda = self._saved

    def _default(self, arch):
        client._ROCM_SDPA_DEFAULT = None
        client.torch.cuda = _cuda(32.0, 31.0, arch=arch)
        return client._rocm_sdpa_default()

    def test_apu_targets_prefer_the_chunked_kernel(self):
        for arch in ("gfx1150", "gfx1150:sramecc+:xnack-", "gfx1151"):
            self.assertEqual(self._default(arch), "0", arch)

    def test_other_rocm_targets_keep_sdpa(self):
        for arch in ("gfx906", "gfx1100", "gfx942"):
            self.assertEqual(self._default(arch), "1", arch)

    def test_unsupported_arch_string_falls_back_to_sdpa(self):
        client._ROCM_SDPA_DEFAULT = None
        client.torch.cuda = SimpleNamespace(get_device_properties=lambda idx=0: (_ for _ in ()).throw(RuntimeError("no device")))
        self.assertEqual(client._rocm_sdpa_default(), "1")

    def test_value_is_cached_after_the_first_probe(self):
        self.assertEqual(self._default("gfx1150"), "0")
        client.torch.cuda = _cuda(32.0, 31.0, arch="gfx1100")
        self.assertEqual(client._rocm_sdpa_default(), "0")


class InstallerRocmIndexTests(unittest.TestCase):
    """The installer must not pin a ROCm 7.2 box to the rocm7.0 wheel index."""

    def setUp(self):
        try:
            import installer
        except Exception as exc:  # pragma: no cover - installer is stdlib-only
            self.skipTest(f"installer unavailable: {exc}")
        self.installer = installer

    def test_rocm7_minors_map_to_their_index(self):
        idx = self.installer._rocm_torch_index
        self.assertTrue(idx("7.2.53211").endswith("/rocm7.2"))
        self.assertTrue(idx("7.1.52801").endswith("/rocm7.1"))
        self.assertTrue(idx("7.0.51831").endswith("/rocm7.0"))

    def test_newer_than_published_clamps_to_newest(self):
        self.assertTrue(self.installer._rocm_torch_index("7.9.1234").endswith("/rocm7.2"))

    def test_rocm6_minors_keep_the_old_mapping(self):
        idx = self.installer._rocm_torch_index
        self.assertTrue(idx("6.4.1").endswith("/rocm6.3"))
        self.assertTrue(idx("6.3.0").endswith("/rocm6.3"))
        self.assertTrue(idx("6.2.4").endswith("/rocm6.2"))
        self.assertTrue(idx("6.1.0").endswith("/rocm6.1"))
        self.assertTrue(idx("5.7.1").endswith("/rocm5.7"))

    def test_unknown_version_uses_the_conservative_index(self):
        self.assertTrue(self.installer._rocm_torch_index("").endswith("/rocm6.1"))
        self.assertTrue(self.installer._rocm_torch_index("garbage").endswith("/rocm6.1"))


class RecommendBatchSizeTests(unittest.TestCase):
    """The ROCm VRAM fraction must not be applied twice on unified memory."""

    def setUp(self):
        self._saved = (client.train_backend, client.train_device, dict(client.memory_config),
                       getattr(client.torch, "cuda", None))
        client.train_device = None
        self._env = mock.patch.dict("os.environ", {
            "CROWDGPT_ROCM_VRAM_FRACTION": "0.85",
            "CROWDGPT_UMA_VRAM_SHARE": "",
            "CROWDGPT_ROCM_BATCH": "",
        })
        self._env.start()

    def tearDown(self):
        self._env.stop()
        backend, device, cfg, cuda = self._saved
        client.train_backend = backend
        client.train_device = device
        client.memory_config.clear()
        client.memory_config.update(cfg)
        client.torch.cuda = cuda

    def _recommend(self, total_gb, free_gb, ram_gb, integrated, sl, arch="gfx1150"):
        client.train_backend = "ROCM"
        client.torch.cuda = _cuda(total_gb, free_gb, integrated=integrated, arch=arch)
        with mock.patch.dict(sys.modules, {"psutil": _fake_psutil(ram_gb)}):
            client.auto_detect_vram_budget()
        return client.recommend_batch_size(sl, force_mode="adamw32")

    def test_unified_memory_is_not_discounted_twice(self):
        # 30.48 GB RAM * 0.75 - 1.0 margin = 21.86 GB budget. The batch-32
        # estimate at seq 512 is 18.77 GB; the old 0.85 factor cut the budget to
        # 18.58 GB and rejected batch 32 by 0.19 GB, even though it measured a
        # 13.17 GB peak on gfx1150.
        best = self._recommend(32.0, 31.84, 30.48, True, 512)
        self.assertIs(client.memory_config.get("unified"), True)
        self.assertEqual(best, 32)
    def test_unified_memory_reaches_batch_16_at_max_seq(self):
        # Seq 2048 estimate for batch 16 is 20.30 GB: inside 21.86, outside the
        # doubly-discounted 18.58.
        self.assertEqual(self._recommend(32.0, 31.84, 30.48, True, 2048), 16)

    def test_discrete_card_still_applies_the_vram_fraction(self):
        # 24 GB card, 23.5 GB free, 64 GB RAM: budget 22.0 GB, fraction -> 18.7,
        # so the 18.77 GB batch-32 estimate stays rejected. Guards against the
        # fix loosening discrete cards, where mem_get_info is the only signal.
        self.assertEqual(self._recommend(24.0, 23.5, 64.0, False, 512, arch="gfx1100"), 16)

    def test_rocm_batch_cap_still_wins(self):
        client.train_backend = "ROCM"
        client.torch.cuda = _cuda(32.0, 31.84, integrated=True)
        with mock.patch.dict(sys.modules, {"psutil": _fake_psutil(30.48)}), \
                mock.patch.dict("os.environ", {"CROWDGPT_ROCM_BATCH": "2"}):
            client.auto_detect_vram_budget()
            self.assertEqual(client.recommend_batch_size(512, force_mode="adamw32"), 2)


if __name__ == "__main__":
    unittest.main()
