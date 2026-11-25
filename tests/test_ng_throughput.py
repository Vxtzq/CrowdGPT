"""Tests for the throughput harness (``tools/bench_throughput.py``).

The harness drives ``client.py``'s production training step, so these tests
check the parts that must not drift: the projection arithmetic and one real
(but heavily trimmed) CPU run. The numbers from a trimmed run are a smoke test,
never a throughput claim -- ``config.trimmedLayers`` records the trim.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - the stdlib-only CI job has neither
    numpy = torch = None

_NEEDS_STACK = unittest.skipIf(torch is None or numpy is None, "torch and numpy are required")


def _total_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (AttributeError, ValueError, OSError):  # not POSIX: unknown, let the test run
        return None


# Even trimmed to one layer, the real model keeps the full vocab embedding and the
# engram table (~580M params): fp32 weights + grads + AdamW state need ~10 GB.
_MIN_RAM_GB = 16
_RAM_GB = _total_ram_gb()
_NEEDS_RAM = unittest.skipIf(
    _RAM_GB is not None and _RAM_GB < _MIN_RAM_GB,
    f"the real-model smoke run needs >= {_MIN_RAM_GB} GB RAM (have {_RAM_GB or 0:.1f} GB)",
)


@_NEEDS_STACK
class ProjectionTests(unittest.TestCase):
    def test_projection_matches_the_definition(self):
        import bench_throughput

        throughput = {"tokensPerSecond": 1000.0, "tokensPerGpuDay": 86_400_000}
        out = bench_throughput.project(throughput, round_steps=500, round_seq=2048, devices=2)
        self.assertEqual(out["tokensPerGpuDay"], 86_400_000)
        self.assertEqual(out["tokensPerDayAtThisSize"], 172_800_000)
        self.assertAlmostEqual(out["roundHours"], round(500 * 2048 / 1000.0 / 3600, 2))

    def test_projection_is_linear_in_devices(self):
        import bench_throughput

        base = bench_throughput.project({"tokensPerSecond": 10.0, "tokensPerGpuDay": 864_000},
                                        round_steps=10, round_seq=100, devices=1)
        more = bench_throughput.project({"tokensPerSecond": 10.0, "tokensPerGpuDay": 864_000},
                                        round_steps=10, round_seq=100, devices=4)
        self.assertEqual(more["tokensPerDayAtThisSize"], 4 * base["tokensPerDayAtThisSize"])


@_NEEDS_STACK
@_NEEDS_RAM
class SmokeTests(unittest.TestCase):
    def test_trimmed_cpu_run_emits_the_schema(self):
        import bench_throughput

        model, dev, backend, cfg = bench_throughput.build("real", seq=16, batch=1, layers=1, device="cpu")
        self.assertEqual(str(dev), "cpu")
        self.assertGreater(cfg["trimmedLayers"], 0)
        client = bench_throughput.load_client_module(REPO_ROOT)
        result = bench_throughput.run(model, client, dev, backend, cfg["seq"], cfg["batch"], 1, 0, 64)
        self.assertGreater(result["tokensPerSecond"], 0)
        self.assertEqual(result["tokens"], 16)
        self.assertIn("tokensPerGpuDay", result)


if __name__ == "__main__":
    unittest.main()
