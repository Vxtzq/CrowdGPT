"""Tests for the held-out evaluation set builder and the reproduction tool.

Both tools are shipped so a third party can re-derive the published loss. The
builder itself needs the HF hub and the Qwen3 tokenizer, so only its pure
selection helper is unit-tested here; the reproduction path is exercised
end to end against a synthetic manifest.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - CI without numpy
    np = None

try:  # needs numpy (crowdgpt_ng.eval); the stdlib-only CI job does not have it
    import build_eval_manifest
except ImportError:  # pragma: no cover - CI path
    build_eval_manifest = None


def _write_manifest(root: Path, *, tokens: int = 4096, seq_len: int = 64, stride: int = 32) -> Path:
    from crowdgpt_ng.eval import build_manifest

    shard = root / "eval" / "synthetic" / "shard-000.bin"
    shard.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rng.integers(0, 256, size=tokens, dtype=np.uint32).tofile(shard)
    manifest = build_manifest(
        name="synthetic",
        version="1",
        shards=[shard],
        seq_len=seq_len,
        stride=stride,
        vocab_size=256,
        tokenizer="test",
        created_at="2026-11-15",
        base_dir=root,
    )
    path = shard.parent / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


@unittest.skipIf(build_eval_manifest is None, "numpy is required for the eval set builder")
class SelectionTests(unittest.TestCase):
    def test_selection_spreads_across_the_queue(self):
        entries = [{"path": f"{i:04d}.txt", "bytes": 400_000} for i in range(400)]
        picked = build_eval_manifest._select(entries, 2_000_000)
        self.assertGreaterEqual(len(picked), 4)
        self.assertLess(len(picked), len(entries))
        self.assertEqual(picked[0], entries[0])
        gaps = {entries.index(b) - entries.index(a) for a, b in zip(picked, picked[1:])}
        self.assertEqual(len(gaps), 1, "the sample must use a single stride")

    def test_small_queue_is_taken_whole(self):
        entries = [{"path": f"{i}.txt", "bytes": 400_000} for i in range(3)]
        picked = build_eval_manifest._select(entries, 20_000_000)
        self.assertEqual(picked, entries)


@unittest.skipIf(np is None, "numpy is required for the eval set")
class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = _write_manifest(self.root)
        import reproduce_eval

        self.repro = reproduce_eval

    def tearDown(self):
        self._tmp.cleanup()

    def test_uniform_model_reproduces_the_known_loss(self):
        import math

        result = self.repro.run(self.manifest, model="uniform", count=8, seed=0, data_root=self.root,
                                device=None, weights=None, engram=None)
        self.assertTrue(result["matchesExpected"])
        self.assertAlmostEqual(float(result["loss"]), math.log(256), places=4)
        self.assertEqual(result["windows"], 8)
        self.assertEqual(result["tokens"], 8 * 64)
        self.assertEqual(result["manifest_sha256"], result["manifest_sha256"].lower())

    def test_tampered_shard_is_rejected_before_scoring(self):
        from crowdgpt_ng.eval import EvalError

        shard = self.root / "eval" / "synthetic" / "shard-000.bin"
        raw = bytearray(shard.read_bytes())
        raw[0] ^= 0xFF
        shard.write_bytes(bytes(raw))
        with self.assertRaises(EvalError):
            self.repro.run(self.manifest, model="uniform", count=1, seed=0, data_root=self.root,
                           device=None, weights=None, engram=None)

    def test_cli_prints_a_report(self):
        code = self.repro.main([
            "--manifest", str(self.manifest),
            "--model", "uniform",
            "--data-root", str(self.root),
            "--count", "4",
        ])
        self.assertEqual(code, 0)

    def test_missing_manifest_is_an_error_not_a_traceback(self):
        code = self.repro.main([
            "--manifest", str(self.root / "nope.json"),
            "--data-root", str(self.root),
        ])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
