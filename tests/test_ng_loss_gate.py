"""Tests for the loss-gate contract (``POST {prefix}/gate``).

The gate is what makes ``LOSS_CHECK_MODE=enforce`` possible: the coordinator
posts the candidate weights it is about to accept and reads back a held-out
loss. These tests pin the wire format and every refusal path without needing a
GPU, a checkpoint or the network.
"""

from __future__ import annotations

import hashlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # the evaluator modules need numpy; the stdlib-only CI job does not have it
    from crowdgpt_ng import loss_gate as lg  # noqa: E402
    from crowdgpt_ng.eval import EvalError, EvalReport, uniform_logits  # noqa: E402
except ImportError:  # pragma: no cover - CI path
    lg = None  # type: ignore[assignment]
    EvalError = EvalReport = uniform_logits = None  # type: ignore[assignment]

_NEEDS_NUMPY = unittest.skipIf(lg is None, "numpy is required for the gate tests")

MODEL_SHA = "a" * 64


def _candidate(**overrides):
    body = {
        "step": 42,
        "round": 7,
        "previousAcceptedLoss": 2.75,
        "candidate": {
            "modelUrl": "https://coordinator.example/api/admin/candidate/model?token=abc",
            "engramUrl": "file:///tmp/engram.bin",
            "format": "bf16",
            "modelSize": 1000,
            "modelSha256": MODEL_SHA,
            "engramSha256": "b" * 64,
        },
    }
    body.update(overrides)
    return body


@_NEEDS_NUMPY
class ParseTests(unittest.TestCase):
    def test_full_request_is_normalised(self):
        parsed = lg.parse_gate_request(_candidate())
        self.assertEqual(parsed["step"], 42)
        self.assertEqual(parsed["round"], 7)
        self.assertEqual(parsed["previousAcceptedLoss"], 2.75)
        self.assertEqual(parsed["candidate"].model_sha256, MODEL_SHA)
        self.assertTrue(parsed["candidate"].engram_url.startswith("file://"))

    def test_first_measurement_has_no_previous_loss(self):
        parsed = lg.parse_gate_request(_candidate(previousAcceptedLoss=None))
        self.assertIsNone(parsed["previousAcceptedLoss"])

    def test_engram_is_optional(self):
        body = _candidate()
        del body["candidate"]["engramUrl"]
        parsed = lg.parse_gate_request(body)
        self.assertIsNone(parsed["candidate"].engram_url)

    def test_rejections_are_specific(self):
        cases = [
            ("not an object", "request body"),
            (_candidate(candidate="nope"), "candidate must be"),
            (_candidate(candidate={"engramUrl": "https://x/y"}),
             "candidate.modelUrl is required"),
            (_candidate(candidate={"modelUrl": "ftp://x/y"}), "http(s):// or file://"),
            (_candidate(candidate={"modelUrl": "https://x/" + "a" * 5000}), "longer than"),
            (_candidate(candidate={"modelUrl": "https://x/y", "modelSha256": "zz"}),
             "64-character hex"),
            (_candidate(candidate={"modelUrl": "https://x/y", "modelSha256": "g" * 64}), "must be hex"),
            (_candidate(step=True), "non-negative integer"),
            (_candidate(step=-1), "non-negative integer"),
            (_candidate(round="7"), "non-negative integer"),
            (_candidate(previousAcceptedLoss="2.7"), "finite number"),
            (_candidate(previousAcceptedLoss=float("nan")), "finite number"),
        ]
        for body, fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(lg.GateError) as ctx:
                    lg.parse_gate_request(body)
                self.assertIn(fragment, str(ctx.exception))


@_NEEDS_NUMPY
class EvaluateGateTests(unittest.TestCase):
    def test_missing_scorer_refuses_without_a_number(self):
        status, body = lg.evaluate_gate(_candidate(), None)
        self.assertEqual(status, 503)
        self.assertNotIn("loss", body)

    def test_invalid_request_is_400_and_never_scores(self):
        calls = []

        def scorer(request):
            calls.append(request)
            return {"loss": 1.0}

        status, body = lg.evaluate_gate({"step": 1}, scorer)
        self.assertEqual(status, 400)
        self.assertEqual(calls, [])

    def test_report_is_returned_with_the_request_echo(self):
        def scorer(request):
            return EvalReport(
                manifest_name="ng-eval-v1", manifest_sha256="c" * 64, model_id="m",
                windows=64, tokens=131072, loss="2.913456", ci_low="2.88", ci_high="2.95",
                seed=0, code_commit="deadbeef",
            )

        status, body = lg.evaluate_gate(_candidate(), scorer)
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["loss"], 2.913456, places=6)
        self.assertAlmostEqual(body["ciLow"], 2.88)
        self.assertEqual(body["windows"], 64)
        self.assertEqual(body["manifestSha256"], "c" * 64)
        self.assertEqual(body["source"], lg.GATE_SOURCE)
        self.assertEqual((body["step"], body["round"]), (42, 7))
        self.assertEqual(body["previousAcceptedLoss"], 2.75)

    def test_plain_mapping_is_accepted(self):
        status, body = lg.evaluate_gate(_candidate(), lambda request: {"loss": 2.5, "windows": 8})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["loss"], 2.5)
        self.assertEqual(body["source"], lg.GATE_SOURCE)

    def test_evaluator_errors_become_502(self):
        def raises_eval_error(request):
            raise EvalError("candidate digest mismatch")

        status, body = lg.evaluate_gate(_candidate(), raises_eval_error)
        self.assertEqual(status, 502)
        self.assertIn("digest mismatch", body["detail"])

        def raises_unknown(request):
            raise RuntimeError("token=secret-value leaked")

        status, body = lg.evaluate_gate(_candidate(), raises_unknown)
        self.assertEqual(status, 502)
        self.assertEqual(body["detail"], "RuntimeError")
        self.assertNotIn("secret-value", str(body))

    def test_bad_results_are_502(self):
        for result in ({"loss": "2.5"}, {"loss": float("inf")}, {}, "2.5"):
            with self.subTest(result=result):
                status, body = lg.evaluate_gate(_candidate(), lambda request, r=result: r)
                self.assertEqual(status, 502)
                self.assertNotIn("loss", body)


@_NEEDS_NUMPY
class FetchTests(unittest.TestCase):
    def _opener(self, payload: bytes):
        def open_url(url, timeout=0):
            return io.BytesIO(payload)
        return open_url

    def test_downloads_and_verifies_the_declared_digest(self):
        payload = b"\x00\x01" * 8
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = lg.Candidate(model_url="https://x/model", model_sha256=digest)
            paths = lg.fetch_candidate(candidate, Path(tmp), opener=self._opener(payload))
            self.assertEqual(Path(paths["model"]).read_bytes(), payload)
            self.assertNotIn("engram", paths)

    def test_digest_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = lg.Candidate(model_url="https://x/model", model_sha256="d" * 64)
            with self.assertRaises(EvalError):
                lg.fetch_candidate(candidate, Path(tmp), opener=self._opener(b"\x00\x01"))

    def test_empty_artifact_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EvalError):
                lg.fetch_candidate(lg.Candidate(model_url="https://x/model"), Path(tmp), opener=self._opener(b""))


@_NEEDS_NUMPY
class ScorerTests(unittest.TestCase):
    def _manifest(self, root: Path, tokens: int = 2048, seq_len: int = 32, stride: int = 16) -> Path:
        import numpy as np
        from crowdgpt_ng.eval import build_manifest

        shard = root / "shard-000.bin"
        (np.arange(tokens, dtype=np.uint32) % 64).tofile(shard)
        manifest = build_manifest(name="synthetic", version="1", shards=[shard], seq_len=seq_len, stride=stride,
                                  vocab_size=64, tokenizer="test", created_at="2026-11-15", base_dir=root)
        path = root / "manifest.json"
        path.write_text(__import__("json").dumps(manifest.to_dict()), encoding="utf-8")
        return path

    def test_gate_scores_a_candidate_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(root)
            seen = []

            def fetch(candidate, dest):
                seen.append(candidate.model_url)
                path = Path(dest) / "model.bin"
                path.write_bytes(b"\x00\x01")
                return {"model": path}

            scorer = lg.make_gate_scorer(manifest_path=manifest, data_root=root, count=4, seed=0,
                                         logits_for=lambda paths, man: uniform_logits(man.vocab_size),
                                         fetch=fetch)
            status, body = lg.evaluate_gate(_candidate(candidate={"modelUrl": "https://x/model"}), scorer)
            self.assertEqual(status, 200)
            self.assertAlmostEqual(body["loss"], math.log(64), places=6)
            self.assertEqual(body["windows"], 4)
            self.assertEqual(len(seen), 1)


def _fastapi_available() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_fastapi_available(), "FastAPI/httpx are needed for the route test")
@_NEEDS_NUMPY
class GateRouteTests(unittest.TestCase):
    def test_gate_route_answers_and_refuses(self):
        from fastapi.testclient import TestClient

        import tools.hivemind_eval_service as hes

        config = hes.ServiceConfig(
            state_path=Path(tempfile.mkdtemp()) / "state.json",
            evaluator_keys={"eval-1": b"k" * 32},
            server_secret=b"s" * 32,
        )
        scorer = lambda request: {"loss": 2.75, "windows": 8}  # noqa: E731
        client = TestClient(hes.create_app(config, gate_scorer=scorer))
        ok = client.post(f"{config.route_prefix}/gate", json=_candidate(candidate={"modelUrl": "https://x/m"}))
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["loss"], 2.75)
        refused = client.post(f"{config.route_prefix}/gate", json={"step": 1})
        self.assertEqual(refused.status_code, 400)
        missing = TestClient(hes.create_app(config, gate_scorer=None)).post(
            f"{config.route_prefix}/gate", json=_candidate(candidate={"modelUrl": "https://x/m"}))
        self.assertEqual(missing.status_code, 503)


if __name__ == "__main__":
    unittest.main()
