"""Stdlib-compatible tests for the additive HiveMind evaluation adapter.

They run under ``python -m unittest discover -s tests`` with no torch, numpy, FastAPI,
network, or 658M checkpoint. The fake loader and a k=0 coordinator exercise the full
job/receipt handler path; the FastAPI route tests skip when the optional stack is absent.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hivemind_loss_check as lc  # noqa: E402
import hivemind_loss_check_service as lcs  # noqa: E402
from tools import hivemind_eval_service as hes  # noqa: E402

KEY_A = bytes.fromhex("11" * 32)
KEY_B = bytes.fromhex("22" * 32)
SECRET = "unit-test-server-secret-0123456789"
BASE = b"\x01\x02\x03\x04"
CANDIDATE = b"\x05\x06\x07\x08"
TOKENS = list(range(64))
WINDOW = 4
N_WINDOWS = 2

try:
    from fastapi.testclient import TestClient

    HAVE_FASTAPI = True
except Exception:  # pragma: no cover - FastAPI/httpx are optional in CI
    TestClient = None  # type: ignore[assignment]
    HAVE_FASTAPI = False

try:
    import numpy as _np  # noqa: F401

    HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is optional in CI
    HAVE_NUMPY = False


def env(**overrides: str | None) -> dict[str, str]:
    values: dict[str, str] = {
        "CROWDGPT_HIVEMIND_EVALUATOR_KEYS": json.dumps({"eval-a": KEY_A.hex()}),
        "CROWDGPT_HIVEMIND_SERVER_SECRET": SECRET,
    }
    values.update(overrides)
    return {k: v for k, v in values.items() if v is not None}


def fake_model_factory() -> object:
    return object()


def config_for(state_path: Path) -> hes.ServiceConfig:
    return hes.ServiceConfig(
        state_path=state_path,
        evaluator_keys={"eval-a": KEY_A, "eval-b": KEY_B},
        server_secret=SECRET.encode(),
        model_factory=None,
    )


class RecordingLoader:
    def __init__(self, base: object = BASE, candidate: object = CANDIDATE, heldout: object = TOKENS) -> None:
        self.base, self.candidate, self.heldout = base, candidate, heldout
        self.calls: list[str] = []

    def __call__(self, job: lcs.LossCheckJob) -> tuple[object, object, object]:
        self.calls.append(job.job_id)
        return self.base, self.candidate, self.heldout


def make_coordinator(tmp: Path, loader: RecordingLoader, **overrides: object) -> lcs.LossCheckCoordinator:
    options: dict[str, object] = {
        "k": 0,
        "min_windows": N_WINDOWS,
        "required_receipts": 1,
        "max_regression_tolerance": 0.0,
        "job_ttl_s": 3600.0,
    }
    options.update(overrides)
    return hes.build_coordinator(config_for(tmp / "state.json"), loader, **options)


def open_test_job(coord: lcs.LossCheckCoordinator, job_id: str = "job-1") -> lcs.LossCheckJob:
    return coord.open_job(
        base_url="mem:base",
        base_sha256=lc.weights_sha256(BASE),
        candidate_url="mem:candidate",
        candidate_sha256=lc.weights_sha256(CANDIDATE),
        heldout_url="mem:heldout",
        heldout_sha256=lc.tokens_sha256(TOKENS),
        window=WINDOW,
        n=N_WINDOWS,
        job_id=job_id,
    )


def receipt_for(job: lcs.LossCheckJob, *, evaluator_id: str = "eval-a", key: bytes = KEY_A,
                before: float = 3.0, after: float = 2.9) -> lc.LossCheckReceipt:
    starts = tuple(lc.heldout_windows(len(TOKENS), job.window, job.n, job.seed))
    receipt = lc.LossCheckReceipt(
        evaluator_id=evaluator_id,
        base_sha256=job.base_sha256,
        candidate_sha256=job.candidate_sha256,
        heldout_sha256=job.heldout_sha256,
        window=job.window,
        seed=job.seed,
        starts=starts,
        losses_before=(before,) * len(starts),
        losses_after=(after,) * len(starts),
        compute_dtype="float32",
        device="cpu",
        round_id=job.job_id,
    )
    return receipt.signed(key)


class TestConfigParsing(unittest.TestCase):
    def test_json_evaluator_keys(self) -> None:
        keys = hes.parse_evaluator_keys(json.dumps({"eval-a": KEY_A.hex(), "eval-b": KEY_B.hex()}))
        self.assertEqual(keys, {"eval-a": KEY_A, "eval-b": KEY_B})

    def test_kv_evaluator_keys_accept_colon_or_equals(self) -> None:
        keys = hes.parse_evaluator_keys(f"eval-a:{KEY_A.hex()}, eval-b={KEY_B.hex()}")
        self.assertEqual(keys, {"eval-a": KEY_A, "eval-b": KEY_B})

    def test_evaluator_key_errors(self) -> None:
        for raw in ("", "{not json", "[]", "eval-a", "eval-a:zz", f"eval-a:{KEY_A.hex()},eval-a:{KEY_B.hex()}"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    hes.parse_evaluator_keys(raw)

    def test_server_secret_forms(self) -> None:
        self.assertEqual(hes.parse_server_secret("hex:" + KEY_A.hex()), KEY_A)
        self.assertEqual(hes.parse_server_secret(KEY_A.hex()), KEY_A)
        self.assertEqual(hes.parse_server_secret(SECRET), SECRET.encode())
        with self.assertRaises(ValueError):
            hes.parse_server_secret("too-short")
        with self.assertRaises(ValueError):
            hes.parse_server_secret("hex:zz")

    def test_from_env_required_and_defaults(self) -> None:
        config = hes.ServiceConfig.from_env(env())
        self.assertEqual(config.evaluator_keys, {"eval-a": KEY_A})
        self.assertEqual(config.server_secret, SECRET.encode())
        self.assertEqual(config.state_path, hes.DEFAULT_STATE_PATH)
        self.assertEqual(config.route_prefix, hes.DEFAULT_ROUTE_PREFIX)
        with self.assertRaises(ValueError):
            hes.ServiceConfig.from_env(env(CROWDGPT_HIVEMIND_EVALUATOR_KEYS=None))
        with self.assertRaises(ValueError):
            hes.ServiceConfig.from_env(env(CROWDGPT_HIVEMIND_SERVER_SECRET=None))

    def test_from_env_state_path_prefix_and_factory(self) -> None:
        config = hes.ServiceConfig.from_env(
            env(
                CROWDGPT_HIVEMIND_STATE_PATH="/tmp/eval-state.json",
                CROWDGPT_HIVEMIND_ROUTE_PREFIX="/custom/loss/",
                CROWDGPT_HIVEMIND_MODEL_FACTORY=f"{__name__}:fake_model_factory",
            )
        )
        self.assertEqual(config.state_path, Path("/tmp/eval-state.json"))
        self.assertEqual(config.route_prefix, "/custom/loss")
        self.assertIs(config.model_factory, fake_model_factory)

    def test_from_env_rejects_bad_factory(self) -> None:
        for spec in ("missing-colon", "no_such_module_xyz:factory", f"{__name__}:not_callable"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    hes.ServiceConfig.from_env(env(CROWDGPT_HIVEMIND_MODEL_FACTORY=spec))

    def test_invalid_route_prefix(self) -> None:
        with self.assertRaises(ValueError):
            hes.ServiceConfig(state_path=Path("s.json"), evaluator_keys={"a": KEY_A}, server_secret=KEY_A, route_prefix="no-slash")

    def test_create_router_requires_fastapi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = config_for(Path(td) / "state.json")
            with mock.patch.object(hes, "APIRouter", None):
                with self.assertRaises(RuntimeError):
                    hes.create_router(config, RecordingLoader())

    def test_safe_summary_masks_secrets(self) -> None:
        config = config_for(Path("/tmp/state.json"))
        summary = config.safe_summary()
        rendered = json.dumps(summary) + repr(config)
        self.assertNotIn(SECRET, rendered)
        self.assertNotIn(KEY_A.hex(), rendered)
        self.assertEqual(summary["evaluator_ids"], ["eval-a", "eval-b"])
        self.assertEqual(summary["server_secret_bytes"], len(SECRET))

    def test_main_print_config(self) -> None:
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env(), clear=True):
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(hes.main(["--print-config"]), 0)
        output = buffer.getvalue()
        self.assertNotIn(SECRET, output)
        self.assertNotIn(KEY_A.hex(), output)
        self.assertEqual(json.loads(output)["evaluator_ids"], ["eval-a"])

    def test_main_without_action_does_not_require_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hes.main([]), 2)


class TestLocalArtifacts(unittest.TestCase):
    def test_resolve_local_path(self) -> None:
        self.assertEqual(hes.resolve_local_path("/tmp/a b"), Path("/tmp/a b"))
        self.assertEqual(hes.resolve_local_path("file:///tmp/a%20b"), Path("/tmp/a b"))
        self.assertEqual(hes.resolve_local_path("file://localhost/tmp/x"), Path("/tmp/x"))
        for bad in ("", "https://example.test/x", "ftp://example.test/x", "file://otherhost/tmp/x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    hes.resolve_local_path(bad)

    def test_read_json_tokens_and_reject_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokens.json"
            path.write_text(json.dumps([1, 2, 3]))
            self.assertEqual(hes.read_heldout_tokens(path), [1, 2, 3])
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path, "0-3")
            path.write_text(json.dumps([1, -2]))
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path)

    def test_read_raw_u32_tokens_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokens.bin"
            path.write_bytes(struct.pack("<4I", 1, 2, 3, 4))
            self.assertEqual(list(hes.read_heldout_tokens(path)), [1, 2, 3, 4])
            self.assertEqual(list(hes.read_heldout_tokens(path, "0-3")), [1])
            self.assertEqual(list(hes.read_heldout_tokens(path, "4-7")), [2])
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path, "0-2")
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path, "0-99")

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_read_npy_tokens(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokens.npy"
            np.save(path, np.array([5, 6, 7], dtype=np.int64))
            tokens = hes.read_heldout_tokens(path)
            self.assertEqual(tokens.dtype, np.uint32)
            self.assertEqual(tokens.tolist(), [5, 6, 7])
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path, "0-3")
            np.save(path, np.array([1.5, 2.5], dtype=np.float32))
            with self.assertRaises(ValueError):
                hes.read_heldout_tokens(path)

    def test_file_loader_round_trip_and_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base, candidate, held = root / "base.safetensors", root / "candidate.safetensors", root / "tokens.json"
            for path in (base, candidate):
                path.write_bytes(b"placeholder")
            held.write_text(json.dumps(TOKENS))
            blobs = {"base": BASE, "candidate": CANDIDATE}
            loader = hes.FileJobLoader(weights_loader=lambda p: blobs[p.stem])
            job = lcs.LossCheckJob(
                job_id="j1", base_url=str(base), base_sha256=lc.weights_sha256(BASE),
                candidate_url=f"file://{candidate}", candidate_sha256=lc.weights_sha256(CANDIDATE),
                heldout_url=str(held), heldout_sha256=lc.tokens_sha256(TOKENS),
                window=WINDOW, n=N_WINDOWS, seed=1, created_at=0.0, expires_at=1.0,
            )
            self.assertEqual(loader(job), (BASE, CANDIDATE, TOKENS))
            broken = lcs.LossCheckJob(**{**job.__dict__, "base_sha256": "0" * 64})
            with self.assertRaises(ValueError):
                loader(broken)
            relaxed = hes.FileJobLoader(weights_loader=lambda p: blobs[p.stem], verify_hashes=False)
            self.assertEqual(relaxed(broken)[0], BASE)


class TestFakeLoaderHandler(unittest.TestCase):
    def test_get_job_and_receipt_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = RecordingLoader()
            coord = make_coordinator(Path(td), loader)
            status, body = coord.handle_get_job("eval-a")
            self.assertEqual((status, body), (204, None))
            self.assertEqual(coord.handle_get_job("unknown")[0], 204)

            job = open_test_job(coord)
            status, body = coord.handle_get_job("eval-a")
            self.assertEqual(status, 200)
            assert body is not None
            self.assertEqual(body["job_id"], job.job_id)
            self.assertEqual(coord.handle_get_job("eval-b")[0], 200)

            received = receipt_for(job)
            status, body = coord.handle_post_receipt(received.to_json())
            self.assertEqual(status, 200)
            self.assertTrue(body["recorded"])
            self.assertTrue(body["decision"]["accepted"])
            self.assertEqual(body["decision"]["evaluators"], ["eval-a"])
            self.assertEqual(loader.calls, [job.job_id])
            self.assertEqual(coord.decision(job.job_id)["accepted"], True)

            self.assertEqual(coord.handle_get_job("eval-a")[0], 204)

    def test_bad_signature_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loader = RecordingLoader()
            coord = make_coordinator(Path(td), loader)
            job = open_test_job(coord)
            forged = receipt_for(job, evaluator_id="eval-a", key=KEY_B)
            status, body = coord.handle_post_receipt(forged.to_json())
            self.assertEqual(status, 200)
            self.assertFalse(body["recorded"])
            self.assertEqual(body["verdict"]["reason"], "REJECTED_BAD_SIGNATURE")
            self.assertEqual(loader.calls, [])

    def test_unknown_job_and_tampered_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            coord = make_coordinator(Path(td), RecordingLoader())
            job = open_test_job(coord)
            missing = receipt_for(job)
            missing = lc.LossCheckReceipt(**{**missing.__dict__, "round_id": "missing"})
            status, body = coord.handle_post_receipt(missing.signed(KEY_A).to_json())
            self.assertEqual(status, 404)
            self.assertIn("unknown job", body["error"])

            receipt = receipt_for(job)
            payload = json.loads(receipt.to_json())
            payload["losses_after"] = [1.0, 1.0]
            status, body = coord.handle_post_receipt(json.dumps(payload))
            self.assertEqual(status, 200)
            self.assertFalse(body["recorded"])
            self.assertEqual(body["verdict"]["reason"], "REJECTED_BAD_SIGNATURE")

    def test_open_job_validation_and_override_plumbing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            coord = make_coordinator(Path(td), RecordingLoader())
            self.assertEqual(coord.path, Path(td) / "state.json")
            self.assertEqual(coord.keys, {"eval-a": KEY_A, "eval-b": KEY_B})
            self.assertEqual(coord.k, 0)
            self.assertEqual(coord.required, 1)
            with self.assertRaises(ValueError):
                coord.open_job(
                    base_url="mem:base", base_sha256=lc.weights_sha256(BASE),
                    candidate_url="mem:candidate", candidate_sha256=lc.weights_sha256(CANDIDATE),
                    heldout_url="mem:heldout", heldout_sha256=lc.tokens_sha256(TOKENS),
                    window=WINDOW, n=N_WINDOWS - 1,
                )
            open_test_job(coord, job_id="dupe")
            with self.assertRaises(ValueError):
                open_test_job(coord, job_id="dupe")


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class TestRouter(unittest.TestCase):
    def test_router_creation_and_http_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = config_for(root / "state.json")
            loader = RecordingLoader()
            coord = make_coordinator(root, loader)
            router = hes.create_router(config, coordinator=coord)
            paths = {route.path for route in router.routes}
            self.assertIn(f"{config.route_prefix}/job", paths)
            self.assertIn(f"{config.route_prefix}/receipt", paths)

            app = hes.create_app(config, coordinator=coord)
            client = TestClient(app)
            response = client.get(f"{config.route_prefix}/job", params={"evaluator_id": "eval-a"})
            self.assertEqual(response.status_code, 204)

            job = open_test_job(coord)
            response = client.get(f"{config.route_prefix}/job", params={"evaluator_id": "eval-a"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["job_id"], job.job_id)

            response = client.post(f"{config.route_prefix}/receipt", content=receipt_for(job).to_json())
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["recorded"])
            self.assertTrue(response.json()["decision"]["accepted"])

    def test_create_router_builds_its_own_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = config_for(Path(td) / "state.json")
            router = hes.create_router(config, RecordingLoader())
            self.assertEqual(len(router.routes), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
