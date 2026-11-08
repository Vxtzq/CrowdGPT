"""Regression tests: a coordinator refusal (HTTP 403 / 426, heartbeat "blocked") must stop the client.

Runs with the standard library only: torch, numpy and requests are replaced by stubs, client.py is
imported from a temporary copy and a temporary working directory (so its checkpoints/ migration never
touches a real folder), and every HTTP call is mocked. No GPU, no training, no network.

    python -m unittest discover -s tests -v
"""
import argparse
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
_real_sleep = time.sleep


def _stub(name, **attrs):
    m = mock.MagicMock(name=name)
    m.__name__ = name
    m.__spec__ = None
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _install_stubs():
    nn = _stub("torch.nn", Module=type("Module", (), {}))
    functional = _stub("torch.nn.functional")
    utils = _stub("torch.utils")
    checkpoint = _stub("torch.utils.checkpoint")
    _stub("torch", nn=nn, utils=utils)
    nn.functional = functional
    utils.checkpoint = checkpoint
    _stub("numpy")
    exc = types.ModuleType("requests.exceptions")
    exc.ChunkedEncodingError = type("ChunkedEncodingError", (Exception,), {})
    sys.modules["requests.exceptions"] = exc
    _stub("requests", exceptions=exc)
    sys.modules["webview"] = None  # force the "no GUI toolkit" branch


_STUBBED = ("torch", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.checkpoint",
            "numpy", "requests", "requests.exceptions", "webview")
_STDIO_BEFORE = (sys.stdout, sys.stderr)


class _NeverClosedBytesIO(io.BytesIO):
    def close(self): pass
_ROOT_HANDLERS_BEFORE = list(logging.getLogger().handlers)


def _load_client():
    tmp = tempfile.mkdtemp(prefix="crowdgpt-test-")
    shutil.copy(REPO / "client.py", tmp)
    saved_stdio, saved_cwd, saved_env = (sys.stdout, sys.stderr), os.getcwd(), set(os.environ)
    root = logging.getLogger(); saved_handlers, saved_level = list(root.handlers), root.level
    missing = object()  # distinguishes "no entry" from an entry whose value is None (a blocked import)
    saved_modules = {name: sys.modules.get(name, missing) for name in _STUBBED}
    _install_stubs()
    spec = importlib.util.spec_from_file_location("crowdgpt_client_under_test", Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        # client.py rewraps sys.stdout/stderr around their .buffer at import time; a wrapper that is later
        # garbage-collected closes that buffer. Give it throwaway streams (whose buffer ignores close())
        # so the real ones are never wrapped, then restore the real ones below.
        fakes = [io.TextIOWrapper(_NeverClosedBytesIO(), encoding="utf-8") for _ in range(2)]
        sys.stdout, sys.stderr = fakes
        # client.py migrates $CWD/checkpoints into its own folder at import time; run it from the temp
        # folder so a developer's real checkpoints/ (saved login, weight cache) is never moved or deleted.
        os.chdir(tmp)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(saved_cwd)
        sys.stdout, sys.stderr = saved_stdio
        root.handlers[:] = saved_handlers; root.setLevel(saved_level)  # undo logging.basicConfig()
        for k in set(os.environ) - saved_env: del os.environ[k]          # undo os.environ.setdefault()
        for name, orig in saved_modules.items():                          # keep the stubs private to this module
            if orig is missing: sys.modules.pop(name, None)
            else: sys.modules[name] = orig
    return mod, tmp


client, _TMP = _load_client()


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


# ── fake HTTP ──────────────────────────────────────────────────────────────────────────────────────

def _response(status, body=None, headers=None, json_error=False):
    r = mock.MagicMock()
    r.status_code = status
    r.headers = headers or {}
    if json_error:
        r.json.side_effect = ValueError("not json"); r.text = "<html>oops</html>"
    else:
        r.json.return_value = body if body is not None else {}
        r.text = json.dumps(body) if body is not None else ""
    return r


# Response shapes taken from Vxtzq/CrowdGPT-Server server.py:
#  - 403 (GET /fl/task, /fl/round_status): HTTPException(403, detail=_banned_reason(...)) -> {"detail": reason}.
#    The reason may be temporary ("subnet temporarily blacklisted", timed bans) or not; no expiry is sent.
#  - 426 (GET /fl/task): _version_gate() -> {"status": "update_required", "minVersion", "message"}, X-Status: update.
#  - GET /fl/heartbeat for a banned client: 200 {"status": "blocked"} (no reason).
_UUID = "3f2b8c1e-9a4d-4e7b-8f10-2c6d5e4a1b90"
# (raw reason sent by the server, what the client shows after _http_detail redacts ids and IPs)
REASONS_403 = ((f"client banned (user:{_UUID}:dev:abc)", "client banned (user:<id>:dev:abc)"),
               ("client banned (ip:203.0.113.7)", "client banned (ip:<ip>)"),
               (f"account suspended ({_UUID})", "account suspended (<id>)"),
               ("subnet temporarily blacklisted", "subnet temporarily blacklisted"))
ROUND_OK = {"current_round": 1, "in_cooldown": False, "is_aggregating": False,
            "max_round_hours": 2, "round_elapsed_hours": 0}
SERVER = "http://coord.invalid"


def _resp_403(raw_reason):
    return _response(403, {"detail": raw_reason})


def _resp_426():
    return _response(426, {"status": "update_required", "minVersion": "9.9",
                           "message": "Client 3.1 is too old; update to 9.9+"}, {"X-Status": "update"})


def _resp_wait():
    return _response(200, {"status": "wait"}, {"X-Status": "wait"})


class _Router:
    """requests.get replacement: routes by path; a route is a response or f(nth_call) -> response."""

    def __init__(self, routes):
        self.routes = routes; self.calls = []; self._lock = threading.Lock()

    def __call__(self, url, *a, **k):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1].split("?", 1)[0]
        with self._lock:
            self.calls.append(path); n = self.calls.count(path)
        route = self.routes[path]
        return route(n) if isinstance(route, types.FunctionType) else route  # a mock response is callable too

    def count(self, path):
        with self._lock: return self.calls.count(path)


def _fast_sleep(stop_event, deadline_s=5.0):
    """time.sleep replacement: yields briefly; past the deadline it sets stop_event so a regression can't hang."""
    end = time.monotonic() + deadline_s; state = {"expired": False}
    def _sleep(*_):
        _real_sleep(0.002)
        if time.monotonic() > end: state["expired"] = True; stop_event.set()
    _sleep.expired = lambda: state["expired"]
    return _sleep


class _Events:
    def __init__(self):
        self.items = []; self._lock = threading.Lock()

    def __call__(self, ev, data=None):
        with self._lock: self.items.append((ev, data))

    def names(self):
        return [ev for ev, _ in self.items]

    def logs(self):
        return [str(d) for ev, d in self.items if ev == "log"]


class _FastEvent(threading.Event):
    def wait(self, timeout=None): return super().wait(0.01 if timeout else timeout)


class _FastHeartbeat(client.HeartbeatManager):
    """The real heartbeat logic, polling every 10 ms instead of 15 s."""
    def __init__(self, *a, **k): super().__init__(*a, **k); self.shutdown = _FastEvent()


def _block_during_weight_stream(stop_event, reason="subnet temporarily blacklisted"):
    """Routes for: round starts, metadata OK, weights stream from the coordinator, and once the stream is
    flowing the heartbeat reports "blocked" (round_status 403 with `reason`)."""
    streaming = threading.Event()
    def stream(n):
        if n == 1:  # metadata request (skip_weights=true)
            return _response(200, {"taskId": "t1"}, {"content-type": "application/json"})
        r = _response(200, headers={"content-length": str(1 << 30)})
        def chunks(*a, **k):
            end = time.monotonic() + 5
            while time.monotonic() < end:
                streaming.set(); yield b"x" * 16; _real_sleep(0.002)
            stop_event.set()  # deadline: the client never noticed the block
        r.iter_content.side_effect = chunks
        return r
    return {
        "/fl/round_status": lambda n: _resp_403(reason) if streaming.is_set() else _response(200, ROUND_OK),
        "/fl/heartbeat": lambda n: _response(200, {"status": "blocked" if streaming.is_set() else "ok"}),
        "/fl/task": stream,
    }


class _NoHeartbeat:
    def __init__(self, *a, **k): self.tid = None; self.rejected = None
    def start(self): pass
    def stop(self): pass
    def should_stop(self): return False
    def commit(self): return self.rejected is None
    def close(self): return self.rejected


# ── tests ──────────────────────────────────────────────────────────────────────────────────────────

class ImportIsolation(unittest.TestCase):
    def test_import_leaves_stdio_and_logging_untouched(self):
        self.assertIs(sys.stdout, _STDIO_BEFORE[0])
        self.assertIs(sys.stderr, _STDIO_BEFORE[1])
        for s in (sys.stdout, sys.stderr):
            self.assertFalse(s.closed)
            if getattr(s, "buffer", None) is not None: self.assertFalse(s.buffer.closed)
            s.write(""); s.flush()
        self.assertEqual(logging.getLogger().handlers, _ROOT_HANDLERS_BEFORE)
        for name in _STUBBED:
            self.assertNotIsInstance(sys.modules.get(name), mock.MagicMock)

    def test_module_table_restore_keeps_blocked_entries(self):
        # sys.modules[name] = None (an import deliberately blocked) must survive another load, not be dropped
        had = "webview" in sys.modules; orig = sys.modules.get("webview")
        sys.modules["webview"] = None
        try:
            _, tmp = _load_client(); shutil.rmtree(tmp, ignore_errors=True)
            self.assertIn("webview", sys.modules); self.assertIsNone(sys.modules["webview"])
        finally:
            if had: sys.modules["webview"] = orig
            else: sys.modules.pop("webview", None)


class FetchRaisesOnRefusal(unittest.TestCase):
    """Both /fl/task fetchers raise CoordinatorRejected once, without retrying."""

    def setUp(self):
        self.se = threading.Event()
        # If a refusal were swallowed, the retry path sleeps: stop the loop there instead of hanging.
        p = mock.patch.object(client.time, "sleep", side_effect=lambda *_: self.se.set())
        p.start(); self.addCleanup(p.stop)

    def _assert_rejected(self, fn, response, status, reason):
        emit = _Events()
        with mock.patch.object(client.requests, "get", return_value=response) as get, \
             mock.patch.object(client, "_clear_saved_token_if_matches") as clear_token:
            with self.assertRaises(client.CoordinatorRejected) as cm:
                fn(emit)
        self.assertEqual(get.call_count, 1, "a refusal must not be retried")
        clear_token.assert_not_called()  # 403/426 are not auth failures: keep the session
        self.assertFalse(self.se.is_set())
        self.assertEqual(cm.exception.status, status)
        self.assertIn(reason, str(cm.exception), "the server's reason must reach the exception")
        return emit

    def _fetchers(self):
        return {"metadata": lambda e: client.fetch_task_metadata(SERVER, {}, self.se, e),
                "weight_stream": lambda e: client.fetch_task(SERVER, {}, "bf16", self.se, e)}

    def test_403_carries_server_reason(self):
        for name, fn in self._fetchers().items():
            for raw, shown in REASONS_403:
                with self.subTest(fetcher=name, reason=raw):
                    emit = self._assert_rejected(fn, _resp_403(raw), 403, shown)
                    self.assertIn(("login_error", {"code": "forbidden"}), emit.items)

    def test_426_carries_server_message(self):
        for name, fn in self._fetchers().items():
            with self.subTest(fetcher=name):
                self._assert_rejected(fn, _resp_426(), 426, "Client 3.1 is too old; update to 9.9+")

    def test_metadata_401_still_auth_rejected(self):
        with mock.patch.object(client.requests, "get", return_value=_response(401)), \
             mock.patch.object(client, "_clear_saved_token_if_matches") as clear_token:
            with self.assertRaises(client.AuthRejected):
                client.fetch_task_metadata(SERVER, {}, self.se, _Events())
        clear_token.assert_called_once()

    def test_metadata_transient_error_still_retried(self):
        responses = [_response(503), _response(200, {"taskId": "t1"}, {"content-type": "application/json"})]
        with mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=responses) as get:
            md = client.fetch_task_metadata(SERVER, {}, self.se, _Events())
        self.assertEqual(md, {"taskId": "t1"})
        self.assertEqual(get.call_count, 2)


class WaitForRound(unittest.TestCase):
    def setUp(self):
        self.se = threading.Event()
        p = mock.patch.object(client.time, "sleep", side_effect=_fast_sleep(self.se)); p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(client, "_clear_saved_token_if_matches"); self.clear_token = p.start(); self.addCleanup(p.stop)

    def test_403_raises_with_reason(self):
        for raw, shown in REASONS_403:
            with self.subTest(reason=raw):
                with mock.patch.object(client.requests, "get", return_value=_resp_403(raw)) as get:
                    with self.assertRaises(client.CoordinatorRejected) as cm:
                        client.wait_for_round(SERVER, {}, self.se, _Events())
                self.assertEqual(get.call_count, 1)
                self.assertEqual(cm.exception.status, 403)
                self.assertIn(shown, str(cm.exception))
                self.assertNotIn(_UUID, str(cm.exception))
        self.clear_token.assert_not_called()
        self.assertFalse(self.se.is_set())

    def test_403_after_waiting_hides_overlay(self):
        cooldown = _response(200, {**ROUND_OK, "in_cooldown": True})
        emit = _Events()
        with mock.patch.object(client.requests, "get", side_effect=[cooldown, _resp_403("subnet temporarily blacklisted")]):
            with self.assertRaises(client.CoordinatorRejected):
                client.wait_for_round(SERVER, {}, self.se, emit)
        names = emit.names()
        self.assertIn("overlay_show", names)
        self.assertGreater(names.index("overlay_hide"), names.index("overlay_show"))

    def test_network_errors_429_and_5xx_still_retried(self):
        seq = [ConnectionError("down"), _response(429), _response(500), _response(502), _response(503),
               _response(504), _response(200, ROUND_OK)]
        with mock.patch.object(client.requests, "get", side_effect=seq) as get:
            st = client.wait_for_round(SERVER, {}, self.se, _Events())
        self.assertEqual(st, ROUND_OK)
        self.assertEqual(get.call_count, len(seq))
        self.assertFalse(self.se.is_set())


class Heartbeat(unittest.TestCase):
    """HeartbeatManager._r run synchronously in the test thread."""

    def _hb(self, routes, tid=None):
        self.se = threading.Event(); self.emit = _Events()
        hb = client.HeartbeatManager(SERVER, {}, 1, self.se, self.emit); hb.tid = tid
        def stop_after(resp):  # end the polling loop after the first round_status check
            def _r(n): hb.shutdown.set(); return resp
            return _r
        routes = {k: (stop_after(v) if k == "/fl/round_status" else v) for k, v in routes.items()}
        self.router = _Router(routes)
        with mock.patch.object(client.requests, "get", side_effect=self.router):
            hb._r()
        return hb

    def test_blocked_uses_round_status_reason(self):
        for raw, shown in REASONS_403:
            with self.subTest(reason=raw):
                hb = self._hb({"/fl/heartbeat": _response(200, {"status": "blocked"}), "/fl/round_status": _resp_403(raw)})
                self.assertTrue(hb.should_stop())
                self.assertIn(shown, hb.rejected); self.assertNotIn(_UUID, hb.rejected)
                self.assertEqual(self.router.count("/fl/heartbeat"), 1)

    def test_blocked_without_reason(self):
        hb = self._hb({"/fl/heartbeat": _response(200, {"status": "blocked"}), "/fl/round_status": _response(200, ROUND_OK)})
        self.assertTrue(hb.should_stop())
        self.assertIn("client blocked by the coordinator", hb.rejected)

    def test_round_status_403_alone_rejects(self):
        hb = self._hb({"/fl/heartbeat": _response(200, {"status": "ok"}), "/fl/round_status": _resp_403("subnet temporarily blacklisted")})
        self.assertTrue(hb.should_stop())
        self.assertIn("subnet temporarily blacklisted", hb.rejected)

    def test_ok_does_not_stop(self):
        hb = self._hb({"/fl/heartbeat": _response(200, {"status": "ok", "task": "active"}), "/fl/round_status": _response(200, ROUND_OK)}, tid="t1")
        self.assertFalse(hb.should_stop()); self.assertIsNone(hb.rejected)

    def test_task_gone_still_stops_without_rejection(self):
        hb = self._hb({"/fl/heartbeat": _response(200, {"status": "ok", "task": "gone"}), "/fl/round_status": _response(200, ROUND_OK)}, tid="t1")
        self.assertTrue(hb.should_stop()); self.assertIsNone(hb.rejected)

    def test_non_json_heartbeat_still_checks_round(self):
        hb = self._hb({"/fl/heartbeat": _response(200, json_error=True), "/fl/round_status": _response(200, {**ROUND_OK, "current_round": 2})})
        self.assertTrue(hb.should_stop()); self.assertIsNone(hb.rejected)


class HeartbeatRoundClose(unittest.TestCase):
    """The real HeartbeatManager thread vs. close()/commit(), ordered with events (no timing assumptions)."""

    def setUp(self):
        self.emit = _Events(); self.in_request = threading.Event(); self.release = threading.Event()
        self.hb = client.HeartbeatManager(SERVER, {}, 1, threading.Event(), self.emit)
        self.hb.shutdown = _FastEvent()  # poll every 10 ms

    def _start(self, round_status, heartbeat=None):
        router = _Router({"/fl/heartbeat": heartbeat or _response(200, {"status": "ok"}), "/fl/round_status": round_status})
        p = mock.patch.object(client.requests, "get", side_effect=router); p.start(); self.addCleanup(p.stop)
        self.hb.start()
        self.addCleanup(lambda: (self.release.set(), self.hb._t.join(5)))

    def _blocked_until(self, event):
        def handler(n):
            self.in_request.set()
            threading.Event.wait(event, 5)  # plain Event.wait: _FastEvent would cut the wait short
            return _resp_403("subnet temporarily blacklisted")
        return handler

    def _refusal_logs(self):
        return [l for l in self.emit.logs() if "refused this client during the round" in l]

    def test_refusal_arriving_while_closing_is_kept(self):
        # the in-flight check returns its 403 only once close() has asked the thread to stop
        self._start(self._blocked_until(self.hb.shutdown))
        self.assertTrue(self.in_request.wait(5))
        reason = self.hb.close()  # joins: the in-flight check finishes and records the refusal first
        self.assertIn("subnet temporarily blacklisted", reason)
        self.assertFalse(self.hb._t.is_alive())
        self.assertEqual(len(self._refusal_logs()), 1)

    def test_check_still_running_after_join_timeout_changes_nothing(self):
        self.hb.JOIN_TIMEOUT = 0.05
        self._start(self._blocked_until(self.release))
        self.assertTrue(self.in_request.wait(5))
        self.assertIsNone(self.hb.close())
        self.assertTrue(self.hb._t.is_alive(), "join must have timed out with the check still running")
        self.release.set(); self.hb._t.join(5)  # now the late 403 arrives
        self.assertFalse(self.hb._t.is_alive())
        self.assertIsNone(self.hb.rejected); self.assertFalse(self.hb.should_stop())
        self.assertEqual(self._refusal_logs(), [])

    BLOCKED = _response(200, {"status": "blocked"})

    def test_blocked_is_recorded_before_the_reason_lookup(self):
        # while the round_status lookup for the reason hangs, the block already prevents the hand-over
        self._start(self._blocked_until(self.release), heartbeat=self.BLOCKED)
        self.assertTrue(self.in_request.wait(5))
        self.assertTrue(self.hb.should_stop())
        self.assertFalse(self.hb.commit(), "a known block must stop the delta hand-over")
        self.release.set(); self.hb._t.join(5)
        self.assertIn("subnet temporarily blacklisted", self.hb.close())  # the reason refines the refusal
        self.assertEqual(len(self._refusal_logs()), 1)

    def test_blocked_survives_a_failed_reason_lookup(self):
        def boom(n): self.in_request.set(); raise ConnectionError("round_status down")
        self._start(boom, heartbeat=self.BLOCKED)
        self.hb._t.join(5); self.assertFalse(self.hb._t.is_alive())
        self.assertTrue(self.hb.should_stop())
        self.assertEqual(self.hb.close(), "coordinator rejected: client blocked by the coordinator")
        self.assertEqual(len(self._refusal_logs()), 1)

    def test_reason_arriving_after_close_keeps_the_recorded_refusal(self):
        self.hb.JOIN_TIMEOUT = 0.05
        self._start(self._blocked_until(self.release), heartbeat=self.BLOCKED)
        self.assertTrue(self.in_request.wait(5))
        self.assertEqual(self.hb.close(), "coordinator rejected: client blocked by the coordinator")
        self.release.set(); self.hb._t.join(5)
        self.assertEqual(self.hb.rejected, "coordinator rejected: client blocked by the coordinator")  # closed: not rewritten
        self.assertEqual(len(self._refusal_logs()), 1)

    def test_close_on_unstarted_or_own_thread_does_not_hang(self):
        self.assertIsNone(self.hb.close())  # never started: no join
        result = []
        hb2 = client.HeartbeatManager(SERVER, {}, 1, threading.Event(), self.emit)
        hb2._t = threading.Thread(target=lambda: result.append(hb2.close()), daemon=True)
        hb2.start(); hb2._t.join(5)  # close() from the heartbeat thread itself must not self-join
        self.assertFalse(hb2._t.is_alive()); self.assertEqual(result, [None])

    def test_refusal_after_commit_is_still_reported(self):
        # the hand-over transfers the file, not the right to report a refusal received before the round closed
        self.assertTrue(self.hb.commit())
        self.assertTrue(self.hb._reject("coordinator rejected: after hand-over"))
        self.assertEqual(self.hb.close(), "coordinator rejected: after hand-over")
        self.assertFalse(self.hb._reject("coordinator rejected: after close"))  # closed: only the next round sees it
        self.assertEqual(self.hb.rejected, "coordinator rejected: after hand-over")

    def test_commit_after_refusal_fails(self):
        self.assertTrue(self.hb._reject("coordinator rejected: early"))
        self.assertFalse(self.hb.commit())
        self.assertEqual(self.hb.close(), "coordinator rejected: early")


class UploadHandOver(unittest.TestCase):
    """_submit_delta_background with a refusal before / while waiting for / after the hand-over to the uploader."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="crowdgpt-upload-", dir=_TMP))
        self.delta = self.dir / "delta-this-round.gz"; self.delta.write_bytes(b"delta")
        self.token_file = self.dir / "auth_token.json"; self.token_file.write_text('{"token": "tok"}')
        self.se = threading.Event(); self.emit = _Events()
        self.hb = client.HeartbeatManager(SERVER, {}, 1, self.se, self.emit)  # not started: _reject() stands in for its thread
        self.uploaded = []; self.upload_gate = threading.Event(); self.upload_entered = threading.Event()
        def fake_upload(srv, h, path, size, meta, emit, se):
            self.uploaded.append(path); self.upload_entered.set(); self.upload_gate.wait(5)
        for target, name, kw in ((client, "_upload_delta_file", {"side_effect": fake_upload}),
                                 (client, "AUTH_TOKEN_FILE", {"new": self.token_file}),
                                 (client, "_upload_thread", {"new": None}),
                                 (client.time, "sleep", {"side_effect": lambda *_: _real_sleep(0.002)})):
            p = mock.patch.object(target, name, **kw); p.start(); self.addCleanup(p.stop)
        self.addCleanup(self.upload_gate.set)

    def _submit(self):
        client._submit_delta_background(SERVER, {}, str(self.delta), 5, {}, self.emit, self.se, hb=self.hb)

    def _assert_discarded(self):
        self.assertEqual(self.uploaded, [], "no new upload may start after a refusal")
        self.assertFalse(self.delta.exists(), "this round's file was not handed over and must be deleted")
        self.assertTrue(self.token_file.exists())
        self.assertIn("Coordinator refused this client; delta discarded, no upload.", self.emit.logs())

    def test_refusal_during_delta_preparation(self):
        self.hb._reject("coordinator rejected: HTTP 403 — subnet temporarily blacklisted")
        self._submit()
        self._assert_discarded()

    def test_refusal_while_waiting_for_previous_upload(self):
        prev_file = self.dir / "delta-previous-round.gz"; prev_file.write_bytes(b"prev")
        prev_done = threading.Event()
        prev = threading.Thread(target=lambda: prev_done.wait(5), daemon=True); prev.start()
        self.addCleanup(lambda: (prev_done.set(), prev.join(5)))
        client._upload_thread = prev
        waiting = threading.Event()
        def emit(ev, data=None):
            self.emit(ev, data)
            if ev == "log" and "Previous upload still in progress" in str(data): waiting.set()
        t = threading.Thread(target=lambda: client._submit_delta_background(SERVER, {}, str(self.delta), 5, {}, emit, self.se, hb=self.hb))
        t.start()
        self.assertTrue(waiting.wait(5))
        self.hb._reject("coordinator rejected: HTTP 403 — subnet temporarily blacklisted")
        t.join(5); self.assertFalse(t.is_alive())
        self._assert_discarded()
        # the previous round's upload is neither cancelled nor robbed of its file
        self.assertTrue(prev.is_alive()); self.assertIs(client._upload_thread, prev); self.assertTrue(prev_file.exists())

    def test_handed_over_file_is_left_to_the_uploader(self):
        self._submit()
        self.assertTrue(self.upload_entered.wait(5), "the uploader thread never started")
        self.assertEqual(self.uploaded, [str(self.delta)])
        uploader = client._upload_thread
        self.assertTrue(self.hb._reject("coordinator rejected: after hand-over"))  # still reported...
        self.assertEqual(self.hb.close(), "coordinator rejected: after hand-over")
        self.assertTrue(self.delta.exists(), "a handed-over file belongs to the uploader")  # ...but the upload is untouched
        self.assertTrue(uploader.is_alive(), "the started upload must not be cancelled")
        self.upload_gate.set(); uploader.join(5); self.assertFalse(uploader.is_alive())


class DeltaFileOwnership(unittest.TestCase):
    """Real prune_hf_cache() and the real uploader: who may delete a delta file, and when."""

    def setUp(self):
        self.ckpt = Path(tempfile.mkdtemp(prefix="crowdgpt-ckpt-", dir=_TMP))
        p = mock.patch.object(client, "CHECKPOINT_DIR", self.ckpt); p.start(); self.addCleanup(p.stop)
        self.old_delta = self.ckpt / "delta-previous-round.gz"; self.old_delta.write_bytes(b"prev")
        two_hours_ago = time.time() - 2 * 3600
        os.utime(self.old_delta, (two_hours_ago, two_hours_ago))
        self.finish = threading.Event(); self.addCleanup(self.finish.set)

    def _active_upload(self):
        t = threading.Thread(target=lambda: threading.Event.wait(self.finish, 10), daemon=True); t.start()
        self.addCleanup(lambda: (self.finish.set(), t.join(5)))
        p = mock.patch.object(client, "_upload_thread", t); p.start(); self.addCleanup(p.stop)
        return t

    def test_prune_keeps_an_old_delta_while_an_upload_runs(self):
        self._active_upload()
        client.prune_hf_cache()
        self.assertTrue(self.old_delta.exists(), "a slow upload's file must not be pruned under it")

    def test_prune_still_removes_stale_deltas_when_idle(self):
        fresh = self.ckpt / "delta-fresh.gz"; fresh.write_bytes(b"new")
        with mock.patch.object(client, "_upload_thread", None):
            client.prune_hf_cache()
        self.assertFalse(self.old_delta.exists()); self.assertTrue(fresh.exists())

    def test_refused_round_cleanup_keeps_the_previous_upload_file(self):
        # the wrapper's finally runs the real prune_hf_cache() before raising the refusal
        self._active_upload()
        hb_cls = WrapperPropagation._RejectedHeartbeat
        with mock.patch.object(client, "wait_for_round", return_value=ROUND_OK), \
             mock.patch.object(client, "HeartbeatManager", hb_cls), \
             mock.patch.object(client, "fetch_task_metadata", return_value=None):
            with self.assertRaises(client.CoordinatorRejected):
                client.run_single_round_wrapper(SERVER, "tok", threading.Event(), _Events())
        self.assertTrue(self.old_delta.exists())

    def test_real_uploader_deletes_its_file_after_success_or_failure(self):
        for status, expected_log in ((200, "✅ Submitted successfully!"), (400, "❌ Round lost")):
            with self.subTest(status=status):
                delta = self.ckpt / f"delta-upload-{status}.gz"; delta.write_bytes(b"delta")
                emit = _Events()
                with mock.patch.object(client.requests, "post", return_value=_response(status, {"detail": "bad"} if status != 200 else {})):
                    client._upload_delta_file(SERVER, {}, str(delta), 5, {}, emit, threading.Event())
                self.assertFalse(delta.exists(), "the uploader owns the file and removes it when done")
                self.assertTrue(any(expected_log in l for l in emit.logs()), emit.logs())


class WrapperPropagation(unittest.TestCase):
    """run_single_round_wrapper turns a heartbeat refusal into CoordinatorRejected in the calling thread."""

    class _RejectedHeartbeat(_NoHeartbeat):
        def __init__(self, *a, **k): super().__init__(); self.rejected = "coordinator rejected: HTTP 403 — subnet temporarily blacklisted"
        def should_stop(self): return True

    def _run(self, fetch_md, hb_cls, se=None):
        with mock.patch.object(client, "wait_for_round", return_value=ROUND_OK), \
             mock.patch.object(client, "HeartbeatManager", hb_cls), \
             mock.patch.object(client, "prune_hf_cache"), \
             mock.patch.object(client, "_submit_delta_background") as submit, \
             mock.patch.object(client, "fetch_task_metadata", side_effect=fetch_md):
            try: return client.run_single_round_wrapper(SERVER, "tok", se or threading.Event(), _Events())
            finally: submit.assert_not_called()

    def _raised(self, *a, **k):
        try: self._run(*a, **k)
        except BaseException as e: return type(e)  # a stray KeyboardInterrupt must fail this test, not the run
        return None

    def test_heartbeat_abort_keyboardinterrupt_raises(self):
        # fetch_task aborts a weight download with KeyboardInterrupt when hb.should_stop() (se not set)
        def abort(*a, **k): raise KeyboardInterrupt("Stop requested")
        self.assertIs(self._raised(abort, self._RejectedHeartbeat), client.CoordinatorRejected)

    def test_user_stop_wins(self):
        se = threading.Event(); se.set()
        def abort(*a, **k): raise KeyboardInterrupt("Stop requested")
        self.assertIs(self._raised(abort, self._RejectedHeartbeat, se=se), KeyboardInterrupt)

    def test_early_return_raises(self):
        with self.assertRaises(client.CoordinatorRejected) as cm:
            self._run(lambda *a, **k: None, self._RejectedHeartbeat)
        self.assertEqual(cm.exception.status, 403)
        self.assertIn("subnet temporarily blacklisted", str(cm.exception))

    def test_other_error_is_named_and_chained(self):
        def boom(*a, **k): raise RuntimeError("mid-round failure")
        with self.assertRaises(client.CoordinatorRejected) as cm:
            self._run(boom, self._RejectedHeartbeat)
        self.assertIn("subnet temporarily blacklisted", str(cm.exception))
        self.assertIn("the round had also failed: RuntimeError: mid-round failure", str(cm.exception))
        self.assertIsInstance(cm.exception.__context__, RuntimeError)  # original traceback kept

    def test_other_exits_are_never_replaced(self):
        for original in (client.CoordinatorRejected("update required: HTTP 426", status=426), SystemExit("exit"), GeneratorExit("gen")):
            with self.subTest(kind=type(original).__name__):
                def boom(*a, **k): raise original
                try: self._run(boom, self._RejectedHeartbeat)
                except BaseException as got: self.assertIs(got, original)
                else: self.fail("exception lost")

    def test_keyboardinterrupt_without_refusal_is_untouched(self):
        def interrupt(*a, **k): raise KeyboardInterrupt("Stop requested")
        self.assertIs(self._raised(interrupt, _NoHeartbeat), KeyboardInterrupt)

    def test_user_stop_with_normal_return_is_not_overridden(self):
        se = threading.Event(); se.set()
        self.assertIsNone(self._raised(lambda *a, **k: None, self._RejectedHeartbeat, se=se))

    def test_user_stop_keeps_original_error(self):
        se = threading.Event(); se.set()
        def boom(*a, **k): raise RuntimeError("mid-round failure")
        self.assertIs(self._raised(boom, self._RejectedHeartbeat, se=se), RuntimeError)

    def test_auth_rejected_wins(self):
        def boom(*a, **k): raise client.AuthRejected("auth rejected: HTTP 401")
        with self.assertRaises(client.AuthRejected):
            self._run(boom, self._RejectedHeartbeat)

    def test_no_rejection_returns_normally(self):
        self.assertIsNone(self._run(lambda *a, **k: None, _NoHeartbeat))


class GuiLoop(unittest.TestCase):
    """Api._rs (the GUI training loop) must stop instead of announcing 'Round complete' and relaunching."""

    MAX_ROUNDS = 3  # safety net so a regression fails instead of looping forever

    def setUp(self):
        self.token_file = Path(_TMP) / "auth_token.json"
        self.token_file.write_text('{"token": "tok"}')
        self.api = client.Api.__new__(client.Api)  # skip __init__: no UI pump thread, no backend probing
        self.api.emit = self.events = _Events()
        self.api.stop_event = threading.Event()
        self.api.auth_token = "tok"; self.api.server_url = SERVER
        self.api.selected_backend = "cpu"; self.api.vram_gb = 0.0
        self.api._check_updates_between_rounds = mock.MagicMock()
        self.rounds = 0
        p = mock.patch.multiple(client, _do_backend_setup=mock.DEFAULT, apply_vram_override=mock.DEFAULT,
                                prune_hf_cache=mock.DEFAULT, _submit_delta_background=mock.DEFAULT,
                                AUTH_TOKEN_FILE=self.token_file)
        self.patched = p.start(); self.addCleanup(p.stop)
        self.sleep = _fast_sleep(self.api.stop_event)
        p = mock.patch.object(client.time, "sleep", side_effect=self.sleep); p.start(); self.addCleanup(p.stop)

    def _run(self, round_fn):
        def wrapper(*a, **k):
            self.rounds += 1
            if self.rounds >= self.MAX_ROUNDS: self.api.stop_event.set()
            return round_fn(*a, **k)
        with mock.patch.object(client, "run_single_round_wrapper", side_effect=wrapper):
            self.api._rs()

    def _run_real_round(self, response):
        """Drive the real run_single_round_wrapper up to the /fl/task request, answered with `response`."""
        real = client.run_single_round_wrapper
        with mock.patch.object(client, "wait_for_round", return_value=ROUND_OK), \
             mock.patch.object(client, "HeartbeatManager", _NoHeartbeat), \
             mock.patch.object(client.requests, "get", return_value=response):
            self._run(real)

    def _run_routed(self, routes, hb_cls=None):
        self.router = _Router(routes)
        with mock.patch.object(client.requests, "get", side_effect=self.router), \
             mock.patch.object(client, "HeartbeatManager", hb_cls or client.HeartbeatManager):
            self._run(client.run_single_round_wrapper)

    def _final_message(self):
        return [l for l in self.events.logs() if l.startswith("⛔") and "Training stopped" in l]

    def _assert_stopped_keeping_session(self, reason=None, update_check=False):
        self.assertFalse(self.sleep.expired(), "hit the test deadline instead of stopping")
        self.assertEqual(self.rounds, 1, "the loop must not relaunch after a refusal")
        logs = self.events.logs()
        self.assertFalse(any("Round complete" in l or "Round failed" in l or "stopped by user" in l for l in logs), logs)
        self.assertNotIn("logged_out", self.events.names())
        self.assertEqual(self.api.auth_token, "tok")
        self.assertTrue(self.token_file.exists())
        self.assertIn(("status", "idle"), self.events.items)
        self.patched["_submit_delta_background"].assert_not_called()
        final = self._final_message()
        self.assertEqual(len(final), 1, final)
        self.assertNotIn("permanent", final[0].lower())
        if reason: self.assertIn(reason, final[0])
        self.assertEqual(self.api._check_updates_between_rounds.call_count, 1 if update_check else 0)
        return final[0]

    def test_fl_task_403_stops_loop_and_shows_reason(self):
        for raw, shown in REASONS_403:
            with self.subTest(reason=raw):
                self.setUp()
                self._run_real_round(_resp_403(raw))
                final = self._assert_stopped_keeping_session(shown)
                self.assertNotIn("update", final.lower())

    def test_fl_task_426_stops_loop_shows_message_and_checks_update(self):
        self.api._check_updates_between_rounds.side_effect = lambda: self.events("update_check")
        self._run_real_round(_resp_426())
        self._assert_stopped_keeping_session("Client 3.1 is too old; update to 9.9+", update_check=True)
        # Start must not be re-enabled (status idle) while the update check is still running
        self.assertGreater(self.events.items.index(("status", "idle")), self.events.names().index("update_check"))

    def test_heartbeat_block_during_weight_download_stops(self):
        self._run_routed(_block_during_weight_stream(self.api.stop_event), hb_cls=_FastHeartbeat)
        self._assert_stopped_keeping_session("subnet temporarily blacklisted")

    def test_round_status_403_stops_loop(self):
        for raw, shown in REASONS_403:
            with self.subTest(reason=raw):
                self.setUp()
                self._run_routed({"/fl/round_status": _resp_403(raw)})
                self._assert_stopped_keeping_session(shown)
                self.assertEqual(self.router.calls, ["/fl/round_status"])

    def test_heartbeat_block_mid_round_stops_without_upload(self):
        self._run_routed({
            "/fl/round_status": lambda n: _response(200, ROUND_OK) if n == 1 else _resp_403("subnet temporarily blacklisted"),
            "/fl/heartbeat": _response(200, {"status": "blocked"}),
            "/fl/task": _resp_wait(),
        })
        self._assert_stopped_keeping_session("subnet temporarily blacklisted")
        self.assertGreaterEqual(self.router.count("/fl/heartbeat"), 1)

    def test_heartbeat_block_without_reason(self):
        self._run_routed({
            "/fl/round_status": _response(200, ROUND_OK),
            "/fl/heartbeat": _response(200, {"status": "blocked"}),
            "/fl/task": _resp_wait(),
        })
        self._assert_stopped_keeping_session("client blocked by the coordinator")

    def test_auth_rejected_still_logs_out(self):
        def boom(*a, **k): raise client.AuthRejected("auth rejected: HTTP 401")
        self._run(boom)
        self.assertEqual(self.rounds, 1)
        self.assertIn("logged_out", self.events.names())
        self.assertIsNone(self.api.auth_token)
        self.assertFalse(self.token_file.exists())

    def test_arch_incompatible_still_stops(self):
        def boom(*a, **k): raise client.ArchIncompatible("GPU 0 has no matching kernels")
        self._run(boom)
        self.assertEqual(self.rounds, 1)
        self.assertNotIn("logged_out", self.events.names())

    def test_other_errors_still_relaunch(self):
        def boom(*a, **k): raise RuntimeError("transient")
        self._run(boom)
        self.assertEqual(self.rounds, self.MAX_ROUNDS)
        self.assertTrue(any("Round failed: transient" in l for l in self.events.logs()))
        self.assertEqual(self.api.auth_token, "tok")

    def test_successful_round_relaunches(self):
        self._run(lambda *a, **k: None)
        self.assertEqual(self.rounds, self.MAX_ROUNDS)
        self.assertTrue(any("Round complete" in l for l in self.events.logs()))


class Cli(unittest.TestCase):
    """_run_headless exits with 7 on a refusal and never reports success (0) for a blocked round."""

    def test_refusal_after_hand_over_before_close_exits_7_with_rounds_1(self):
        # Last (only) round: the delta is handed to the uploader, then - while close() is ending the round - the
        # heartbeat receives a 403. The refusal must still give exit 7; the started upload and its file are untouched.
        main, made = threading.current_thread(), []
        hb_entered, committed = threading.Event(), threading.Event()
        class HB(client.HeartbeatManager):
            def __init__(s, *a, **k): super().__init__(*a, **k); s.shutdown = _FastEvent(); made.append(s)
            def commit(s):
                ok = super().commit(); committed.set(); return ok
        def round_status(n):
            if threading.current_thread() is main: return _response(200, ROUND_OK)
            hb_entered.set()
            threading.Event.wait(committed, 5)           # after the hand-over...
            threading.Event.wait(made[-1].shutdown, 5)   # ...and once close() has started
            return _resp_403("subnet temporarily blacklisted")
        delta = Path(_TMP) / "delta-last-round.gz"
        uploaded, upload_gate, file_seen_by_uploader = [], threading.Event(), []
        def fake_upload(srv, h, path, size, meta, emit, se):
            uploaded.append(path); threading.Event.wait(upload_gate, 10); file_seen_by_uploader.append(Path(path).exists())
        def end_of_training(srv, h, se, emit, hb=None):
            # stands in for a finished training run: write the delta, hand it over, end the round
            self.assertTrue(threading.Event.wait(hb_entered, 5), "heartbeat check never started")
            delta.write_bytes(b"delta")
            client._submit_delta_background(srv, h, str(delta), 5, {}, emit, se, hb=hb)
            return None
        real_join = client._join_upload
        def join_upload(*a, **k): upload_gate.set(); return real_join(*a, **k)
        self.addCleanup(lambda: (upload_gate.set(), delta.unlink(missing_ok=True)))
        with mock.patch.object(client, "fetch_task_metadata", side_effect=end_of_training), \
             mock.patch.object(client, "_upload_delta_file", side_effect=fake_upload), \
             mock.patch.object(client, "_upload_thread", None), \
             mock.patch.object(client, "_join_upload", side_effect=join_upload):
            rc = self._run_cli({"/fl/round_status": round_status, "/fl/heartbeat": _response(200, {"status": "ok"})},
                               rounds=1, hb_cls=HB, real_submit=True)
            uploader = client._upload_thread
        self.assertEqual(rc, 7, "a refusal received before the last round closed must not end in success")
        self.assertEqual(uploaded, [str(delta)], "the delta was handed over before the refusal")
        self.assertEqual(file_seen_by_uploader, [True], "the handed-over file must not be deleted by the round")
        self.assertFalse(uploader.is_alive(), "the CLI waits for the upload instead of killing it")
        self.assertTrue(any("delta already handed to the uploader is left to finish" in l for l in self.events.logs()))

    def test_known_limitation_refusal_after_close_timeout_on_last_round_exits_0(self):
        # NOT solved (documented limitation): if the heartbeat's 403 arrives only after close() gave up waiting
        # (JOIN_TIMEOUT), the closed last round can't be changed and the CLI exits 0. With more rounds left, the
        # next wait_for_round gets the 403 and exits 7 (see the test above with rounds=2).
        release = threading.Event(); self.addCleanup(release.set)
        def hb_round_status():
            threading.Event.wait(release, 30); return _resp_403("subnet temporarily blacklisted")
        rc = self._run_cli(self._race_routes(hb_round_status), rounds=1, hb_cls=self._capturing_hb(join_timeout=0.05))
        self.assertEqual(rc, 0)
        release.set(); self.hbs[0]._t.join(5)
        self.assertIsNone(self.hbs[0].rejected)

    def _run_cli(self, routes, rounds=1, hb_cls=None, real_submit=False):
        control = client.HeadlessRunControl(None)
        control.halt_event = _FastEvent()  # the 10 s pause between CLI rounds becomes 10 ms
        if callable(routes): routes = routes(control.halt_event)
        self.router = _Router(routes); self.events = _Events()
        args = argparse.Namespace(password=None, token="tok", anon=False, user=None, server=SERVER, backend="cpu",
                                  vram_gb=0.0, no_tokenizer=True, dry_run=False, rounds=rounds, no_update=True)
        patched = dict(_do_backend_setup=mock.DEFAULT, apply_vram_override=mock.DEFAULT, prune_hf_cache=mock.DEFAULT,
                       _check_updates_between_rounds=mock.DEFAULT, _clear_saved_token_if_matches=mock.DEFAULT)
        if not real_submit: patched["_submit_delta_background"] = mock.DEFAULT
        with mock.patch.multiple(client, **patched) as m, \
             mock.patch.object(client.signal, "signal") as sig, \
             mock.patch.object(client.time, "sleep", side_effect=(sleep := _fast_sleep(control.halt_event))), \
             mock.patch.object(client.requests, "get", side_effect=self.router), \
             mock.patch.object(client, "HeartbeatManager", hb_cls or client.HeartbeatManager):
            rc = client._run_headless(args, self.events, control=control)
        self.assertTrue(sig.called)  # handlers were "installed" on the mock, not in the test process
        self.assertFalse(sleep.expired(), "hit the test deadline instead of stopping")
        if not real_submit: m["_submit_delta_background"].assert_not_called()
        m["_clear_saved_token_if_matches"].assert_not_called()
        return rc

    def test_round_status_403_exits_7(self):
        rc = self._run_cli({"/fl/round_status": _resp_403("subnet temporarily blacklisted")})
        self.assertEqual(rc, 7)
        self.assertTrue(any("subnet temporarily blacklisted" in l for l in self.events.logs()))

    def test_heartbeat_block_mid_round_exits_7(self):
        rc = self._run_cli({
            "/fl/round_status": lambda n: _response(200, ROUND_OK) if n == 1 else _resp_403("subnet temporarily blacklisted"),
            "/fl/heartbeat": _response(200, {"status": "blocked"}),
            "/fl/task": _resp_wait(),
        })
        self.assertEqual(rc, 7)

    def test_heartbeat_block_during_weight_download_exits_7(self):
        rc = self._run_cli(_block_during_weight_stream, hb_cls=_FastHeartbeat)
        self.assertEqual(rc, 7)
        self.assertTrue(any("subnet temporarily blacklisted" in l for l in self.events.logs()))

    def test_fl_task_403_and_426_exit_7(self):
        for name, resp in (("403", _resp_403("subnet temporarily blacklisted")), ("426", _resp_426())):
            with self.subTest(status=name):
                rc = self._run_cli({"/fl/round_status": _response(200, ROUND_OK),
                                    "/fl/heartbeat": _response(200, {"status": "ok"}), "/fl/task": resp})
                self.assertEqual(rc, 7)

    def _race_routes(self, hb_round_status, main_round_status=lambda n: _response(200, ROUND_OK)):
        """round_status answered per caller: the CLI thread (wait_for_round) vs. the heartbeat thread."""
        main, main_calls, hb_entered = threading.current_thread(), [], threading.Event()
        def round_status(n):
            if threading.current_thread() is main:
                main_calls.append(1); return main_round_status(len(main_calls))
            hb_entered.set(); return hb_round_status()
        def task(n):  # handshake: the round may only end once the heartbeat check is in flight
            if n == 1: self.assertTrue(threading.Event.wait(hb_entered, 5), "heartbeat check never started")
            return _response(404)  # three 4xx -> the round ends normally, without a task
        return {"/fl/round_status": round_status, "/fl/heartbeat": _response(200, {"status": "ok"}), "/fl/task": task}

    def _capturing_hb(self, join_timeout=None):
        made = self.hbs = []
        class HB(client.HeartbeatManager):
            def __init__(s, *a, **k):
                super().__init__(*a, **k); s.shutdown = _FastEvent(); made.append(s)
                if join_timeout is not None: s.JOIN_TIMEOUT = join_timeout
        return HB

    def test_refusal_at_round_end_exits_7(self):
        # The heartbeat's 403 lands exactly while the round is closing (after hb.close() asked it to stop).
        def hb_round_status():
            threading.Event.wait(self.hbs[-1].shutdown, 5)
            return _resp_403("subnet temporarily blacklisted")
        rc = self._run_cli(self._race_routes(hb_round_status), hb_cls=self._capturing_hb())
        self.assertEqual(rc, 7, "a refusal recorded before the round closed must not end in success")
        self.assertFalse(self.hbs[0]._t.is_alive())

    def test_refusal_after_join_timeout_is_dropped_and_next_round_exits_7(self):
        # The heartbeat check is still hanging when the round closes (join times out); its late 403 must not
        # change the closed round. The next round's wait_for_round sees the block and exits 7.
        release = threading.Event(); self.addCleanup(release.set)
        def hb_round_status():
            threading.Event.wait(release, 30); return _resp_403("subnet temporarily blacklisted")
        routes = self._race_routes(hb_round_status, lambda n: _response(200, ROUND_OK) if n == 1 else _resp_403("subnet temporarily blacklisted"))
        rc = self._run_cli(routes, rounds=2, hb_cls=self._capturing_hb(join_timeout=0.05))
        self.assertEqual(rc, 7)
        hb = self.hbs[0]
        self.assertTrue(hb._t.is_alive(), "the late check must still be running when the CLI returns")
        release.set(); hb._t.join(5)
        self.assertFalse(hb._t.is_alive())
        self.assertIsNone(hb.rejected)
        self.assertFalse(any("refused this client during the round" in l for l in self.events.logs()))

    def test_refusal_waits_for_previous_upload_before_exit_7(self):
        # A daemon upload from an earlier round must not be killed by the process exit that follows rc 7.
        prev_file = Path(_TMP) / "delta-previous-round.gz"; prev_file.write_bytes(b"prev")
        started, finish, done = threading.Event(), threading.Event(), threading.Event()
        def previous_upload():
            started.set(); threading.Event.wait(finish, 10); done.set()
        prev = threading.Thread(target=previous_upload, daemon=True); prev.start(); self.assertTrue(started.wait(5))
        self.addCleanup(lambda: (finish.set(), prev.join(5), prev_file.unlink(missing_ok=True)))
        joining = []
        real_join = client._join_upload
        def join_upload(*a, **k):  # the CLI must be waiting for the upload before it may finish
            joining.append(1); finish.set(); return real_join(*a, **k)
        with mock.patch.object(client, "_upload_thread", prev), mock.patch.object(client, "_join_upload", side_effect=join_upload):
            rc = self._run_cli({"/fl/round_status": _resp_403("subnet temporarily blacklisted")})
        self.assertEqual(rc, 7)
        self.assertEqual(joining, [1])
        self.assertTrue(done.is_set(), "rc 7 was returned before the previous upload finished")
        self.assertTrue(prev_file.exists(), "the previous round's file is not ours to delete")

    def test_normal_round_still_exits_0(self):
        rc = self._run_cli({
            "/fl/round_status": _response(200, ROUND_OK),
            "/fl/heartbeat": _response(200, {"status": "ok"}),
            "/fl/task": _response(404),  # three 4xx -> no task this round, round ends normally
        })
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
