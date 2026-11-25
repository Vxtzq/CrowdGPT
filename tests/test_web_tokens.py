"""No token in any event streamed to browsers: neither the coordinator auth token nor the web UI token.

Drives the real Api paths that touch the auth token into a bound WebSink and scans every SSE frame.

    python -m unittest discover -s tests -v
"""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import crowd_web as cw
from test_coordinator_rejected import client
from test_web_ui import _drain, _LoopThread, _subscribe

AUTH_MARKER = "auth-tok-3f9a1c7e5b2d8046"
WEB_MARKER = "web-tok-71c4e09a2b5f6d38"
MARKERS = (AUTH_MARKER, WEB_MARKER)


def _leaks(frames):
    """(marker, frame) for every frame that contains a token marker."""
    return [(s, f) for f in frames for s in MARKERS if s in f]


class NoTokenInEvents(unittest.TestCase):
    def setUp(self):
        self.lt = _LoopThread()
        self.addCleanup(self.lt.close)
        self.sink = cw.WebSink(queue_size=1024)
        self.sink.bind(self.lt.loop)
        self.q = self.lt.run(_subscribe(self.sink))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for patcher in (
            mock.patch("builtins.print"),
            mock.patch.object(client, "AUTH_TOKEN_FILE", self.tmp / "auth_token.json"),
            mock.patch.object(client, "_save_auth_token"),
            mock.patch.object(client, "do_login", return_value=(AUTH_MARKER, None)),
            mock.patch.object(client, "get_available_backends", return_value={"cpu": True}),
            mock.patch.object(client, "get_gpu_list", return_value=[]),
            mock.patch.object(client, "_load_saved_gpu_idx", return_value=None),
            mock.patch.object(client, "apply_vram_override"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.api = self._api()

    def _api(self):
        api = client.Api.__new__(client.Api)  # skip __init__: no backend probing, no files
        api._lp = 0.0
        api.stop_event, api.thread, api._activity_monitor = threading.Event(), None, None
        api.auth_token, api.username, api.server_url = None, None, "https://coord.example"
        api.selected_backend, api.vram_gb = "cpu", 0.0
        api._vram_file = self.tmp / "selected_vram.json"
        api._ui_ready, api._resume = threading.Event(), False
        api._saved = {"token": AUTH_MARKER, "username": "alice", "server": api.server_url, "remember": True}
        api._add_sink(self.sink)
        return api

    def _frames(self):
        return self.lt.run(_drain(self.q))

    def _drive(self):
        api = self.api
        api._mark_ui_ready()                       # restored session: sets auth_token, emits login_success
        api._do_login("alice", "pw")               # fresh login: sets auth_token, emits login_success
        self.assertEqual(api.auth_token, AUTH_MARKER)
        with mock.patch.object(client, "do_login", return_value=(None, "bad_credentials")):
            api._do_login("alice", "wrong")        # login_error
        api.get_backends()                         # backends, vram, gpus
        api.set_vram(4)                            # log
        api.emit("status", "training")
        api.emit("stats", {"loss": 3.2, "step": 7, "user": api.username, "server": api.server_url})
        api.emit("log", f"Connected to {api.server_url} as {api.username}")
        api.logout()                               # log, logged_out

    def test_no_token_in_any_frame(self):
        self._drive()
        frames = self._frames()
        events = {f.split('"ev": "', 1)[1].split('"', 1)[0] for f in frames}
        self.assertTrue({"login_success", "login_error", "backends", "gpus", "vram",
                         "status", "stats", "log", "logged_out"} <= events, events)
        self.assertEqual(_leaks(frames), [])

    def test_injected_leak_is_caught(self):
        self._drive()
        self.api.emit("login_success", {"username": "alice", "token": AUTH_MARKER})
        self.api.emit("log", f"web UI: http://0.0.0.0:8765/?token={WEB_MARKER}")
        leaked = {s for s, _ in _leaks(self._frames())}
        self.assertEqual(leaked, set(MARKERS))


if __name__ == "__main__":
    unittest.main()
