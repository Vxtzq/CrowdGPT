"""The --web entry point: routing from main and run_web's startup and ordered shutdown (WEB-13).

Reuses the stubbed client import from test_coordinator_rejected: no torch, no GUI, no network.
The server, Api and process-wide helpers are faked; FastAPI isn't needed.

    python -m unittest discover -s tests -v
"""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import crowd_web as cw
from test_coordinator_rejected import client

_TEMPLATE = "<html><img src='__LOGO__'><img src='__LOGO_DARK__'>v__VERSION__<script>page()</script></html>"


def _parse(*argv):
    with mock.patch("sys.argv", ["client.py", *argv]):
        return client.parse_args()


class PickRunner(unittest.TestCase):
    def test_routes(self):
        cases = [(("--web",), client.run_web), (("--web", "--headless"), client.run_web),
                 (("--headless",), client.run_headless), ((), client.run_gui)]
        for argv, runner in cases:
            with self.subTest(argv=argv):
                self.assertIs(client._pick_runner(_parse(*argv)), runner)


class _Thread:
    def __init__(self, calls): self.calls = calls
    def is_alive(self): return True
    def join(self, timeout=None): self.calls.append(("thread.join", timeout))


class _FakeApi:
    def __init__(self, calls):
        self.calls = calls; self.sinks = []; self.events = []
        self.server_url = None; self.max_rounds = 0; self._saved = None
        self.auth_token = None; self.username = None; self.selected_backend = "cpu"
        self.thread = _Thread(calls)
        self.start = mock.MagicMock()
    def _add_sink(self, sink): self.sinks.append(sink)
    def emit(self, ev, data): self.events.append((ev, data))
    def stop(self): self.calls.append(("api.stop",))


class _FakeServer:
    def __init__(self, calls, app, host, port):
        self.calls, self.app, self.host, self.port = calls, app, host, port
    def start(self, timeout=10.0):
        self.calls.append(("server.start", self.host, self.port)); return (self.host, self.port)
    def stop(self, timeout=10.0): self.calls.append(("server.stop",))


class RunWeb(unittest.TestCase):
    def setUp(self):
        self.calls = []; self.apis = []; self.servers = []; self.app_opts = []
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.token_file = Path(tmp.name) / "web_token"

        def make_api():
            api = _FakeApi(self.calls); self.apis.append(api); return api

        def make_server(app, host, port):
            srv = _FakeServer(self.calls, app, host, port); self.servers.append(srv); return srv

        def make_app(api, opts):
            self.app_opts.append(opts); return "app"

        patches = [
            mock.patch.object(client, "Api", side_effect=make_api),
            mock.patch.object(client, "start_tokenizer_loader"),
            mock.patch.object(client, "_prewarm_gpu"),
            mock.patch.object(client, "_start_stall_watchdog"),
            mock.patch.object(client, "_install_stop_signal_handlers"),
            mock.patch.object(client, "_load_client_html_template", return_value=_TEMPLATE),
            mock.patch.object(client, "_join_upload", side_effect=lambda *a, **k: self.calls.append(("join_upload", k.get("timeout")))),
            mock.patch.object(client, "do_login", return_value=(None, "network")),
            mock.patch.object(client, "WEB_TOKEN_FILE", self.token_file),
            mock.patch.object(cw, "WebServer", side_effect=make_server),
            mock.patch.object(cw, "create_web_app", side_effect=make_app),
            mock.patch.dict(client.os.environ, {}, clear=False),
        ]
        for p in patches: p.start(); self.addCleanup(p.stop)
        for name in ("CROWDGPT_WEB_HOST", "CROWDGPT_WEB_PORT", "CROWDGPT_WEB_AUTH", "CROWDGPT_WEB_TOKEN",
                     "CROWDGPT_WEB_ALLOWED_HOSTS", "CROWDGPT_TOKEN", "CROWDGPT_PASSWORD"):
            client.os.environ.pop(name, None)

    def _run(self, *argv):
        stop = threading.Event(); stop.set()  # SIGTERM already arrived: run_web starts, then shuts down
        return client.run_web(_parse("--web", *argv), stop_event=stop)

    def test_shutdown_runs_in_order(self):
        self.assertEqual(self._run(), 0)
        names = [c[0] for c in self.calls]
        self.assertEqual(names, ["server.start", "api.stop", "thread.join", "join_upload", "server.stop"])
        join_timeout = self.calls[2][1]; upload_timeout = self.calls[3][1]
        self.assertLess(join_timeout + upload_timeout, 120)  # inside compose's stop_grace_period: 2m

    def test_server_gets_page_sink_and_checks(self):
        self._run("--server", "https://coord.example", "--rounds", "3")
        api, opts, srv = self.apis[0], self.app_opts[0], self.servers[0]
        self.assertEqual((api.server_url, api.max_rounds), ("https://coord.example", 3))
        self.assertEqual(len(api.sinks), 1)
        self.assertIs(opts["sink"], api.sinks[0])
        self.assertIsInstance(opts["sink"], cw.WebSink)
        self.assertIn("window.__CROWD_WEB__", opts["html"])
        self.assertIn("v" + client.CLIENT_VERSION, opts["html"])
        self.assertNotIn("__LOGO__", opts["html"])
        self.assertIn("127.0.0.1", opts["allowed_hosts"])
        self.assertIsNone(opts["token"])  # auto on loopback: no token
        self.assertEqual((srv.host, srv.port), ("127.0.0.1", 8765))

    def test_token_required_off_loopback(self):
        self._run("--web-host", "10.0.0.5")
        token = self.app_opts[0]["token"]
        self.assertTrue(token)
        self.assertEqual(self.token_file.read_text().strip(), token)
        self.assertIn("10.0.0.5", self.app_opts[0]["allowed_hosts"])

    def test_star_turns_host_check_off(self):
        client.os.environ["CROWDGPT_WEB_ALLOWED_HOSTS"] = "*"
        self._run()
        self.assertIsNone(self.app_opts[0]["allowed_hosts"])

    def test_cli_credentials_log_in_without_starting(self):
        self._run("--token", "tok-cli")
        api = self.apis[0]
        self.assertEqual(api.auth_token, "tok-cli")
        self.assertIn("login_success", [ev for ev, _ in api.events])
        api.start.assert_not_called()

    def test_bad_config_returns_2_before_starting(self):
        self.assertEqual(self._run("--web-port", "0"), 2)
        self.assertEqual((self.apis, self.servers), ([], []))

    def test_missing_page_returns_4(self):
        client._load_client_html_template.return_value = None
        self.assertEqual(self._run(), 4)
        self.assertEqual(self.servers, [])


if __name__ == "__main__":
    unittest.main()
