"""Web UI: config, server, auth and event streaming.

Reuses the stubbed client import from test_coordinator_rejected: no torch, no GUI, no network.

    python -m unittest discover -s tests -v
"""
import argparse
import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from test_coordinator_rejected import _load_client, client


def _args(**kw):
    base = {"web": False, "web_host": None, "web_port": None, "web_auth": None, "web_token": None}
    base.update(kw)
    return argparse.Namespace(**base)


class ResolveWebConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = client.resolve_web_config(_args(), {})
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 8765)
        self.assertEqual(cfg.auth, "auto")
        self.assertIsNone(cfg.token)
        self.assertEqual(cfg.allowed_hosts, ())
        self.assertFalse(cfg.any_host)

    def test_env_used_when_no_flag(self):
        env = {"CROWDGPT_WEB_HOST": "0.0.0.0", "CROWDGPT_WEB_PORT": "9000",
               "CROWDGPT_WEB_AUTH": "token", "CROWDGPT_WEB_TOKEN": "e" * 16}
        cfg = client.resolve_web_config(_args(), env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("0.0.0.0", 9000, "token", "e" * 16))

    def test_flag_beats_env(self):
        env = {"CROWDGPT_WEB_HOST": "0.0.0.0", "CROWDGPT_WEB_PORT": "9000",
               "CROWDGPT_WEB_AUTH": "token", "CROWDGPT_WEB_TOKEN": "e" * 16}
        args = _args(web_host="::1", web_port=8000, web_auth="none", web_token="f" * 20)
        cfg = client.resolve_web_config(args, env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("::1", 8000, "none", "f" * 20))

    def test_empty_env_counts_as_unset(self):
        env = {"CROWDGPT_WEB_HOST": "", "CROWDGPT_WEB_PORT": "", "CROWDGPT_WEB_AUTH": "",
               "CROWDGPT_WEB_TOKEN": "", "CROWDGPT_WEB_ALLOWED_HOSTS": ""}
        cfg = client.resolve_web_config(_args(), env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("127.0.0.1", 8765, "auto", None))

    def test_auth_mode_is_case_insensitive(self):
        cfg = client.resolve_web_config(_args(), {"CROWDGPT_WEB_AUTH": " Token "})
        self.assertEqual(cfg.auth, "token")

    def test_invalid_auth_mode(self):
        for src in (_args(web_auth="off"), ):
            with self.assertRaisesRegex(ValueError, "auth"):
                client.resolve_web_config(src, {})
        with self.assertRaisesRegex(ValueError, "auth"):
            client.resolve_web_config(_args(), {"CROWDGPT_WEB_AUTH": "nope"})

    def test_port_out_of_range(self):
        for bad in (0, -1, 65536):
            with self.subTest(port=bad), self.assertRaisesRegex(ValueError, "port"):
                client.resolve_web_config(_args(web_port=bad), {})
        for bad in ("0", "70000", "abc"):
            with self.subTest(env_port=bad), self.assertRaisesRegex(ValueError, "port"):
                client.resolve_web_config(_args(), {"CROWDGPT_WEB_PORT": bad})

    def test_port_bounds_accepted(self):
        self.assertEqual(client.resolve_web_config(_args(web_port=1), {}).port, 1)
        self.assertEqual(client.resolve_web_config(_args(web_port=65535), {}).port, 65535)

    def test_short_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "16"):
            client.resolve_web_config(_args(web_token="x" * 15), {})
        with self.assertRaisesRegex(ValueError, "16"):
            client.resolve_web_config(_args(), {"CROWDGPT_WEB_TOKEN": "1234"})

    def test_token_of_16_accepted(self):
        self.assertEqual(client.resolve_web_config(_args(web_token="x" * 16), {}).token, "x" * 16)

    def test_allowed_hosts_parsed(self):
        env = {"CROWDGPT_WEB_ALLOWED_HOSTS": " Crowd.Example.com , ,box.lan,crowd.example.com "}
        cfg = client.resolve_web_config(_args(), env)
        self.assertEqual(cfg.allowed_hosts, ("crowd.example.com", "box.lan"))
        self.assertFalse(cfg.any_host)

    def test_allowed_hosts_star(self):
        cfg = client.resolve_web_config(_args(), {"CROWDGPT_WEB_ALLOWED_HOSTS": "box.lan, *"})
        self.assertTrue(cfg.any_host)


class WebFlags(unittest.TestCase):
    def _parse(self, argv):
        from unittest import mock
        with mock.patch("sys.argv", ["client.py", *argv]), mock.patch.dict("os.environ", {}, clear=False):
            return client.parse_args()

    def test_flags_parse(self):
        a = self._parse(["--web", "--web-host", "::", "--web-port", "9001",
                         "--web-auth", "none", "--web-token", "t" * 16])
        self.assertTrue(a.web)
        self.assertEqual((a.web_host, a.web_port, a.web_auth, a.web_token), ("::", 9001, "none", "t" * 16))

    def test_flags_default_to_none(self):
        a = self._parse([])
        self.assertFalse(a.web)
        self.assertEqual((a.web_host, a.web_port, a.web_auth, a.web_token), (None, None, None, None))


class ResolveWebToken(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "web_token"

    def tearDown(self):
        self._tmp.cleanup()

    def test_override_wins_over_saved_file(self):
        self.path.write_text("saved-token-from-file-123")
        self.assertEqual(client.resolve_web_token("override-token-abcdef", self.path), "override-token-abcdef")
        self.assertEqual(self.path.read_text(), "saved-token-from-file-123")

    def test_override_does_not_create_file(self):
        client.resolve_web_token("override-token-abcdef", self.path)
        self.assertFalse(self.path.exists())

    def test_saved_file_used_and_stripped(self):
        self.path.write_text("saved-token-from-file-123\n")
        self.assertEqual(client.resolve_web_token(None, self.path), "saved-token-from-file-123")

    def test_generated_when_no_file(self):
        token = client.resolve_web_token(None, self.path)
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(self.path.read_text().strip(), token)

    def test_generated_file_mode_0600(self):
        client.resolve_web_token(None, self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_generated_token_reused(self):
        first = client.resolve_web_token(None, self.path)
        self.assertEqual(client.resolve_web_token(None, self.path), first)

    def test_empty_file_regenerated(self):
        self.path.write_text("  \n")
        token = client.resolve_web_token(None, self.path)
        self.assertTrue(token)
        self.assertEqual(self.path.read_text().strip(), token)

    def test_default_path_is_checkpoint_web_token(self):
        self.assertEqual(client.WEB_TOKEN_FILE, client.CHECKPOINT_DIR / "web_token")


HOSTS = {  # host -> loopback?
    "127.0.0.1": True, "::1": True, "localhost": True,
    "0.0.0.0": False, "::": False, "192.168.1.20": False,
}


class IsLoopback(unittest.TestCase):
    def test_table(self):
        for host, want in HOSTS.items():
            with self.subTest(host=host):
                self.assertIs(client.is_loopback(host), want)

    def test_variants(self):
        cases = {"127.8.9.10": True, "LocalHost": True, "[::1]": True, " ::1 ": True,
                 "::ffff:127.0.0.1": True, "example.com": False, "": False}
        for host, want in cases.items():
            with self.subTest(host=host):
                self.assertIs(client.is_loopback(host), want)


class AuthRequired(unittest.TestCase):
    def test_matrix(self):
        for mode in client.WEB_AUTH_MODES:
            for host, loop in HOSTS.items():
                want = {"token": True, "none": False, "auto": not loop}[mode]
                with self.subTest(mode=mode, host=host):
                    self.assertIs(client.auth_required(mode, host), want)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            client.auth_required("open", "127.0.0.1")


class WebUrl(unittest.TestCase):
    def test_formats_bound_address(self):
        cases = [
            (("127.0.0.1", 8765), "http://127.0.0.1:8765"),
            (("0.0.0.0", 80), "http://0.0.0.0:80"),
            (("::1", 8765, 0, 0), "http://[::1]:8765"),
            (("::", 9000, 0, 0), "http://[::]:9000"),
        ]
        for sockname, want in cases:
            with self.subTest(sockname=sockname):
                self.assertEqual(client.web_url(sockname), want)


_TEMPLATE = (Path(__file__).resolve().parent.parent / "client.html").read_text(encoding="utf-8")
_SHIM_MARK = "window.__CROWD_WEB__"


class RenderClientHtml(unittest.TestCase):
    def test_placeholders_replaced(self):
        out = client.render_client_html(_TEMPLATE)
        for ph in ("__LOGO__", "__LOGO_DARK__", "__VERSION__"):
            self.assertNotIn(ph, out)

    def test_matches_previous_gui_output(self):
        want = (_TEMPLATE.replace("__LOGO__", client.LOGO_URI_LIGHT)
                .replace("__LOGO_DARK__", client.LOGO_URI_DARK)
                .replace("__VERSION__", client.CLIENT_VERSION))
        self.assertEqual(client.render_client_html(_TEMPLATE), want)

    def test_no_shim_in_gui_output(self):
        self.assertNotIn(_SHIM_MARK, client.render_client_html(_TEMPLATE))


class RenderWebHtml(unittest.TestCase):
    def test_placeholders_replaced(self):
        out = client.render_web_html(_TEMPLATE, {})
        for ph in ("__LOGO__", "__LOGO_DARK__", "__VERSION__"):
            self.assertNotIn(ph, out)

    def test_shim_once_before_first_page_script(self):
        out = client.render_web_html(_TEMPLATE, {})
        self.assertEqual(out.count(_SHIM_MARK), 1)
        page_script = out.lower().index("<script", out.index(_SHIM_MARK))
        self.assertLess(out.index(_SHIM_MARK), page_script)
        self.assertEqual(out.lower().count("<script"), _TEMPLATE.lower().count("<script") + 1)

    def test_shim_wires_bridge(self):
        out = client.render_web_html(_TEMPLATE, {})
        for needle in ("Proxy", "'/api/'", "EventSource('/events')", "handleEvent", "pywebviewready"):
            self.assertIn(needle, out)

    def test_rest_of_page_unchanged(self):
        out = client.render_web_html(_TEMPLATE, {})
        start = out.index("<script")
        end = out.index("</script>", start) + len("</script>")
        self.assertEqual(out[:start] + out[end:], client.render_client_html(_TEMPLATE))

    def test_opts_cannot_close_script(self):
        out = client.render_web_html(_TEMPLATE, {"x": "</script><b>"})
        self.assertNotIn("</script><b>", out)
        self.assertIn(_SHIM_MARK, out)


class ImportWebStack(unittest.TestCase):
    _BLOCKED = ("fastapi", "uvicorn")

    def setUp(self):
        missing = object()
        saved = {name: sys.modules.get(name, missing) for name in self._BLOCKED}
        for name in self._BLOCKED:
            sys.modules[name] = None  # an import of it raises ImportError

        def restore():
            for name, orig in saved.items():
                if orig is missing: sys.modules.pop(name, None)
                else: sys.modules[name] = orig
        self.addCleanup(restore)

    def test_client_imports_without_web_stack(self):
        mod, tmp = _load_client()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with self.assertRaisesRegex(RuntimeError, "pip install fastapi uvicorn"):
            mod._import_web_stack()


_HAVE_WEB = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "uvicorn"))


class _FakeApi:
    thread = None


class _Alive:
    def is_alive(self): return True


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebApp(unittest.TestCase):
    def _client(self, api=None):
        from fastapi.testclient import TestClient
        app = client.create_web_app(api or _FakeApi(), {"html": "<html>page</html>"})
        return TestClient(app)

    def test_root_serves_page(self):
        r = self._client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertEqual(r.text, "<html>page</html>")

    def test_healthz_idle(self):
        r = self._client().get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "status": "idle"})

    def test_healthz_running(self):
        api = _FakeApi(); api.thread = _Alive()
        self.assertEqual(self._client(api).get("/healthz").json(), {"ok": True, "status": "running"})

    def test_unknown_and_docs_are_404(self):
        c = self._client()
        for path in ("/nope", "/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                self.assertEqual(c.get(path).status_code, 404)


class _RecordingApi(_FakeApi):
    """Every page-facing Api method, recording its calls; plus private ones that must stay unreachable."""
    def __init__(self): self.calls = []
    def _rec(self, name, *a): self.calls.append((name, a))
    def get_backends(self): self._rec("get_backends")
    def set_gpu(self, idx): self._rec("set_gpu", idx)
    def set_vram(self, gb): self._rec("set_vram", gb)
    def set_backend(self, backend_name): self._rec("set_backend", backend_name)
    def login(self, s, u, p, remember=False): self._rec("login", s, u, p, remember)
    def register(self, s, u, email, p, remember=False): self._rec("register", s, u, email, p, remember)
    def login_anon(self): self._rec("login_anon")
    def start(self): self._rec("start")
    def stop(self): self._rec("stop")
    def logout(self): self._rec("logout")
    def emit(self, ev, data): self._rec("emit", ev, data)
    def _rs(self): self._rec("_rs")
    def _do_login(self, *a): self._rec("_do_login", *a)


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebApiDispatch(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.api = _RecordingApi()
        self.c = TestClient(client.create_web_app(self.api, {"html": "x"}))

    def test_allowed_calls_arrive_with_args(self):
        cases = [
            ("get_backends", []), ("set_gpu", [1]), ("set_vram", [6.5]), ("set_backend", ["cpu"]),
            ("login", ["https://s", "u", "pw", True]), ("register", ["https://s", "u", "e@x", "pw"]),
            ("login_anon", []), ("start", []), ("stop", []), ("logout", []),
        ]
        for name, args in cases:
            with self.subTest(name=name):
                self.api.calls.clear()
                r = self.c.post(f"/api/{name}", json=args)
                self.assertEqual(r.status_code, 204)
                self.assertEqual(self.api.calls[0][0], name)
                self.assertEqual(list(self.api.calls[0][1][:len(args)]), args)

    def test_unlisted_names_are_404(self):
        for name in ("_rs", "_do_login", "emit", "nope", "__init__"):
            with self.subTest(name=name):
                self.assertEqual(self.c.post(f"/api/{name}", json=[]).status_code, 404)
        self.assertEqual(self.api.calls, [])

    def test_non_array_body_is_400(self):
        for body in (b'{"a": 1}', b'"x"', b"1", b"not json", b""):
            with self.subTest(body=body):
                r = self.c.post("/api/stop", content=body, headers={"content-type": "application/json"})
                self.assertEqual(r.status_code, 400)
        self.assertEqual(self.api.calls, [])

    def test_wrong_arg_count_is_400(self):
        self.assertEqual(self.c.post("/api/set_gpu", json=[]).status_code, 400)
        self.assertEqual(self.c.post("/api/stop", json=[1]).status_code, 400)
        self.assertEqual(self.api.calls, [])

    def test_get_is_not_allowed(self):
        self.assertEqual(self.c.get("/api/stop").status_code, 405)
        self.assertEqual(self.api.calls, [])


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebServerThread(unittest.TestCase):
    def test_start_fetch_stop(self):
        import json
        import time
        import urllib.request
        app = client.create_web_app(_FakeApi(), {"html": "x"})
        srv = client.WebServer(app, "127.0.0.1", 0)
        host, port = srv.start(timeout=5)[:2]
        self.addCleanup(srv.stop)
        self.assertEqual(host, "127.0.0.1")
        self.assertGreater(port, 0)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
            self.assertEqual(json.loads(r.read()), {"ok": True, "status": "idle"})
        t0 = time.monotonic()
        srv.stop(timeout=5)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertFalse(srv.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
