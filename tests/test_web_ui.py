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

import crowd_web as cw
from test_coordinator_rejected import _load_client, client


def _args(**kw):
    base = {"web": False, "web_host": None, "web_port": None, "web_auth": None, "web_token": None}
    base.update(kw)
    return argparse.Namespace(**base)


class ResolveWebConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = cw.resolve_web_config(_args(), {})
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 8765)
        self.assertEqual(cfg.auth, "auto")
        self.assertIsNone(cfg.token)
        self.assertEqual(cfg.allowed_hosts, ())
        self.assertFalse(cfg.any_host)

    def test_env_used_when_no_flag(self):
        env = {"CROWDGPT_WEB_HOST": "0.0.0.0", "CROWDGPT_WEB_PORT": "9000",
               "CROWDGPT_WEB_AUTH": "token", "CROWDGPT_WEB_TOKEN": "e" * 16}
        cfg = cw.resolve_web_config(_args(), env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("0.0.0.0", 9000, "token", "e" * 16))

    def test_flag_beats_env(self):
        env = {"CROWDGPT_WEB_HOST": "0.0.0.0", "CROWDGPT_WEB_PORT": "9000",
               "CROWDGPT_WEB_AUTH": "token", "CROWDGPT_WEB_TOKEN": "e" * 16}
        args = _args(web_host="::1", web_port=8000, web_auth="none", web_token="f" * 20)
        cfg = cw.resolve_web_config(args, env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("::1", 8000, "none", "f" * 20))

    def test_empty_env_counts_as_unset(self):
        env = {"CROWDGPT_WEB_HOST": "", "CROWDGPT_WEB_PORT": "", "CROWDGPT_WEB_AUTH": "",
               "CROWDGPT_WEB_TOKEN": "", "CROWDGPT_WEB_ALLOWED_HOSTS": ""}
        cfg = cw.resolve_web_config(_args(), env)
        self.assertEqual((cfg.host, cfg.port, cfg.auth, cfg.token), ("127.0.0.1", 8765, "auto", None))

    def test_auth_mode_is_case_insensitive(self):
        cfg = cw.resolve_web_config(_args(), {"CROWDGPT_WEB_AUTH": " Token "})
        self.assertEqual(cfg.auth, "token")

    def test_invalid_auth_mode(self):
        for src in (_args(web_auth="off"), ):
            with self.assertRaisesRegex(ValueError, "auth"):
                cw.resolve_web_config(src, {})
        with self.assertRaisesRegex(ValueError, "auth"):
            cw.resolve_web_config(_args(), {"CROWDGPT_WEB_AUTH": "nope"})

    def test_port_out_of_range(self):
        for bad in (0, -1, 65536):
            with self.subTest(port=bad), self.assertRaisesRegex(ValueError, "port"):
                cw.resolve_web_config(_args(web_port=bad), {})
        for bad in ("0", "70000", "abc"):
            with self.subTest(env_port=bad), self.assertRaisesRegex(ValueError, "port"):
                cw.resolve_web_config(_args(), {"CROWDGPT_WEB_PORT": bad})

    def test_port_bounds_accepted(self):
        self.assertEqual(cw.resolve_web_config(_args(web_port=1), {}).port, 1)
        self.assertEqual(cw.resolve_web_config(_args(web_port=65535), {}).port, 65535)

    def test_short_token_rejected(self):
        with self.assertRaisesRegex(ValueError, "16"):
            cw.resolve_web_config(_args(web_token="x" * 15), {})
        with self.assertRaisesRegex(ValueError, "16"):
            cw.resolve_web_config(_args(), {"CROWDGPT_WEB_TOKEN": "1234"})

    def test_token_of_16_accepted(self):
        self.assertEqual(cw.resolve_web_config(_args(web_token="x" * 16), {}).token, "x" * 16)

    def test_allowed_hosts_parsed(self):
        env = {"CROWDGPT_WEB_ALLOWED_HOSTS": " Crowd.Example.com , ,box.lan,crowd.example.com "}
        cfg = cw.resolve_web_config(_args(), env)
        self.assertEqual(cfg.allowed_hosts, ("crowd.example.com", "box.lan"))
        self.assertFalse(cfg.any_host)

    def test_allowed_hosts_star(self):
        cfg = cw.resolve_web_config(_args(), {"CROWDGPT_WEB_ALLOWED_HOSTS": "box.lan, *"})
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
        self.assertEqual(cw.resolve_web_token("override-token-abcdef", self.path), "override-token-abcdef")
        self.assertEqual(self.path.read_text(), "saved-token-from-file-123")

    def test_override_does_not_create_file(self):
        cw.resolve_web_token("override-token-abcdef", self.path)
        self.assertFalse(self.path.exists())

    def test_saved_file_used_and_stripped(self):
        self.path.write_text("saved-token-from-file-123\n")
        self.assertEqual(cw.resolve_web_token(None, self.path), "saved-token-from-file-123")

    def test_generated_when_no_file(self):
        token = cw.resolve_web_token(None, self.path)
        self.assertGreaterEqual(len(token), 40)
        self.assertEqual(self.path.read_text().strip(), token)

    def test_generated_file_mode_0600(self):
        cw.resolve_web_token(None, self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_generated_token_reused(self):
        first = cw.resolve_web_token(None, self.path)
        self.assertEqual(cw.resolve_web_token(None, self.path), first)

    def test_empty_file_regenerated(self):
        self.path.write_text("  \n")
        token = cw.resolve_web_token(None, self.path)
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
                self.assertIs(cw.is_loopback(host), want)

    def test_variants(self):
        cases = {"127.8.9.10": True, "LocalHost": True, "[::1]": True, " ::1 ": True,
                 "::ffff:127.0.0.1": True, "example.com": False, "": False}
        for host, want in cases.items():
            with self.subTest(host=host):
                self.assertIs(cw.is_loopback(host), want)


class AuthRequired(unittest.TestCase):
    def test_matrix(self):
        for mode in cw.WEB_AUTH_MODES:
            for host, loop in HOSTS.items():
                want = {"token": True, "none": False, "auto": not loop}[mode]
                with self.subTest(mode=mode, host=host):
                    self.assertIs(cw.auth_required(mode, host), want)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            cw.auth_required("open", "127.0.0.1")


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
                self.assertEqual(cw.web_url(sockname), want)


_TEMPLATE = (Path(__file__).resolve().parent.parent / "client.html").read_text(encoding="utf-8")
_SHIM_MARK = "window.__CROWD_WEB__"
_HTML = client.render_client_html(_TEMPLATE)


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
        out = cw.render_web_html(_HTML, {})
        for ph in ("__LOGO__", "__LOGO_DARK__", "__VERSION__"):
            self.assertNotIn(ph, out)

    def test_shim_once_before_first_page_script(self):
        out = cw.render_web_html(_HTML, {})
        self.assertEqual(out.count(_SHIM_MARK), 1)
        page_script = out.lower().index("<script", out.index(_SHIM_MARK))
        self.assertLess(out.index(_SHIM_MARK), page_script)
        self.assertEqual(out.lower().count("<script"), _TEMPLATE.lower().count("<script") + 1)

    def test_shim_wires_bridge(self):
        out = cw.render_web_html(_HTML, {})
        for needle in ("Proxy", "'/api/'", "EventSource('/events')", "handleEvent", "pywebviewready"):
            self.assertIn(needle, out)

    def test_rest_of_page_unchanged(self):
        out = cw.render_web_html(_HTML, {})
        start = out.index("<script")
        end = out.index("</script>", start) + len("</script>")
        self.assertEqual(out[:start] + out[end:], _HTML)

    def test_opts_cannot_close_script(self):
        out = cw.render_web_html(_HTML, {"x": "</script><b>"})
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

    def test_raises_clear_error_without_web_stack(self):
        with self.assertRaisesRegex(RuntimeError, "pip install fastapi uvicorn"):
            cw._import_web_stack()

    def test_client_imports_without_web_stack(self):
        mod, tmp = _load_client()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.assertTrue(callable(mod.parse_args))


class ImportCrowdWeb(unittest.TestCase):
    """The updater ships only client.py and client.html, so crowd_web.py may be missing."""
    def setUp(self):
        missing = object()
        saved = sys.modules.get("crowd_web", missing)
        sys.modules["crowd_web"] = None  # an import of it raises ImportError

        def restore():
            if saved is missing: sys.modules.pop("crowd_web", None)
            else: sys.modules["crowd_web"] = saved
        self.addCleanup(restore)

    def test_client_runs_without_crowd_web(self):
        from unittest import mock
        mod, tmp = _load_client()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with mock.patch("sys.argv", ["client.py", "--headless"]):
            self.assertTrue(mod.parse_args().headless)
        self.assertIn("<", mod.render_client_html("<html>__VERSION__</html>"))
        with self.assertRaisesRegex(RuntimeError, "crowd_web.py"):
            mod._import_crowd_web()

    def test_client_has_no_web_code_left(self):
        for name in ("WebSink", "WebServer", "create_web_app", "render_web_html", "resolve_web_config",
                     "resolve_web_token", "is_loopback", "auth_required", "web_url", "_import_web_stack"):
            self.assertFalse(hasattr(client, name), name)


_HAVE_WEB = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx", "uvicorn"))


class _FakeApi:
    thread = None


class _Alive:
    def is_alive(self): return True


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebApp(unittest.TestCase):
    def _client(self, api=None):
        from fastapi.testclient import TestClient
        app = cw.create_web_app(api or _FakeApi(), {"html": "<html>page</html>"})
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
        self.c = TestClient(cw.create_web_app(self.api, {"html": "x"}))

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


_TOKEN = "t" * 20 + "-good"


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebTokenAuth(unittest.TestCase):
    def _client(self, token=_TOKEN):
        from fastapi.testclient import TestClient
        self.api = _RecordingApi()
        opts = {"html": "<html>page</html>", "sink": cw.WebSink(), "token": token}
        return TestClient(cw.create_web_app(self.api, opts))

    def test_no_auth_needed_leaves_routes_open(self):
        c = self._client(token=None)
        self.assertEqual(c.get("/").text, "<html>page</html>")
        self.assertEqual(c.post("/api/stop", json=[]).status_code, 204)

    def test_good_query_token_sets_cookie_and_redirects(self):
        c = self._client()
        r = c.get("/", params={"token": _TOKEN}, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/")
        cookie = r.headers["set-cookie"]
        self.assertIn(f"crowdgpt_web_token={_TOKEN}", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertIn("Path=/", cookie)
        self.assertEqual(c.get("/").text, "<html>page</html>")  # the cookie now carries the auth

    def test_bad_query_token_gets_token_page_and_no_cookie(self):
        r = self._client().get("/", params={"token": "wrong"}, follow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("set-cookie", r.headers)
        self.assertIn('name="token"', r.text)

    def test_cookie_is_accepted(self):
        c = self._client()
        c.cookies.set("crowdgpt_web_token", _TOKEN)
        self.assertEqual(c.get("/").text, "<html>page</html>")
        self.assertEqual(c.post("/api/stop", json=[]).status_code, 204)

    def test_bad_cookie_is_rejected(self):
        c = self._client()
        c.cookies.set("crowdgpt_web_token", "wrong")
        self.assertEqual(c.post("/api/stop", json=[]).status_code, 401)
        self.assertEqual(self.api.calls, [])

    def test_bearer_is_accepted(self):
        c = self._client()
        r = c.post("/api/stop", json=[], headers={"Authorization": f"Bearer {_TOKEN}"})
        self.assertEqual(r.status_code, 204)
        self.assertEqual(self.api.calls, [("stop", ())])

    def test_bad_or_empty_bearer_is_rejected(self):
        c = self._client()
        for value in ("Bearer wrong", "Bearer ", "Basic " + _TOKEN, _TOKEN):
            with self.subTest(value=value):
                r = c.post("/api/stop", json=[], headers={"Authorization": value})
                self.assertEqual(r.status_code, 401)
        self.assertEqual(self.api.calls, [])

    def test_missing_token_on_each_route(self):
        c = self._client()
        page = c.get("/")
        self.assertEqual(page.status_code, 401)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn('name="token"', page.text)
        self.assertNotIn("page</html>", page.text)
        for method, path in (("POST", "/api/stop"), ("GET", "/events"), ("GET", "/nope")):
            with self.subTest(path=path):
                r = c.request(method, path, json=[] if method == "POST" else None)
                self.assertEqual(r.status_code, 401)
                self.assertNotIn('name="token"', r.text)
        self.assertEqual(self.api.calls, [])

    def test_healthz_stays_open_and_minimal(self):
        r = self._client().get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "status": "idle"})


class WebAllowedHosts(unittest.TestCase):
    def _hosts(self, bind, extra=()):
        from unittest import mock
        infos = [(None, None, None, "", ("192.168.1.20", 0)), (None, None, None, "", ("fe80::1", 0, 0, 0))]
        with mock.patch.object(cw.socket, "gethostname", return_value="Box"), \
                mock.patch.object(cw.socket, "getfqdn", return_value="box.lan"), \
                mock.patch.object(cw.socket, "getaddrinfo", return_value=infos):
            return cw.web_allowed_hosts(bind, extra)

    def test_loopback_bind(self):
        self.assertEqual(self._hosts("127.0.0.1"), {"localhost", "127.0.0.1", "::1"})

    def test_lan_ip_bind_adds_it(self):
        self.assertEqual(self._hosts("10.0.0.5"), {"localhost", "127.0.0.1", "::1", "10.0.0.5"})

    def test_wildcard_binds_add_hostname_and_interfaces(self):
        for bind in ("0.0.0.0", "::", "[::]"):
            with self.subTest(bind=bind):
                self.assertEqual(self._hosts(bind), {"localhost", "127.0.0.1", "::1", "box", "box.lan",
                                                     "192.168.1.20", "fe80::1"})

    def test_extra_hosts_and_star(self):
        self.assertIn("stats.example.com", self._hosts("127.0.0.1", ("stats.example.com",)))
        self.assertIsNone(self._hosts("127.0.0.1", ("stats.example.com", "*")))

    def test_lookup_failure_keeps_names(self):
        import socket
        from unittest import mock
        with mock.patch.object(cw.socket, "gethostname", return_value="box"), \
                mock.patch.object(cw.socket, "getfqdn", return_value="box"), \
                mock.patch.object(cw.socket, "getaddrinfo", side_effect=socket.gaierror):
            self.assertEqual(cw.web_allowed_hosts("0.0.0.0", ()), {"localhost", "127.0.0.1", "::1", "box"})


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebHostOriginChecks(unittest.TestCase):
    def _client(self, allowed=frozenset({"localhost", "127.0.0.1", "::1"}), base="http://127.0.0.1:8765", **extra):
        from fastapi.testclient import TestClient
        self.api = _RecordingApi()
        opts = {"html": "x", "allowed_hosts": allowed, **extra}
        return TestClient(cw.create_web_app(self.api, opts), base_url=base)

    def test_same_origin_post_accepted(self):
        r = self._client().post("/api/stop", json=[], headers={"Origin": "http://127.0.0.1:8765"})
        self.assertEqual(r.status_code, 204)

    def test_post_without_origin_accepted(self):
        self.assertEqual(self._client().post("/api/stop", json=[]).status_code, 204)

    def test_cross_site_post_rejected(self):
        c = self._client()
        for origin in ("http://evil.example", "null", "http://127.0.0.1.evil.example:8765"):
            with self.subTest(origin=origin):
                self.assertEqual(c.post("/api/stop", json=[], headers={"Origin": origin}).status_code, 403)
        self.assertEqual(self.api.calls, [])

    def test_rebinding_host_rejected_on_every_route(self):
        c = self._client(base="http://evil.example:8765")
        for method, path in (("GET", "/"), ("GET", "/healthz"), ("POST", "/api/stop")):
            with self.subTest(path=path):
                self.assertEqual(c.request(method, path, json=[] if method == "POST" else None).status_code, 400)
        self.assertEqual(self.api.calls, [])

    def test_ipv6_host_accepted(self):
        self.assertEqual(self._client(base="http://[::1]:8765").get("/healthz").status_code, 200)

    def test_extra_allowed_host_accepted_behind_rewriting_proxy(self):
        allowed = frozenset({"localhost", "127.0.0.1", "::1", "stats.example.com"})
        c = self._client(allowed=allowed)  # the proxy rewrote Host to 127.0.0.1:8765
        r = c.post("/api/stop", json=[], headers={"Origin": "https://stats.example.com"})
        self.assertEqual(r.status_code, 204)
        self.assertEqual(self._client(allowed=allowed, base="https://stats.example.com").get("/").status_code, 200)

    def test_star_turns_both_checks_off(self):
        c = self._client(allowed=None, base="http://evil.example")
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.post("/api/stop", json=[], headers={"Origin": "http://other.example"}).status_code, 204)

    def test_host_check_runs_before_token(self):
        c = self._client(base="http://evil.example", token="t" * 20)
        self.assertEqual(c.get("/").status_code, 400)


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebServerThread(unittest.TestCase):
    def test_start_fetch_stop(self):
        import json
        import time
        import urllib.request
        app = cw.create_web_app(_FakeApi(), {"html": "x"})
        srv = cw.WebServer(app, "127.0.0.1", 0)
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


class _LoopThread:
    """A running asyncio loop on its own thread, standing in for uvicorn's."""
    def __init__(self):
        import asyncio
        import threading
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro):
        import asyncio
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(5)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


async def _drain(q):
    out = []
    while not q.empty(): out.append(q.get_nowait())
    return out


def _frames(raw):
    """(id, payload) per SSE frame string."""
    import json
    out = []
    for frame in raw:
        lines = dict(line.split(": ", 1) for line in frame.strip().split("\n"))
        out.append((int(lines["id"]), json.loads(lines["data"])))
    return out


async def _subscribe(sink): return sink.subscribe()


class WebSinkFanOut(unittest.TestCase):
    def setUp(self):
        self.lt = _LoopThread()
        self.addCleanup(self.lt.close)

    def test_send_before_bind_is_noop(self):
        sink = cw.WebSink()
        sink.send("log", "early")  # no loop yet: must not raise
        sink.bind(self.lt.loop)
        q = self.lt.run(_subscribe(sink))
        self.assertEqual(self.lt.run(_drain(q)), [])

    def test_two_subscribers_same_events_in_order(self):
        sink = cw.WebSink()
        sink.bind(self.lt.loop)
        a, b = self.lt.run(_subscribe(sink)), self.lt.run(_subscribe(sink))
        for i in range(5): sink.send("log", f"line {i}")  # from this thread, not the loop's
        sink.send("stats", {"loss": 1.5})
        want = [(i + 1, {"ev": "log", "data": f"line {i}"}) for i in range(5)] + [(6, {"ev": "stats", "data": {"loss": 1.5}})]
        self.assertEqual(_frames(self.lt.run(_drain(a))), want)
        self.assertEqual(_frames(self.lt.run(_drain(b))), want)

    def test_stalled_subscriber_drops_oldest_and_blocks_nobody(self):
        import time
        sink = cw.WebSink(queue_size=4)
        sink.bind(self.lt.loop)
        stalled, live = self.lt.run(_subscribe(sink)), self.lt.run(_subscribe(sink))
        t0 = time.monotonic()
        for i in range(100): sink.send("log", i)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual([p["data"] for _, p in _frames(self.lt.run(_drain(live)))], [96, 97, 98, 99])
        for i in range(100, 103): sink.send("log", i)
        self.assertEqual([p["data"] for _, p in _frames(self.lt.run(_drain(live)))], [100, 101, 102])
        self.assertEqual([p["data"] for _, p in _frames(self.lt.run(_drain(stalled)))], [99, 100, 101, 102])

    def test_unsubscribe_stops_delivery(self):
        sink = cw.WebSink()
        sink.bind(self.lt.loop)
        q = self.lt.run(_subscribe(sink))
        self.assertEqual(sink.subscriber_count(), 1)
        sink.unsubscribe(q)
        self.assertEqual(sink.subscriber_count(), 0)
        sink.send("log", "x")
        self.assertEqual(self.lt.run(_drain(q)), [])


async def _backlog(sink, last_id): return sink.backlog(last_id)


class WebSinkReplay(unittest.TestCase):
    """Late joiners get the state snapshot and log tail; reconnects get only what they missed."""
    def setUp(self):
        self.lt = _LoopThread()
        self.addCleanup(self.lt.close)

    def _sink(self, **kw):
        sink = cw.WebSink(**kw)
        sink.bind(self.lt.loop)
        return sink

    def _replay(self, sink, last_id=None):
        return _frames(self.lt.run(_backlog(sink, last_id)))

    def test_late_join_gets_state_and_log_tail_in_order(self):
        sink = self._sink()
        sink.send("login_success", {"username": "alice"})   # 1
        sink.send("log", "a")                                # 2
        sink.send("status", "training")                      # 3
        sink.send("login_error", {"code": "network"})        # 4: transient, not replayed
        sink.send("stats", {"loss": 2.0})                    # 5
        sink.send("log", "b")                                # 6
        self.assertEqual(self._replay(sink), [
            (1, {"ev": "login_success", "data": {"username": "alice"}}),
            (2, {"ev": "log", "data": "a"}),
            (3, {"ev": "status", "data": "training"}),
            (5, {"ev": "stats", "data": {"loss": 2.0}}),
            (6, {"ev": "log", "data": "b"}),
        ])

    def test_latest_event_per_state_group(self):
        sink = self._sink()
        for ev, data in [("status", "idle"), ("login_success", {"username": "a"}), ("stats", {"loss": 3}),
                         ("overlay_show", {"title": "x"}), ("started", {}), ("status", "training"),
                         ("cal_stats", {"loss": 2}), ("logged_out", {}), ("overlay_hide", {}), ("stopped", {})]:
            sink.send(ev, data)
        self.assertEqual([p["ev"] for _, p in self._replay(sink)],
                         ["status", "cal_stats", "logged_out", "overlay_hide", "stopped"])

    def test_resume_gets_only_what_was_missed(self):
        sink = self._sink()
        sink.send("status", "training")      # 1
        sink.send("log", "seen")              # 2
        sink.send("stats", {"loss": 3})       # 3: the browser saw up to here
        sink.send("log", "missed")            # 4
        sink.send("stats", {"loss": 2})       # 5
        self.assertEqual(self._replay(sink, 3), [
            (4, {"ev": "log", "data": "missed"}),
            (5, {"ev": "stats", "data": {"loss": 2}}),
        ])
        self.assertEqual(self._replay(sink, 5), [])

    def test_resume_past_ring_buffer_gets_snapshot(self):
        sink = self._sink(log_size=3)
        sink.send("status", "training")                  # 1
        for i in range(6): sink.send("log", i)            # 2..7; 2..4 leave the ring
        snap = [(1, {"ev": "status", "data": "training"})] + [(n, {"ev": "log", "data": n - 2}) for n in (5, 6, 7)]
        self.assertEqual(self._replay(sink, 2), snap)    # 3 and 4 are gone
        self.assertEqual(self._replay(sink, 4), snap[1:])  # nothing after 4 was lost

    def test_unknown_last_id_gets_snapshot(self):
        sink = self._sink()
        sink.send("status", "training")
        sink.send("log", "x")
        snap = self._replay(sink)
        self.assertEqual(self._replay(sink, 99), snap)   # ahead of the sink: a server restart
        self.assertEqual(len(snap), 2)

    def test_log_ring_holds_500_lines(self):
        sink = self._sink()
        for i in range(600): sink.send("log", i)
        tail = self._replay(sink)
        self.assertEqual([p["data"] for _, p in tail], list(range(100, 600)))


@unittest.skipUnless(_HAVE_WEB, "needs fastapi, uvicorn and httpx")
class WebEventsStream(unittest.TestCase):
    def _wait(self, cond, timeout=5):
        import time
        deadline = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > deadline: self.fail("condition not met in time")
            time.sleep(0.02)

    def test_stream_frames_heartbeat_and_disconnect(self):
        import json
        import urllib.request
        sink = cw.WebSink()
        app = cw.create_web_app(_FakeApi(), {"html": "x", "sink": sink, "heartbeat": 0.1})
        srv = cw.WebServer(app, "127.0.0.1", 0)
        port = srv.start(timeout=5)[1]
        self.addCleanup(srv.stop)
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=5)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self._wait(lambda: sink.subscriber_count() == 1)
        self.assertEqual(r.readline(), b": ping\n")  # heartbeat before any event
        self.assertEqual(r.readline(), b"\n")
        sink.send("status", "training")
        lines = []
        while len(lines) < 2:
            line = r.readline().decode()
            if line.startswith(":") or line == "\n": continue
            lines.append(line.rstrip("\n"))
        self.assertEqual(lines[0], "id: 1")
        self.assertEqual(json.loads(lines[1].removeprefix("data: ")), {"ev": "status", "data": "training"})
        r.close()
        self._wait(lambda: sink.subscriber_count() == 0)

    def _read_frames(self, r, n, timeout=5):
        """The next n event frames; heartbeats keep the socket alive, so the deadline is ours."""
        import json
        import time
        deadline = time.monotonic() + timeout
        out, cur = [], {}
        while len(out) < n:
            if time.monotonic() > deadline: self.fail(f"got {out} before the deadline, wanted {n} frames")
            line = r.readline().decode().rstrip("\n")
            if line.startswith(":"): continue
            if line == "":
                if cur: out.append((int(cur["id"]), json.loads(cur["data"]))); cur = {}
                continue
            k, v = line.split(": ", 1); cur[k] = v
        return out

    def test_stream_replays_from_last_event_id(self):
        import urllib.request
        sink = cw.WebSink()
        app = cw.create_web_app(_FakeApi(), {"html": "x", "sink": sink, "heartbeat": 0.1})
        srv = cw.WebServer(app, "127.0.0.1", 0)
        port = srv.start(timeout=5)[1]
        self.addCleanup(srv.stop)
        sink.send("status", "training")   # 1
        sink.send("log", "seen")           # 2
        sink.send("log", "missed")         # 3
        url = f"http://127.0.0.1:{port}/events"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Last-Event-ID": "2"}), timeout=5) as r:
            self.assertEqual(self._read_frames(r, 1), [(3, {"ev": "log", "data": "missed"})])
            sink.send("log", "live")       # 4
            self.assertEqual(self._read_frames(r, 1), [(4, {"ev": "log", "data": "live"})])
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Last-Event-ID": "junk"}), timeout=5) as r:
            self.assertEqual([i for i, _ in self._read_frames(r, 4)], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
