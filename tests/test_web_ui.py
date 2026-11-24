"""Web UI: config, server, auth and event streaming.

Reuses the stubbed client import from test_coordinator_rejected: no torch, no GUI, no network.

    python -m unittest discover -s tests -v
"""
import argparse
import unittest

from test_coordinator_rejected import client


def _args(**kw):
    base = dict(web=False, web_host=None, web_port=None, web_auth=None, web_token=None)
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


if __name__ == "__main__":
    unittest.main()
