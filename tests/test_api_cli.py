"""The web entry point logs in from CLI credentials (crowd_web) and caps Api rounds (WEB-12).

Reuses the stubbed client import from test_coordinator_rejected: no torch, no GUI, no network.

    python -m unittest discover -s tests -v
"""
import argparse
import threading
import unittest
from unittest import mock

import crowd_web as cw
from test_coordinator_rejected import SERVER, _Events, _fast_sleep, client


def _args(**kw):
    base = {"anon": False, "token": None, "user": None, "password": None}
    return argparse.Namespace(**{**base, **kw})


class _FakeApi:
    def __init__(self):
        self.server_url = SERVER; self.auth_token = None; self.username = None
        self.emit = _Events(); self.start = mock.MagicMock()


class CliLogin(unittest.TestCase):
    def setUp(self):
        self.api = _FakeApi()
        self.do_login = mock.MagicMock(return_value=("tok-pw", None))

    def _login(self, env=None, saved=None, autostart=False, **kw):
        creds = cw.cli_credentials(_args(**kw), env or {})
        return cw.web_autologin(self.api, creds, saved, self.do_login, autostart=autostart)

    def _success(self):
        return [d for ev, d in self.api.emit.items if ev == "login_success"]

    def test_anon(self):
        self.assertTrue(self._login(anon=True, token="t", user="u", password="p"))
        self.assertIsNone(self.api.auth_token)
        self.assertEqual(self._success()[0]["username"], "anonymous")
        self.do_login.assert_not_called()

    def test_token_from_flag_or_env(self):
        self.assertTrue(self._login(token="tok-flag", user="u", password="p"))
        self.assertEqual(self.api.auth_token, "tok-flag")
        self.do_login.assert_not_called()
        self.api = _FakeApi()
        self.assertTrue(self._login(env={"CROWDGPT_TOKEN": "tok-env"}))
        self.assertEqual(self.api.auth_token, "tok-env")

    def test_user_and_password_from_flag_or_env(self):
        self.assertTrue(self._login(user="u", password="pw"))
        self.do_login.assert_called_once_with(SERVER, "u", "pw")
        self.assertEqual((self.api.auth_token, self.api.username), ("tok-pw", "u"))
        self.assertEqual(self._success()[0]["username"], "u")
        self.api = _FakeApi()
        self.assertTrue(self._login(env={"CROWDGPT_PASSWORD": "envpw"}, user="u"))
        self.assertEqual(self.do_login.call_args.args, (SERVER, "u", "envpw"))

    def test_failed_login_emits_error_and_does_not_start(self):
        self.do_login.return_value = (None, "invalid_credentials")
        self.assertFalse(self._login(autostart=True, user="u", password="bad"))
        self.assertIn(("login_error", {"code": "invalid_credentials"}), self.api.emit.items)
        self.assertEqual(self._success(), [])
        self.assertIsNone(self.api.auth_token)
        self.api.start.assert_not_called()

    def test_user_without_password_is_an_error(self):
        self.assertFalse(self._login(autostart=True, user="u"))
        self.assertIn(("login_error", {"code": "missing_password"}), self.api.emit.items)
        self.do_login.assert_not_called()
        self.api.start.assert_not_called()

    def test_saved_session_for_the_same_server(self):
        saved = {"token": "tok-saved", "username": "sam", "server": SERVER + "/"}
        self.assertTrue(self._login(saved=saved))
        self.assertEqual((self.api.auth_token, self.api.username), ("tok-saved", "sam"))

    def test_saved_session_for_another_server_is_ignored(self):
        saved = {"token": "tok-saved", "username": "sam", "server": "https://elsewhere"}
        self.assertFalse(self._login(saved=saved, autostart=True))
        self.assertIsNone(self.api.auth_token)
        self.assertEqual(self.api.emit.items, [])
        self.api.start.assert_not_called()

    def test_no_credentials_is_quiet(self):
        self.assertFalse(self._login(autostart=True))
        self.assertEqual(self.api.emit.items, [])
        self.api.start.assert_not_called()

    def test_autostart_starts_once_after_login(self):
        self.assertTrue(self._login(autostart=True, token="t"))
        self.api.start.assert_called_once_with()
        self.api = _FakeApi()
        self.assertTrue(self._login(token="t"))
        self.api.start.assert_not_called()

    def test_cli_credentials_are_not_saved(self):
        with mock.patch.object(client, "_save_auth_token") as save:
            for kw in ({"token": "t"}, {"user": "u", "password": "p"}, {"anon": True}):
                with self.subTest(kw=kw):
                    self.api = _FakeApi()
                    self._login(**kw)
        save.assert_not_called()


class RoundCap(unittest.TestCase):
    SAFETY = 6  # a regression fails instead of looping forever

    def setUp(self):
        self.api = client.Api.__new__(client.Api)
        self.api.emit = self.events = _Events()
        self.api.stop_event = threading.Event()
        self.api.auth_token = "tok"; self.api.server_url = SERVER
        self.api.selected_backend = "cpu"; self.api.vram_gb = 0.0
        self.api._check_updates_between_rounds = mock.MagicMock()
        p = mock.patch.multiple(client, _do_backend_setup=mock.DEFAULT, apply_vram_override=mock.DEFAULT,
                                _gpu_preflight=mock.DEFAULT)
        self.patched = p.start(); self.addCleanup(p.stop)
        self.patched["_gpu_preflight"].return_value = True
        p = mock.patch.object(client.time, "sleep", side_effect=_fast_sleep(self.api.stop_event))
        p.start(); self.addCleanup(p.stop)
        self.rounds = 0

    def _run(self, fail=False):
        def round_fn(*_a, **_k):
            self.rounds += 1
            if self.rounds >= self.SAFETY: self.api.stop_event.set()
            if fail: raise RuntimeError("boom")
        with mock.patch.object(client, "run_single_round_wrapper", side_effect=round_fn):
            self.api._rs()

    def test_stops_after_max_rounds(self):
        self.api.max_rounds = 2
        self._run()
        self.assertEqual(self.rounds, 2)
        self.assertTrue(any("Reached 2 rounds" in line for line in self.events.logs()), self.events.logs())
        self.assertEqual(self.events.items[-2], ("status", "idle"))

    def test_failed_rounds_count_too(self):
        self.api.max_rounds = 2
        self._run(fail=True)
        self.assertEqual(self.rounds, 2)

    def test_zero_means_no_cap(self):
        self._run()  # no max_rounds set: the class default applies
        self.assertEqual(self.rounds, self.SAFETY)


if __name__ == "__main__":
    unittest.main()
