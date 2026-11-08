"""Tests for `--login`: log in, save the session, exit.

Loads its own stubbed copy of client.py through test_coordinator_rejected, so no network, GPU or real checkpoints.

    python -m unittest discover -s tests -v
"""
import argparse
import io
import json
import os
import shutil
import unittest
from unittest import mock

from test_coordinator_rejected import SERVER, _load_client

client, _TMP = _load_client()


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


def _args(**kw):
    return argparse.Namespace(**{"user": "alice", "password": "hunter22", "server": SERVER, **kw})


class RunLogin(unittest.TestCase):
    def setUp(self):
        client.AUTH_TOKEN_FILE.unlink(missing_ok=True)
        self.addCleanup(client.AUTH_TOKEN_FILE.unlink, missing_ok=True)

    def test_success_saves_a_session_that_headless_and_the_app_can_reuse(self):
        with mock.patch.object(client, "do_login", return_value=("tok-123", None)) as login:
            self.assertEqual(client.run_login(_args()), 0)
        login.assert_called_once_with(SERVER, "alice", "hunter22")
        saved = json.loads(client.AUTH_TOKEN_FILE.read_text())
        self.assertEqual(saved, {"token": "tok-123", "username": "alice", "server": SERVER, "remember": True})

    def test_rejected_login_keeps_the_previous_session(self):
        client._save_auth_token(SERVER, "alice", "old-token", True)
        with mock.patch.object(client, "do_login", return_value=(None, "invalid_credentials")):
            self.assertEqual(client.run_login(_args(password="wrong")), 1)
        self.assertEqual(json.loads(client.AUTH_TOKEN_FILE.read_text())["token"], "old-token")

    def test_without_a_password_or_terminal_it_fails_instead_of_hanging(self):
        env = {k: v for k, v in os.environ.items() if k != "CROWDGPT_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("sys.stdin", io.StringIO("")), \
                mock.patch.object(client, "do_login") as login:
            self.assertEqual(client.run_login(_args(password=None)), 2)
        login.assert_not_called()
        self.assertFalse(client.AUTH_TOKEN_FILE.exists())


if __name__ == "__main__":
    unittest.main()
