"""Orchestrator failover tests (CROWDGPT_ORCHESTRATORS).

Standard library only: the client itself is imported through the torch/numpy/requests stubs from
test_coordinator_rejected, and every HTTP call is mocked. No GPU, no network.

A connection error/timeout must move to the next orchestrator; an HTTP response (including 4xx) must be
returned unchanged so the existing 401/403/426 handling still classifies it.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

try:
    import test_coordinator_rejected as base
except ImportError:
    from tests import test_coordinator_rejected as base

client = base.client

A = "https://a.example"
B = "https://b.example"
C = "https://c.example"
ROUND_OK = {"current_round": 1, "in_cooldown": False, "is_aggregating": False,
            "max_round_hours": 2, "round_elapsed_hours": 0}


class _Reply:
    """Minimal requests.Response stand-in."""

    def __init__(self, status=200, body=None, headers=None, json_error=False):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self._json_error = json_error
        self.closed = False

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._body if self._body is not None else {}

    @property
    def text(self):
        return "" if self._body is None else str(self._body)

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Events:
    def __init__(self):
        self.items = []
        self._lock = threading.Lock()

    def __call__(self, event, data=None):
        with self._lock:
            self.items.append((event, data))

    def logs(self):
        return [str(data) for event, data in self.items if event == "log"]


def _env(*urls):
    return mock.patch.dict(os.environ, {"CROWDGPT_ORCHESTRATORS": ",".join(urls)})


class OrchestratorUrls(unittest.TestCase):
    def test_primary_first_then_env_without_duplicates(self):
        with _env(f" {B}/ ", C, B):
            self.assertEqual(client.orchestrator_urls(A), [A, B, C])

    def test_defaults_to_the_builtin_server(self):
        with mock.patch.dict(os.environ, {"CROWDGPT_ORCHESTRATORS": ""}):
            self.assertEqual(client.orchestrator_urls(), [client.DEFAULT_SERVER])

    def test_failover_url_rotates_and_preserves_the_path(self):
        with _env(B, C):
            urls = client.orchestrator_urls(A)
        self.assertEqual(client._failover_url(f"{A}/fl/submit", urls), f"{B}/fl/submit")
        self.assertEqual(client._failover_url(f"{C}/fl/task?format=bf16", urls), f"{A}/fl/task?format=bf16")

    def test_failover_url_leaves_unknown_urls_alone(self):
        with _env(B):
            urls = client.orchestrator_urls(A)
        self.assertEqual(client._failover_url("https://other.example/fl/submit", urls),
                         "https://other.example/fl/submit")


class GetFailover(unittest.TestCase):
    def test_connection_error_moves_to_the_next_orchestrator(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if url.startswith(A):
                raise OSError("connection refused")
            return _Reply(200)

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get) as get:
            reply = client._get_with_failover(A, "/fl/round_status", headers={"X": "1"},
                                              timeout=30, retries=2)
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(calls, [f"{A}/fl/round_status", f"{B}/fl/round_status"])
        self.assertEqual(get.call_args.kwargs["headers"], {"X": "1"})
        self.assertEqual(get.call_args.kwargs["timeout"], 30)

    def test_http_response_is_returned_unchanged_without_rotating(self):
        forbidden = _Reply(403, {"detail": "subnet temporarily blacklisted"})
        with _env(B), mock.patch.object(client.requests, "get", return_value=forbidden) as get:
            reply = client._get_with_failover(A, "/fl/round_status", retries=4)
        self.assertIs(reply, forbidden)
        self.assertEqual(get.call_count, 1)
        self.assertTrue(get.call_args.args[0].startswith(A))

    def test_params_are_forwarded(self):
        with _env(B), mock.patch.object(client.requests, "get", return_value=_Reply(200)) as get:
            client._get_with_failover(A, "/fl/heartbeat", params={"taskId": "t1"})
        self.assertEqual(get.call_args.kwargs["params"], {"taskId": "t1"})

    def test_every_attempt_exhausted_raises(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            raise OSError("refused")

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get):
            with self.assertRaises(RuntimeError) as cm:
                client._get_with_failover(A, "/fl/round_status", retries=2)
        self.assertIn("/fl/round_status", str(cm.exception))
        self.assertEqual(calls, [f"{A}/fl/round_status", f"{B}/fl/round_status"])

    def test_stop_already_set_never_requests(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.object(client.requests, "get") as get:
            with self.assertRaises(KeyboardInterrupt):
                client._get_with_failover(A, "/fl/round_status", se=stop)
        get.assert_not_called()

    def test_stop_during_a_retry_aborts_at_once(self):
        stop = threading.Event()

        def fake_get(url, **kwargs):
            stop.set()          # the user stops the client while the request is failing
            raise OSError("refused")

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get) as get:
            with self.assertRaises(KeyboardInterrupt):
                client._get_with_failover(A, "/fl/round_status", se=stop, retries=4)
        self.assertEqual(get.call_count, 1)


class CallersUseFailover(unittest.TestCase):
    def test_wait_for_round_moves_off_a_dead_primary(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if url.startswith(A):
                raise OSError("no route to host")
            return _Reply(200, dict(ROUND_OK))

        events = _Events()
        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get):
            status = client.wait_for_round(A, {}, threading.Event(), events)
        self.assertEqual(status, ROUND_OK)
        self.assertEqual(calls, [f"{A}/fl/round_status", f"{B}/fl/round_status"])

    def test_fetch_task_metadata_moves_off_a_dead_primary(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if url.startswith(A):
                raise OSError("connection refused")
            return _Reply(200, {"taskId": "t1"}, {"content-type": "application/json"})

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get):
            md = client.fetch_task_metadata(A, {}, threading.Event(), _Events())
        self.assertEqual(md, {"taskId": "t1"})
        self.assertEqual(calls, [f"{A}/fl/task?format=bf16&skip_weights=true",
                                 f"{B}/fl/task?format=bf16&skip_weights=true"])

    def test_fetch_task_metadata_403_is_not_rotated(self):
        responses = [_Reply(403, {"detail": "account suspended"}), _Reply(200, {"taskId": "nope"},
                                                                         {"content-type": "application/json"})]
        with _env(B), mock.patch.object(client.requests, "get", side_effect=responses) as get:
            with self.assertRaises(client.CoordinatorRejected) as cm:
                client.fetch_task_metadata(A, {}, threading.Event(), _Events())
        self.assertEqual(cm.exception.status, 403)
        self.assertEqual(get.call_count, 1, "an HTTP refusal must be classified, not retried elsewhere")

    def test_fetch_task_streams_from_the_next_orchestrator(self):
        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            if url.startswith(A):
                raise OSError("connection refused")
            return _Reply(403, {"detail": "subnet temporarily blacklisted"})

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get):
            with self.assertRaises(client.CoordinatorRejected) as cm:
                client.fetch_task(A, {}, "bf16", threading.Event(), _Events())
        # the failover moved to B, and B's 403 kept its existing refusal classification
        self.assertEqual(cm.exception.status, 403)
        self.assertIn("subnet temporarily blacklisted", str(cm.exception))
        self.assertEqual([url for url, _ in calls], [f"{A}/fl/task?format=bf16", f"{B}/fl/task?format=bf16"])
        second = calls[1][1]
        self.assertTrue(second["stream"])
        self.assertEqual(second["timeout"], (client.STREAM_CONNECT_TIMEOUT, client.STREAM_READ_TIMEOUT))
        self.assertEqual(second["timeout"][1], 120, "the streamed read timeout stays 120s")

    def test_heartbeat_moves_off_a_dead_primary(self):
        stop = threading.Event()
        events = _Events()
        hb = client.HeartbeatManager(A, {}, 1, stop, events)
        hb.tid = "task-9"
        hit_b = threading.Event()
        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            if "/fl/heartbeat" not in url:
                return _Reply(200, dict(ROUND_OK))
            if url.startswith(A):
                raise OSError("connection refused")
            hit_b.set()
            return _Reply(200, {"status": "ok"})

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "get", side_effect=fake_get):
            hb.start()
            try:
                self.assertTrue(hit_b.wait(5), "the heartbeat never reached the second orchestrator")
            finally:
                hb.stop()
                hb._t.join(5)
        self.assertFalse(hb._t.is_alive())
        self.assertTrue(any(url.startswith(B) for url, _ in calls))
        b_call = next(kwargs for url, kwargs in calls
                      if url.startswith(B) and "/fl/heartbeat" in url)
        self.assertEqual(b_call["params"], {"taskId": "task-9"})

    def test_post_rotates_on_a_failed_attempt(self):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            if url.startswith(A):
                raise OSError("connection refused")
            return _Reply(200, {})

        with _env(B), mock.patch.object(client.time, "sleep"), \
             mock.patch.object(client.requests, "post", side_effect=fake_post):
            reply = client._post_json_with_retry(f"{A}/fl/submit/init", {}, {}, _Events(),
                                                 threading.Event(), retries=2, srv=A)
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(calls, [f"{A}/fl/submit/init", f"{B}/fl/submit/init"])


if __name__ == "__main__":
    unittest.main()
