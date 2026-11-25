"""Api.emit fans each event out to every registered sink (desktop window today, browsers later).

Reuses the stubbed client import from test_coordinator_rejected: no torch, no GUI, no network.

    python -m unittest discover -s tests -v
"""
import json
import threading
import unittest
from unittest import mock

from test_coordinator_rejected import client


class _Sink:
    def __init__(self): self.got = []
    def send(self, ev, data): self.got.append((ev, data))


def _bare_api():
    api = client.Api.__new__(client.Api)  # skip __init__: no backend probing, no files
    api._lp = 0.0
    return api


class EmitFanOut(unittest.TestCase):
    def setUp(self):
        self.api = _bare_api()
        p = mock.patch("builtins.print"); p.start(); self.addCleanup(p.stop)

    def test_every_sink_gets_sanitized_event(self):
        a, b = _Sink(), _Sink()
        self.api._add_sink(a); self.api._add_sink(b)
        self.api.emit("status", {"loss": float("nan"), 3: (1, 2)})
        expected = [("status", {"loss": 0.0, "3": [1, 2]})]
        self.assertEqual(a.got, expected)
        self.assertEqual(b.got, expected)

    def test_zero_sinks_is_a_noop(self):
        with mock.patch("builtins.print") as p:
            self.api.emit("log", "hello")  # must not raise, must not print
        p.assert_not_called()

    def test_failing_sink_does_not_starve_the_others(self):
        bad, good = mock.Mock(), _Sink()
        bad.send.side_effect = RuntimeError("boom")
        self.api._add_sink(bad); self.api._add_sink(good)
        self.api.emit("log", "x")
        self.assertEqual(good.got, [("log", "x")])

    def test_stats_throttle_still_applies(self):
        s = _Sink(); self.api._add_sink(s)
        with mock.patch.object(client.time, "time", side_effect=[100.0, 100.05, 100.2]):
            for i in range(3): self.api.emit("stats", {"i": i})
        self.assertEqual([d["i"] for _, d in s.got], [0, 2])

    def test_preview_loss_throttle_still_applies(self):
        s = _Sink(); self.api._add_sink(s)
        with mock.patch.object(client.time, "time", side_effect=[100.0, 100.3, 100.5]):
            for i in range(3): self.api.emit("preview_loss", i)
        self.assertEqual([d for _, d in s.got], [0, 2])


class PywebviewSinkPump(unittest.TestCase):
    def test_waits_for_ready_then_evaluates_same_script(self):
        window, ready, done = mock.Mock(), threading.Event(), threading.Event()
        window.evaluate_js.side_effect = lambda s: done.set()
        sink = client.PywebviewSink(window, ready)
        sink.send("status", {"a": 1})
        self.assertFalse(done.wait(0.3), "pump must wait for the page to load")
        ready.set()
        self.assertTrue(done.wait(2.0))
        window.evaluate_js.assert_called_once_with(
            f'window.handleEvent({json.dumps("status")}, {json.dumps({"a": 1})});')


if __name__ == "__main__":
    unittest.main()
