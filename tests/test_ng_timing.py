"""Tests for the opt-in client timing log (P0.5)."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client_module(tmp_path_factory):
    sandbox = tmp_path_factory.mktemp("timing-client")
    shutil.copy2(ROOT / "client.py", sandbox / "client.py")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("ng_timing_client", sandbox / "client.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    yield module
    sys.modules.pop("ng_timing_client", None)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_unset_is_inert(client_module, tmp_path, monkeypatch):
    monkeypatch.delenv("CROWDGPT_TIMING_LOG", raising=False)
    recorder = client_module.TimingRecorder.from_env()
    recorder.set_round(7)
    recorder.phase("queue", 1.25, bytes=10, tokens=20)
    recorder.finish(ok=True)
    assert list(tmp_path.iterdir()) == []
    assert recorder.enabled is False


def test_set_writes_one_row_per_phase(client_module, tmp_path, monkeypatch):
    path = tmp_path / "nested" / "timing.jsonl"
    monkeypatch.setenv("CROWDGPT_TIMING_LOG", str(path))
    recorder = client_module.TimingRecorder.from_env()
    recorder.set_round(338)
    recorder.phase("queue", 0.5)
    recorder.phase("weights", 2.0, bytes=2_015_576_064)
    recorder.phase("train", 3.0, tokens=4_400_000)
    recorder.phase("upload", 4.0, bytes=1_386_000_000, tokens=4_400_000, ok=False)
    recorder.finish(ok=False, error="TimeoutError")
    recorder.finish(ok=True)  # idempotent: no second total row

    rows = _rows(path)
    assert [row["phase"] for row in rows] == ["queue", "weights", "train", "upload", "total"]
    assert all(row["schema"] == "ng-timing-1" for row in rows)
    assert all(row["round"] == 338 for row in rows)
    assert rows[1]["bytes"] == 2_015_576_064
    assert rows[2]["tokens"] == 4_400_000
    assert rows[3]["ok"] is False
    assert rows[4]["ok"] is False
    assert rows[4]["error"] == "TimeoutError"
    assert path.stat().st_mode & 0o777 == 0o600


def test_rows_contain_no_identity_fields(client_module, tmp_path, monkeypatch):
    path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("CROWDGPT_TIMING_LOG", str(path))
    recorder = client_module.TimingRecorder.from_env()
    recorder.set_round(1)
    recorder.phase("queue", 0.1, bytes=1, tokens=2, ok=True)
    recorder.finish(ok=True)

    text = path.read_text(encoding="utf-8")
    for forbidden in ("10.0.0.", "Bearer", "Authorization", "secret", "http://", "https://"):
        assert forbidden not in text
    assert set(_rows(path)[0]) == {
        "schema", "ts", "round", "phase", "seconds", "bytes", "tokens", "ok",
    }


def test_write_failure_does_not_raise(client_module, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    recorder = client_module.TimingRecorder(path=blocker / "timing.jsonl", round_id=1)
    recorder.phase("queue", 0.1)
    recorder.finish(ok=False, error="OverflowError")
    assert recorder.enabled is True
