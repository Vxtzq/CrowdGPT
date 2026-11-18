"""DiLoCo round loop: averaging, staleness cap, round closure, checkpoint/resume."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# The repo's CI job is deliberately standard-library only (no pytest, no torch),
# and runs `python -m unittest discover`. Skip this module there instead of
# erroring on import; the tests run under pytest locally and in any env with deps.
try:  # pragma: no cover - depends on the environment, not on the code
    import numpy as np  # noqa: F401
    import pytest  # noqa: F401
    import torch  # noqa: F401
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs pytest, torch and numpy: {_missing}")
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_diloco import DilocoConfig, DilocoCoordinator, DilocoRoundError  # noqa: E402

N = 16


def _coord(**kw) -> DilocoCoordinator:
    cfg = DilocoConfig(**kw)
    return DilocoCoordinator(cfg, total_params=N)


def test_identical_peers_leave_theta_unchanged():
    c = _coord(min_peers=2)
    theta = np.ones(N, dtype=np.float32)
    c.submit("a", 0, theta.copy())
    c.submit("b", 0, theta.copy())
    assert c.ready() and c.peers_ready() == 2
    new, stats = c.finalize(theta)
    # theta - mean(theta_k) == 0, so the outer step is a no-op even with momentum.
    assert np.allclose(new, theta, atol=1e-6)
    assert stats["peers"] == 2 and stats["innerSteps"] == 25


def test_min_peers_is_enforced():
    c = _coord(min_peers=2)
    c.submit("a", 0, np.zeros(N, dtype=np.float32))
    with pytest.raises(DilocoRoundError):
        c.finalize(np.zeros(N, dtype=np.float32))


def test_one_stale_round_is_accepted_and_two_beyond_is_dropped():
    c = _coord(min_peers=2, max_stale_steps=2)
    c.submit("a", 2, np.zeros(N, dtype=np.float32))      # 2 ahead: allowed
    assert c.peers_ready() == 1
    c.submit("b", 3, np.zeros(N, dtype=np.float32))      # 3 ahead: dropped
    assert c.dropped_stale == 1 and c.peers_ready() == 1


def test_a_closed_round_cannot_be_used_again():
    c = _coord(min_peers=2)
    theta = np.zeros(N, dtype=np.float32)
    c.submit("a", 0, theta)
    c.submit("b", 0, theta)
    c.finalize(theta)
    assert c.round_index == 1 and c.rounds_completed == 1
    with pytest.raises(DilocoRoundError):
        c.submit("late", 0, theta)
    with pytest.raises(DilocoRoundError):
        c.finalize(theta)                                 # nothing pending


def test_outer_step_moves_toward_the_peer_average():
    c = _coord(min_peers=2, outer_lr=0.7, outer_momentum=0.0, nesterov=False)
    theta = np.zeros(N, dtype=np.float32)
    c.submit("a", 0, np.full(N, 2.0, dtype=np.float32))
    c.submit("b", 0, np.full(N, 2.0, dtype=np.float32))
    new, _ = c.finalize(theta)
    # grad = theta - mean = -2 ; non-nesterov step: theta - lr*grad = 0 + 1.4
    assert np.allclose(new, np.full(N, 1.4), atol=1e-6)


def test_checkpoint_and_resume_restore_the_round_and_momentum():
    c = _coord(min_peers=2)
    theta = np.zeros(N, dtype=np.float32)
    c.submit("a", 0, np.full(N, 1.0, dtype=np.float32))
    c.submit("b", 0, np.full(N, 1.0, dtype=np.float32))
    c.finalize(theta)
    prev = None if c.outer._prev is None else np.asarray(c.outer._prev).copy()
    cp = c.checkpoint(tokens_seen=1234, inner_steps_done=25)

    fresh = _coord(min_peers=2)
    fresh.resume(cp)
    assert fresh.round_index == c.round_index == 1
    assert cp.tokens_seen == 1234 and cp.inner_steps_done == 25
    if prev is None:
        assert fresh.outer._prev is None
    else:
        assert np.allclose(np.asarray(fresh.outer._prev), prev, atol=1e-6)


def test_checkpoint_survives_a_dict_round_trip():
    c = _coord(min_peers=2)
    theta = np.zeros(N, dtype=np.float32)
    c.submit("a", 0, np.full(N, 3.0, dtype=np.float32))
    c.submit("b", 0, np.full(N, 3.0, dtype=np.float32))
    c.finalize(theta)
    cp = c.checkpoint(tokens_seen=99)
    again = type(cp).from_dict(cp.to_dict())
    assert again.round_index == cp.round_index
    assert again.tokens_seen == 99
