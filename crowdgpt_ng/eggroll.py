"""A small, dependency-free EGGROLL-style mechanism prototype.

This is a *spike*, not a production optimizer. It implements the core idea from
the EGGROLL paper (low-rank perturbations at inference cost) on a toy linear
model so we can check convergence, determinism, and the size of a scalar-only
client upload before touching the real training loop.

The production question is whether the same mechanism works on the 658M model;
this module only proves the mechanics and gives us a testable baseline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

Matrix = list[list[float]]
Vector = list[float]


@dataclass(frozen=True)
class EggrollConfig:
    population: int = 16
    rank: int = 2
    sigma: float = 0.1
    lr: float = 0.1
    seed: int = 1


@dataclass(frozen=True)
class LowRankPerturbation:
    """A low-rank matrix perturbation plus a small bias perturbation."""

    a: Matrix
    b: Matrix
    bias: Vector
    seed: int

    @property
    def out_dim(self) -> int:
        return len(self.a)

    @property
    def in_dim(self) -> int:
        return len(self.b)


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        return []
    inner = len(b)
    cols = len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(len(a))]


def _transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)] if a else []


def _mse(w: Matrix, bias: Vector, data: Sequence[tuple[Vector, Vector]]) -> float:
    total = 0.0
    for x, y in data:
        pred = [sum(w[i][j] * x[j] for j in range(len(x))) + bias[i] for i in range(len(w))]
        total += sum((pred[i] - y[i]) ** 2 for i in range(len(y)))
    return total / (len(data) * len(data[0][1]))


def _scaled_perturbation(pert: LowRankPerturbation, sigma: float) -> Matrix:
    # A B^T / sqrt(rank) keeps the perturbation scale comparable to a
    # full-rank Gaussian draw with the same sigma.
    rank = len(pert.a[0]) if pert.a else 1
    scale = sigma / math.sqrt(max(1, rank))
    bt = _transpose(pert.b)
    return [[scale * v for v in row] for row in _matmul(pert.a, bt)]


def sample_perturbation(rng: random.Random, out_dim: int, in_dim: int, rank: int) -> LowRankPerturbation:
    if out_dim <= 0 or in_dim <= 0 or rank <= 0:
        raise ValueError("out_dim, in_dim, and rank must be positive")
    a = [[rng.gauss(0.0, 1.0) for _ in range(rank)] for _ in range(out_dim)]
    b = [[rng.gauss(0.0, 1.0) for _ in range(rank)] for _ in range(in_dim)]
    bias = [rng.gauss(0.0, 1.0) for _ in range(out_dim)]
    return LowRankPerturbation(a=a, b=b, bias=bias, seed=rng.randrange(1 << 31))


def perturbed(w: Matrix, bias: Vector, pert: LowRankPerturbation, sigma: float) -> tuple[Matrix, Vector]:
    delta = _scaled_perturbation(pert, sigma)
    w2 = [[w[i][j] + delta[i][j] for j in range(len(w[i]))] for i in range(len(w))]
    b2 = [bias[i] + sigma * pert.bias[i] for i in range(len(bias))]
    return w2, b2


def _standardize(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def eggroll_step(
    w: Matrix,
    bias: Vector,
    data: Sequence[tuple[Vector, Vector]],
    config: EggrollConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[Matrix, Vector, dict]:
    """One EGGROLL/ES update on a toy linear model.

    The population is evaluated at W + sigma * perturbation. The resulting
    scalar fitnesses are standardized and combined into a low-rank update.
    """
    cfg = config or EggrollConfig()
    rng = rng or random.Random(cfg.seed)
    out_dim, in_dim = len(w), len(w[0]) if w else 0
    perts = [sample_perturbation(rng, out_dim, in_dim, cfg.rank) for _ in range(cfg.population)]
    losses = []
    for pert in perts:
        wp, bp = perturbed(w, bias, pert, cfg.sigma)
        losses.append(_mse(wp, bp, data))
    scores = _standardize(losses)
    n = max(1, cfg.population)
    rank_scale = math.sqrt(max(1, cfg.rank))
    gw = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
    gb = [0.0 for _ in range(out_dim)]
    for score, pert in zip(scores, perts):
        delta = _matmul(pert.a, _transpose(pert.b))
        coef = score / (n * cfg.sigma * rank_scale)
        for i in range(out_dim):
            for j in range(in_dim):
                gw[i][j] += coef * delta[i][j]
            gb[i] += score * pert.bias[i] / (n * cfg.sigma)
    w2 = [[w[i][j] - cfg.lr * gw[i][j] for j in range(in_dim)] for i in range(out_dim)]
    b2 = [bias[i] - cfg.lr * gb[i] for i in range(out_dim)]
    dense_params = out_dim * in_dim + out_dim
    scalar_bytes = 8 + 4 * n
    dense_bytes = dense_params * 2
    diagnostics = {
        "mean_loss": sum(losses) / n,
        "best_loss": min(losses),
        "population": n,
        "rank": cfg.rank,
        "scalar_bytes": scalar_bytes,
        "dense_bytes": dense_bytes,
        "compression": dense_bytes / max(1, scalar_bytes),
        "dense_params": dense_params,
    }
    return w2, b2, diagnostics


def benchmark(
    steps: int = 100,
    config: EggrollConfig | None = None,
    data: Sequence[tuple[Vector, Vector]] | None = None,
) -> list[dict]:
    """Run the toy model and return one history row per step."""
    cfg = config or EggrollConfig()
    if data is None:
        data = _toy_data()
    w = [[0.0 for _ in range(len(data[0][0]))] for _ in range(len(data[0][1]))]
    b = [0.0 for _ in range(len(data[0][1]))]
    rng = random.Random(cfg.seed)
    history = []
    for step in range(steps):
        w, b, diag = eggroll_step(w, b, data, cfg, rng)
        row = {"step": step + 1, "loss": _mse(w, b, data), **diag}
        history.append(row)
    return history


def _toy_data() -> list[tuple[Vector, Vector]]:
    # Fixed linear regression problem: y = 2*x0 - 3*x1 + 0.5.
    rows = []
    for i in range(32):
        x = [(i % 7) / 7.0 - 0.5, (i % 5) / 5.0 - 0.5, 1.0]
        rows.append((x, [2.0 * x[0] - 3.0 * x[1] + 0.5]))
    return rows
