#!/usr/bin/env python3
"""Benchmark the EGGROLL / low-rank-ES mechanism on a small Transformer LM.

This is a *scaled-down* mechanism benchmark, not a run of the 658M production
model and not a production optimizer. It answers three questions the toy
linear-regression spike in ``crowdgpt_ng/eggroll.py`` cannot:

1. Does the low-rank perturbation + scalar-fitness ES update reduce the loss of
   a real (if tiny) next-token Transformer, with no gradients and no dense
   parameter deltas?
2. What does one EGGROLL round cost in tokens/s and peak GPU memory at a size
   that still fits a single consumer GPU?
3. How large is a scalar-only upload compared with a dense bf16 parameter
   upload for the same model?

The EGGROLL trick used here is that a perturbation ``dW = (sigma/sqrt(r)) A B^T``
never has to be materialised: for a linear layer ``y = x W^T`` the perturbed
activation is ``x W^T + (sigma/sqrt(r)) (x B) A^T``, i.e. two thin matmuls.
The forward pass below applies perturbations that way. The update is the usual
ES estimator ``theta <- theta - (lr / (n * sigma)) sum_i z_i dW_i`` with
z-scored fitness, where ``z_i`` is the standardised scalar loss of member i.

Only ``torch`` is required; the module is importable without a GPU.

    python tools/bench_eggroll_model.py --device cuda --steps 60 \
        --population 32 --rank 4 --sigma 0.02 --lr 0.05 \
        --dim 64 --layers 2 --heads 4 --seq 32 --vocab 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the small Transformer LM."""

    dim: int = 64
    layers: int = 2
    heads: int = 4
    vocab: int = 64
    seq: int = 32
    batch: int = 4

    def __post_init__(self) -> None:
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        for name in ("dim", "layers", "heads", "vocab", "seq", "batch"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EggrollOptions:
    """ES / EGGROLL hyperparameters."""

    population: int = 32
    rank: int = 4
    sigma: float = 0.02
    lr: float = 0.05
    steps: int = 60
    seed: int = 1

    def __post_init__(self) -> None:
        for name in ("population", "rank", "steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sigma <= 0 or self.lr <= 0:
            raise ValueError("sigma and lr must be positive")


@dataclass
class Perturbation:
    """One population member's perturbation, kept in low-rank form."""

    low: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)  # 2D: (out, r), (in, r)
    full: dict[str, torch.Tensor] = field(default_factory=dict)  # 1D: (n,)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def _layer_names(layer: int) -> list[str]:
    return [
        f"l{layer}.wq",
        f"l{layer}.wk",
        f"l{layer}.wv",
        f"l{layer}.wo",
        f"l{layer}.w1",
        f"l{layer}.w2",
        f"l{layer}.ln1_w",
        f"l{layer}.ln1_b",
        f"l{layer}.ln2_w",
        f"l{layer}.ln2_b",
    ]


def init_params(cfg: ModelConfig, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create the parameter dict on CPU (for device-independent init), then move it."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = cfg.dim

    def matrix(out_dim: int, in_dim: int) -> torch.Tensor:
        return (torch.randn(out_dim, in_dim, generator=gen) * 0.02).to(device)

    def vector(n: int, fill: float) -> torch.Tensor:
        return torch.full((n,), fill, device=device)

    params: dict[str, torch.Tensor] = {
        "tok_emb": matrix(cfg.vocab, d),  # also the tied output head
        "pos_emb": matrix(cfg.seq, d),
        "lnf_w": vector(d, 1.0),
        "lnf_b": vector(d, 0.0),
    }
    for layer in range(cfg.layers):
        for name in _layer_names(layer):
            if name.endswith(("ln1_w", "ln2_w")):
                params[name] = vector(d, 1.0)
            elif name.endswith(("ln1_b", "ln2_b")):
                params[name] = vector(d, 0.0)
            elif name.endswith(("w1",)):
                params[name] = matrix(4 * d, d)
            elif name.endswith(("w2",)):
                params[name] = matrix(d, 4 * d)
            else:
                params[name] = matrix(d, d)
    return params


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #


def sample_perturbation(
    params: dict[str, torch.Tensor],
    rank: int,
    gen: torch.Generator,
) -> Perturbation:
    """Draw a low-rank (2D) or full-rank (1D) Gaussian perturbation."""
    pert = Perturbation()
    for name, value in params.items():
        if value.dim() == 2:
            out_dim, in_dim = value.shape
            a = torch.randn(out_dim, rank, generator=gen, device=value.device)
            b = torch.randn(in_dim, rank, generator=gen, device=value.device)
            pert.low[name] = (a, b)
        else:
            pert.full[name] = torch.randn(value.shape, generator=gen, device=value.device)
    return pert


def materialize_delta(pert: Perturbation, name: str, value: torch.Tensor, sigma: float, rank: int) -> torch.Tensor:
    """Dense form of one member's perturbation, for checks and for the update."""
    if name in pert.low:
        a, b = pert.low[name]
        return (sigma / math.sqrt(rank)) * (a @ b.transpose(-1, -2))
    return sigma * pert.full[name]


# --------------------------------------------------------------------------- #
# Forward pass with fused low-rank perturbations
# --------------------------------------------------------------------------- #


def _linear(
    x: torch.Tensor,
    base: torch.Tensor,
    pert: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
) -> torch.Tensor:
    """y = x @ base^T, plus the low-rank term without materialising base + delta."""
    y = x @ base.transpose(-1, -2)
    if pert is not None:
        a, b = pert
        y = y + scale * ((x @ b) @ a.transpose(-1, -2))
    return y


def _embedding(
    table: torch.Tensor,
    ids: torch.Tensor,
    pert: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
) -> torch.Tensor:
    """Look up rows of table + (scale * A B^T) without building the full delta."""
    out = F.embedding(ids, table)
    if pert is not None:
        a, b = pert
        out = out + scale * (a[ids] @ b.transpose(-1, -2))
    return out


def _norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pert_w: torch.Tensor | None,
    pert_b: torch.Tensor | None,
    sigma: float,
) -> torch.Tensor:
    if pert_w is not None:
        weight = weight + sigma * pert_w
    if pert_b is not None:
        bias = bias + sigma * pert_b
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-5)


def forward(
    params: dict[str, torch.Tensor],
    cfg: ModelConfig,
    tokens: torch.Tensor,
    pert: Perturbation | None = None,
    sigma: float = 0.0,
    rank: int = 1,
) -> torch.Tensor:
    """Next-token logits for ``tokens`` (batch, seq) under an optional perturbation."""
    scale = sigma / math.sqrt(rank) if pert is not None else 0.0
    low = pert.low if pert is not None else {}
    full = pert.full if pert is not None else {}
    batch, seq = tokens.shape
    head_dim = cfg.dim // cfg.heads

    x = _embedding(params["tok_emb"], tokens, low.get("tok_emb"), scale)
    positions = torch.arange(seq, device=tokens.device)
    x = x + _embedding(params["pos_emb"], positions, low.get("pos_emb"), scale)

    for layer in range(cfg.layers):
        prefix = f"l{layer}."
        h = _norm(
            x,
            params[prefix + "ln1_w"],
            params[prefix + "ln1_b"],
            full.get(prefix + "ln1_w"),
            full.get(prefix + "ln1_b"),
            sigma,
        )
        q = _linear(h, params[prefix + "wq"], low.get(prefix + "wq"), scale)
        k = _linear(h, params[prefix + "wk"], low.get(prefix + "wk"), scale)
        v = _linear(h, params[prefix + "wv"], low.get(prefix + "wv"), scale)
        q = q.view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, cfg.heads, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=tokens.device).tril()
        att = att.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        mixed = (att @ v).transpose(1, 2).reshape(batch, seq, cfg.dim)
        x = x + _linear(mixed, params[prefix + "wo"], low.get(prefix + "wo"), scale)

        h = _norm(
            x,
            params[prefix + "ln2_w"],
            params[prefix + "ln2_b"],
            full.get(prefix + "ln2_w"),
            full.get(prefix + "ln2_b"),
            sigma,
        )
        h = _linear(h, params[prefix + "w1"], low.get(prefix + "w1"), scale)
        h = F.gelu(h)
        x = x + _linear(h, params[prefix + "w2"], low.get(prefix + "w2"), scale)

    x = _norm(
        x,
        params["lnf_w"],
        params["lnf_b"],
        full.get("lnf_w"),
        full.get("lnf_b"),
        sigma,
    )
    return _linear(x, params["tok_emb"], low.get("tok_emb"), scale)  # tied head


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
    )


# --------------------------------------------------------------------------- #
# Synthetic task
# --------------------------------------------------------------------------- #


def make_batch(cfg: ModelConfig, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Deterministic synthetic task: token[t+1] = (token[t] + 1) mod vocab.

    A local successor rule is learnable by a tiny Transformer, so a falling loss
    measures the optimizer rather than the difficulty of the data.
    """
    starts = torch.randint(0, cfg.vocab, (cfg.batch, 1), generator=gen, device=device)
    offsets = torch.arange(cfg.seq, device=device).unsqueeze(0)
    return (starts + offsets) % cfg.vocab


# --------------------------------------------------------------------------- #
# EGGROLL update
# --------------------------------------------------------------------------- #


def evaluate(
    params: dict[str, torch.Tensor],
    cfg: ModelConfig,
    tokens: torch.Tensor,
) -> float:
    with torch.inference_mode():
        return float(next_token_loss(forward(params, cfg, tokens), tokens).item())


def eggroll_step(
    params: dict[str, torch.Tensor],
    cfg: ModelConfig,
    options: EggrollOptions,
    gen: torch.Generator,
    tokens: torch.Tensor,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    """One ES/EGGROLL step; returns member losses and the updated parameters."""
    members = [sample_perturbation(params, options.rank, gen) for _ in range(options.population)]
    with torch.inference_mode():
        losses = [
            float(next_token_loss(forward(params, cfg, tokens, p, options.sigma, options.rank), tokens).item())
            for p in members
        ]
        stacked = torch.tensor(losses, device=tokens.device)
        # z-scored fitness; population std is 1 by construction, so this is the
        # standard ES gradient estimate with a unit-variance scalar per member.
        scores = (stacked - stacked.mean()) / (stacked.std(unbiased=False) + 1e-8)
        score_list = scores.tolist()
        n = options.population
        matrix_coef = options.lr / (n * math.sqrt(options.rank))
        vector_coef = options.lr / n
        updated: dict[str, torch.Tensor] = {}
        for name, value in params.items():
            if name in members[0].low:
                acc = torch.zeros_like(value)
                for score, member in zip(score_list, members):
                    a, b = member.low[name]
                    acc = acc + score * (a @ b.transpose(-1, -2))
                updated[name] = value - matrix_coef * acc
            else:
                acc = torch.zeros_like(value)
                for score, member in zip(score_list, members):
                    acc = acc + score * member.full[name]
                updated[name] = value - vector_coef * acc
    return losses, updated


# --------------------------------------------------------------------------- #
# Payload accounting
# --------------------------------------------------------------------------- #


def payload_projection(
    params: dict[str, torch.Tensor], population: int, rank: int
) -> dict[str, float]:
    """Compare a dense bf16 upload with the scalar-only EGGROLL client report.

    This is arithmetic on this benchmark's parameter count, not a measurement of
    the 658M model. The scalar payload is what a client would send per round:
    one float32 fitness plus an 8-byte member/nonce id per population member.
    The factor payload assumes every member's A/B factors were shipped instead
    of being regenerated from a shared seed; it is included to show where the
    compression actually comes from.
    """
    dense_params = sum(t.numel() for t in params.values())
    factor_params = sum(
        (t.shape[0] + t.shape[1]) * rank if t.dim() == 2 else t.numel() for t in params.values()
    )
    dense_bytes = dense_params * 2  # bf16
    scalar_bytes = population * (4 + 8)
    # Per-round traffic if the population's A/B factors were shipped instead of
    # regenerated from seeds: this is NOT the EGGROLL client upload.
    factor_bytes = population * factor_params * 4  # fp32 factors
    return {
        "model_params": float(dense_params),
        "dense_bytes": float(dense_bytes),
        "scalar_bytes": float(scalar_bytes),
        "scalar_compression": dense_bytes / max(1.0, float(scalar_bytes)),
        "factor_bytes": float(factor_bytes),
        "factor_compression": dense_bytes / max(1.0, float(factor_bytes)),
    }


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #


def run_benchmark(
    model: ModelConfig | None = None,
    options: EggrollOptions | None = None,
    device: str = "cpu",
) -> dict:
    """Train the small LM with EGGROLL for ``options.steps`` and report metrics."""
    model = model or ModelConfig()
    options = options or EggrollOptions()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    dev = torch.device(device)

    params = init_params(model, options.seed, dev)
    init_gen = torch.Generator(device=dev).manual_seed(options.seed)
    data_gen = torch.Generator(device=dev).manual_seed(options.seed + 1)
    eval_tokens = make_batch(model, torch.Generator(device=dev).manual_seed(options.seed + 2), dev)

    first_loss = evaluate(params, model, eval_tokens)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    step_losses: list[float] = []
    for _ in range(options.steps):
        tokens = make_batch(model, data_gen, dev)
        losses, params = eggroll_step(params, model, options, init_gen, tokens)
        step_losses.append(sum(losses) / len(losses))

    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    last_loss = evaluate(params, model, eval_tokens)

    tokens_processed = options.steps * options.population * model.batch * model.seq
    payload = payload_projection(params, options.population, options.rank)
    result = {
        "schema": "ng-eggroll-model-bench-1",
        "device": device,
        "model": {
            "dim": model.dim,
            "layers": model.layers,
            "heads": model.heads,
            "vocab": model.vocab,
            "seq": model.seq,
            "batch": model.batch,
            "params": int(payload["model_params"]),
        },
        "eggroll": {
            "population": options.population,
            "rank": options.rank,
            "sigma": options.sigma,
            "lr": options.lr,
            "steps": options.steps,
            "seed": options.seed,
        },
        "first_loss": round(first_loss, 6),
        "last_loss": round(last_loss, 6),
        "loss_ratio": round(last_loss / max(1e-12, first_loss), 6),
        "tokens_processed": tokens_processed,
        "seconds": round(seconds, 4),
        "tokens_per_second": round(tokens_processed / max(1e-9, seconds), 2),
        "mean_member_loss_first": round(step_losses[0], 6) if step_losses else None,
        "mean_member_loss_last": round(step_losses[-1], 6) if step_losses else None,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if dev.type == "cuda" else None
        ),
        **{key: round(value, 2) if isinstance(value, float) else value for key, value in payload.items()},
        "note": (
            "scaled-down Transformer benchmark; the 658M model is not run and "
            "payload figures are arithmetic, not production measurements"
        ),
    }
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = ModelConfig(
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        vocab=args.vocab,
        seq=args.seq,
        batch=args.batch,
    )
    options = EggrollOptions(
        population=args.population,
        rank=args.rank,
        sigma=args.sigma,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps(run_benchmark(model, options, args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
