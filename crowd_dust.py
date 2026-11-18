"""DUST as a selectable v2 paradigm: perturbation-only gradients.

The estimator never calls ``backward`` on the network. For each draw it perturbs
every ``nn.Linear`` output with its own antithetic noise pair, measures the two
losses, and uses the loss difference as the scalar coefficient of that noise
direction. With per-layer independent noise this is a SPSA/ES estimate of the
gradient; it costs ``2 * population`` forwards per step and no backward.

Ported from the v1.5 prototype (``crowd_v15/dust.py``) which measured 90 tok/s and
a plateau at 6.686 held-out after 24 h at 306 M, because ``3 * population``
forwards buy only ``batch * seq`` evidence tokens. It is here because the founder
reports toy-scale DUST+ReLoRA results and wants the paradigm available.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class DustError(ValueError):
    """Raised for an unusable DUST configuration."""


@dataclass(frozen=True)
class DustConfig:
    population: int = 16
    sigma: float = 0.01
    seed: int = 0
    chunk: int = 4

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise DustError("population must be positive")
        if self.sigma <= 0:
            raise DustError("sigma must be positive")
        if self.chunk <= 0:
            raise DustError("chunk must be positive")


def derive_seed(base_seed: int, step: int, draw: int, name: str) -> int:
    """Stable seed per (run, step, draw, layer) so a run is reproducible."""
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{int(step)}:{int(draw)}:{name}".encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") >> 1


def linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Every ``nn.Linear`` in the model, keyed by its qualified name."""
    return {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}


class _Perturb:
    """Add ``sign * sigma * noise[name]`` to each Linear's output, and record its input."""

    def __init__(self, layers: dict[str, nn.Linear], noise: dict[str, torch.Tensor], sign: float):
        self.layers, self.noise, self.sign = layers, noise, sign
        self.inputs: dict[str, torch.Tensor] = {}
        self.handles: list = []

    def _pre(self, name: str):
        def hook(_mod, args):
            self.inputs[name] = args[0].detach()
        return hook

    def _post(self, name: str):
        def hook(_mod, _args, out):
            n = self.noise.get(name)
            return out if n is None else out + self.sign * n
        return hook

    def __enter__(self):
        for name, layer in self.layers.items():
            self.handles.append(layer.register_forward_pre_hook(self._pre(name)))
            self.handles.append(layer.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def _draw(layers, rows: int, seq: int, *, cfg: DustConfig, step: int, draw: int, device, dtype):
    noise = {}
    for name, layer in layers.items():
        gen = torch.Generator(device="cpu").manual_seed(derive_seed(cfg.seed, step, draw, name))
        noise[name] = torch.randn((rows, seq, layer.out_features), generator=gen).to(
            device=device, dtype=dtype
        )
    return noise


def dust_step(model, x, y, *, cfg: DustConfig = DustConfig(), step: int = 0, loss_fn=None) -> float:
    """Fill ``.grad`` on every parameter from a population of perturbation draws.

    Returns the mean loss of the clean forward, for logging. ``loss_fn(logits, y)``
    defaults to cross-entropy over the last two dims flattened.
    """
    layers = linear_layers(model)
    if not layers:
        raise DustError("model has no nn.Linear layers to perturb")
    loss_fn = loss_fn or (lambda logits, tgt: F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt[:, :-1].reshape(-1)))
    rows, seq = x.shape[0], x.shape[1]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    clean = 0.0
    for start in range(0, cfg.population, cfg.chunk):
        for draw in range(start, min(start + cfg.chunk, cfg.population)):
            noise = _draw(layers, rows, seq, cfg=cfg, step=step, draw=draw, device=device, dtype=dtype)
            with torch.no_grad():
                with _Perturb(layers, noise, +1.0) as plus:
                    loss_plus = float(loss_fn(model(x), y))
                plus_inputs = plus.inputs
                with _Perturb(layers, noise, -1.0):
                    loss_minus = float(loss_fn(model(x), y))
            coef = (loss_plus - loss_minus) / (2.0 * cfg.sigma * cfg.population)
            for name, layer in layers.items():
                weight = layer.weight
                inp = plus_inputs.get(name)
                if inp is None:
                    # A layer the forward never reached (or a CPU-parked head) has
                    # no input to correlate with, so it gets no estimate rather
                    # than a wrong one.
                    continue
                # The real 1.01B model parks some layers on CPU (head/streamed
                # blocks), so the estimate has to be built on the *weight's*
                # device: mixing a cuda noise draw with a cpu weight raises
                # "Expected all tensors to be on the same device" (found by the
                # real-model smoke, invisible to the CPU unit tests).
                wdev = weight.device
                n = noise[name].to(wdev, torch.float32)
                xi = inp.to(wdev, torch.float32)
                # dL/dW estimate for this layer: coef * sum_{b,t} noise[b,t,out] * x[b,t,in]
                contrib = torch.einsum("bto,bti->oi", n, xi)
                if weight.grad is None:
                    weight.grad = torch.zeros_like(weight)
                weight.grad += (coef * contrib).to(dtype=weight.dtype)
                if layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad += (coef * n.sum(dim=(0, 1))).to(dtype=layer.bias.dtype)
            if draw == start:
                clean = 0.5 * (loss_plus + loss_minus)
    return clean
