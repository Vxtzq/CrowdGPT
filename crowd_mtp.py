"""Multi-token prediction: extra heads that must predict tokens 2..k ahead.

Why a volunteer swarm wants this (and why it is not free):

* Every round the client ships a *compressed* view of what it learned, and the
  server merges deltas from peers that trained on other shards. A gradient that
  only ever fits the next token is a narrow signal to average, and a slice-local
  objective (MOSAIC/SOLO) is narrower still. Predicting further ahead forces the
  trunk to encode information that survives aggregation instead of features that
  only help one position.
* It is a *training-signal* feature: the extra heads are client-local and never
  reach the wire, exactly like the SOLO heads. `_LOCAL_ONLY_PREFIXES` in
  `client.py` skips them, so the server's flat vector is unchanged.

Shape of the implementation (deliberately cheaper than DeepSeek-V3's MTP):

* one `nn.Linear(dim, dim)` projection per extra depth, each applied to the
  *same* trunk hidden state, and the **shared** readout (`head_weight()`, which
  is the tied embedding) produces the logits. That keeps the parameter cost at
  ``k*dim*dim`` and keeps the embedding as the single output space, which is
  what makes the auxiliary loss comparable to the main one.
* depth 1 predicts ``y[t+2]``, depth k predicts ``y[t+k+1]``: the main head
  already covers ``y[t+1]``, so the projections never duplicate it.
* the projections are initialised to the identity, so at step 0 the auxiliary
  loss is the ordinary next-token loss at a longer offset rather than noise.

The loss is added to the main one by `client.py` behind `CROWDGPT_MTP_TOKENS`
(default 0 = off). It costs one extra forward pass over the same batch, because
the trunk hidden state is not exposed by `_fwl`; a production build that reuses
the main pass's hidden state would pay nothing extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

MAX_TOKENS = 8
DEFAULT_WEIGHT = 0.2


@dataclass(frozen=True)
class MTPConfig:
    """``tokens`` = number of *extra* depths (4 means y[t+2]..y[t+5])."""

    tokens: int = 4
    weight: float = DEFAULT_WEIGHT
    loss_chunk: int = 256

    def __post_init__(self):
        if not (1 <= int(self.tokens) <= MAX_TOKENS):
            raise ValueError(f"MTP needs 1..{MAX_TOKENS} extra depths, got {self.tokens}")
        if not (0.0 <= float(self.weight) <= 10.0):
            raise ValueError(f"MTP weight must be in [0, 10], got {self.weight}")


def attach(model, cfg: MTPConfig = MTPConfig()):
    """Add ``cfg.tokens`` projections and return a description of what was built."""
    dim = int(model.head_weight().shape[1])
    heads = {}
    for k in range(1, int(cfg.tokens) + 1):
        lin = torch.nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.eye(dim))
        heads[str(k)] = lin
    model.mtp_projs = torch.nn.ModuleDict(heads)
    # Registered on the model's own device so the projections follow the trunk.
    device = getattr(model, "main_device", None)
    if device is not None:
        model.mtp_projs = model.mtp_projs.to(device)
    return {"mtp": True, "depths": int(cfg.tokens), "weight": float(cfg.weight),
            "params": int(cfg.tokens) * dim * dim}


def is_attached(model) -> bool:
    projs = getattr(model, "mtp_projs", None)
    return isinstance(projs, torch.nn.ModuleDict) and len(projs) > 0


def predict_logits(model, hidden, depth: int):
    """Logits for the token ``depth + 1`` ahead, from the trunk hidden state."""
    proj = model.mtp_projs[str(int(depth))]
    return proj(hidden) @ model.head_weight(hidden.dtype).t()


def mtp_loss(model, x, y, cfg: MTPConfig = MTPConfig(), *, hidden=None):
    """Weighted mean cross-entropy over the extra depths.

    ``hidden`` lets a caller that already has ``ln_f(run_blocks(embed(x)))``
    reuse it; otherwise one extra forward pass is run. Returns a scalar tensor
    that is ``0`` (and does not need a backward) when the model has no
    projections attached, so the call site can stay unconditional.
    """
    if not is_attached(model):
        return torch.zeros((), device=getattr(model, "main_device", None))
    if hidden is None:
        hidden = model.ln_f(model.run_blocks(model.embed(x)))
    vocab = int(model.head_weight().shape[0])
    total = None
    used = 0
    for depth in range(1, int(cfg.tokens) + 1):
        if y.shape[1] <= depth:
            continue
        # position t predicts y[t + depth]; the main head owns depth 0.
        logits = predict_logits(model, hidden, depth)[:, : y.shape[1] - depth, :]
        target = y[:, depth:]
        loss_k = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
        total = loss_k if total is None else total + loss_k
        used += 1
    if total is None:                       # batch shorter than the deepest shift
        return torch.zeros((), device=hidden.device)
    return (total / used) * float(cfg.weight)


def parameters(model) -> Iterable[torch.nn.Parameter]:
    """The projections' parameters (never on the wire, but they do train)."""
    projs = getattr(model, "mtp_projs", None)
    if projs is None:
        return ()
    return tuple(p for p in projs.parameters() if p.requires_grad)


def config_from_env(env=None) -> MTPConfig | None:
    """``CROWDGPT_MTP_TOKENS`` / ``CROWDGPT_MTP_WEIGHT``; None when disabled."""
    import os

    env = os.environ if env is None else env
    raw = str(env.get("CROWDGPT_MTP_TOKENS") or "").strip()
    if not raw:
        return None
    try:
        tokens = int(raw)
    except ValueError:
        return None
    if tokens <= 0:
        return None
    try:
        weight = float(str(env.get("CROWDGPT_MTP_WEIGHT") or DEFAULT_WEIGHT))
    except ValueError:
        weight = DEFAULT_WEIGHT
    try:
        return MTPConfig(tokens=tokens, weight=weight)
    except ValueError:
        return None
