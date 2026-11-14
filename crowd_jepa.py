"""Opt-in LLM-JEPA auxiliary loss for Crowd-v1 (Huang, LeCun, Balestriero, arXiv:2509.14252).

    L = gamma * CE_next_token + lambda * (1 - cos(h_text, h_code))

h_text is the last-layer hidden state (after ln_f) at the last token of view 1 (optionally followed by k predictor tokens),
h_code the one at the last token of view 2. Both views are packed in one sequence with a block-causal mask: causal inside
each view, no attention across views, so one forward pass gives both embeddings (see the wiring note below for where the
next-token loss comes from).
As in the reference implementation (rbalestr-lab/llm-jepa, Apache-2.0, finetune.py), there is no stop-gradient: the
gradient flows through both views; the next-token term prevents collapse (the paper reports gamma = 0 only yields empty
outputs). Inference is unchanged: nothing here is used at generation time.

NOT wired into client.py. Nothing changes for contributors unless the coordinator opts in after a pilot
(docs/LLM_JEPA_PROPOSAL.md). Standalone on purpose: torch only, imported lazily so the stdlib CI can import this file.

Wiring sketch for SFT (for a pilot, not applied). As in the reference finetune.py, the next-token loss comes from the
NORMAL causal pass over [prompt ; answer] (the answer must see the prompt); the packed block-causal pass only provides
the two embeddings:
    ce = next_token_ce(model, model.forward_features(prompt_and_answer_ids), answer_targets)   # usual SFT loss
    ids, mask, (i_text, i_code), _ = pack_two_views(prompt_ids, answer_ids, k_pred=0)
    h = model.forward_features(ids, attn_mask=mask)          # SDPA with attn_mask=mask instead of is_causal=True
    loss, j = combined_loss(ce, h, i_text, i_code, lam=1.0, apply_jepa=jepa_step(seed, step, alpha=0.5))
The `targets` returned by `pack_two_views` (each view predicting only itself) are for the other use, where both views
are standalone texts (e.g. docstring and function body in pretraining); there a single packed pass serves both terms.

Cohort mode needs every client of a pool to compute the same thing, so the loss-dropout decision is a deterministic
function of (seed, step), never of a local RNG (`jepa_step`).
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

IGNORE = -100


def _torch() -> Any:
    import torch  # lazy: the stdlib CI imports this module with torch stubbed

    return torch


def jepa_step(seed: int, step: int, alpha: float) -> bool:
    """Loss dropout (paper §3.3): skip the JEPA term on a fraction alpha of steps, deterministically per (seed, step).

    Keep lambda * (1 - alpha) roughly constant when changing alpha (the paper's guidance). alpha = 0 always applies it."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return True
    h = hashlib.sha256(f"crowd-jepa:{int(seed)}:{int(step)}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return u >= alpha


def pack_two_views(text_ids: Sequence[int], code_ids: Sequence[int], *, k_pred: int = 0, pred_token_id: int | None = None,
                   max_len: int | None = None):
    """Pack view 1 (+ k predictor tokens) and view 2 into one sequence.

    Returns (ids [1, T] long, attn_mask [1, 1, T, T] bool -- True = may attend, (i_text, i_code) last-token indices,
    targets [1, T] long with IGNORE where no next-token target applies: the last token of each view (its successor is in
    the other view or absent) and the predictor tokens (they are not text)."""
    torch = _torch()
    if k_pred < 0:
        raise ValueError("k_pred must be >= 0")
    if k_pred > 0 and pred_token_id is None:
        raise ValueError("pred_token_id is required when k_pred > 0")
    t, c = [int(x) for x in text_ids], [int(x) for x in code_ids]
    if not t or not c:
        raise ValueError("both views must be non-empty")
    v1 = t + [int(pred_token_id)] * k_pred if k_pred else t
    T = len(v1) + len(c)
    if max_len is not None and T > max_len:
        raise ValueError(f"packed length {T} exceeds max_len {max_len}")
    ids = torch.tensor([v1 + c], dtype=torch.long)
    n1 = len(v1)
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[:n1, :n1] = torch.ones(n1, n1, dtype=torch.bool).tril()
    mask[n1:, n1:] = torch.ones(T - n1, T - n1, dtype=torch.bool).tril()
    tgt = torch.full((1, T), IGNORE, dtype=torch.long)
    if len(t) > 1:
        tgt[0, : len(t) - 1] = torch.tensor(t[1:], dtype=torch.long)   # view 1 predicts its own next tokens
    if len(c) > 1:
        tgt[0, n1: T - 1] = torch.tensor(c[1:], dtype=torch.long)      # view 2 likewise; boundary and predictor slots ignored
    return ids, mask[None, None], (n1 - 1, T - 1), tgt


def jepa_term(hidden, i_text, i_code, *, stopgrad: bool = False, center: bool = False):
    """1 - cos(h[i_text], h[i_code]) averaged over the batch. hidden: [B, T, D] final hidden states (after ln_f).
    i_text / i_code: an int, or one index per row (packed batches of different lengths).

    Anti-collapse options, both measured on the HiveMind T4 (Crowd-v1 + LoRA r64, NL-RX; see docs/LLM_JEPA_PROPOSAL.md §5).
    With the plain term at lambda = 1, 1 - cos fell to ~0.002 and general held-out loss rose by +0.76..0.96 nat:
      stopgrad: no gradient through the target (view-2) embedding; only view 1 moves (JEPA / BYOL style).
      center:   subtract the batch mean of both views first, so "every state points the same way" no longer pays."""
    torch = _torch()
    r = torch.arange(hidden.shape[0], device=hidden.device)
    a = hidden[r, i_text, :].float()
    b = hidden[r, i_code, :].float()
    if stopgrad:
        b = b.detach()
    if center:
        mu = torch.cat([a, b]).mean(0, keepdim=True)
        a, b = a - mu, b - mu
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()


def combined_loss(ce, hidden, i_text, i_code, *, lam: float = 0.1, gamma: float = 1.0, apply_jepa: bool = True,
                  stopgrad: bool = True, center: bool = False):
    """gamma * ce + lam * jepa_term, or gamma * ce on a loss-dropout step. Returns (loss, detached jepa value or None).

    Defaults follow the T4 autoresearch screen, not the paper's lambda = 1: lam = 0.1 with stopgrad (use loss dropout
    alpha = 0.5 via `jepa_step`) was the best variant that kept general loss at the SFT level. It still did not beat plain
    SFT at equal compute, so this stays opt-in."""
    if lam < 0 or gamma <= 0:
        raise ValueError("need lam >= 0 and gamma > 0 (gamma = 0 collapses, per the paper)")
    if not apply_jepa or lam == 0:
        return gamma * ce, None
    j = jepa_term(hidden, i_text, i_code, stopgrad=stopgrad, center=center)
    return gamma * ce + lam * j, j.detach()
