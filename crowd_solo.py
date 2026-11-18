"""SOLO — shared-output local learning for the Crowd-v1 client.

This implements Algorithm 1 of Yin, Wang, Pan & Li, "SOLO: Pretraining
Billion-Parameter Language Models with Shared-Output Local Learning"
(arXiv:2609.35440v1). The full text ships in ``research/`` next to this file;
Appendix A is the step implemented here.

Why the project needs it
------------------------
A v1 client backpropagates through all 24 blocks in one pass, so every block's
activations must stay alive until the single backward finishes. SOLO splits the
trunk into ``K`` contiguous modules and gives each one its own loss. Modules
hand each other ``sg(h)`` (stop-gradient) activations, so no gradient crosses a
module boundary and one module's activations are freed by its local backward
before the next module even runs. That is what turns saved activation memory
into throughput: a stage holds one module's worth instead of the whole trunk's.

What makes it different from ordinary local learning
----------------------------------------------------
A private-head local rule gives module ``k`` its own readout ``W_k`` and then
has to hope ``W_k`` points the same way as the terminal readout. SOLO instead
projects the module output back into the readout's input space with a small
head ``phi_k`` and scores it through ``sg(W)`` — a copy of the *terminal*
readout with the gradient stopped. The local gradient therefore returns through
the same ``W`` backprop would use, up to the learned temperature ``tau_k`` and
the refresh delay, which is why the paper measures its cosine to the backprop
gradient at 0.70 versus 0.52 for a private head and 0.32 for a random one.

Only the last module is trained by the true loss through the true readout, so
only that step updates ``W`` (and, with weight tying, the embedding).

Design notes for this codebase
------------------------------
* Heads live on the model as ``solo_heads`` so the existing optimizer builder
  and gradient clipper pick them up, but they are **client-local**: the delta
  writers skip them, because the paper discards heads at inference and shipping
  them would change the base-weight layout the server validates against.
* ``W`` is read with ``.detach()`` inside the step. In one process that is
  exactly Algorithm 1 line 1 (``Wbar <- sg(W)``) at a refresh period of one
  step, because the optimizer only touches ``W`` after the accumulation window.
* The terminal readout is bias-free and weight-tied to the embedding in v1, so
  the final loss is the only path that moves it — same as the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_MODULES = 4
"""Modules to split the trunk into. The founder's target shape is a 1B model
trained as 4 x 250M chunks, and v1's 24 blocks over 4 modules is 6 blocks and
~83M trunk params each (plus a share of the 233M tied readout)."""

LOCAL_ONLY_PREFIXES = ("engram.", "solo_heads.")
"""Parameters that are not part of the uploaded base-weight vector."""


def is_uploaded_param(name: str) -> bool:
    """True when ``name`` belongs in the base-weight vector the server holds."""
    return not name.startswith(LOCAL_ONLY_PREFIXES)


def split_layers(n_layers: int, modules: int) -> list[tuple[int, int]]:
    """Contiguous, balanced ``[lo, hi)`` block ranges — one per module.

    The last range is the terminal module, which keeps the true loss and the
    true readout. Ranges are contiguous so a module is a depth interval of the
    trunk, which is what makes the split a pipeline stage boundary.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if modules < 1:
        raise ValueError("modules must be positive")
    modules = min(int(modules), int(n_layers))
    base, extra = divmod(int(n_layers), modules)
    out, lo = [], 0
    for i in range(modules):
        hi = lo + base + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


class SoloHead(nn.Module):
    """``phi_k``, ``tau_k`` and ``b_k`` for one non-terminal module.

    ``phi_k`` is ``Linear(dim, dim)`` behind its own LayerNorm so that a module's
    local loss cannot reach the shared ``ln_f`` (that would be a cross-module
    coupling the method exists to avoid). ``tau_k`` starts at 1, which is the
    value at which the readout term of the gap to backprop vanishes.
    """

    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.phi = nn.Linear(dim, dim)
        self.tau = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.phi(self.ln(h))

    def extra_repr(self) -> str:
        return f"dim={self.phi.in_features}, vocab={self.bias.numel()}"


def build_solo_heads(dim: int, vocab: int, modules: int) -> nn.ModuleDict:
    """One head per non-terminal module (the terminal module keeps the readout)."""
    if modules < 2:
        raise ValueError("SOLO needs at least two modules")
    return nn.ModuleDict({str(k): SoloHead(dim, vocab) for k in range(modules - 1)})


def chunked_solo_ce(
    xs: torch.Tensor,
    w: torch.Tensor,
    ys: torch.Tensor,
    *,
    tau: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    pos_step: int = 64,
) -> torch.Tensor:
    """Cross-entropy of ``xs @ w.T`` without materialising all logits at once.

    ``xs`` is ``(B, T, D)``, ``w`` is ``(V, D)``. Position slices keep the peak
    allocation at ``B * pos_step * V`` floats, so a 151,669-token vocabulary
    never has to sit in memory for every position simultaneously. Returns the
    mean over ``B * T`` predictions, matching ``reduction='mean'``.
    """
    if xs.dim() != 3:
        raise ValueError("xs must be (B, T, D)")
    b, t, _ = xs.shape
    if t == 0:
        raise ValueError("cannot score an empty sequence")
    step = max(1, int(pos_step))
    total = xs.new_zeros(())
    for i in range(0, t, step):
        x = xs[:, i:i + step, :]
        y = ys[:, i:i + step]
        # No torch.nn.functional here on purpose: cross-entropy is assembled
        # from primitives so that a stubbed or monkeypatched `F` cannot change
        # what the estimator computes. logsumexp - gather IS cross-entropy and
        # is numerically stable without materialising log-softmax.
        logits = torch.matmul(x, w.t()).float()
        if tau is not None:
            logits = logits * tau
        if bias is not None:
            logits = logits + bias
        lse = torch.logsumexp(logits, dim=-1)
        picked = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        total = total + (lse - picked).sum()
    return total / (b * t)


def solo_step(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    splits: list[tuple[int, int]],
    loss_scale: float = 1.0,
    loss_chunk: int = 256,
    use_chunked: bool = True,
    use_ckpt: bool = True,
    stream_dtype: torch.dtype | None = None,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """One SOLO micro-step: run Algorithm 1 and populate gradients.

    Returns ``(loss, local_losses)`` where ``loss`` is the terminal cross-entropy
    scaled by ``loss_scale`` (the same convention ``_fwl`` uses, so the round
    loop's logging and gradient accumulation are unchanged) and
    ``local_losses`` are the unscaled per-module losses for diagnostics.
    """
    heads = getattr(model, "solo_heads", None)
    if heads is None:
        raise RuntimeError("model has no solo_heads; build them before stepping")
    if len(splits) < 2:
        raise ValueError("SOLO needs at least two splits")
    if len(heads) != len(splits) - 1:
        raise ValueError(f"{len(heads)} heads for {len(splits)} splits")

    dev = model.main_device
    ctx = (
        torch.autocast(device_type=dev.type, dtype=autocast_dtype)
        if (use_autocast and autocast_dtype is not None)
        else _null_ctx()
    )

    with ctx:
        h = model.embed(x)
        y_dev = y.to(h.device)
        local: list[float] = []
        for k in range(len(splits) - 1):
            lo, hi = splits[k]
            h = model.run_blocks(
                h, lo, hi, use_chunked=use_chunked, use_ckpt=use_ckpt,
                stream_dtype=stream_dtype,
            )
            head = heads[str(k)]
            # sg(W): a read-only copy of the terminal readout, exactly as in
            # Algorithm 1 line 1. Only the terminal step below may move it.
            w_bar = model.head_weight(stream_dtype).detach()
            proj = head.project(h)
            loss_k = chunked_solo_ce(
                proj, w_bar, y_dev, tau=head.tau, bias=head.bias, pos_step=loss_chunk,
            )
            (loss_k * loss_scale).backward()
            local.append(float(loss_k.detach()))
            # Free this module's graph and hand the next module a constant.
            h = h.detach()
            proj = None
            loss_k = None

        lo, hi = splits[-1]
        h = model.run_blocks(
            h, lo, hi, use_chunked=use_chunked, use_ckpt=use_ckpt,
            stream_dtype=stream_dtype,
        )
        h = model.ln_f(h)
        w = model.head_weight(stream_dtype)
        final = chunked_solo_ce(h, w, y_dev, pos_step=loss_chunk)
        (final * loss_scale).backward()
        loss = (final * loss_scale).detach()

    return loss, local


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def reference_local_gradient(
    xs: torch.Tensor, w: torch.Tensor, ys: torch.Tensor
) -> torch.Tensor:
    """``W^T delta`` for a set of hidden states, with ``delta`` from the loss.

    Only used by the tests and by the diagnostic that checks the local gradient
    really is the backprop gradient when ``tau = 1`` and the readout is current.
    """
    xs = xs.detach().clone().requires_grad_(True)
    loss = chunked_solo_ce(xs, w, ys)
    loss.backward()
    return xs.grad


def solo_module_step(model, x, y, *, splits, module, loss_scale=1.0, loss_chunk=256,
                     use_ckpt=True, stream_dtype=None):
    """One module-parallel step: a peer trains only module ``module`` of ``splits``.

    This is the shape Crowd actually runs, because its peers are independent
    replicas rather than stages of one co-located pipeline. A peer spends a
    redundant forward over the blocks before its own module (under ``no_grad``,
    so no activations are kept), then a normal forward+backward inside the
    module it owns. Nothing crosses peers: no activation pipe, no gradient
    return, no barrier. That redundancy is what buys the throughput -- the
    prefix is one forward instead of a full forward+backward.
    """
    heads = getattr(model, "solo_heads", None)
    if heads is None:
        raise RuntimeError("model has no solo_heads; build them before stepping")
    if not 0 <= int(module) < len(splits):
        raise ValueError(f"module {module} is outside {len(splits)} splits")
    module = int(module)
    terminal = module == len(splits) - 1
    if not terminal and str(module) not in heads:
        raise ValueError(f"no solo head for module {module}")
    lo, hi = splits[module]
    with torch.no_grad():
        h = model.embed(x)
        if lo > 0:
            h = model.run_blocks(h, 0, lo, use_ckpt=False, stream_dtype=stream_dtype)
    h = h.detach()
    h = model.run_blocks(h, lo, hi, use_ckpt=use_ckpt, stream_dtype=stream_dtype)
    ydev = y.to(h.device)
    if terminal:
        loss = chunked_solo_ce(model.ln_f(h), model.head_weight(stream_dtype), ydev,
                               pos_step=loss_chunk)
    else:
        head = heads[str(module)]
        wbar = model.head_weight(stream_dtype).detach()
        loss = chunked_solo_ce(head.project(h), wbar, ydev, tau=head.tau, bias=head.bias,
                               pos_step=loss_chunk)
    (loss * loss_scale).backward()
    return (loss * loss_scale).detach(), [float(loss.detach())]


def uploaded_for_module(name, splits, module):
    """Whether a peer training ``module`` may upload ``name``.

    A module owns its own blocks and nothing else; the terminal module also owns
    the readout, the final norm and the embedding, because those are what its
    true loss actually moves. ``splits``/``module`` of ``None`` means the peer is
    training the whole model, so the answer is the legacy one.
    """
    if splits is None or module is None:
        return True
    if name.startswith(LOCAL_ONLY_PREFIXES):
        return False
    if name.startswith("blocks."):
        try:
            idx = int(name.split(".")[1])
        except (IndexError, ValueError):
            return False
        lo, hi = splits[int(module)]
        return lo <= idx < hi
    return int(module) == len(splits) - 1
