"""Held-out loss check for CrowdGPT (`Vxtzq/Crowd-v1`), computed on a community GPU and verified on a CPU-only backend.

CrowdGPT's coordinator decides whether a merged update may become the new checkpoint by comparing the held-out loss
before and after it (`LossRatchetGate` in `hivemind_bridge.py`). That needs forward passes of a 658M-parameter model,
which a 2-core CPU server cannot afford for every round. Here the split is:

1. An evaluator with a GPU computes the loss of the base and the candidate weights on N held-out windows, one number per
   window, and signs a receipt that binds those numbers to the exact weights and data (SHA-256 of each).
2. The backend checks the signature and hashes, then recomputes only k of the 2N numbers on its CPU. Which k is derived
   from the receipt's commitment and a secret the evaluator never sees, so the evaluator cannot know in advance which
   windows will be checked. If every recomputed value agrees within `tol`, the means go to the ratchet.

What this does and does not guarantee: an evaluator that falsifies m of the 2N numbers is caught with probability
1 - C(2N-m, k) / C(2N, k) (`detection_probability`). Falsifying a few windows by a large amount can slip through; a large
change in the mean needs many falsified numbers, or large lies on a few, and the backend also rejects any window whose
claimed loss is outside [0, ln(vocab) + margin]. For stronger assurance use two independent evaluators
(`receipts_agree`). Agreement across hardware is a tolerance, not equality: GPU and CPU kernels differ; measure it on
your own hardware before choosing `tol` (`calibrate` in HiveMind's `scripts/crowdgpt_loss_check.py`).

Standalone on purpose (torch, numpy, standard library; safetensors only for loading files) so the same file can sit in
the CrowdGPT repository next to `hivemind_bridge.py`. The model below is the inference path of `SotaGPT` in CrowdGPT's
`client.py` (LayerNorm, grouped-query attention with RoPE, SwiGLU, tied head); the engram table is not read by the
forward pass there (`ENGRAM_ACTIVE = False`), so it is not part of the check.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECEIPT_SCHEMA = "hivemind.crowdgpt.loss_check.v1"
CROWD_V1_CONFIG: dict[str, int | bool] = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
CROWD_V1_MODEL_SIZE = 657_787_392
TOKEN_DTYPE = np.uint32  # the shards on Vxtzq/CrowdGPT hold uint32 tokens (client.py StreamingShardDataset), whatever DATA.md says


# ---------------------------------------------------------------------------------------------------------------- model


def _rope(head_dim: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.einsum("i,j->ij", torch.arange(seq_len, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None].to(device), e.sin()[None, None].to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), -1)


class _Attention(nn.Module):
    def __init__(self, c: Mapping[str, int | bool]) -> None:
        super().__init__()
        self.nh, self.nkv, self.hd, dim = int(c["nHeads"]), int(c["nKvHeads"]), int(c["headDim"]), int(c["dim"])
        self.wq = nn.Linear(dim, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(dim, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(dim, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q = self.wq(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        ct, st = cos[:, :, :t, :], sin[:, :, :t, :]
        q = q * ct + _rotate_half(q) * st
        k = k * ct + _rotate_half(k) * st
        rep = self.nh // self.nkv
        k = k.unsqueeze(2).expand(b, self.nkv, rep, t, self.hd).reshape(b, self.nh, t, self.hd)
        v = v.unsqueeze(2).expand(b, self.nkv, rep, t, self.hd).reshape(b, self.nh, t, self.hd)
        if q.dtype != v.dtype or k.dtype != v.dtype:  # fp32 RoPE tables promote q and k; SotaGPT casts them back the same way
            q, k = q.to(v.dtype), k.to(v.dtype)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(b, t, c))


class _SwiGLU(nn.Module):
    def __init__(self, c: Mapping[str, int | bool]) -> None:
        super().__init__()
        dim, hid = int(c["dim"]), int(c["mlpHidden"])
        self.w1 = nn.Linear(dim, hid, bias=False)
        self.w2 = nn.Linear(dim, hid, bias=False)
        self.w3 = nn.Linear(hid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class _Block(nn.Module):
    def __init__(self, c: Mapping[str, int | bool]) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(int(c["dim"]))
        self.attn = _Attention(c)
        self.ln_2 = nn.LayerNorm(int(c["dim"]))
        self.mlp = _SwiGLU(c)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin)
        return x + self.mlp(self.ln_2(x))


class CrowdV1(nn.Module):
    """Inference-only SotaGPT. Parameter order matches `SotaGPT.named_parameters()` minus the engram table, so the flat
    `weights` tensor of `model_bf16.safetensors` loads with the same offsets as `SotaGPT.load_base_weights`."""

    def __init__(self, config: Mapping[str, int | bool] = CROWD_V1_CONFIG) -> None:
        super().__init__()
        self.config = dict(config)
        dim, vocab = int(config["dim"]), int(config["vocabSize"])
        self.wte = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([_Block(config) for _ in range(int(config["nLayers"]))])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        if config.get("weightTying", True):
            self.wte.weight = self.lm_head.weight

    def n_flat(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_flat(self, flat: torch.Tensor) -> None:
        flat = flat.reshape(-1)
        if flat.numel() != self.n_flat():
            raise ValueError(f"flat weights have {flat.numel()} values, the model needs {self.n_flat()}")
        o = 0
        with torch.no_grad():
            for p in self.parameters():  # deduplicated: the tied head appears once, as wte.weight
                s = p.numel()
                p.copy_(flat[o: o + s].view(p.shape).to(p.dtype))
                o += s

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cos, sin = _rope(int(self.config["headDim"]), idx.shape[1], idx.device)
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return F.linear(self.ln_f(x), self.lm_head.weight)


def load_flat_safetensors(path: Path | str) -> torch.Tensor:
    """The single flat tensor of a CrowdGPT `model_*.safetensors` file (key `weights`)."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    key = "weights" if "weights" in tensors else ("flat" if "flat" in tensors else None)
    if key is None:
        if len(tensors) != 1:
            raise ValueError(f"cannot identify the weights tensor among {sorted(tensors)[:10]}")
        key = next(iter(tensors))
    return tensors[key].reshape(-1)


def weights_sha256(flat: torch.Tensor) -> str:
    """SHA-256 of the weights as bf16 bytes: the identity both sides agree on, whatever dtype each computes in."""
    b = flat.detach().to("cpu", dtype=torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def tokens_sha256(tokens: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- the data


def heldout_windows(n_tokens: int, window: int, n: int, seed: int) -> list[int]:
    """`n` distinct, non-overlapping window starts in a held-out token array, chosen by `seed` (same on both sides)."""
    slots = n_tokens // window
    if window < 2 or n < 1 or slots < n:
        raise ValueError(f"need window >= 2, n >= 1 and at least n={n} windows of {window} tokens (have {slots})")
    rng = np.random.default_rng(seed)
    return sorted(int(s) * window for s in rng.choice(slots, size=n, replace=False))


def window_losses(model: nn.Module, tokens: np.ndarray, starts: Sequence[int], window: int, *, device: torch.device | str = "cpu",
                  dtype: torch.dtype = torch.float32, batch: int = 1, stop: Callable[[], bool] | None = None) -> list[float]:
    """Mean next-token cross-entropy (nats) of each window: input tokens[s : s+window-1], targets tokens[s+1 : s+window],
    the same split as `StreamingShardDataset.get_batch`. `stop()` is asked before each batch; True raises `Interrupted`
    (so a shared GPU can be handed back to inference between windows)."""
    vocab = int(getattr(model, "config", {}).get("vocabSize", 0)) or None
    arr = np.asarray(tokens)
    if vocab is not None and len(arr) and int(arr.max()) >= vocab:
        raise ValueError("held-out tokens contain ids outside the vocabulary")
    model = model.to(device=device, dtype=dtype).train(False)  # inference mode for dropout/norm layers (same as Module.eval)
    out: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(starts), batch):
            if stop is not None and stop():
                raise Interrupted(f"stopped after {len(out)} of {len(starts)} windows")
            chunk = np.stack([arr[s: s + window] for s in starts[i: i + batch]]).astype(np.int64)
            t = torch.from_numpy(chunk).to(device)
            logits = model(t[:, :-1]).float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1), reduction="none")
            out.extend(float(v) for v in ce.view(t.shape[0], -1).mean(dim=1).cpu())
    return out


class Interrupted(RuntimeError):
    """The evaluation was stopped on request before it finished."""


# ---------------------------------------------------------------------------------------------------------- the receipt


@dataclass(frozen=True)
class LossCheckReceipt:
    """What an evaluator signs. `losses_before[i]` / `losses_after[i]` belong to window `starts[i]`."""

    evaluator_id: str
    base_sha256: str
    candidate_sha256: str
    heldout_sha256: str
    window: int
    seed: int
    starts: tuple[int, ...]
    losses_before: tuple[float, ...]
    losses_after: tuple[float, ...]
    compute_dtype: str
    device: str
    round_id: str = ""
    signature_hex: str = ""
    schema: str = field(default=RECEIPT_SCHEMA)

    def __post_init__(self) -> None:
        if not (len(self.starts) == len(self.losses_before) == len(self.losses_after)) or not self.starts:
            raise ValueError("starts, losses_before and losses_after must be non-empty and of equal length")

    def payload(self) -> bytes:
        body = asdict(self)
        body.pop("signature_hex")
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    def commitment(self) -> str:
        return hashlib.sha256(self.payload()).hexdigest()

    def signed(self, key: bytes) -> "LossCheckReceipt":
        return _replace(self, signature_hex=hmac.new(key, self.payload(), hashlib.sha256).hexdigest())

    def verify_signature(self, key: bytes) -> bool:
        return bool(self.signature_hex) and hmac.compare_digest(hmac.new(key, self.payload(), hashlib.sha256).hexdigest(), self.signature_hex)

    @property
    def mean_before(self) -> float:
        return float(np.mean(self.losses_before))

    @property
    def mean_after(self) -> float:
        return float(np.mean(self.losses_after))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "LossCheckReceipt":
        d = json.loads(text)
        if d.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"unknown receipt schema {d.get('schema')!r}")
        for k in ("starts", "losses_before", "losses_after"):
            d[k] = tuple(d[k])
        return cls(**d)


def _replace(r: LossCheckReceipt, **kw: object) -> LossCheckReceipt:
    d = asdict(r) | kw
    return LossCheckReceipt(**d)


def make_receipt(*, evaluator_id: str, key: bytes, base: torch.Tensor, candidate: torch.Tensor, heldout: np.ndarray, window: int, n: int,
                 seed: int, model_factory: Callable[[], nn.Module], device: torch.device | str, dtype: torch.dtype = torch.float32,
                 round_id: str = "", batch: int = 1, stop: Callable[[], bool] | None = None) -> LossCheckReceipt:
    """Evaluator side: losses of the base and the candidate on the same n windows, signed."""
    starts = heldout_windows(len(heldout), window, n, seed)
    losses = []
    for flat in (base, candidate):
        m = model_factory()
        m.load_flat(flat)
        losses.append(tuple(window_losses(m, heldout, starts, window, device=device, dtype=dtype, batch=batch, stop=stop)))
        del m
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    r = LossCheckReceipt(evaluator_id=evaluator_id, base_sha256=weights_sha256(base), candidate_sha256=weights_sha256(candidate),
                         heldout_sha256=tokens_sha256(heldout), window=window, seed=seed, starts=tuple(starts), losses_before=losses[0],
                         losses_after=losses[1], compute_dtype=str(dtype).replace("torch.", ""), device=str(device), round_id=round_id)
    return r.signed(key)


# ---------------------------------------------------------------------------------------------------- the CPU backend side


def spot_check_picks(commitment: str, server_secret: bytes, n_windows: int, k: int) -> list[tuple[str, int]]:
    """k distinct (which, window index) pairs, which in {"before", "after"}, derived from HMAC(server_secret, commitment).
    The evaluator never holds `server_secret`, so it cannot predict the picks when it computes the losses."""
    total = 2 * n_windows
    k = max(0, min(k, total))
    seed = int.from_bytes(hmac.new(server_secret, commitment.encode(), hashlib.sha256).digest()[:8], "big")
    flat = np.random.default_rng(seed).choice(total, size=k, replace=False)
    return [("before" if int(j) < n_windows else "after", int(j) % n_windows) for j in sorted(int(x) for x in flat)]


def detection_probability(n_numbers: int, falsified: int, k: int) -> float:
    """P(at least one of `falsified` wrong numbers among the k spot-checked of `n_numbers`), sampling without replacement."""
    if falsified <= 0 or k <= 0:
        return 0.0
    if falsified + k > n_numbers:
        return 1.0
    return 1.0 - math.comb(n_numbers - falsified, k) / math.comb(n_numbers, k)


@dataclass(frozen=True)
class LossCheckVerdict:
    accepted: bool
    reason: str
    loss_before: float = float("nan")
    loss_after: float = float("nan")
    delta_loss: float = float("nan")
    checked: tuple[dict, ...] = ()
    detail: dict = field(default_factory=dict)


def verify_receipt(receipt: LossCheckReceipt, *, evaluator_keys: Mapping[str, bytes], server_secret: bytes, expected_base_sha256: str,
                   expected_candidate_sha256: str, heldout: np.ndarray, recompute: Callable[[str, list[int]], list[float]], k: int = 2,
                   tol: float = 2e-3, max_regression_tolerance: float = 0.0, min_windows: int = 8) -> LossCheckVerdict:
    """Backend side, CPU only. Every structural check happens before any forward pass; `recompute(which, starts)` returns the
    losses of the "before" (base) or "after" (candidate) weights on those window starts and is called at most twice.

    Accepts only when: the evaluator is known and its signature is valid; the receipt names the weights the backend expects
    and the backend's own held-out data; the windows are the ones the seed selects; every claimed loss is finite and in
    [0, ln(vocab) + 5]; every spot-checked number agrees within `tol`; and mean(after) - mean(before) <= the tolerance
    (the same rule as `LossRatchetGate`)."""
    key = evaluator_keys.get(receipt.evaluator_id)
    if key is None:
        return LossCheckVerdict(False, "REJECTED_UNKNOWN_EVALUATOR")
    if not receipt.verify_signature(key):
        return LossCheckVerdict(False, "REJECTED_BAD_SIGNATURE")
    if receipt.base_sha256 != expected_base_sha256 or receipt.candidate_sha256 != expected_candidate_sha256:
        return LossCheckVerdict(False, "REJECTED_WRONG_WEIGHTS")
    if receipt.heldout_sha256 != tokens_sha256(heldout):
        return LossCheckVerdict(False, "REJECTED_WRONG_HELDOUT_DATA")
    if len(receipt.starts) < min_windows:
        return LossCheckVerdict(False, f"REJECTED_TOO_FEW_WINDOWS ({len(receipt.starts)} < {min_windows})")
    if list(receipt.starts) != heldout_windows(len(heldout), receipt.window, len(receipt.starts), receipt.seed):
        return LossCheckVerdict(False, "REJECTED_WINDOWS_NOT_FROM_SEED")
    ceiling = math.log(int(CROWD_V1_CONFIG["vocabSize"])) + 5.0
    for v in (*receipt.losses_before, *receipt.losses_after):
        if not (math.isfinite(v) and 0.0 <= v <= ceiling):
            return LossCheckVerdict(False, "REJECTED_IMPLAUSIBLE_LOSS")
    picks = spot_check_picks(receipt.commitment(), server_secret, len(receipt.starts), k)
    checked: list[dict] = []
    for which in ("before", "after"):
        idx = [i for w, i in picks if w == which]
        if not idx:
            continue
        got = recompute(which, [receipt.starts[i] for i in idx])
        claimed = receipt.losses_before if which == "before" else receipt.losses_after
        for i, g in zip(idx, got):
            checked.append({"which": which, "window": i, "start": receipt.starts[i], "claimed": claimed[i], "recomputed": float(g),
                            "abs_diff": abs(float(g) - claimed[i])})
    bad = [c for c in checked if not c["abs_diff"] <= tol]
    mb, ma = receipt.mean_before, receipt.mean_after
    detail = {"commitment": receipt.commitment(), "k": len(checked), "tol": tol, "n_windows": len(receipt.starts),
              "max_abs_diff": max((c["abs_diff"] for c in checked), default=0.0)}
    if bad:
        c = bad[0]
        return LossCheckVerdict(False, f"REJECTED_SPOT_CHECK ({c['which']} window {c['window']}: claimed {c['claimed']:.5f}, "
                                f"recomputed {c['recomputed']:.5f})", mb, ma, ma - mb, tuple(checked), detail)
    delta = ma - mb
    if delta <= max_regression_tolerance:
        return LossCheckVerdict(True, f"ACCEPTED (delta_loss={delta:+.5f} <= tol={max_regression_tolerance:.5f})", mb, ma, delta, tuple(checked), detail)
    return LossCheckVerdict(False, f"HELD_PREVIOUS_HEAD (delta_loss={delta:+.5f} > tol={max_regression_tolerance:.5f})", mb, ma, delta,
                            tuple(checked), detail)


def receipts_agree(a: LossCheckReceipt, b: LossCheckReceipt, tol: float = 2e-3) -> tuple[bool, str]:
    """Two independent evaluators on the same weights, data and windows: every number within `tol`."""
    if a.evaluator_id == b.evaluator_id:
        return False, "same evaluator"
    same = (a.base_sha256, a.candidate_sha256, a.heldout_sha256, a.starts) == (b.base_sha256, b.candidate_sha256, b.heldout_sha256, b.starts)
    if not same:
        return False, "the receipts are about different weights, data or windows"
    worst = max(abs(x - y) for x, y in zip((*a.losses_before, *a.losses_after), (*b.losses_before, *b.losses_after)))
    return worst <= tol, f"max abs diff {worst:.6f} (tol {tol})"


def cpu_recompute_fn(model_factory: Callable[[], nn.Module], base: torch.Tensor, candidate: torch.Tensor, heldout: np.ndarray, window: int,
                     dtype: torch.dtype = torch.float32) -> Callable[[str, list[int]], list[float]]:
    """A `recompute` for `verify_receipt` that loads the base or candidate weights on CPU only when a pick needs them."""

    def run(which: str, starts: list[int]) -> list[float]:
        m = model_factory()
        m.load_flat(base if which == "before" else candidate)
        return window_losses(m, heldout, starts, window, device="cpu", dtype=dtype)

    return run
