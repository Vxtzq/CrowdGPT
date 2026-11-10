"""Held-out loss check for CrowdGPT (`Vxtzq/Crowd-v1`), computed on a community GPU and verified on a CPU-only backend.

Supports both:
- Flat `model_bf16.safetensors` checkpoints (`CROWD_V1_CONFIG`, 657,787,392 trunk parameters)
- Hugging Face `Vxtzq/Crowd-v1/model.safetensors` (`Step 348+`) with `CrowdEngram` (`CROWD_V1_ENGRAM_CONFIG`,
  `engramBuckets=227,865`, 350,000,640 bigram table parameters).

Receipt signing, HMAC commitment, `spot_check_picks`, `detection_probability`, `receipts_agree`, and structural
`verify_receipt` checks work with Python's standard library alone (zero GPU / PyTorch import requirement), while
`CrowdV1` and `window_losses` use PyTorch + NumPy when installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - stdlib-only backend / CI mode
    np = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - stdlib-only backend / CI mode
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

RECEIPT_SCHEMA = "hivemind.crowdgpt.loss_check.v1"
CROWD_V1_CONFIG: dict[str, int | bool] = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
CROWD_V1_ENGRAM_BUCKETS = 227_865
CROWD_V1_ENGRAM_CONFIG: dict[str, int | bool] = {
    **CROWD_V1_CONFIG,
    "engramBuckets": CROWD_V1_ENGRAM_BUCKETS,
}
CROWD_V1_MODEL_SIZE = 657_787_392
CROWD_V1_ENGRAM_SIZE = CROWD_V1_ENGRAM_BUCKETS * 1536
TOKEN_DTYPE = np.uint32 if np is not None else "uint32"


def _require_torch() -> None:
    if torch is None or nn is None or F is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for CrowdV1 forward passes.")


# ---------------------------------------------------------------------------------------------------------------- model
if nn is not None:

    def _rope(head_dim: int, seq_len: int, device: Any) -> tuple[Any, Any]:
        inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        f = torch.einsum("i,j->ij", torch.arange(seq_len, dtype=torch.float32), inv)
        e = torch.cat((f, f), -1)
        return e.cos()[None, None].to(device), e.sin()[None, None].to(device)

    def _rotate_half(x: Any) -> Any:
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), -1)

    class _Attention(nn.Module):
        def __init__(self, c: Mapping[str, int | bool]) -> None:
            super().__init__()
            self.nh, self.nkv, self.hd, dim = int(c["nHeads"]), int(c["nKvHeads"]), int(c["headDim"]), int(c["dim"])
            self.wq = nn.Linear(dim, self.nh * self.hd, bias=False)
            self.wk = nn.Linear(dim, self.nkv * self.hd, bias=False)
            self.wv = nn.Linear(dim, self.nkv * self.hd, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
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
            if q.dtype != v.dtype or k.dtype != v.dtype:
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

        def forward(self, x: Any) -> Any:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

    class _Block(nn.Module):
        def __init__(self, c: Mapping[str, int | bool]) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(int(c["dim"]))
            self.attn = _Attention(c)
            self.ln_2 = nn.LayerNorm(int(c["dim"]))
            self.mlp = _SwiGLU(c)

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            x = x + self.attn(self.ln_1(x), cos, sin)
            return x + self.mlp(self.ln_2(x))

    class _Engram(nn.Module):
        """Bigram hash lookup `(prev * 1_000_003 + idx) % buckets` matching `modeling_crowd.CrowdEngram`."""

        def __init__(self, buckets: int, dim: int) -> None:
            super().__init__()
            self.table = nn.Embedding(buckets, dim)

        def forward(self, idx: Any) -> Any:
            prev = torch.cat((torch.zeros_like(idx[:, :1]), idx[:, :-1]), dim=1)
            h = (prev.to(torch.int64) * 1_000_003 + idx.to(torch.int64)) % self.table.num_embeddings
            u, inv = torch.unique(h.flatten(), return_inverse=True)
            looked = self.table(u.to(self.table.weight.device)).to(idx.device)
            return looked[inv].view(*idx.shape, -1)

    class CrowdV1(nn.Module):
        """Inference-only SotaGPT / `CrowdForCausalLM` with optional `CrowdEngram` (`engramBuckets=227865`)."""

        def __init__(self, config: Mapping[str, int | bool] = CROWD_V1_CONFIG) -> None:
            super().__init__()
            self.config = dict(config)
            dim, vocab = int(config["dim"]), int(config["vocabSize"])
            raw_buckets = config.get("engramBuckets", config.get("engram_buckets", 0))
            if isinstance(raw_buckets, bool) or not isinstance(raw_buckets, int) or raw_buckets < 0:
                raise ValueError("engramBuckets must be a non-negative integer")
            self.engram_buckets = int(raw_buckets)
            self.wte = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([_Block(config) for _ in range(int(config["nLayers"]))])
            self.ln_f = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            if config.get("weightTying", True):
                self.wte.weight = self.lm_head.weight
            self.engram = _Engram(self.engram_buckets, dim) if self.engram_buckets > 0 else None

        def _base_parameters(self) -> list[Any]:
            out = [self.wte.weight]
            for blk in self.blocks:
                out.extend(list(blk.parameters()))
            out.extend([self.ln_f.weight, self.ln_f.bias])
            if not self.config.get("weightTying", True):
                out.append(self.lm_head.weight)
            return out

        def n_base_flat(self) -> int:
            return sum(p.numel() for p in self._base_parameters())

        def n_flat(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def load_flat(self, flat: Any, engram: Any = None) -> None:
            flat = flat.reshape(-1)
            base_params = self._base_parameters()
            n_base = sum(p.numel() for p in base_params)
            n_total = self.n_flat()
            if engram is not None:
                if flat.numel() != n_base:
                    raise ValueError(f"flat trunk weights have {flat.numel()} values, trunk needs {n_base}")
                if self.engram is None:
                    raise ValueError("engram tensor supplied, but engramBuckets is 0")
                eng_flat = engram.reshape(-1)
                if eng_flat.numel() != self.engram.table.weight.numel():
                    raise ValueError(f"engram weights have {eng_flat.numel()} values")
                targets = [*base_params, self.engram.table.weight]
                source = torch.cat((flat, eng_flat.to(flat.device, dtype=flat.dtype)))
            elif self.engram is not None and flat.numel() == n_base:
                targets = base_params
                source = flat
            else:
                if flat.numel() != n_total:
                    raise ValueError(f"flat weights have {flat.numel()} values, the model needs {n_total}")
                targets = list(self.parameters())
                source = flat
            o = 0
            with torch.no_grad():
                for p in targets:
                    s = p.numel()
                    p.copy_(source[o: o + s].view(p.shape).to(p.dtype))
                    o += s

        def forward(self, idx: Any) -> Any:
            cos, sin = _rope(int(self.config["headDim"]), idx.shape[1], idx.device)
            x = self.wte(idx)
            if self.engram is not None:
                x = x + self.engram(idx).to(device=x.device, dtype=x.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            return F.linear(self.ln_f(x), self.lm_head.weight)
else:
    CrowdV1 = None  # type: ignore[assignment]


def load_flat_safetensors(path: Path | str) -> Any:
    """The single flat tensor of a CrowdGPT `model_*.safetensors` file (key `weights`)."""
    _require_torch()
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    key = "weights" if "weights" in tensors else ("flat" if "flat" in tensors else None)
    if key is None:
        if len(tensors) != 1:
            raise ValueError(f"cannot identify the weights tensor among {sorted(tensors)[:10]}")
        key = next(iter(tensors))
    return tensors[key].reshape(-1)


def load_named_crowd_v1_safetensors(
    path: Path | str,
    *,
    n_layers: int = 24,
    weight_tying: bool = True,
) -> tuple[Any, Any]:
    """Extract `(flat_trunk, flat_engram)` from Hugging Face `Vxtzq/Crowd-v1/model.safetensors` (`Step 348+`)."""
    _require_torch()
    from safetensors.torch import load_file

    sd = load_file(str(path), device="cpu")
    parts = [sd["wte.weight"].reshape(-1)]
    for i in range(int(n_layers)):
        pfx = f"blocks.{i}"
        parts.extend(
            [
                sd[f"{pfx}.ln1.weight"].reshape(-1),
                sd[f"{pfx}.ln1.bias"].reshape(-1),
                sd[f"{pfx}.attn.q.weight"].reshape(-1),
                sd[f"{pfx}.attn.k.weight"].reshape(-1),
                sd[f"{pfx}.attn.v.weight"].reshape(-1),
                sd[f"{pfx}.attn.o.weight"].reshape(-1),
                sd[f"{pfx}.ln2.weight"].reshape(-1),
                sd[f"{pfx}.ln2.bias"].reshape(-1),
                sd[f"{pfx}.mlp.w1.weight"].reshape(-1),
                sd[f"{pfx}.mlp.w2.weight"].reshape(-1),
                sd[f"{pfx}.mlp.w3.weight"].reshape(-1),
            ]
        )
    parts.extend([sd["ln.weight"].reshape(-1), sd["ln.bias"].reshape(-1)])
    if not weight_tying and "head.weight" in sd:
        parts.append(sd["head.weight"].reshape(-1))
    engram = sd["engram.table.weight"].reshape(-1) if "engram.table.weight" in sd else None
    return torch.cat(parts), engram


def weights_sha256(flat: Any) -> str:
    """SHA-256 of the weights as bf16 bytes (or raw bytes when passed bytes in stdlib-only tests)."""
    if isinstance(flat, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(flat)).hexdigest()
    _require_torch()
    b = flat.detach().to("cpu", dtype=torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def tokens_sha256(tokens: Any) -> str:
    if np is not None and isinstance(tokens, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(tokens, dtype=np.uint32).tobytes()).hexdigest()
    import struct as _struct

    seq = [int(x) for x in tokens]
    return hashlib.sha256(_struct.pack(f"<{len(seq)}I", *seq)).hexdigest()


def heldout_windows(n_tokens: int, window: int, n: int, seed: int) -> list[int]:
    """`n` distinct, non-overlapping window starts in a held-out token array, chosen by `seed`."""
    slots = n_tokens // window
    if window < 2 or n < 1 or slots < n:
        raise ValueError(f"need window >= 2, n >= 1 and at least n={n} windows of {window} tokens (have {slots})")
    if np is not None:
        rng = np.random.default_rng(seed)
        return sorted(int(s) * window for s in rng.choice(slots, size=n, replace=False))
    py_rng = random.Random(seed)
    return sorted(int(s) * window for s in py_rng.sample(range(slots), n))


def window_losses(
    model: Any,
    tokens: Any,
    starts: Sequence[int],
    window: int,
    *,
    device: Any = "cpu",
    dtype: Any = None,
    batch: int = 1,
    stop: Callable[[], bool] | None = None,
) -> list[float]:
    _require_torch()
    if dtype is None:
        dtype = torch.float32
    vocab = int(getattr(model, "config", {}).get("vocabSize", 0)) or None
    arr = np.asarray(tokens)
    if vocab is not None and len(arr) and int(arr.max()) >= vocab:
        raise ValueError("held-out tokens contain ids outside the vocabulary")
    model = model.to(device=device, dtype=dtype).train(False)
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
        return float(sum(self.losses_before) / len(self.losses_before))

    @property
    def mean_after(self) -> float:
        return float(sum(self.losses_after) / len(self.losses_after))

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


def make_receipt(
    *,
    evaluator_id: str,
    key: bytes,
    base: Any,
    candidate: Any,
    heldout: Any,
    window: int,
    n: int,
    seed: int,
    model_factory: Callable[[], Any],
    device: Any,
    dtype: Any = None,
    round_id: str = "",
    batch: int = 1,
    stop: Callable[[], bool] | None = None,
) -> LossCheckReceipt:
    _require_torch()
    if dtype is None:
        dtype = torch.float32
    starts = heldout_windows(len(heldout), window, n, seed)
    losses = []
    for flat in (base, candidate):
        m = model_factory()
        m.load_flat(flat)
        losses.append(tuple(window_losses(m, heldout, starts, window, device=device, dtype=dtype, batch=batch, stop=stop)))
        del m
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    r = LossCheckReceipt(
        evaluator_id=evaluator_id,
        base_sha256=weights_sha256(base),
        candidate_sha256=weights_sha256(candidate),
        heldout_sha256=tokens_sha256(heldout),
        window=window,
        seed=seed,
        starts=tuple(starts),
        losses_before=losses[0],
        losses_after=losses[1],
        compute_dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        round_id=round_id,
    )
    return r.signed(key)


def spot_check_picks(commitment: str, server_secret: bytes, n_windows: int, k: int) -> list[tuple[str, int]]:
    total = 2 * n_windows
    k = max(0, min(k, total))
    seed = int.from_bytes(hmac.new(server_secret, commitment.encode(), hashlib.sha256).digest()[:8], "big")
    if np is not None:
        flat = [int(x) for x in np.random.default_rng(seed).choice(total, size=k, replace=False)]
    else:
        flat = random.Random(seed).sample(range(total), k)
    return [("before" if j < n_windows else "after", j % n_windows) for j in sorted(flat)]


def detection_probability(n_numbers: int, falsified: int, k: int) -> float:
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


def verify_receipt(
    receipt: LossCheckReceipt,
    *,
    evaluator_keys: Mapping[str, bytes],
    server_secret: bytes,
    expected_base_sha256: str,
    expected_candidate_sha256: str,
    heldout: Any,
    recompute: Callable[[str, list[int]], list[float]],
    k: int = 2,
    tol: float = 2e-3,
    max_regression_tolerance: float = 0.0,
    min_windows: int = 8,
) -> LossCheckVerdict:
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
        if len(got) != len(idx) or not all(math.isfinite(float(g)) for g in got):
            return LossCheckVerdict(False, "REJECTED_RECOMPUTE_INCOMPLETE")
        claimed = receipt.losses_before if which == "before" else receipt.losses_after
        for i, g in zip(idx, got):
            checked.append(
                {
                    "which": which,
                    "window": i,
                    "start": receipt.starts[i],
                    "claimed": claimed[i],
                    "recomputed": float(g),
                    "abs_diff": abs(float(g) - claimed[i]),
                }
            )
    bad = [c for c in checked if not c["abs_diff"] <= tol]
    mb, ma = receipt.mean_before, receipt.mean_after
    detail = {
        "commitment": receipt.commitment(),
        "k": len(checked),
        "tol": tol,
        "n_windows": len(receipt.starts),
        "max_abs_diff": max((c["abs_diff"] for c in checked), default=0.0),
    }
    if bad:
        c = bad[0]
        return LossCheckVerdict(
            False,
            f"REJECTED_SPOT_CHECK ({c['which']} window {c['window']}: claimed {c['claimed']:.5f}, recomputed {c['recomputed']:.5f})",
            mb,
            ma,
            ma - mb,
            tuple(checked),
            detail,
        )
    delta = ma - mb
    if delta <= max_regression_tolerance:
        return LossCheckVerdict(
            True,
            f"ACCEPTED (delta_loss={delta:+.5f} <= tol={max_regression_tolerance:.5f})",
            mb,
            ma,
            delta,
            tuple(checked),
            detail,
        )
    return LossCheckVerdict(
        False,
        f"HELD_PREVIOUS_HEAD (delta_loss={delta:+.5f} > tol={max_regression_tolerance:.5f})",
        mb,
        ma,
        delta,
        tuple(checked),
        detail,
    )


def receipts_agree(a: LossCheckReceipt, b: LossCheckReceipt, tol: float = 2e-3) -> tuple[bool, str]:
    if a.evaluator_id == b.evaluator_id:
        return False, "same evaluator"
    same = (a.base_sha256, a.candidate_sha256, a.heldout_sha256, a.starts) == (
        b.base_sha256,
        b.candidate_sha256,
        b.heldout_sha256,
        b.starts,
    )
    if not same:
        return False, "the receipts are about different weights, data or windows"
    worst = max(abs(x - y) for x, y in zip((*a.losses_before, *a.losses_after), (*b.losses_before, *b.losses_after)))
    return worst <= tol, f"max abs diff {worst:.6f} (tol {tol})"


def cpu_recompute_fn(
    model_factory: Callable[[], Any] | None,
    base: Any,
    candidate: Any,
    heldout: Any,
    window: int,
    dtype: Any = None,
) -> Callable[[str, list[int]], list[float]]:
    def run(which: str, starts: list[int]) -> list[float]:
        _require_torch()
        if model_factory is None:
            raise RuntimeError("model_factory (or PyTorch CrowdV1) is required when k > 0")
        eff_dtype = torch.float32 if dtype is None else dtype
        m = model_factory()
        m.load_flat(base if which == "before" else candidate)
        return window_losses(m, heldout, starts, window, device="cpu", dtype=eff_dtype)

    return run
