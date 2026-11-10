"""CrowdGPT x HiveMind DiLoCo, Multi-Krum, CrossCheck & Fleet Header Bridge.

Zero-extra-dependency integration module for `Vxtzq/CrowdGPT` (`server.crowdgpt.net` & `client.py` v3.1):
1. Pure-stdlib client/coordinator protocol helpers (`build_crowdgpt_client_hardware_headers`,
   `parse_crowdgpt_task_crosscheck`, `build_crowdgpt_submit_metadata`, `classify_gpu_precision_cohort`)
   so both stdlib-only coordinator environments and CI runners can validate `user:<uuid>:dev:<device>`
   (`X-Client-Device: hm-<handle>-<gpu>`) multiplexing and `crossCheck` cohort submissions without
   requiring PyTorch at import time.
2. Exact binary packer/unpacker (`pack_crowdgpt_delta_stream` / `unpack_crowdgpt_delta_stream`) for
   CrowdGPT's gzipped `/fl/submit` delta stream (`<I` header length + JSON metadata + `bf16` delta + sparse engram).
3. Chunked Multi-Krum Byzantine filtering (`multi_krum_select_and_aggregate`), Anti-Lobotomy Weight RMS
   Collapse Guard (`check_weight_rms_collapse_guard`), and Held-Out Loss Ratchet (`LossRatchetGate`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import statistics
import struct
import sys
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - stdlib-only CI / lightweight coordinator mode
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - stdlib-only CI / lightweight coordinator mode
    torch = None  # type: ignore[assignment]

DEFAULT_HIVEMIND_GATEWAY_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
CROWDGPT_HF_REPO_ID = "Vxtzq/Crowd-v1"
HIVEMIND_CROWD_MODEL_ID = "hivemind/crowd-v1"
MAX_TOKEN_WEIGHT_MEDIAN_MULTIPLE: float = 1.25


def _require_torch_numpy() -> None:
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for tensor delta packing and Multi-Krum operations.")


# ---------------------------------------------------------------------------
# 1. Pure-stdlib v3.1 Hardware Headers, Precision Cohorts & CrossCheck Protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuPrecisionCohort:
    """Hardware-stratified precision & optimizer cohort assignment (`Vxtzq/CrowdGPT#44`, `#51`)."""

    vendor: str
    compute_capability: tuple[int, int]
    effective_dtype: str
    optimizer_family: str
    pre_ampere_fp32_enforced: bool
    cohort_key: str


def classify_gpu_precision_cohort(
    vendor: str,
    compute_capability: tuple[int, int] = (0, 0),
    *,
    requested_dtype: str = "auto",
    optimizer_family: str = "adamw_8bit",
) -> GpuPrecisionCohort:
    """Assign a volunteer GPU to a homogeneous numerical cohort and enforce `fp32` on pre-Ampere CUDA (`sm < 80`)."""
    if not isinstance(vendor, str) or not vendor.strip():
        raise ValueError("vendor must be a non-empty string.")
    if (
        not isinstance(compute_capability, tuple)
        or len(compute_capability) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in compute_capability)
    ):
        raise ValueError("compute_capability must be a 2-tuple of non-negative integers.")
    norm_vendor = vendor.strip().lower()
    major, minor = int(compute_capability[0]), int(compute_capability[1])
    req = requested_dtype.strip().lower()
    opt = optimizer_family.strip().lower() or "adamw"

    pre_ampere_enforced = False
    if norm_vendor in ("nvidia", "cuda"):
        norm_vendor = "nvidia"
        if major < 8:
            effective_dtype = "fp32"
            pre_ampere_enforced = req in ("auto", "bf16", "bfloat16")
            arch_tier = f"sm{major}{minor}_pre_ampere"
        else:
            effective_dtype = "fp32" if req in ("fp32", "float32") else "bf16"
            arch_tier = "sm80_plus_ampere"
    elif norm_vendor in ("amd", "rocm", "hip"):
        norm_vendor = "amd_rocm"
        effective_dtype = "bf16" if req in ("bf16", "bfloat16") else "fp32"
        arch_tier = "rocm"
    elif norm_vendor in ("apple", "mps", "metal"):
        norm_vendor = "apple_mps"
        effective_dtype = "fp32"
        arch_tier = "mps"
    else:
        effective_dtype = "fp32"
        arch_tier = "generic"

    cohort_key = f"{norm_vendor}:{arch_tier}:{effective_dtype}:{opt}"
    return GpuPrecisionCohort(
        vendor=norm_vendor,
        compute_capability=(major, minor),
        effective_dtype=effective_dtype,
        optimizer_family=opt,
        pre_ampere_fp32_enforced=pre_ampere_enforced,
        cohort_key=cohort_key,
    )


@dataclass(frozen=True)
class CrowdGPTCrossCheckSpec:
    """Exact cross-client cohort specification issued by `server.crowdgpt.net` `/fl/task` (`client.py:_parse_task_meta`)."""

    task_id: str
    global_step: int
    seq_len: int
    cohort_mode: bool
    target_seed: int | None
    target_local_steps: int | None
    target_batch_size: int | None
    target_accum_steps: int | None
    target_autocast: str | None
    target_optimizer: str
    effective_slot: int


def build_crowdgpt_client_hardware_headers(
    *,
    hiveminder_handle: str,
    device_suffix: str = "gpu0",
    backend: str = "CUDA",
    batch_size: int = 1,
    vram_gb: float = 8.0,
    gpu_name: str = "NVIDIA GeForce RTX 2070",
    autocast_dtype: str = "fp32",
    native_optim: str = "adamw8bit",
    can_fp32: bool = True,
    can_8bit: bool = True,
    engram_active: bool = True,
    compute_capability: tuple[int, int] = (7, 5),
    client_version: str = "3.1.0",
) -> dict[str, str]:
    """Build the exact 12 `X-Client-*` headers required by `server.crowdgpt.net` (`client.py:_hardware_headers`).

    Because CrowdGPT keys active per-client tasks on `user:<uuid>:dev:<device>` (`client.py:337`),
    setting a deterministic `X-Client-Device: hm-<handle>-<device_suffix>` lets multiple HiveMinder
    GPUs share one fleet account (`hivemind-swarm`) OR use personal CrowdGPT accounts without colliding.
    """
    handle_clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(hiveminder_handle or "").strip())
    if not handle_clean:
        raise ValueError("hiveminder_handle must be a non-empty string.")
    dev_clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(device_suffix or "gpu0").strip()) or "gpu0"
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1.")
    if isinstance(vram_gb, bool) or not isinstance(vram_gb, (int, float)) or not math.isfinite(float(vram_gb)) or float(vram_gb) <= 0.0:
        raise ValueError("vram_gb must be a finite positive number.")
    if (
        not isinstance(compute_capability, tuple)
        or len(compute_capability) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in compute_capability)
    ):
        raise ValueError("compute_capability must be a 2-tuple of non-negative integers.")
    device_id = f"hm-{handle_clean[:32]}-{dev_clean[:16]}"
    cc_str = f"{compute_capability[0]}.{compute_capability[1]}"
    return {
        "X-Client-Backend": str(backend or "CUDA").strip().upper(),
        "X-Client-Batch-Size": str(int(batch_size)),
        "X-Client-Vram-Gb": f"{float(vram_gb):.2f}",
        "X-Client-Gpu-Name": str(gpu_name or "UNKNOWN").strip(),
        "X-Client-Autocast": str(autocast_dtype or "fp32").strip().lower(),
        "X-Client-Optim": str(native_optim or "adamw8bit").strip(),
        "X-Client-Optim-Can-Fp32": "1" if can_fp32 else "0",
        "X-Client-Optim-Can-8bit": "1" if can_8bit else "0",
        "X-Client-Engram": "1" if engram_active else "0",
        "X-Client-Device": device_id,
        "X-Client-Compute-Cap": cc_str,
        "X-Client-Version": str(client_version or "3.1.0").strip(),
    }


def parse_crowdgpt_task_crosscheck(
    task_metadata: Mapping[str, Any],
    *,
    default_seq_len: int = 2048,
) -> CrowdGPTCrossCheckSpec:
    """Parse `/fl/task` metadata and enforce `crossCheck` cohort rules (`client.py:_parse_task_meta`)."""
    if not isinstance(task_metadata, Mapping):
        raise TypeError("task_metadata must be a Mapping.")
    tid = str(task_metadata.get("taskId") or "").strip()
    if not tid:
        raise ValueError("task_metadata must include a non-empty taskId.")
    gs = int(task_metadata.get("globalStep") or 0)
    cc_cfg = task_metadata.get("crossCheck")
    if not isinstance(cc_cfg, Mapping):
        cc_cfg = {}

    cohort_mode = bool(cc_cfg.get("enabled", False))
    cc_seed = cc_cfg.get("targetSeed") if cohort_mode else None
    cc_steps = cc_cfg.get("targetLocalSteps") if cohort_mode else None
    cc_bs = cc_cfg.get("targetBatchSize") if cohort_mode else None
    cc_accum = cc_cfg.get("targetAccumSteps") if cohort_mode else None
    cc_autocast = cc_cfg.get("targetAutocast") if cohort_mode else None
    cc_optim = str((cc_cfg.get("targetOptimizer") if cohort_mode else None) or "any")

    if cohort_mode and (
        cc_seed is None
        or cc_steps is None
        or cc_bs is None
        or cc_accum is None
        or cc_autocast is None
    ):
        cohort_mode = False
        cc_seed = cc_steps = cc_bs = cc_accum = cc_autocast = None
        cc_optim = "any"

    shard_cfg = task_metadata.get("shardConfig")
    raw_slot = int(shard_cfg.get("slot", 0)) if isinstance(shard_cfg, Mapping) else 0
    effective_slot = 0 if cohort_mode else raw_slot

    return CrowdGPTCrossCheckSpec(
        task_id=tid,
        global_step=gs,
        seq_len=int(default_seq_len),
        cohort_mode=cohort_mode,
        target_seed=int(cc_seed) if cc_seed is not None else None,
        target_local_steps=int(cc_steps) if cc_steps is not None else None,
        target_batch_size=int(cc_bs) if cc_bs is not None else None,
        target_accum_steps=int(cc_accum) if cc_accum is not None else None,
        target_autocast=str(cc_autocast) if cc_autocast is not None else None,
        target_optimizer=cc_optim,
        effective_slot=effective_slot,
    )


def build_crowdgpt_submit_metadata(
    *,
    spec: CrowdGPTCrossCheckSpec,
    mean_loss: float,
    last_step_loss: float,
    loss_std: float,
    loss_samples: int,
    actual_steps: int,
    actual_batch_size: int,
    tokens_processed: int,
    has_engram: bool = False,
    engram_sparse_count: int = 0,
) -> dict[str, Any]:
    """Build and verify the exact `/fl/submit` JSON header required by CrowdGPT cross-client verification (`client.py:2905-2914`)."""
    if not isinstance(spec, CrowdGPTCrossCheckSpec):
        raise TypeError("spec must be a CrowdGPTCrossCheckSpec.")
    for name, val in (("mean_loss", mean_loss), ("last_step_loss", last_step_loss), ("loss_std", loss_std)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite number.")
    if float(loss_std) < 0.0:
        raise ValueError("loss_std must be non-negative.")
    for int_name, int_val, min_v in (
        ("loss_samples", loss_samples, 1),
        ("actual_steps", actual_steps, 1),
        ("actual_batch_size", actual_batch_size, 1),
        ("tokens_processed", tokens_processed, 1),
        ("engram_sparse_count", engram_sparse_count, 0),
    ):
        if isinstance(int_val, bool) or not isinstance(int_val, int) or int_val < min_v:
            raise ValueError(f"{int_name} must be an integer >= {min_v}.")

    if spec.cohort_mode:
        if spec.target_local_steps is not None and int(actual_steps) != int(spec.target_local_steps):
            raise ValueError(
                f"Cross-check cohort requires exact targetLocalSteps={spec.target_local_steps}, got actual_steps={actual_steps}."
            )
        if spec.target_batch_size is not None and int(actual_batch_size) != int(spec.target_batch_size):
            raise ValueError(
                f"Cross-check cohort requires exact targetBatchSize={spec.target_batch_size}, got actual_batch_size={actual_batch_size}."
            )

    return {
        "taskId": spec.task_id,
        "loss": float(mean_loss),
        "localSteps": int(actual_steps),
        "tokensProcessed": int(tokens_processed),
        "loraRank": 0,
        "isDelta": True,
        "weightFormat": "bf16",
        "hasEngram": bool(has_engram and engram_sparse_count > 0),
        "engramSparseCount": int(engram_sparse_count) if has_engram else 0,
        "crossCheckUsed": bool(spec.cohort_mode),
        "actualSeed": int(spec.target_seed) if (spec.cohort_mode and spec.target_seed is not None) else None,
        "actualSteps": int(actual_steps),
        "actualBatchSize": int(actual_batch_size),
        "lastLoss": float(last_step_loss),
        "lossStd": float(loss_std),
        "lossSamples": int(loss_samples),
    }


# ---------------------------------------------------------------------------
# 2. Gzipped Delta Stream, Multi-Krum, RMS Guard & Loss Ratchet (PyTorch/NumPy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrowdGPTDeltaPayload:
    """Unpacked representation of a CrowdGPT client delta upload."""

    metadata: dict[str, Any]
    base_delta_bf16: Any
    engram_indices: Any
    engram_values_uint16: Any
    payload_sha256: str


def pack_crowdgpt_delta_stream(
    metadata: Mapping[str, Any],
    base_delta_bf16: Any,
    engram_indices: Any = None,
    engram_values_uint16: Any = None,
    *,
    compresslevel: int = 6,
) -> bytes:
    """Pack a CrowdGPT delta in the exact wire format produced by `client.py`."""
    _require_torch_numpy()
    if base_delta_bf16.dtype != torch.bfloat16:
        base_delta_bf16 = base_delta_bf16.to(torch.bfloat16)
    flat_uint16 = (
        base_delta_bf16.detach()
        .to("cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    si = (
        np.ascontiguousarray(engram_indices, dtype=np.uint32)
        if engram_indices is not None and len(engram_indices) > 0
        else np.array([], dtype=np.uint32)
    )
    sv = (
        np.ascontiguousarray(engram_values_uint16, dtype=np.uint16)
        if engram_values_uint16 is not None and len(engram_values_uint16) > 0
        else np.array([], dtype=np.uint16)
    )
    meta_dict = dict(metadata)
    meta_dict.setdefault("loraRank", 0)
    meta_dict.setdefault("isDelta", True)
    meta_dict.setdefault("weightFormat", "bf16")
    meta_dict["hasEngram"] = bool(len(si) > 0)
    meta_dict["engramSparseCount"] = int(len(si))

    pl_bytes = json.dumps(meta_dict, separators=(",", ":")).encode("utf-8")
    header = struct.pack("<I", len(pl_bytes)) + pl_bytes

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=compresslevel) as gz:
        gz.write(header)
        for arr in (flat_uint16, si, sv):
            mv = memoryview(np.ascontiguousarray(arr)).cast("B")
            gz.write(mv)
    return buf.getvalue()


def unpack_crowdgpt_delta_stream(
    compressed_data: bytes,
    *,
    engram_row_dim: int = 1536,
) -> CrowdGPTDeltaPayload:
    """Unpack a gzipped CrowdGPT client delta stream and verify its structure."""
    _require_torch_numpy()
    payload_sha256 = hashlib.sha256(compressed_data).hexdigest()
    raw = gzip.decompress(compressed_data)
    if len(raw) < 4:
        raise ValueError("CrowdGPT delta stream is shorter than the 4-byte header prefix.")

    meta_len = struct.unpack("<I", raw[:4])[0]
    meta_end = 4 + meta_len
    if len(raw) < meta_end:
        raise ValueError("CrowdGPT delta stream truncated inside JSON metadata header.")

    metadata = json.loads(raw[4:meta_end].decode("utf-8"))
    body = memoryview(raw)[meta_end:]

    has_engram = bool(metadata.get("hasEngram", False))
    sparse_count = int(metadata.get("engramSparseCount", 0)) if has_engram else 0
    engram_idx_bytes = sparse_count * 4
    engram_val_bytes = sparse_count * engram_row_dim * 2
    engram_total_bytes = engram_idx_bytes + engram_val_bytes

    if len(body) < engram_total_bytes:
        raise ValueError("CrowdGPT delta stream smaller than declared sparse engram tail.")

    base_bytes_len = len(body) - engram_total_bytes
    if base_bytes_len % 2 != 0:
        raise ValueError("CrowdGPT bf16 base delta byte length must be divisible by 2.")

    base_int16 = np.frombuffer(body[:base_bytes_len], dtype=np.int16).copy()
    base_bf16 = torch.from_numpy(base_int16).view(torch.bfloat16)

    if sparse_count > 0:
        si = np.frombuffer(
            body[base_bytes_len : base_bytes_len + engram_idx_bytes], dtype=np.uint32
        ).copy()
        sv = np.frombuffer(
            body[base_bytes_len + engram_idx_bytes :], dtype=np.uint16
        ).copy()
    else:
        si = np.array([], dtype=np.uint32)
        sv = np.array([], dtype=np.uint16)

    return CrowdGPTDeltaPayload(
        metadata=metadata,
        base_delta_bf16=base_bf16,
        engram_indices=si,
        engram_values_uint16=sv,
        payload_sha256=payload_sha256,
    )


def _flatten_delta(delta: Any) -> Any:
    _require_torch_numpy()
    if isinstance(delta, torch.Tensor):
        return delta.detach().to("cpu", dtype=torch.float32).reshape(-1)
    parts = [
        delta[k].detach().to("cpu", dtype=torch.float32).reshape(-1)
        for k in sorted(delta.keys())
    ]
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.float32)


def chunked_pairwise_sq_l2(
    deltas: Sequence[Any],
    *,
    chunk_elements: int = 1_048_576,
) -> Any:
    """Compute the N x N pairwise squared L2 distance matrix in memory-bounded chunks."""
    _require_torch_numpy()
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int) or chunk_elements < 1:
        raise ValueError("chunk_elements must be an integer >= 1.")
    n = len(deltas)
    if n == 0:
        raise ValueError("deltas must be non-empty.")
    flat = [_flatten_delta(d) for d in deltas]
    dim = flat[0].numel()
    for idx, vec in enumerate(flat):
        if vec.numel() != dim:
            raise ValueError(f"Delta {idx} has {vec.numel()} elements, expected {dim}.")
        if not bool(torch.isfinite(vec).all()):
            raise ValueError(f"Delta {idx} contains non-finite (NaN/Inf) values.")

    dist_sq = torch.zeros((n, n), dtype=torch.float64)
    for start in range(0, dim, chunk_elements):
        end = min(dim, start + chunk_elements)
        block = torch.stack([v[start:end] for v in flat], dim=0).to(torch.float64)
        norms = (block * block).sum(dim=1, keepdim=True)
        gram = block @ block.T
        chunk_d2 = torch.clamp(norms + norms.T - 2.0 * gram, min=0.0)
        dist_sq += chunk_d2
    dist_sq.fill_diagonal_(0.0)
    return dist_sq.to(torch.float32)


@dataclass(frozen=True)
class MultiKrumAggregationResult:
    """Result of Multi-Krum filtering and token-weighted DiLoCo aggregation."""

    selected_indices: list[int]
    rejected_indices: list[int]
    krum_scores: list[float]
    aggregated_delta_bf16: Any
    effective_tokens: int
    byzantine_filter_active: bool = True


def _validate_and_clamp_token_weights(
    token_weights: Sequence[int | float],
    expected_len: int,
) -> list[float]:
    if len(token_weights) != expected_len:
        raise ValueError("token_weights length must match deltas length.")
    parsed: list[float] = []
    for w in token_weights:
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"token_weights must be finite and positive, got {w!r}")
        fw = float(w)
        if not math.isfinite(fw) or fw <= 0.0:
            raise ValueError(f"token_weights must be finite and positive, got {w!r}")
        parsed.append(max(1.0, fw))
    med_w = float(statistics.median(parsed))
    cap_w = max(1.0, MAX_TOKEN_WEIGHT_MEDIAN_MULTIPLE * med_w)
    return [min(w, cap_w) for w in parsed]


def multi_krum_select_and_aggregate(
    deltas: Sequence[Any],
    *,
    f_byzantine: int = 1,
    m_select: int | None = None,
    token_weights: Sequence[int | float] | None = None,
    chunk_elements: int = 1_048_576,
) -> MultiKrumAggregationResult:
    """Select `m_select` Byzantine-resilient deltas via Multi-Krum and aggregate in bf16."""
    _require_torch_numpy()
    n = len(deltas)
    if n == 0:
        raise ValueError("At least one delta is required.")

    clamped_all_w = (
        _validate_and_clamp_token_weights(token_weights, n)
        if token_weights is not None
        else None
    )
    flat = [_flatten_delta(d) for d in deltas]
    if n == 1:
        tok = int(round(clamped_all_w[0])) if clamped_all_w is not None else 1
        return MultiKrumAggregationResult(
            selected_indices=[0],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=flat[0].to(torch.bfloat16),
            effective_tokens=max(1, tok),
            byzantine_filter_active=False,
        )

    eff_f = min(max(0, f_byzantine), max(0, (n - 3) // 2))
    neighbors_k = max(1, n - eff_f - 2)
    m = m_select if m_select is not None else max(1, n - eff_f)
    m = min(n, max(1, m))

    dist_sq = chunked_pairwise_sq_l2(flat, chunk_elements=chunk_elements)
    scores: list[float] = []
    for i in range(n):
        row = torch.cat([dist_sq[i, :i], dist_sq[i, i + 1 :]])
        nearest, _ = torch.topk(row, k=min(neighbors_k, row.numel()), largest=False)
        scores.append(float(nearest.sum().item()))

    ranked = sorted(range(n), key=lambda idx: (scores[idx], idx))
    selected = sorted(ranked[:m])
    rejected = sorted(ranked[m:])

    raw_w = [clamped_all_w[i] for i in selected] if clamped_all_w is not None else [1.0 for _ in selected]
    total_w = sum(raw_w)
    agg = torch.zeros_like(flat[0], dtype=torch.float32)
    for idx, w in zip(selected, raw_w):
        agg.add_(flat[idx], alpha=w / total_w)

    eff_tokens = int(round(sum(raw_w))) if clamped_all_w is not None else len(selected)
    return MultiKrumAggregationResult(
        selected_indices=selected,
        rejected_indices=rejected,
        krum_scores=scores,
        aggregated_delta_bf16=agg.to(torch.bfloat16),
        effective_tokens=eff_tokens,
        byzantine_filter_active=bool(eff_f > 0),
    )


@dataclass(frozen=True)
class WeightRmsGuardDecision:
    """Anti-Lobotomy Weight RMS Collapse & Step-282 Halving Guard (`Vxtzq/CrowdGPT#36`, `#42`)."""

    accepted: bool
    rms_old: float
    rms_new: float
    rms_ratio: float
    halving_residual_rms: float
    reason: str


def check_weight_rms_collapse_guard(
    old_weights: Any,
    new_weights: Any,
    *,
    min_rms_ratio: float = 0.80,
    max_rms_ratio: float = 1.25,
    halving_residual_tol: float = 1e-3,
) -> WeightRmsGuardDecision:
    """Reject aggregated checkpoints whose global RMS norm collapses, explodes, or halves (`0.5 * old`)."""
    _require_torch_numpy()
    flat_old = _flatten_delta(old_weights)
    flat_new = _flatten_delta(new_weights)
    if flat_old.numel() == 0 or flat_old.numel() != flat_new.numel():
        raise ValueError("old_weights and new_weights must be non-empty and have identical element counts.")
    if not (bool(torch.isfinite(flat_old).all()) and bool(torch.isfinite(flat_new).all())):
        return WeightRmsGuardDecision(False, float("nan"), float("nan"), float("nan"), float("nan"), "REJECTED_NON_FINITE_WEIGHTS")
    old_f64 = flat_old.to(torch.float64)
    new_f64 = flat_new.to(torch.float64)
    rms_old = float(torch.sqrt(torch.mean(old_f64 * old_f64)).item())
    rms_new = float(torch.sqrt(torch.mean(new_f64 * new_f64)).item())
    if rms_old <= 1e-12:
        raise ValueError("old_weights RMS norm must be strictly positive.")
    rms_ratio = rms_new / rms_old
    diff_half = new_f64 - 0.5 * old_f64
    halving_residual_rms = float(torch.sqrt(torch.mean(diff_half * diff_half)).item())
    if halving_residual_rms <= halving_residual_tol and rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(False, rms_old, rms_new, rms_ratio, halving_residual_rms, "REJECTED_ZERO_PARTICIPANT_WEIGHT_HALVING")
    if rms_ratio < min_rms_ratio:
        return WeightRmsGuardDecision(False, rms_old, rms_new, rms_ratio, halving_residual_rms, "REJECTED_RMS_COLLAPSE")
    if rms_ratio > max_rms_ratio:
        return WeightRmsGuardDecision(False, rms_old, rms_new, rms_ratio, halving_residual_rms, "REJECTED_RMS_EXPLOSION")
    return WeightRmsGuardDecision(True, rms_old, rms_new, rms_ratio, halving_residual_rms, "ACCEPTED_RMS_HEALTHY")


@dataclass(frozen=True)
class LossRatchetDecision:
    """Audit receipt for a held-out Loss Ratchet evaluation."""

    accepted: bool
    loss_before: float
    loss_after: float
    delta_loss: float
    checkpoint_sha256: str
    reason: str


class LossRatchetGate:
    """Monotonic held-out validation gate for outer DiLoCo / CrowdGPT merges."""

    def __init__(self, *, max_regression_tolerance: float = 0.0) -> None:
        self.max_regression_tolerance = float(max_regression_tolerance)

    def evaluate(
        self,
        *,
        loss_before: float,
        loss_after: float,
        candidate_delta_bf16: Any,
    ) -> LossRatchetDecision:
        _require_torch_numpy()
        raw_bytes = (
            candidate_delta_bf16.detach()
            .to("cpu", dtype=torch.bfloat16)
            .contiguous()
            .view(torch.int16)
            .numpy()
            .tobytes()
        )
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if not (math.isfinite(loss_before) and math.isfinite(loss_after)):
            return LossRatchetDecision(
                accepted=False,
                loss_before=float(loss_before),
                loss_after=float(loss_after),
                delta_loss=float("nan"),
                checkpoint_sha256=digest,
                reason="REJECTED_NON_FINITE_LOSS",
            )
        delta_loss = float(loss_after - loss_before)
        accepted = delta_loss <= self.max_regression_tolerance
        reason = (
            f"ACCEPTED (delta_loss={delta_loss:+.5f} <= tol={self.max_regression_tolerance:.5f})"
            if accepted
            else f"HELD_PREVIOUS_HEAD (delta_loss={delta_loss:+.5f} > tol={self.max_regression_tolerance:.5f})"
        )
        return LossRatchetDecision(
            accepted=accepted,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            delta_loss=delta_loss,
            checkpoint_sha256=digest,
            reason=reason,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI helper for generating CrowdGPT v3.1 `user:<uuid>:dev:<device>` headers & precision cohorts."""
    ap = argparse.ArgumentParser(description="CrowdGPT x HiveMind Bridge CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_hdr = sub.add_parser("headers", help="Emit X-Client-* hardware headers for a HiveMind GPU")
    p_hdr.add_argument("--handle", required=True, help="HiveMinder contributor handle")
    p_hdr.add_argument("--device", default="gpu0", help="GPU device suffix (e.g. rtx2070, t4)")
    p_hdr.add_argument("--gpu-name", default="NVIDIA GeForce RTX 2070")
    p_hdr.add_argument("--vram-gb", type=float, default=8.0)
    p_hdr.add_argument("--sm", default="7.5", help="CUDA compute capability major.minor")
    p_hdr.add_argument("--batch-size", type=int, default=1)

    args = ap.parse_args(argv)
    if args.cmd == "headers":
        maj, _, min_s = str(args.sm).partition(".")
        cc = (int(maj or 7), int(min_s or 5))
        cohort = classify_gpu_precision_cohort("nvidia", cc)
        headers = build_crowdgpt_client_hardware_headers(
            hiveminder_handle=args.handle,
            device_suffix=args.device,
            batch_size=args.batch_size,
            vram_gb=args.vram_gb,
            gpu_name=args.gpu_name,
            autocast_dtype=cohort.effective_dtype,
            compute_capability=cc,
        )
        print(json.dumps({"cohort": asdict(cohort), "headers": headers}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
