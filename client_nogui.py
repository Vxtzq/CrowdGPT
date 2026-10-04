#!/usr/bin/env python3
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import io
import gzip
import json
import time
import struct
import logging
import hashlib
import math
import gc
import threading
import argparse
import tempfile
import subprocess
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.checkpoint import checkpoint

from rich.console import Console
from rich.table import Table
from rich.live import Live

import http.client
from requests.exceptions import ChunkedEncodingError

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)
console = Console()

# ============ AUTO-UPDATE ============
CLIENT_VERSION = "0.5"   # bump alongside docs/version.txt to release an update
VERSION_URL = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/version.txt"
CLIENT_URL  = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client_nogui.py"
UPDATE_TIMEOUT_VERSION = 5     # seconds for version.txt fetch
UPDATE_TIMEOUT_CLIENT  = 60    # seconds for client_nogui.py fetch
UPDATE_CHECK_BETWEEN_ROUNDS = True


def _parse_version(v):
    """'0.5' -> (0,5), '1.2.3-beta' -> (1,2,3), garbage -> (0,)"""
    try:
        parts = str(v).strip().split("-")[0].split(".")
        return tuple(int(x) for x in parts[:4])
    except Exception:
        return (0,)


def _fetch_remote_version():
    try:
        r = requests.get(VERSION_URL, timeout=UPDATE_TIMEOUT_VERSION)
        r.raise_for_status()
        line = r.text.strip().splitlines()[0].strip()
        return line or None
    except Exception as e:
        log.info(f"Version check skipped: {e}")
        return None


def _download_new_client(expected_version):
    try:
        r = requests.get(CLIENT_URL, timeout=UPDATE_TIMEOUT_CLIENT)
        r.raise_for_status()
        code = r.text
    except Exception as e:
        log.warning(f"client_nogui.py download failed: {e}")
        return None

    # Sanity 1: it must be valid Python
    try:
        compile(code, "client_nogui.py", "exec")
    except SyntaxError as e:
        log.warning(f"Downloaded client_nogui.py failed syntax check: {e}")
        return None

    # Sanity 2: it must declare the version we expected from version.txt
    markers = [
        f'CLIENT_VERSION = "{expected_version}"',
        f"CLIENT_VERSION = '{expected_version}'",
    ]
    if not any(m in code for m in markers):
        log.warning(f"Downloaded client_nogui.py does not declare version {expected_version}")
        return None

    return code


def _apply_update(new_code):
    """Atomically replace the running client_nogui.py. Keeps a .bak for rollback."""
    try:
        current = Path(__file__).resolve()
    except Exception:
        return False

    backup = current.with_name(current.name + ".bak")
    tmp = current.with_name(current.name + ".new")

    try:
        try:
            shutil.copy2(str(current), str(backup))
        except Exception:
            pass

        tmp.write_text(new_code, encoding="utf-8")

        try:
            st = current.stat()
            os.chmod(str(tmp), st.st_mode)
        except Exception:
            pass

        os.replace(str(tmp), str(current))  # atomic on POSIX & Windows
        return True
    except Exception as e:
        log.warning(f"Failed to apply update: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def check_for_update():
    """
    Returns:
        True  -> update written to disk; caller should restart the process
        False -> no update needed, or check/download failed
    """
    remote = _fetch_remote_version()
    if not remote:
        return False

    if _parse_version(remote) <= _parse_version(CLIENT_VERSION):
        return False

    log.info(f"Update available: {CLIENT_VERSION} -> {remote}")
    new_code = _download_new_client(remote)
    if not new_code:
        return False

    if not _apply_update(new_code):
        return False

    log.info(f"client_nogui.py updated to {remote}")
    return True


def _restart_self():
    """Re-exec this script with the same interpreter and args. Never returns on POSIX."""
    python = sys.executable
    script = str(Path(__file__).resolve())
    args = [python, script] + sys.argv[1:]

    if os.name == "nt":
        # On Windows, detach a fresh process and exit the current one
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008)  # DETACHED_PROCESS
        sys.exit(0)
    else:
        os.execv(python, args)  # never returns


# ============ CONFIG ============
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SERVER = "https://server.crowdgpt.net"

DATASET_REPO_ID = "Vxtzq/CrowdGPT"
MODEL_REPO_ID = "Vxtzq/Crowd-v1"

MODEL_CONFIG = {
    "vocabSize": 151669,
    "dim": 1536,
    "nLayers": 24,
    "nHeads": 16,
    "nKvHeads": 4,
    "headDim": 96,
    "maxSeqLen": 2048,
    "mlpHidden": 2560,
    "weightTying": True,
    "architecture": "SotaGPT",
}

VOCAB_SIZE = MODEL_CONFIG["vocabSize"]
DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]
N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]
HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]
MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865

LOSS_CHUNK = 256
UPLOAD_BUFFER_MIN = 25
TPS_DEGRADATION = 0.85
DATASET_PAUSE_PER_ADVANCE = 8
CALIBRATION_STEPS = 15

MAX_CHUNK_RETRY = 4

# Upload defaults, can be overridden by CLI args.
DEFAULT_CHUNK_SIZE_MB = 64
DEFAULT_MAX_DIRECT_UPLOAD_MB = 80
SERVER_MAX_CHUNK_MB = 80


def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM) * 2 + DIM * DIM + DIM * 2
        size += DIM * MLP_HIDDEN * 2 + MLP_HIDDEN * DIM
    size += DIM * 2
    return size


EXPECTED_MODEL_SIZE = _calc_model_size()
EXPECTED_ENGRAM_SIZE = ENG_NUM_BUCKETS * DIM

ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

memory_config = {
    "ram_gb": 12,
    "safety_margin_gb": 1.0,
    "is_auto_detected": False,
}

train_device, train_backend = None, None


# ============ HARDWARE ============
def detect_training_backend(force=None):
    if force and force != "auto":
        name = force.lower()

        if name == "cpu":
            return torch.device('cpu'), "CPU"

        if name == "cuda" and torch.cuda.is_available():
            return torch.device('cuda'), "CUDA"

        if name == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return torch.device('cuda'), "ROCM"

        if name == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), "MPS"

        if name == "xpu":
            try:
                import intel_extension_for_pytorch  # noqa: F401
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    return torch.device('xpu'), "XPU"
            except ImportError:
                pass

        if name == "directml":
            try:
                import torch_directml
                if torch_directml.is_available():
                    return torch_directml.device(0), "DIRECTML"
            except ImportError:
                pass

        raise Exception(f"Backend '{force}' not available")

    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"

    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return torch.device('cuda'), "ROCM"

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"

    try:
        import intel_extension_for_pytorch  # noqa: F401
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu'), "XPU"
    except ImportError:
        pass

    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(0), "DIRECTML"
    except ImportError:
        pass

    return torch.device('cpu'), "CPU"


def auto_detect_vram_budget():
    global train_backend

    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            usable = max(1.0, min(free_b / 1024**3, total_b / 1024**3 - 2.0) - 1.0)
            memory_config.update({
                "ram_gb": round(usable, 2),
                "is_auto_detected": True,
            })
            return
        except Exception:
            pass

    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024**3
            memory_config.update({
                "ram_gb": round(max(1.0, total_gb * 0.75 - 2.0), 2),
                "is_auto_detected": True,
            })
            return
        except Exception:
            pass

    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        memory_config.update({
            "ram_gb": round(max(1.0, avail - 2.0), 2),
            "is_auto_detected": True,
        })
    except ImportError:
        pass


def has_bitsandbytes():
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def estimate_vram_bytes(batch_size, seq_len, use_8bit):
    model_bytes = EXPECTED_MODEL_SIZE * 2
    optim_bytes = EXPECTED_MODEL_SIZE * (4 if use_8bit else 12)
    grad_bytes = EXPECTED_MODEL_SIZE * 4

    act_per_block = batch_size * seq_len * DIM * 2
    attn_act = batch_size * N_HEADS * 64 * seq_len * 4
    act_bytes = N_LAYERS * (act_per_block + attn_act)

    logits_bytes = batch_size * LOSS_CHUNK * VOCAB_SIZE * 4
    engram_bytes = batch_size * seq_len * DIM * 2
    safety = int(0.8 * 1024**3)

    return int(model_bytes + optim_bytes + grad_bytes + act_bytes + logits_bytes + engram_bytes + safety)


def recommend_batch_size(seq_len=2048):
    use_8bit = has_bitsandbytes()
    budget = int(memory_config["ram_gb"] * 1024**3)
    best_bs = 1

    for bs in ALLOWED_BATCH_SIZES:
        est = estimate_vram_bytes(bs, seq_len, use_8bit)
        if est <= budget:
            best_bs = bs
        else:
            break

    log.info(
        f"VRAM budget: {memory_config['ram_gb']:.1f}GB | "
        f"8-bit optim: {use_8bit} | Recommended BS: {best_bs}"
    )
    return best_bs


# ============ MODEL ============
def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.nrep = N_HEADS, N_KV_HEADS, N_HEADS // N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, use_chunked=True):
        B, T, C = x.size()

        q = self.wq(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)

        ct, st = cos[:, :, :T, :], sin[:, :, :T, :]
        q = q * ct + rotate_half(q) * st
        k = k * ct + rotate_half(k) * st

        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)

        if use_chunked:
            return self._chunked(q, k, v, B, T, C)

        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        m = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        a = a.masked_fill(m == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)

            q_chunk = q[:, :, i:e, :]
            k_chunk = k[:, :, :e, :]
            v_chunk = v[:, :, :e, :]

            aw = (q_chunk @ k_chunk.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))

            row_idx = torch.arange(i, e, device=q.device).unsqueeze(1)
            col_idx = torch.arange(e, device=q.device).unsqueeze(0)
            mask = (col_idx <= row_idx).unsqueeze(0).unsqueeze(0)

            aw = aw.masked_fill(~mask, float('-inf'))
            attn = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
            chunks.append(attn @ v_chunk)

        out = torch.cat(chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ln_2 = nn.LayerNorm(DIM)
        self.mlp = SwiGLU()

    def forward(self, x, cos, sin, use_chunked=True):
        x = x + self.attn(self.ln_1(x), cos, sin, use_chunked)
        return x + self.mlp(self.ln_2(x))


class EngramMemory(nn.Module):
    def __init__(self, backend_name, device):
        super().__init__()
        self.device = device
        self.table = nn.Embedding(ENG_NUM_BUCKETS, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

        self.use_async = device.type in ('cuda',) and torch.cuda.is_available()
        self.transfer_stream = torch.cuda.Stream(device=device) if self.use_async else None

    def forward(self, idx):
        prev_x = torch.cat([torch.zeros_like(idx[:, :1]), idx[:, :-1]], dim=1)
        hash_idx = (prev_x * 1000003 + idx) % ENG_NUM_BUCKETS

        unique_indices, inverse_map = torch.unique(hash_idx.flatten(), return_inverse=True)
        unique_cpu = unique_indices.cpu()
        cached_rows = self.table(unique_cpu)

        if self.use_async:
            with torch.cuda.stream(self.transfer_stream):
                cached_rows_gpu = cached_rows.to(self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        else:
            cached_rows_gpu = cached_rows.to(self.device)

        B, T = idx.shape
        return cached_rows_gpu[inverse_map].view(B, T, DIM)


class SotaGPT(nn.Module):
    def __init__(self, backend_name, device):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory(backend_name, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)

        if MODEL_CONFIG["weightTying"]:
            self.wte.weight = self.lm_head.weight

        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos", cm)
        self.register_buffer("freqs_sin", sm)

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = p.numel()
                p.data.copy_(ft[o:o + s].view(p.shape).to(p.device))
                o += s


# ============ METADATA PARSING ============
def _parse_metadata_response(r):
    """
    Parses either:
    - JSON metadata from /fl/task?skip_weights=true
    - legacy binary metadata: 4-byte len + JSON
    Returns dict or None if server says wait.
    """
    try:
        if r.headers.get("X-Status") == "wait":
            return None

        ct = (r.headers.get("content-type") or "").lower()

        if "json" in ct:
            data = r.json()
            if data.get("status") == "wait":
                return None
            return data

        raw = r.content
        if len(raw) < 4:
            return None

        ml = struct.unpack('<I', raw[:4])[0]
        if len(raw) < 4 + ml:
            return None

        data = json.loads(raw[4:4 + ml].decode('utf-8'))
        if data.get("status") == "wait":
            return None

        return data

    except Exception:
        return None


# ============ DATASET ============
def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            ChunkedEncodingError,
            http.client.IncompleteRead,
        ) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"Network drop (attempt {attempt + 1}/{retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500

    def __init__(self, repo_id, chunk_idx, sub_size=10 * 1024 * 1024, tps=65, slot=0, auth_token=None):
        self.repo_id = repo_id
        self.ci = chunk_idx
        self.sub_size = sub_size
        self.tps = tps
        self.auth_token = auth_token

        self._name_fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover_chunk_size(chunk_idx)
        self.off = (slot * sub_size) % self.chunk_size

        self.data = None
        self.n = 0
        self.steps_used = 0

        self._pf_thread = None
        self._pf_result = None
        self._lock = threading.Lock()

        self._load_subchunk(self.off)
        self._start_prefetch(self.off + self.sub_size)

    def _chunk_url(self):
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/"
            + self._name_fmt.format(self.ci)
        )

    def _discover_chunk_size(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)

            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        size = int(r.headers.get('content-length', 0))
                        if size > 0:
                            self._name_fmt = fmt
                            return size
                except Exception:
                    time.sleep(1)

        raise RuntimeError(f"Could not find chunk {idx}")

    def _fetch_slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(
            self._chunk_url(),
            headers={'Range': f'bytes={off}-{end}'},
            timeout=120,
        ).content

    def _set_data(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data = tk
        self.n = len(tk) // self.tps
        self.steps_used = 0

    def _load_subchunk(self, off):
        self._set_data(self._fetch_slice(off))
        self.off = off

    def _start_prefetch(self, off):
        if off >= self.chunk_size:
            return

        with self._lock:
            self._pf_result = None

        def worker():
            try:
                raw = self._fetch_slice(off)
                with self._lock:
                    self._pf_result = (off, raw)
            except Exception:
                with self._lock:
                    self._pf_result = None

        self._pf_thread = threading.Thread(target=worker, daemon=True)
        self._pf_thread.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, server_url=None, fmt="bf16"):
        next_off = self.off + self.sub_size

        if next_off < self.chunk_size:
            if self._pf_thread:
                self._pf_thread.join(timeout=120)
                self._pf_thread = None

            with self._lock:
                res = self._pf_result
                self._pf_result = None

            if res and res[0] == next_off:
                self.off = next_off
                self._set_data(res[1])
            else:
                self._load_subchunk(next_off)

            self._start_prefetch(next_off + self.sub_size)
            return True

        self.request_new_chunk(server_url, fmt)
        self.chunk_size = self._discover_chunk_size(self.ci)
        self.off = 0
        self._load_subchunk(0)
        self._start_prefetch(self.sub_size)
        return True

    def request_new_chunk(self, server_url, fmt="bf16"):
        if not server_url:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(
                f"{server_url}/fl/task?format={fmt}&skip_weights=true",
                headers=headers,
                timeout=30,
            )

            md = _parse_metadata_response(r)
            if not md:
                return False

            new_idx = md.get("datasetConfig", {}).get("chunkIdx", self.ci)
            if new_idx != self.ci:
                self.ci = new_idx
                return True

        except Exception:
            pass

        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0:
            self.advance()

        self.steps_used += 1

        rng = np.random.RandomState(seed)
        starts = rng.randint(0, self.n, size=bs) * self.tps

        inp = np.stack([self.data[s:s + self.tps - 1] for s in starts])
        tgt = np.stack([self.data[s + 1:s + self.tps] for s in starts])

        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ============ WEIGHT LOADING ============
def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)

    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()


def load_weights_from_temp(path, fmt="bf16"):
    bpp = 2 if fmt in ("bf16", "fp16") else 4

    model_bytes = EXPECTED_MODEL_SIZE * bpp
    engram_bytes = EXPECTED_ENGRAM_SIZE * bpp
    required = model_bytes + engram_bytes

    size = os.path.getsize(path)
    if size < required:
        raise RuntimeError(
            f"Incomplete weight payload file: got {size} bytes, expected at least {required} bytes"
        )

    if fmt == "fp16":
        iw = np.fromfile(
            path,
            dtype=np.uint16,
            count=EXPECTED_MODEL_SIZE,
            offset=0,
        ).view(np.float16).astype(np.float32)

        ie = np.fromfile(
            path,
            dtype=np.uint16,
            count=EXPECTED_ENGRAM_SIZE,
            offset=model_bytes,
        ).view(np.float16).astype(np.float32)

        return iw, ie

    if fmt == "fp32":
        iw = np.fromfile(path, dtype=np.float32, count=EXPECTED_MODEL_SIZE, offset=0)
        ie = np.fromfile(path, dtype=np.float32, count=EXPECTED_ENGRAM_SIZE, offset=model_bytes)
        return iw, ie

    # Default bf16.
    iw_u16 = np.fromfile(path, dtype=np.uint16, count=EXPECTED_MODEL_SIZE, offset=0)
    iw = torch.from_numpy(iw_u16.copy()).view(torch.bfloat16).to(torch.float32).numpy()
    del iw_u16
    gc.collect()

    ie_u16 = np.fromfile(path, dtype=np.uint16, count=EXPECTED_ENGRAM_SIZE, offset=model_bytes)
    ie = torch.from_numpy(ie_u16.copy()).view(torch.bfloat16).to(torch.float32).numpy()
    del ie_u16
    gc.collect()

    return iw, ie


def _hf_cache_path(revision, filename):
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(revision or "main"))
    d = CHECKPOINT_DIR / "hf_weights" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _download_with_progress(url, dest, emit_hb=None, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is None or sz == int(expected_size):
            log.info(f"Using cached weight file: {dest}")
            return dest

        try:
            dest.unlink()
        except Exception:
            pass

    hf_token = os.getenv("HF_TOKEN")
    req_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    last_err = None

    for attempt in range(3):
        if emit_hb and emit_hb.should_stop():
            raise KeyboardInterrupt("Stop requested during weight download")

        tmp = dest.with_suffix(dest.suffix + ".part")

        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

        try:
            log.info(f"Downloading {label}: {url}")

            r = requests.get(
                url,
                headers=req_headers,
                stream=True,
                timeout=(15, 3600),
                allow_redirects=True,
            )
            r.raise_for_status()

            total = int(expected_size) if expected_size else int(r.headers.get("content-length", 0) or 0)
            done = 0
            next_pct = 10

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if emit_hb and emit_hb.should_stop():
                        raise KeyboardInterrupt("Stop requested during weight download")

                    if not chunk:
                        continue

                    f.write(chunk)
                    done += len(chunk)

                    if total > 0:
                        pct = done * 100 // total
                        if pct >= next_pct:
                            log.info(
                                f"{label}: {pct}% "
                                f"({done / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)"
                            )
                            next_pct = min(100, next_pct + 10)

            if expected_size and done != int(expected_size):
                raise RuntimeError(
                    f"Downloaded size mismatch for {label}: got {done}, expected {expected_size}"
                )

            os.replace(tmp, dest)
            log.info(f"Finished downloading {label}: {dest}")
            return dest

        except KeyboardInterrupt:
            raise

        except Exception as e:
            last_err = e

            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

            if attempt < 2:
                log.warning(f"Download retry {attempt + 1} for {label}: {e}")
                time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"Failed to download {label}: {last_err}")


def _load_safetensors_flat(path, expected_numel, label="weights"):
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: safetensors. Install with: pip install safetensors"
        ) from e

    tensors = load_file(str(path))

    if "weights" in tensors:
        t = tensors["weights"]
    elif "flat" in tensors:
        t = tensors["flat"]
    elif len(tensors) == 1:
        t = next(iter(tensors.values()))
    else:
        raise RuntimeError(
            f"{label}: could not identify expected tensor key. "
            f"Expected 'weights' or 'flat'. Found keys: {list(tensors.keys())[:20]}"
        )

    arr = t.to(torch.float32).flatten().contiguous().cpu().numpy()

    del tensors, t
    gc.collect()

    if arr.size != int(expected_numel):
        raise RuntimeError(
            f"{label} element count mismatch: got {arr.size}, expected {expected_numel}."
        )

    return arr


def fetch_weights_from_hf(md, hb=None):
    wcfg = md.get("weights") or {}

    if wcfg.get("source") != "huggingface":
        raise RuntimeError("Task metadata does not advertise Hugging Face weights.")

    repo = wcfg.get("repo") or MODEL_REPO_ID
    revision = wcfg.get("revision") or "main"
    files = wcfg.get("files") or {}

    model_info = files.get("model") or {}
    engram_info = files.get("engram") or {}

    model_url = model_info.get("url") or (
        f"https://huggingface.co/{repo}/resolve/{revision}/model_bf16.safetensors"
    )
    engram_url = engram_info.get("url") or (
        f"https://huggingface.co/{repo}/resolve/{revision}/engram_bf16.safetensors"
    )

    model_size = model_info.get("size")
    engram_size = engram_info.get("size")

    if not model_size:
        try:
            hr = requests.head(model_url, allow_redirects=True, timeout=30)
            model_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            model_size = None

    if not engram_size:
        try:
            hr = requests.head(engram_url, allow_redirects=True, timeout=30)
            engram_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            engram_size = None

    model_path = _hf_cache_path(revision, "model_bf16.safetensors")
    engram_path = _hf_cache_path(revision, "engram_bf16.safetensors")

    _download_with_progress(
        model_url,
        model_path,
        emit_hb=hb,
        expected_size=model_size,
        label="model_bf16.safetensors",
    )

    _download_with_progress(
        engram_url,
        engram_path,
        emit_hb=hb,
        expected_size=engram_size,
        label="engram_bf16.safetensors",
    )

    log.info("Loading weights from safetensors...")

    iw = _load_safetensors_flat(model_path, EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(engram_path, EXPECTED_ENGRAM_SIZE, "engram weights")

    return iw, ie


def fetch_task_metadata(server_url, headers, hb=None):
    while True:
        if hb and hb.should_stop():
            return None

        try:
            r = requests.get(
                f"{server_url}/fl/task?format=bf16&skip_weights=true",
                headers=headers,
                timeout=30,
            )

            md = _parse_metadata_response(r)
            if md is None:
                time.sleep(10)
                continue

            return md

        except KeyboardInterrupt:
            raise

        except Exception as e:
            if hb and hb.should_stop():
                return None

            log.error(f"Task metadata fetch failed: {e}")
            time.sleep(10)


def _read_exact_raw(resp, n):
    buf = bytearray()

    while len(buf) < n:
        chunk = resp.raw.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)

    return bytes(buf)


def fetch_task_and_weights_binary(server_url, headers, precision, hb=None):
    """
    Fallback: legacy coordinator binary stream.
    Returns metadata, initial_weights, initial_engram_weights.
    """
    while True:
        if hb and hb.should_stop():
            return None, None, None

        r = None
        temp_path = None

        try:
            r = requests.get(
                f"{server_url}/fl/task?format={precision}",
                headers=headers,
                timeout=(15, 3600),
                stream=True,
            )

            if r.headers.get("X-Status") == "wait":
                r.close()
                log.info("Server says wait during binary task fetch.")
                time.sleep(10)
                continue

            if r.status_code != 200:
                r.close()
                time.sleep(10)
                continue

            meta_len_bytes = _read_exact_raw(r, 4)
            if len(meta_len_bytes) < 4:
                raise RuntimeError("Incomplete metadata length")

            meta_len = struct.unpack('<I', meta_len_bytes)[0]

            meta_bytes = _read_exact_raw(r, meta_len)
            if len(meta_bytes) < meta_len:
                raise RuntimeError("Incomplete metadata payload")

            metadata = json.loads(meta_bytes.decode('utf-8'))

            if metadata.get("status") == "wait":
                r.close()
                time.sleep(10)
                continue

            fd, temp_path = tempfile.mkstemp(
                prefix="crowdgpt_weights_",
                suffix=".bin",
                dir=str(CHECKPOINT_DIR),
            )
            os.close(fd)

            with open(temp_path, 'wb') as f:
                while True:
                    if hb and hb.should_stop():
                        raise KeyboardInterrupt("Stop requested during weight download")

                    chunk = r.raw.read(1024 * 1024)
                    if not chunk:
                        break

                    f.write(chunk)

            r.close()
            r = None

            weight_format = metadata.get("weightFormat", precision)
            iw, ie = load_weights_from_temp(temp_path, weight_format)

            return metadata, iw, ie

        except KeyboardInterrupt:
            raise

        except Exception as e:
            log.error(f"Binary task fetch failed: {e}")
            time.sleep(15)

        finally:
            try:
                if r is not None:
                    r.close()
            except Exception:
                pass

            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass


def fetch_initial_weights(server_url, headers, args, hb=None):
    """
    Preferred flow:
    1. Get metadata only.
    2. If HF weights are advertised, download from HF.
    3. Otherwise fallback to coordinator binary stream.
    """
    md = fetch_task_metadata(server_url, headers, hb=hb)
    if md is None:
        return None, None, None

    iw = None
    ie = None

    if not args.no_hf_weights:
        wcfg = md.get("weights") or {}
        if wcfg.get("source") == "huggingface":
            try:
                log.info("Downloading weights from Hugging Face...")
                iw, ie = fetch_weights_from_hf(md, hb=hb)
                log.info("HF weight download complete.")
            except KeyboardInterrupt:
                return None, None, None
            except Exception as e:
                log.warning(f"Hugging Face weight download failed: {e}")
                log.info("Falling back to coordinator weight stream...")
                iw = None
                ie = None

    if iw is None or ie is None:
        try:
            log.info("Downloading weights from coordinator...")
            md2, iw, ie = fetch_task_and_weights_binary(server_url, headers, args.precision, hb=hb)
        except KeyboardInterrupt:
            return None, None, None

        if md2 is None:
            return None, None, None

        md = md2

    return md, iw, ie


# ============ AUTH ============
def authenticate(server_url, username, password, email=None):
    if not username or not password:
        return None

    try:
        r = requests.post(
            f"{server_url}/auth/login",
            json={"username": username, "password": password},
            timeout=30,
        )

        if r.status_code == 200:
            return r.json().get("token")

        email = email or f"{username}@crowdgpt.local"

        r = requests.post(
            f"{server_url}/auth/register",
            json={
                "username": username,
                "password": password,
                "email": email,
            },
            timeout=30,
        )

        if r.status_code == 200:
            return r.json().get("token")

        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text

        log.warning(f"Auth/register failed: {detail[:200]}")

    except Exception as e:
        log.warning(f"Authentication error: {e}")

    return None


# ============ DASHBOARD ============
def create_dashboard(
    step,
    target_steps,
    loss,
    tps,
    lr,
    global_step,
    backend_name,
    batch_size,
    seq_len,
    current_round,
    time_remaining_min,
    accum_steps=1,
    tokens_per_opt_step=None,
):
    table = Table(title=f"Round {current_round} Training", expand=True, border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Progress", f"{step}/{target_steps} ({step / max(1, target_steps) * 100:.1f}%)")
    table.add_row("Loss", f"{loss:.4f}" if loss > 0 else "—")
    table.add_row("Tokens/s", f"{tps:.0f}" if tps else "—")
    table.add_row("LR", f"{lr:.2e}" if lr else "—")
    table.add_row("Global Step", str(global_step))
    table.add_row("Round", str(current_round))
    table.add_row("Time Left", f"{time_remaining_min:.0f} min")
    table.add_row("Backend", backend_name)
    table.add_row("Batch / SeqLen", f"{batch_size} / {seq_len}")

    if accum_steps > 1:
        table.add_row(
            "Grad Accum",
            f"{accum_steps}x (~{tokens_per_opt_step:,} tok/opt-step)"
            if tokens_per_opt_step else
            f"{accum_steps}x"
        )

    return table


# ============ HEARTBEAT ============
class HeartbeatManager:
    def __init__(self, server_url, headers, current_round):
        self.server_url = server_url
        self.headers = headers
        self.current_round = current_round

        self.stop_training = threading.Event()
        self.shutdown = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.shutdown.set()

    def should_stop(self):
        return self.stop_training.is_set()

    def _run(self):
        while not self.shutdown.is_set():
            try:
                requests.get(f"{self.server_url}/fl/heartbeat", headers=self.headers, timeout=10)

                r = requests.get(f"{self.server_url}/fl/round_status", headers=self.headers, timeout=10)
                if r.status_code == 200:
                    status = r.json()

                    if status.get("current_round", self.current_round) != self.current_round:
                        log.warning("Server moved to next round. Stopping!")
                        self.stop_training.set()

                    remaining_min = (
                        status.get("max_round_hours", 2) -
                        status.get("round_elapsed_hours", 0)
                    ) * 60

                    if remaining_min <= 3:
                        log.warning(f"ULTIMATUM: {remaining_min:.0f} min left! Submitting NOW.")
                        self.stop_training.set()

            except Exception:
                pass

            self.shutdown.wait(timeout=15)


# ============ ROUND WAIT ============
def wait_for_round(server_url, headers):
    while True:
        try:
            r = requests.get(f"{server_url}/fl/round_status", headers=headers, timeout=30)

            if r.status_code == 200:
                status = r.json()

                if status.get("is_aggregating"):
                    log.info("Server is aggregating and uploading to HF. Waiting 30s...")
                    time.sleep(30)
                    continue

                if not status.get("in_cooldown", False):
                    return status

                log.info("Server in cooldown. Waiting 30s...")
                time.sleep(30)
            else:
                time.sleep(10)

        except Exception:
            time.sleep(10)


# ============ FORWARD / LOSS ============
def _chunk_ce(model, x_slice, y_slice):
    logits = model.lm_head(x_slice)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y_slice.reshape(-1))


def _forward_and_loss(model, x, y, seq_len, use_autocast, autocast_dtype, loss_scale=1.0):
    def fwd():
        x_emb = model.wte(x)

        if use_autocast:
            with torch.autocast(device_type='cuda', enabled=False):
                eng_out = model.engram(x)
        else:
            eng_out = model.engram(x)

        x_emb = x_emb + eng_out.to(x_emb.dtype)

        for b in model.blocks:
            x_emb = checkpoint(
                b,
                x_emb,
                model.freqs_cos,
                model.freqs_sin,
                True,
                use_reentrant=False,
            )

        return model.ln_f(x_emb)

    n_chunks = max(1, math.ceil(seq_len / LOSS_CHUNK))

    if use_autocast:
        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
            x_emb = fwd()
            loss = sum(
                checkpoint(
                    _chunk_ce,
                    model,
                    x_emb[:, i:i + LOSS_CHUNK, :],
                    y[:, i:i + LOSS_CHUNK],
                    use_reentrant=False,
                )
                for i in range(0, seq_len, LOSS_CHUNK)
            ) / n_chunks
    else:
        x_emb = fwd()
        loss = sum(
            checkpoint(
                _chunk_ce,
                model,
                x_emb[:, i:i + LOSS_CHUNK, :],
                y[:, i:i + LOSS_CHUNK],
                use_reentrant=False,
            )
            for i in range(0, seq_len, LOSS_CHUNK)
        ) / n_chunks

    return loss * loss_scale


# ============ DELTA COMPUTATION ============
def compute_base_delta_bf16(model, initial_weights):
    """
    Memory-friendlier replacement for:
        np.concatenate([...]) - initial_weights
    Returns uint16 bf16 flat array.
    """
    out = np.empty(EXPECTED_MODEL_SIZE, dtype=np.uint16)
    offset = 0

    for n, p in model.named_parameters():
        if n.startswith('engram.'):
            continue

        numel = p.numel()

        flat = p.detach().float().flatten().cpu().numpy()
        delta = flat - initial_weights[offset:offset + numel]

        out[offset:offset + numel] = (
            torch.from_numpy(delta)
            .to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )

        offset += numel

        del flat, delta
        gc.collect()

    if offset != EXPECTED_MODEL_SIZE:
        raise RuntimeError(
            f"Base delta size mismatch: produced {offset}, expected {EXPECTED_MODEL_SIZE}"
        )

    return out


def compute_engram_sparse_delta(model, initial_engram_weights):
    """
    Computes sparse engram delta without materializing full engram delta.
    Returns:
      sparse_indices: np.uint32 array
      sparse_values: np.uint16 bf16 array shape (count, DIM), or empty (0, DIM)
    """
    weight = model.engram.table.weight.data.cpu()
    init_table = torch.from_numpy(initial_engram_weights).view(ENG_NUM_BUCKETS, DIM)

    row_norms = torch.empty(ENG_NUM_BUCKETS, dtype=torch.float32)

    chunk_rows = 8192
    for s in range(0, ENG_NUM_BUCKETS, chunk_rows):
        e = min(s + chunk_rows, ENG_NUM_BUCKETS)

        d = weight[s:e] - init_table[s:e]
        row_norms[s:e] = d.abs().sum(dim=1)

        del d
        gc.collect()

    k = max(1, int(ENG_NUM_BUCKETS * 0.10))
    topk_values, topk_indices = torch.topk(row_norms, k)

    active = topk_values > 1e-8
    active_indices = topk_indices[active]

    if active_indices.numel() == 0:
        del weight, init_table, row_norms, topk_values, topk_indices, active
        gc.collect()
        return (
            np.array([], dtype=np.uint32),
            np.empty((0, DIM), dtype=np.uint16),
        )

    delta_sel = weight[active_indices] - init_table[active_indices]

    sparse_values = (
        delta_sel
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
    )

    sparse_indices = active_indices.cpu().numpy().astype(np.uint32)

    del weight, init_table, row_norms, topk_values, topk_indices, active, active_indices, delta_sel
    gc.collect()

    return sparse_indices, sparse_values


# ============ UPLOAD HELPERS ============
def _hash_file(path, chunk_size=4 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def _create_compressed_payload_file(meta, delta_base_bf16, sparse_indices, sparse_values):
    """
    Writes gzip payload to disk instead of holding full compressed bytes in RAM.
    Payload format expected by server:
      4-byte JSON length
      JSON metadata
      base delta bf16 bytes
      engram indices uint32 bytes, only if count > 0
      engram values bf16 bytes, only if count > 0
    """
    fd, path = tempfile.mkstemp(
        prefix="crowdgpt_delta_",
        suffix=".gz",
        dir=str(CHECKPOINT_DIR),
    )
    os.close(fd)

    payload_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=2) as gz:
            gz.write(struct.pack("<I", len(payload_json)))
            gz.write(payload_json)

            chunk_elems = 5_000_000
            arr = delta_base_bf16

            for start in range(0, len(arr), chunk_elems):
                gz.write(arr[start:start + chunk_elems].tobytes())

            count = int(len(sparse_indices))
            if count > 0:
                gz.write(np.ascontiguousarray(sparse_indices).tobytes())

                vals = np.ascontiguousarray(sparse_values)
                row_chunk = 10_000

                for start in range(0, len(vals), row_chunk):
                    gz.write(vals[start:start + row_chunk].tobytes())

    size = os.path.getsize(path)
    checksum = _hash_file(path)

    return path, size, checksum


def _post_json_with_retry(url, payload, headers, timeout=60, retries=4):
    last_err = None

    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)

            if r.status_code == 503:
                last_err = RuntimeError("server busy 503")
                log.warning("Server busy/aggregating, waiting before retry...")
                time.sleep(10)
                continue

            r.raise_for_status()
            return r

        except KeyboardInterrupt:
            raise

        except Exception as e:
            last_err = e

            if attempt < retries - 1:
                log.warning(f"POST retry {attempt + 1}: {e}")
                time.sleep(min(20, 2 ** (attempt + 1)))

    raise RuntimeError(f"POST {url} failed: {last_err}")


def _chunked_submit_file(server_url, headers, path, size, checksum, meta, chunk_size):
    chunks = max(1, math.ceil(size / chunk_size))

    init_payload = {
        **meta,
        "totalBytes": int(size),
        "chunkSize": int(chunk_size),
        "chunks": int(chunks),
        "checksum": checksum,
    }

    log.info(
        f"Initializing chunked upload: {chunks} chunks, "
        f"{size / (1024 * 1024):.1f} MB total"
    )

    r = _post_json_with_retry(
        f"{server_url}/fl/submit/init",
        init_payload,
        headers,
        timeout=60,
        retries=4,
    )

    j = r.json()
    upload_id = j.get("uploadId")

    if not upload_id:
        raise RuntimeError("Server did not return uploadId")

    for idx in range(chunks):
        start = idx * chunk_size
        read_size = min(chunk_size, size - start)

        attempt = 0

        while True:
            try:
                with open(path, 'rb') as f:
                    f.seek(start)
                    part = f.read(read_size)

                if not part:
                    raise RuntimeError(f"Empty chunk {idx}")

                part_sha = hashlib.sha256(part).hexdigest()

                rr = requests.post(
                    f"{server_url}/fl/submit/chunk?uploadId={upload_id}&index={idx}",
                    data=part,
                    headers={
                        **headers,
                        "Content-Type": "application/octet-stream",
                        "X-Chunk-Sha256": part_sha,
                    },
                    timeout=300,
                )

                if rr.status_code == 413:
                    raise RuntimeError("CHUNK_TOO_LARGE")

                rr.raise_for_status()
                break

            except KeyboardInterrupt:
                raise

            except RuntimeError:
                raise

            except Exception as e:
                attempt += 1

                if attempt >= MAX_CHUNK_RETRY:
                    raise

                log.warning(f"Chunk {idx + 1}/{chunks} failed, retry {attempt}: {e}")
                time.sleep(min(20, 2 ** attempt))

        done = min(size, start + read_size)
        log.info(
            f"Uploaded chunk {idx + 1}/{chunks} "
            f"({done / (1024 * 1024):.1f}/{size / (1024 * 1024):.1f} MB)"
        )

    log.info("Finalizing chunked upload...")

    return _post_json_with_retry(
        f"{server_url}/fl/submit/finalize",
        {"uploadId": upload_id},
        headers,
        timeout=900,
        retries=2,
    )


def chunked_submit_file(server_url, headers, path, size, checksum, meta, chunk_size):
    cs = int(chunk_size)

    while True:
        try:
            return _chunked_submit_file(
                server_url,
                headers,
                path,
                size,
                checksum,
                meta,
                cs,
            )

        except Exception as e:
            msg = str(e)

            if ("CHUNK_TOO_LARGE" in msg or "413" in msg) and cs > 8 * 1024 * 1024:
                cs = max(8 * 1024 * 1024, cs // 2)
                log.warning(
                    f"Chunk too large. Reducing chunk size to "
                    f"{cs / (1024 * 1024):.0f} MB and retrying."
                )
                continue

            raise


def direct_submit_file(server_url, headers, path, size):
    with open(path, 'rb') as f:
        data = f.read()

    return requests.post(
        f"{server_url}/fl/submit",
        headers={
            **headers,
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "gzip",
        },
        data=data,
        timeout=900,
    )


def submit_delta_file(
    server_url,
    headers,
    path,
    size,
    checksum,
    meta,
    chunk_size,
    max_direct_bytes,
):
    if size <= max_direct_bytes:
        try:
            r = direct_submit_file(server_url, headers, path, size)

            if r.status_code == 413:
                raise RuntimeError("CHUNK_TOO_LARGE")

            if r.status_code != 200:
                log.error(f"Direct submit failed: {r.text[:300]}")
                return r

            return r

        except RuntimeError as e:
            if "CHUNK_TOO_LARGE" not in str(e):
                raise

            log.warning("Direct upload rejected by edge/proxy (413). Falling back to chunked upload...")

        except Exception as e:
            if size > 8 * 1024 * 1024:
                log.warning(f"Direct upload failed ({e}). Falling back to chunked upload...")
            else:
                raise

    return chunked_submit_file(
        server_url,
        headers,
        path,
        size,
        checksum,
        meta,
        chunk_size,
    )


# ============ MAIN TRAINING ROUND ============
def run_single_round(args, auth_token=None):
    global train_device, train_backend

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    model = None
    optimizer_base = None
    optimizer_engram = None
    base_params = None
    engram_params = None

    initial_weights = None
    initial_engram_weights = None
    dataset_shard = None

    task_id = None
    global_step = 0
    current_round = 0

    step = CALIBRATION_STEPS
    total_tok = 0
    loss_history = []

    log.info("Checking round status...")
    round_status = wait_for_round(args.server, headers)

    current_round = round_status["current_round"]

    hb = HeartbeatManager(args.server, headers, current_round)
    hb.start()

    try:
        log.info("Fetching task metadata / weights...")
        metadata, initial_weights, initial_engram_weights = fetch_initial_weights(
            args.server,
            headers,
            args,
            hb=hb,
        )

        if metadata is None:
            log.warning("Round ended or task fetch was interrupted. Restarting cycle.")
            return

        task_id = metadata.get("taskId")
        global_step = metadata.get("globalStep", 0)

        seq_len = (
            args.seq_len
            if args.seq_len > 0
            else metadata.get("modelConfig", {}).get("maxSeqLen", MAX_SEQ_LEN)
        )

        if hb.should_stop():
            log.warning("Round ended during weight fetch. Restarting cycle.")
            return

        # Re-sync remaining round time after potentially long download.
        try:
            rs = requests.get(f"{args.server}/fl/round_status", headers=headers, timeout=15)
            if rs.status_code == 200:
                fresh = rs.json()
                remaining_hours = max(
                    0.05,
                    fresh.get("max_round_hours", 2.0) - fresh.get("round_elapsed_hours", 0)
                )
                log.info(f"Round timer re-synced: {remaining_hours * 60:.0f} min remaining")
            else:
                remaining_hours = max(
                    0.1,
                    round_status.get("max_round_hours", 2.0) - round_status.get("round_elapsed_hours", 0)
                )
        except Exception:
            remaining_hours = max(
                0.1,
                round_status.get("max_round_hours", 2.0) - round_status.get("round_elapsed_hours", 0)
            )

        ds_cfg = metadata.get("datasetConfig", {})
        shard_cfg = metadata.get("shardConfig", {})

        dataset_shard = StreamingShardDataset(
            ds_cfg.get("repoId", DATASET_REPO_ID),
            ds_cfg.get("chunkIdx", 0),
            ds_cfg.get("subChunkSize", 10 * 1024 * 1024),
            ds_cfg.get("tokensPerSample", seq_len + 1),
            shard_cfg.get("slot", 0),
            auth_token,
        )

        batch_size = args.batch_size if args.batch_size > 0 else recommend_batch_size(seq_len)

        while batch_size >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                gc.collect()

                log.info(f"Initializing model with BS={batch_size}...")

                model = SotaGPT(train_backend, train_device).to(train_device)
                model.engram.table.to('cpu')

                model.load_base_weights(initial_weights)

                init_engram_table = torch.from_numpy(initial_engram_weights).view(
                    ENG_NUM_BUCKETS,
                    DIM,
                )
                model.engram.table.weight.data.copy_(init_engram_table)

                model.train()
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if batch_size <= 1:
                        log.error("OOM even at batch size 1. Cannot continue.")
                        raise

                    log.warning(f"CUDA OOM at BS={batch_size}. Halving batch size...")

                    if model is not None:
                        del model
                        model = None

                    if train_backend in ("CUDA", "ROCM"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                    gc.collect()
                    batch_size = max(1, batch_size // 2)
                else:
                    raise

        if model is None:
            log.error("Failed to initialize model.")
            return

        micro_batch_tokens = batch_size * seq_len
        accum_steps = max(1, round(args.tokens_per_step / micro_batch_tokens))
        loss_scale = 1.0 / accum_steps

        def halve_accum():
            nonlocal accum_steps, loss_scale

            if accum_steps <= 1:
                raise RuntimeError("OOM with accumulation=1; cannot reduce further.")

            accum_steps = max(1, accum_steps // 2)
            loss_scale = 1.0 / accum_steps

            optimizer_base.zero_grad(set_to_none=True)
            gc.collect()

            if train_backend in ("CUDA", "ROCM"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            log.warning(
                f"OOM in accumulation window -> accum now {accum_steps}x "
                f"({accum_steps * micro_batch_tokens:,} tok/opt-step)"
                + (" [vanilla mode]" if accum_steps == 1 else "")
            )

        base_params = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
        engram_params = list(model.engram.parameters())

        try:
            import bitsandbytes as bnb
            optimizer_base = bnb.optim.AdamW8bit(
                base_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )
        except ImportError:
            optimizer_base = torch.optim.AdamW(
                base_params,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )

        try:
            optimizer_engram = torch.optim.SparseAdam(engram_params, lr=args.lr)
        except Exception:
            optimizer_engram = torch.optim.AdamW(
                engram_params,
                lr=args.lr,
                weight_decay=0.01,
            )

        use_autocast = (
            args.precision == "bf16"
            and train_backend in ("CUDA", "ROCM")
            and torch.cuda.is_bf16_supported()
        )
        autocast_dtype = torch.bfloat16 if use_autocast else None

        log.info(
            f"Grad accumulation: {accum_steps}x micro-batches = "
            f"{accum_steps * micro_batch_tokens:,} tok/opt-step"
        )

        # ============ CALIBRATION ============
        log.info(
            f"Calibrating TPS ({CALIBRATION_STEPS} opt-steps, "
            f"BS={batch_size}, accum={accum_steps})..."
        )

        cal_start = time.time()
        ci = 0

        while ci < CALIBRATION_STEPS:
            if hb.should_stop():
                log.warning("Round ended during calibration. Restarting cycle.")
                return

            optimizer_base.zero_grad(set_to_none=True)

            try:
                accum_loss_sum = 0.0

                for mi in range(accum_steps):
                    if dataset_shard.needs_new_subchunk():
                        dataset_shard.advance(args.server, args.precision)

                    seed = (hash("cal") % 10000) + ci * 1000 + mi
                    x, y = dataset_shard.get_batch(batch_size, seed=seed)
                    x, y = x.to(train_device), y.to(train_device)

                    loss = _forward_and_loss(
                        model,
                        x,
                        y,
                        seq_len,
                        use_autocast,
                        autocast_dtype,
                        loss_scale,
                    )

                    loss.backward()

                    optimizer_engram.step()
                    optimizer_engram.zero_grad(set_to_none=True)

                    accum_loss_sum += float(loss.item()) * accum_steps
                    total_tok += x.numel()

                    del loss, x, y

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise

                halve_accum()
                continue

            torch.nn.utils.clip_grad_norm_(base_params, 1.0)
            optimizer_base.step()
            optimizer_base.zero_grad(set_to_none=True)

            gc.collect()
            ci += 1

        cal_elapsed = time.time() - cal_start
        measured_tps = total_tok / max(cal_elapsed, 1.0)

        seconds_per_step = cal_elapsed / CALIBRATION_STEPS
        effective_sps = seconds_per_step / TPS_DEGRADATION

        log.info(
            f"TPS: {measured_tps:.0f} | "
            f"{seconds_per_step:.3f}s/opt-step "
            f"(effective: {effective_sps:.3f}s)"
        )

        upload_buffer_sec = UPLOAD_BUFFER_MIN * 60
        available_sec = max(
            60,
            remaining_hours * 3600 - cal_elapsed - upload_buffer_sec,
        )

        # Estimate dataset pause overhead more accurately with accum.
        opt_steps_est = available_sec / max(effective_sps, 1e-6)
        micro_steps_est = opt_steps_est * accum_steps
        advances_est = int(micro_steps_est / StreamingShardDataset.STEPS_PER_SUBCHUNK)
        pause_sec = advances_est * DATASET_PAUSE_PER_ADVANCE

        training_budget_sec = max(60, available_sec - pause_sec)
        target_steps = max(50, int(training_budget_sec / max(effective_sps, 1e-6)))

        estimated_train_min = (target_steps * effective_sps) / 60

        log.info(
            f"Budget: {training_budget_sec / 60:.0f} min | "
            f"Target: {target_steps} opt-steps "
            f"(~{estimated_train_min:.0f} min, "
            f"~{target_steps * accum_steps * micro_batch_tokens:,} tokens)"
        )

        step = CALIBRATION_STEPS
        train_start = time.time()
        deadline = train_start + training_budget_sec

        total_target = target_steps + CALIBRATION_STEPS

        with Live(
            create_dashboard(
                step,
                total_target,
                0,
                measured_tps,
                args.lr,
                global_step,
                train_backend,
                batch_size,
                seq_len,
                current_round,
                estimated_train_min,
                accum_steps,
                accum_steps * micro_batch_tokens,
            ),
            console=console,
            refresh_per_second=2,
            screen=False,
        ) as live:

            while step < total_target:
                if hb.should_stop():
                    log.warning("Stopped by heartbeat/ultimatum.")
                    break

                if time.time() >= deadline:
                    log.info("Time budget reached. Stopping training.")
                    break

                optimizer_base.zero_grad(set_to_none=True)

                try:
                    accum_loss_sum = 0.0
                    completed = 0

                    for mi in range(accum_steps):
                        if hb.should_stop() or time.time() >= deadline:
                            break

                        if dataset_shard.needs_new_subchunk():
                            dataset_shard.advance(args.server, args.precision)

                        seed = (hash("t") % 10000) + step * 1000 + mi
                        x, y = dataset_shard.get_batch(batch_size, seed=seed)
                        x, y = x.to(train_device), y.to(train_device)

                        loss = _forward_and_loss(
                            model,
                            x,
                            y,
                            seq_len,
                            use_autocast,
                            autocast_dtype,
                            loss_scale,
                        )

                        loss.backward()

                        optimizer_engram.step()
                        optimizer_engram.zero_grad(set_to_none=True)

                        accum_loss_sum += float(loss.item()) * accum_steps
                        total_tok += x.numel()
                        completed += 1

                        del loss, x, y

                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise

                    halve_accum()
                    continue

                if completed == 0:
                    optimizer_base.zero_grad(set_to_none=True)
                    break

                torch.nn.utils.clip_grad_norm_(base_params, 1.0)
                optimizer_base.step()
                optimizer_base.zero_grad(set_to_none=True)

                lv = accum_loss_sum / completed
                if math.isnan(lv) or lv <= 0:
                    lv = 10.0

                loss_history.append(lv)
                step += 1

                elapsed = time.time() - train_start
                cur_tps = total_tok / max(elapsed + cal_elapsed, 1.0)
                time_left = max(0, (deadline - time.time()) / 60)

                live.update(
                    create_dashboard(
                        step,
                        total_target,
                        lv,
                        cur_tps,
                        optimizer_base.param_groups[0]['lr'],
                        global_step,
                        train_backend,
                        batch_size,
                        seq_len,
                        current_round,
                        time_left,
                        accum_steps,
                        accum_steps * micro_batch_tokens,
                    )
                )

    finally:
        hb.stop()

        if dataset_shard is not None:
            del dataset_shard
            dataset_shard = None

        gc.collect()

        if train_backend in ("CUDA", "ROCM"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ============ UPLOAD PHASE ============
    actual_training_steps = step - CALIBRATION_STEPS

    if task_id is None or actual_training_steps <= 0 or not loss_history:
        log.warning("Round ended before actual training started. Skipping upload.")

        if model is not None:
            del model
            model = None

        gc.collect()
        return

    final_loss = float(loss_history[-1]) if loss_history else 10.0
    log.info(f"Done: {actual_training_steps} opt-steps, loss {final_loss:.4f}")

    # Free optimizer state before delta computation.
    try:
        if optimizer_base is not None:
            optimizer_base.zero_grad(set_to_none=True)
        if optimizer_engram is not None:
            optimizer_engram.zero_grad(set_to_none=True)
    except Exception:
        pass

    try:
        del optimizer_base, optimizer_engram, base_params, engram_params
    except Exception:
        pass

    optimizer_base = None
    optimizer_engram = None
    base_params = None
    engram_params = None

    gc.collect()

    if train_backend in ("CUDA", "ROCM"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    log.info("Computing base delta...")
    delta_base_bf16 = compute_base_delta_bf16(model, initial_weights)

    del initial_weights
    initial_weights = None
    gc.collect()

    log.info("Computing sparse engram delta...")
    sparse_indices, sparse_values = compute_engram_sparse_delta(model, initial_engram_weights)

    del initial_engram_weights
    initial_engram_weights = None

    del model
    model = None

    gc.collect()

    if train_backend in ("CUDA", "ROCM"):
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    has_engram = bool(len(sparse_indices) > 0)
    engram_sparse_count = int(len(sparse_indices))

    # We always submit delta as bf16, regardless of downloaded weight format.
    meta = {
        "taskId": task_id,
        "loss": final_loss,
        "localSteps": int(step),
        "tokensProcessed": int(total_tok),
        "loraRank": 0,
        "isDelta": True,
        "weightFormat": "bf16",
        "hasEngram": has_engram,
        "engramSparseCount": engram_sparse_count,
    }

    log.info("Compressing delta payload to disk...")

    path = None

    try:
        path, size, checksum = _create_compressed_payload_file(
            meta,
            delta_base_bf16,
            sparse_indices,
            sparse_values,
        )

        del delta_base_bf16, sparse_indices, sparse_values
        delta_base_bf16 = None
        sparse_indices = None
        sparse_values = None

        gc.collect()

        chunk_size = int(args.chunk_size_mb * 1024 * 1024)
        chunk_size = max(1 * 1024 * 1024, min(SERVER_MAX_CHUNK_MB * 1024 * 1024, chunk_size))

        max_direct_bytes = int(args.max_direct_mb * 1024 * 1024)
        max_direct_bytes = max(0, max_direct_bytes)

        log.info(
            f"Uploading delta: {size / (1024 * 1024):.1f} MB compressed, "
            f"chunk size {chunk_size / (1024 * 1024):.0f} MB, "
            f"max direct {max_direct_bytes / (1024 * 1024):.0f} MB"
        )

        r = submit_delta_file(
            args.server,
            headers,
            path,
            size,
            checksum,
            meta,
            chunk_size,
            max_direct_bytes,
        )

        if r is not None and r.status_code == 200:
            log.info("Submitted successfully!")
        elif r is not None:
            log.error(f"Submit failed: {r.text[:300]}")

    except KeyboardInterrupt:
        log.warning("Upload aborted by user.")

    except Exception as e:
        log.error(f"Upload failed: {e}")

    finally:
        if path is not None:
            try:
                os.unlink(path)
            except Exception:
                pass

        gc.collect()


# ============ SWARM NODE LOOP ============
def _check_updates_between_rounds(args):
    """Between rounds, check for updates. If applied, restart the process."""
    if getattr(args, "no_update", False):
        return

    try:
        if check_for_update():
            log.info("🔄 Updated to a new version — restarting...")
            time.sleep(2)
            _restart_self()  # never returns on POSIX
    except Exception as e:
        log.warning(f"Update check failed: {e}")


def run_swarm_node(args):
    global train_device, train_backend

    log.info("=" * 60)
    log.info(f"CrowdGPT CLI Client v{CLIENT_VERSION} - HF-Direct + Chunked Upload Edition")
    log.info("=" * 60)

    username = args.username or os.environ.get("CROWDGPT_USERNAME")
    password = args.password or os.environ.get("CROWDGPT_PASSWORD")
    email = args.email or os.environ.get("CROWDGPT_EMAIL")

    auth_token = None

    if username and password:
        auth_token = authenticate(args.server, username, password, email=email)
        if auth_token:
            log.info(f"Authenticated as {username}")
        else:
            log.warning("Auth failed, running anonymously.")
    else:
        log.info("Running anonymously.")

    try:
        train_device, train_backend = detect_training_backend(args.backend)
        log.info(f"Backend: {train_backend} ({train_device})")
    except Exception as e:
        log.error(f"Backend error: {e}")
        sys.exit(1)

    auto_detect_vram_budget()
    log.info(f"VRAM Budget: {memory_config['ram_gb']:.1f} GB")

    round_count = 0

    while True:
        round_count += 1

        log.info("=" * 50)
        log.info(f"Round cycle #{round_count}")
        log.info("=" * 50)

        try:
            run_single_round(args, auth_token)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            log.error(f"Round failed: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(15)
            continue

        if args.single:
            break

        # Between rounds: safe place to apply an update and restart
        if UPDATE_CHECK_BETWEEN_ROUNDS:
            _check_updates_between_rounds(args)

        log.info("Waiting 10s for next round...")
        time.sleep(10)


# ============ ENTRYPOINT ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CrowdGPT CLI Client - HF-Direct + Chunked Upload + Auto-Update Edition"
    )

    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=131072,
        help="Target tokens per optimizer step (grad accum). 4096 = vanilla.",
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--email", default=None)

    parser.add_argument(
        "--no-hf-weights",
        action="store_true",
        help="Disable Hugging Face direct weight downloads and force coordinator fallback.",
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MB,
        help=f"Chunked upload chunk size in MB. Server max is {SERVER_MAX_CHUNK_MB} MB.",
    )

    parser.add_argument(
        "--max-direct-mb",
        type=int,
        default=DEFAULT_MAX_DIRECT_UPLOAD_MB,
        help="Maximum compressed delta size to send via direct /fl/submit before chunking.",
    )

    parser.add_argument(
        "--single",
        action="store_true",
        help="Run one round and exit.",
    )

    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip auto-update checks entirely.",
    )

    args = parser.parse_args()

    # ---- Auto-update check, before any heavy work ----
    print(f"CrowdGPT CLI client v{CLIENT_VERSION}")

    if not args.no_update:
        try:
            if check_for_update():
                print("Update applied. Restarting...")
                _restart_self()
                # If _restart_self somehow returns, continue normally
        except Exception as e:
            log.warning(f"Auto-update check error: {e}")

    try:
        run_swarm_node(args)
    except KeyboardInterrupt:
        log.info("Disconnected.")
