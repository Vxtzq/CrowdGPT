#!/usr/bin/env python3
"""
CrowdGPT Client - Continuous FL Node (OOM-Proof Edition)
"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys, io, gzip, json, time, struct, logging, hashlib, math, gc, threading, argparse, subprocess
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
console = Console()

# ============ CONFIG ============
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_REPO_ID = "Vxtzq/CrowdGPT"
MODEL_REPO_ID = "Vxtzq/Crowd-v1"

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
    "architecture": "SotaGPT"
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

# Time budget constants
UPLOAD_BUFFER_MIN = 12        # Reserve 12 min for delta computation + upload
TPS_DEGRADATION = 0.85        # TPS drops ~15% over long runs
DATASET_PAUSE_PER_ADVANCE = 8 # Seconds lost per subchunk advance
CALIBRATION_STEPS = 15        # Warmup steps to measure real TPS

def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM) * 2 + DIM * DIM + DIM * 2
        size += DIM * MLP_HIDDEN * 2 + MLP_HIDDEN * DIM
    size += DIM * 2
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

memory_config = {"ram_gb": 12, "safety_margin_gb": 1.0, "is_auto_detected": False}
train_device, train_backend = None, None

# ============ HARDWARE ============
def detect_training_backend(force=None):
    if force and force != "auto":
        name = force.lower()
        if name == "cpu": return torch.device('cpu'), "CPU"
        if name == "cuda" and torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
        if name == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return torch.device('cuda'), "ROCM"
        if name == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), "MPS"
        if name == "directml":
            try:
                import torch_directml
                if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
            except ImportError: pass
        raise Exception(f"Backend '{force}' not available")

    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return torch.device('cuda'), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"
    try:
        import torch_directml
        if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
    except ImportError: pass
    return torch.device('cpu'), "CPU"

def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            usable = max(1.0, min(free_b / 1024**3, total_b / 1024**3 - 2.0) - 1.0)
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True})
            return
        except Exception: pass
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        memory_config.update({"ram_gb": round(max(1.0, avail - 2.0), 2), "is_auto_detected": True})
    except ImportError: pass

def has_bitsandbytes():
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False

def estimate_vram_bytes(batch_size, seq_len, use_8bit):
    """Accurate VRAM estimate including gradients and full checkpoint activations"""
    # 1. Model weights (BF16)
    model_bytes = EXPECTED_MODEL_SIZE * 2  
    # 2. Optimizer states (8-bit AdamW = ~4 bytes/param, FP32 AdamW = ~12 bytes/param)
    optim_bytes = EXPECTED_MODEL_SIZE * (4 if use_8bit else 12)
    # 3. Gradients (FP32) - ALWAYS present during backward pass!
    grad_bytes = EXPECTED_MODEL_SIZE * 4  
    
    # 4. Activations: Gradient checkpointing stores the input to EVERY block
    act_per_block = batch_size * seq_len * DIM * 2
    attn_act = batch_size * N_HEADS * 64 * seq_len * 4
    act_bytes = N_LAYERS * (act_per_block + attn_act)
    
    # 5. Logits chunk & Engram cache
    logits_bytes = batch_size * 256 * VOCAB_SIZE * 4
    engram_bytes = batch_size * seq_len * DIM * 2
    
    # 6. CUDA context + safety margin
    safety = int(0.8 * 1024**3)
    
    return int(model_bytes + optim_bytes + grad_bytes + act_bytes + logits_bytes + engram_bytes + safety)

def recommend_batch_size(seq_len=2048):
    """Pick largest batch size that fits VRAM"""
    use_8bit = has_bitsandbytes()
    budget = int(memory_config["ram_gb"] * 1024**3)
    best_bs = 1
    for bs in ALLOWED_BATCH_SIZES:
        est = estimate_vram_bytes(bs, seq_len, use_8bit)
        if est <= budget:
            best_bs = bs
        else:
            break
    log.info(f"VRAM budget: {memory_config['ram_gb']:.1f}GB | 8-bit optim: {use_8bit} | Recommended BS: {best_bs}")
    return best_bs

# ============ MODEL ============
def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)

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
        if use_chunked: return self._chunked(q, k, v, B, T, C)
        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        m = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        a = a.masked_fill(m == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        m = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)
            aw = (q[:, :, i:e, :] @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            aw = aw.masked_fill(m[:, :, i:e, :] == 0, float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v)
        return self.wo(torch.cat(chunks, 2).transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

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
        if MODEL_CONFIG["weightTying"]: self.wte.weight = self.lm_head.weight
        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos", cm)
        self.register_buffer("freqs_sin", sm)

    def get_base_weights(self):
        return np.concatenate([p.detach().float().flatten().cpu().numpy() for n, p in self.named_parameters() if not n.startswith('engram.')])

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = p.numel()
                p.data.copy_(ft[o:o+s].view(p.shape).to(p.device))
                o += s

# ============ DATASET ============
import http.client
from requests.exceptions import ChunkedEncodingError

def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ChunkedEncodingError, http.client.IncompleteRead) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"Network drop (attempt {attempt+1}/{retries}), retrying in {wait}s")
                time.sleep(wait)
            else: raise

class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500
    def __init__(self, repo_id, chunk_idx, sub_size=10*1024*1024, tps=65, slot=0, auth_token=None):
        self.repo_id, self.ci, self.sub_size, self.tps, self.auth_token = repo_id, chunk_idx, sub_size, tps, auth_token
        self._name_fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover_chunk_size(chunk_idx)
        self.off = (slot * sub_size) % self.chunk_size
        self.data, self.n, self.steps_used = None, 0, 0
        self._pf_thread, self._pf_result, self._lock = None, None, threading.Lock()
        self._load_subchunk(self.off)
        self._start_prefetch(self.off + self.sub_size)

    def _chunk_url(self):
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._name_fmt.format(self.ci)

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
                except Exception: time.sleep(1)
        raise RuntimeError(f"Could not find chunk {idx}")

    def _fetch_slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(self._chunk_url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set_data(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data, self.n, self.steps_used = tk, len(tk) // self.tps, 0

    def _load_subchunk(self, off):
        self._set_data(self._fetch_slice(off))
        self.off = off

    def _start_prefetch(self, off):
        if off >= self.chunk_size: return
        with self._lock: self._pf_result = None
        def worker():
            try:
                raw = self._fetch_slice(off)
                with self._lock: self._pf_result = (off, raw)
            except Exception:
                with self._lock: self._pf_result = None
        self._pf_thread = threading.Thread(target=worker, daemon=True)
        self._pf_thread.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, server_url=None, fmt="bf16"):
        next_off = self.off + self.sub_size
        if next_off < self.chunk_size:
            if self._pf_thread: self._pf_thread.join(timeout=120); self._pf_thread = None
            with self._lock: res = self._pf_result; self._pf_result = None
            if res and res[0] == next_off:
                self.off = next_off; self._set_data(res[1])
            else: self._load_subchunk(next_off)
            self._start_prefetch(next_off + self.sub_size)
            return True
        self.request_new_chunk(server_url, fmt)
        self.chunk_size = self._discover_chunk_size(self.ci)
        self.off = 0
        self._load_subchunk(0)
        self._start_prefetch(self.sub_size)
        return True

    def request_new_chunk(self, server_url, fmt="bf16"):
        if not server_url: return False
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{server_url}/fl/task?format={fmt}&skip_weights=true", headers=headers, timeout=30)
            if r.status_code == 200:
                raw = r.content
                ml = struct.unpack('<I', raw[:4])[0]
                new_idx = json.loads(raw[4:4+ml].decode()).get("datasetConfig", {}).get("chunkIdx", self.ci)
                if new_idx != self.ci: self.ci = new_idx
                return True
        except Exception: pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0: self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        starts = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[s:s+self.tps-1] for s in starts])
        tgt = np.stack([self.data[s+1:s+self.tps] for s in starts])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# ============ WEIGHTS ============
import zlib
def decompress_weights(raw, fmt="bf16"):
    if len(raw) >= 2 and raw[0] == 0x78: raw = zlib.decompress(raw)
    if fmt == "fp16": return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()

# ============ AUTH ============
def authenticate(server_url, username, password):
    if not username or not password: return None
    try:
        r = requests.post(f"{server_url}/auth/login", json={"username": username, "password": password}, timeout=30)
        if r.status_code == 200: return r.json().get("token")
        r = requests.post(f"{server_url}/auth/register", json={"username": username, "password": password, "email": f"{username}@crowdgpt.local"}, timeout=30)
        if r.status_code == 200: return r.json().get("token")
    except Exception: pass
    return None

# ============ DASHBOARD ============
def create_dashboard(step, target_steps, loss, tps, lr, global_step, backend_name, batch_size, seq_len, current_round, time_remaining_min):
    table = Table(title=f"Round {current_round} Training", expand=True, border_style="dim")
    table.add_column("Metric", style="bold"); table.add_column("Value", justify="right")
    table.add_row("Progress", f"{step}/{target_steps} ({step/max(1,target_steps)*100:.1f}%)")
    table.add_row("Loss", f"{loss:.4f}" if loss > 0 else "—")
    table.add_row("Tokens/s", f"{tps:.0f}" if tps else "—")
    table.add_row("LR", f"{lr:.2e}" if lr else "—")
    table.add_row("Global Step", str(global_step))
    table.add_row("Round", str(current_round))
    table.add_row("Time Left", f"{time_remaining_min:.0f} min")
    table.add_row("Backend", backend_name)
    table.add_row("Batch / SeqLen", f"{batch_size} / {seq_len}")
    return table

# ============ HEARTBEAT ============
class HeartbeatManager:
    def __init__(self, server_url, headers, current_round):
        self.server_url, self.headers, self.current_round = server_url, headers, current_round
        self.stop_training = threading.Event()
        self.shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self._thread.start()
    def stop(self): self.shutdown.set()
    def should_stop(self): return self.stop_training.is_set()

    def _run(self):
        while not self.shutdown.is_set():
            try:
                requests.get(f"{self.server_url}/fl/heartbeat", headers=self.headers, timeout=15)
                r = requests.get(f"{self.server_url}/fl/round_status", headers=self.headers, timeout=15)
                if r.status_code == 200:
                    status = r.json()
                    if status.get("current_round", self.current_round) != self.current_round:
                        log.warning("Server moved to next round. Stopping!")
                        self.stop_training.set()
                    remaining_min = (status.get("max_round_hours", 2) - status.get("round_elapsed_hours", 0)) * 60
                    if remaining_min <= 3:
                        log.warning(f"ULTIMATUM: {remaining_min:.0f} min left! Submitting NOW.")
                        self.stop_training.set()
            except Exception: pass
            self.shutdown.wait(timeout=120)

# ============ MAIN TRAINING ============
def wait_for_round(server_url, headers):
    while True:
        try:
            r = requests.get(f"{server_url}/fl/round_status", headers=headers, timeout=30)
            if r.status_code == 200:
                status = r.json()
                if status.get("is_aggregating"):
                    log.info("⏳ Server is aggregating and uploading to HF. Waiting 30s...")
                    time.sleep(30)
                    continue
                if not status.get("in_cooldown", False):
                    return status
                log.info("Server in cooldown. Waiting 30s...")
                time.sleep(30)
            else: time.sleep(10)
        except Exception: time.sleep(10)

def fetch_task_and_weights(server_url, headers, precision):
    while True:
        try:
            r = requests.get(f"{server_url}/fl/task?format={precision}", headers=headers, timeout=(15, 3600))
            if r.headers.get("X-Status") == "wait":
                try:
                    body = r.json()
                    log.info(f"⏳ Server says wait: {body.get('message', '')}")
                except Exception:
                    log.info("⏳ Server says wait (aggregating or cooldown)")
                time.sleep(30); continue
            raw = r.content
            ml = struct.unpack('<I', raw[:4])[0]
            metadata = json.loads(raw[4:4+ml].decode())
            weights_bytes = raw[4+ml:]
            return metadata, weights_bytes
        except Exception as e:
            log.error(f"Task fetch failed: {e}")
            time.sleep(15)

def run_single_round(args, auth_token=None):
    global train_device, train_backend
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    # 1. Wait for server
    log.info("Checking round status...")
    round_status = wait_for_round(args.server, headers)
    current_round = round_status["current_round"]
    max_round_hours = round_status.get("max_round_hours", 2.0)

    # 2. Fetch fresh weights
    log.info("Downloading fresh weights...")
    metadata, weights_bytes = fetch_task_and_weights(args.server, headers, args.precision)
    task_id = metadata['taskId']
    global_step = metadata['globalStep']
    server_round = metadata.get('currentRound', current_round)
    weight_format = metadata.get('weightFormat', 'bf16')
    seq_len = args.seq_len if args.seq_len > 0 else 2048

    # 3. RE-SYNC round timer
    try:
        rs = requests.get(f"{args.server}/fl/round_status", headers=headers, timeout=15)
        if rs.status_code == 200:
            fresh = rs.json()
            round_elapsed = fresh.get("round_elapsed_hours", 0)
            remaining_hours = max(0.05, fresh.get("max_round_hours", 2.0) - round_elapsed)
            log.info(f"Round timer re-synced: {remaining_hours*60:.0f} min remaining")
    except Exception:
        remaining_hours = max(0.1, max_round_hours - round_status.get("round_elapsed_hours", 0))

    # 4. Decompress weights
    # 4. Decompress weights (Base + Engram)
    bytes_per_param = 2 if weight_format in ("bf16", "fp16") else 4
    base_bytes_len = EXPECTED_MODEL_SIZE * bytes_per_param
    
    initial_weights = decompress_weights(weights_bytes[:base_bytes_len], weight_format)
    initial_engram_weights = decompress_weights(weights_bytes[base_bytes_len:], weight_format)
    del weights_bytes; gc.collect()

    # 5. Setup dataset
    ds_cfg = metadata.get("datasetConfig", {})
    shard_cfg = metadata.get("shardConfig", {})
    dataset_shard = StreamingShardDataset(
        ds_cfg.get("repoId", DATASET_REPO_ID), ds_cfg.get("chunkIdx", 0),
        ds_cfg.get("subChunkSize", 10*1024*1024), ds_cfg.get("tokensPerSample", seq_len+1),
        shard_cfg.get("slot", 0), auth_token)

    # 6. Setup model with OOM FALLBACK
    batch_size = args.batch_size if args.batch_size > 0 else recommend_batch_size(seq_len)
    
    total_tok = 0
    loss_history = []
    
    while batch_size >= 1:
        try:
            if train_backend in ("CUDA", "ROCM"):
                try: torch.cuda.empty_cache()
                except: pass
            gc.collect()
            
            log.info(f"Initializing model with BS={batch_size}...")
            model = SotaGPT(train_backend, train_device).to(train_device)
            model.engram.table.to('cpu')
            model.load_base_weights(initial_weights)
            
            # 🚨 CRITICAL FIX: Load Engram weights from server!
            ft_eng = torch.from_numpy(initial_engram_weights) if not isinstance(initial_engram_weights, torch.Tensor) else initial_engram_weights
            model.engram.table.weight.data.copy_(ft_eng.view(model.engram.table.weight.shape))
            
            model.train()

            base_params = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
            engram_params = list(model.engram.parameters())

            try:
                import bitsandbytes as bnb
                optimizer_base = bnb.optim.AdamW8bit(base_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            except ImportError:
                optimizer_base = torch.optim.AdamW(base_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            try:
                optimizer_engram = torch.optim.SparseAdam(engram_params, lr=args.lr)
            except Exception:
                optimizer_engram = torch.optim.AdamW(engram_params, lr=args.lr, weight_decay=0.01)

            initial_engram_weights = model.engram.table.weight.data.cpu().clone()
            use_autocast = args.precision == "bf16" and train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported()
            autocast_dtype = torch.bfloat16 if use_autocast else None

            # 7. TPS Calibration
            log.info(f"Calibrating TPS ({CALIBRATION_STEPS} steps, BS={batch_size})...")
            loss_history = []
            total_tok = 0
            cal_start = time.time()

            for ci in range(1, CALIBRATION_STEPS + 1):
                if dataset_shard.needs_new_subchunk(): dataset_shard.advance(args.server, args.precision)
                x, y = dataset_shard.get_batch(batch_size, seed=hash("cal") % 10000 + ci)
                x, y = x.to(train_device), y.to(train_device)
                
                def fwd():
                    x_emb = model.wte(x)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', enabled=False): eng_out = model.engram(x)
                    else: eng_out = model.engram(x)
                    x_emb = x_emb + eng_out.to(x_emb.dtype)
                    for b in model.blocks:
                        x_emb = checkpoint(b, x_emb, model.freqs_cos, model.freqs_sin, True, use_reentrant=True)
                    return model.ln_f(x_emb)

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        x_emb = fwd()
                        loss = sum(F.cross_entropy(model.lm_head(x_emb[:, i:i+256, :]).reshape(-1, VOCAB_SIZE), y[:, i:i+256].reshape(-1)) for i in range(0, seq_len, 256)) / max(1, math.ceil(seq_len/256))
                        loss.backward()
                else:
                    x_emb = fwd()
                    loss = sum(F.cross_entropy(model.lm_head(x_emb[:, i:i+256, :]).reshape(-1, VOCAB_SIZE), y[:, i:i+256].reshape(-1)) for i in range(0, seq_len, 256)) / max(1, math.ceil(seq_len/256))
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(base_params, 1.0)
                optimizer_base.step(); optimizer_engram.step()
                optimizer_base.zero_grad(set_to_none=True)
                try: optimizer_engram.zero_grad(set_to_none=True)
                except Exception: optimizer_engram.zero_grad()

                lv = float(loss.item())
                if math.isnan(lv) or lv <= 0: lv = 10.0
                total_tok += x.numel()
                loss_history.append(lv)
                
            # If we get here, calibration succeeded! Break the loop.
            break 
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning(f"CUDA OOM at BS={batch_size}. Halving batch size and retrying...")
                del model, optimizer_base, optimizer_engram
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except: pass
                gc.collect()
                batch_size = max(1, batch_size // 2)
            else:
                raise

    cal_elapsed = time.time() - cal_start
    if cal_elapsed <= 0 or total_tok == 0: raise Exception("Calibration failed")

    measured_tps = total_tok / cal_elapsed
    tokens_per_step = batch_size * seq_len
    seconds_per_step = cal_elapsed / CALIBRATION_STEPS

    # 8. TIME BUDGET CALCULATION
    effective_sps = seconds_per_step / TPS_DEGRADATION

    dataset_advances_remaining = int((remaining_hours * 3600) / effective_sps / StreamingShardDataset.STEPS_PER_SUBCHUNK)
    dataset_pause_total = dataset_advances_remaining * DATASET_PAUSE_PER_ADVANCE

    training_budget_sec = (remaining_hours * 3600) - cal_elapsed - (UPLOAD_BUFFER_MIN * 60) - dataset_pause_total
    training_budget_sec = max(60, training_budget_sec)

    target_steps = int(training_budget_sec / effective_sps)
    target_steps = max(50, min(target_steps, 500_000))
    estimated_train_min = (target_steps * effective_sps) / 60

    log.info(f"TPS: {measured_tps:.0f} | {seconds_per_step:.3f}s/step (effective: {effective_sps:.3f}s)")
    log.info(f"Budget: {training_budget_sec/60:.0f} min | Dataset pauses: ~{dataset_pause_total}s | Target: {target_steps} steps (~{estimated_train_min:.0f} min)")

    # 9. Start heartbeat
    hb = HeartbeatManager(args.server, headers, server_round)
    hb.start()

    # 10. Training loop
    step = CALIBRATION_STEPS
    train_start = time.time()
    deadline = train_start + training_budget_sec

    with Live(create_dashboard(step, target_steps + CALIBRATION_STEPS, loss_history[-1] if loss_history else 0,
                               measured_tps, args.lr, global_step, train_backend, batch_size, seq_len,
                               current_round, estimated_train_min),
              console=console, refresh_per_second=2, screen=False) as live:

        for _ in range(target_steps):
            if hb.should_stop():
                log.warning("Stopped by heartbeat/ultimatum.")
                break
            if time.time() >= deadline:
                log.info("Time budget reached. Stopping training.")
                break

            if dataset_shard.needs_new_subchunk(): dataset_shard.advance(args.server, args.precision)
            x, y = dataset_shard.get_batch(batch_size, seed=hash("t") % 10000 + step)
            x, y = x.to(train_device), y.to(train_device)

            try:
                def fwd():
                    x_emb = model.wte(x)
                    if use_autocast:
                        with torch.autocast(device_type='cuda', enabled=False): eng_out = model.engram(x)
                    else: eng_out = model.engram(x)
                    x_emb = x_emb + eng_out.to(x_emb.dtype)
                    for b in model.blocks:
                        x_emb = checkpoint(b, x_emb, model.freqs_cos, model.freqs_sin, True, use_reentrant=True)
                    return model.ln_f(x_emb)

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        x_emb = fwd()
                        loss = sum(F.cross_entropy(model.lm_head(x_emb[:, i:i+256, :]).reshape(-1, VOCAB_SIZE), y[:, i:i+256].reshape(-1)) for i in range(0, seq_len, 256)) / max(1, math.ceil(seq_len/256))
                        loss.backward()
                else:
                    x_emb = fwd()
                    loss = sum(F.cross_entropy(model.lm_head(x_emb[:, i:i+256, :]).reshape(-1, VOCAB_SIZE), y[:, i:i+256].reshape(-1)) for i in range(0, seq_len, 256)) / max(1, math.ceil(seq_len/256))
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(base_params, 1.0)
                optimizer_base.step(); optimizer_engram.step()
                optimizer_base.zero_grad(set_to_none=True)
                try: optimizer_engram.zero_grad(set_to_none=True)
                except Exception: optimizer_engram.zero_grad()

                lv = float(loss.item())
                if math.isnan(lv) or lv <= 0: lv = 10.0
                total_tok += x.numel()
                loss_history.append(lv)
                step += 1

                elapsed = time.time() - train_start
                cur_tps = total_tok / max(elapsed + cal_elapsed, 1)
                time_left = max(0, (deadline - time.time()) / 60)
                live.update(create_dashboard(step, target_steps + CALIBRATION_STEPS, lv, cur_tps,
                                             optimizer_base.param_groups[0]['lr'], global_step,
                                             train_backend, batch_size, seq_len, current_round, time_left))
            except Exception as e:
                log.error(f"Training error: {e}"); break

    hb.stop()

    # 11. Compute deltas
    final_loss = float(loss_history[-1]) if loss_history else 10.0
    log.info(f"Done: {step} steps, loss {final_loss:.4f}")

    delta_base_bf16 = torch.from_numpy(model.get_base_weights() - initial_weights).to(torch.bfloat16).view(torch.uint16).numpy()
    engram_delta = model.engram.table.weight.data.cpu() - initial_engram_weights

    # Top-K sparsity (10%)
    row_norms = engram_delta.abs().sum(dim=1)
    k = max(1, int(len(row_norms) * 0.10))
    topk_values, topk_indices = torch.topk(row_norms, k)
    active_mask = topk_values > 1e-8
    active_indices = topk_indices[active_mask]
    sparse_indices = active_indices.cpu().numpy().astype(np.uint32) if len(active_indices) > 0 else np.array([], dtype=np.uint32)
    sparse_values = engram_delta[active_indices].to(torch.bfloat16).view(torch.uint16).numpy() if len(active_indices) > 0 else np.array([], dtype=np.uint16)

    # 12. Upload
    payload = json.dumps({
        "taskId": task_id, "loss": final_loss, "localSteps": step,
        "tokensProcessed": total_tok, "loraRank": 0, "isDelta": True,
        "weightFormat": "bf16", "hasEngram": True, "engramSparseCount": len(sparse_indices)
    }).encode()

    binary = struct.pack('<I', len(payload)) + payload
    binary += np.ascontiguousarray(delta_base_bf16).tobytes()
    binary += np.ascontiguousarray(sparse_indices).tobytes()
    binary += np.ascontiguousarray(sparse_values).tobytes()

    log.info(f"Uploading (Base: {len(delta_base_bf16)/1024/1024:.1f}MB, Engram: {len(sparse_indices)} rows)...")

    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=10, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    r = session.post(f"{args.server}/fl/submit",
                     headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **headers},
                     data=gzip.compress(binary, compresslevel=2), timeout=600)
    if r.status_code == 200: log.info("Submitted successfully!")
    else: raise Exception(f"Submit failed: {r.text[:300]}")

    del model, initial_weights, delta_base_bf16, binary
    gc.collect()
    if train_backend in ("CUDA", "ROCM"):
        try: torch.cuda.empty_cache()
        except Exception: pass

def run_swarm_node(args):
    global train_device, train_backend
    log.info("=" * 60)
    log.info("CrowdGPT Continuous FL Node (OOM-Proof Edition)")
    log.info("=" * 60)

    username = args.username or os.environ.get("CROWDGPT_USERNAME")
    password = args.password or os.environ.get("CROWDGPT_PASSWORD")
    auth_token = None
    if username and password:
        auth_token = authenticate(args.server, username, password)
        if auth_token: log.info(f"Authenticated as {username}")
        else: log.warning("Auth failed, running anonymously.")
    else: log.info("Running anonymously.")

    try:
        train_device, train_backend = detect_training_backend(args.backend)
        log.info(f"Backend: {train_backend} ({train_device})")
    except Exception as e:
        log.error(f"Backend error: {e}"); sys.exit(1)

    auto_detect_vram_budget()
    log.info(f"VRAM Budget: {memory_config['ram_gb']:.1f} GB")

    round_count = 0
    while True:
        round_count += 1
        log.info(f"{'='*50}")
        log.info(f"Round cycle #{round_count}")
        log.info(f"{'='*50}")
        try:
            run_single_round(args, auth_token)
        except KeyboardInterrupt: raise
        except Exception as e:
            log.error(f"Round failed: {e}")
            time.sleep(15); continue
        if args.single: break
        log.info("Waiting 10s for next round...")
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://api.crowdgpt.net:5006")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--single", action="store_true")
    try: run_swarm_node(parser.parse_args())
    except KeyboardInterrupt: log.info("Disconnected.")
