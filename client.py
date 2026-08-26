#!/usr/bin/env python3
"""
CrowdGPT Client - Distributed LLM Training

"""

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import io
import gzip
import zlib
import json
import time
import struct
import logging
import hashlib
import math
import gc
import threading
import argparse
import subprocess
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

# ============ LOGGING & CONSOLE ============
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

HF_WEIGHTS_URL_BF16 = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/model_bf16.safetensors"
HF_WEIGHTS_URL_FP32 = f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/model_fp32.safetensors"
LOCAL_SAFETENSORS_CACHE = CHECKPOINT_DIR / "hf_weights_bf16.safetensors"
LOCAL_FP32_CACHE = CHECKPOINT_DIR / "hf_weights_fp32.bin"

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

def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2
        size += DIM * (N_HEADS * HEAD_DIM)
        size += DIM * (N_KV_HEADS * HEAD_DIM) * 2
        size += DIM * DIM
        size += DIM * 2
        size += DIM * MLP_HIDDEN * 2
        size += MLP_HIDDEN * DIM
    size += DIM * 2
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()

ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

memory_config = {"ram_gb": 12, "safety_margin_gb": 1.0, "is_auto_detected": False, "precision": "bf16"}
train_device, train_backend = None, None

BANNER = """
[bold cyan]CrowdGPT[/bold cyan] [dim]::[/dim] [magenta]Distributed LLM Training Node (Crowd-v1)[/magenta]
[dim]═══════════════════════════════════════════════════════════════[/dim]
"""

def print_welcome():
    console.print(BANNER)

# ============ CUDA AUTO-DETECTION ============
def check_cuda_installed():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def suggest_cuda_install():
    console.print("\n[bold red]⚠ PyTorch does NOT detect your GPU![/bold red]")
    console.print("[white]pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/white]\n")

# ============ HARDWARE & MEMORY ============
def check_backend_available(name):
    name = name.lower()
    if name == "cpu": 
        return True, torch.device('cpu'), "CPU"
    if name == "cuda":
        if torch.cuda.is_available() and not (hasattr(torch.version,'hip') and torch.version.hip):
            return True, torch.device('cuda'), f"CUDA: {torch.cuda.get_device_name(0)}"
        return False, None, "CUDA not available"
    if name == "rocm":
        if hasattr(torch.version,'hip') and torch.version.hip and torch.cuda.is_available():
            return True, torch.device('cuda'), f"ROCm: {torch.cuda.get_device_name(0)}"
        return False, None, "ROCm not available"
    if name == "mps":
        if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
            return True, torch.device('mps'), "MPS (Apple Silicon)"
        return False, None, "MPS not available"
    if name in ("xpu","intel"):
        try:
            import intel_extension_for_pytorch
            if hasattr(torch,'xpu') and torch.xpu.is_available():
                return True, torch.device('xpu'), f"XPU: {torch.xpu.get_device_name(0)}"
        except ImportError: pass
        return False, None, "XPU not available"
    if name == "directml":
        try:
            import torch_directml
            if torch_directml.is_available():
                return True, torch_directml.device(0), f"DirectML: {torch_directml.device_name(0)}"
        except ImportError: pass
        return False, None, "DirectML not available"
    return False, None, f"Unknown: {name}"

def detect_training_backend(force=None):
    if force and force != "auto":
        ok, dev, info = check_backend_available(force.lower())
        if not ok:
            raise Exception(f"'{force}': {info}")
        return dev, force.upper()
    
    # Auto-detect priority: CUDA -> ROCm -> XPU -> MPS -> DirectML -> CPU
    if torch.cuda.is_available() and not (hasattr(torch.version,'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"
    if hasattr(torch.version,'hip') and torch.version.hip and torch.cuda.is_available():
        return torch.device('cuda'), "ROCM"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch,'xpu') and torch.xpu.is_available():
            return torch.device('xpu'), "XPU"
    except ImportError: pass
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(0), "DIRECTML"
    except ImportError: pass
    
    if check_cuda_installed():
        suggest_cuda_install()
        
    return torch.device('cpu'), "CPU"

def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            total_gb = total_b / 1024**3
            free_gb = free_b / 1024**3
            
            os_reserve_gb = 2.0
            safety_margin_gb = 1.0
            
            usable = max(1.0, min(free_gb, total_gb - os_reserve_gb) - safety_margin_gb)
            
            memory_config.update({
                "ram_gb": round(usable, 2), 
                "safety_margin_gb": safety_margin_gb + os_reserve_gb, 
                "is_auto_detected": True
            })
            return
        except Exception:
            pass
    if train_backend == "DIRECTML":
        try:
            import torch_directml
            if torch_directml.is_available():
                memory_config.update({"ram_gb": 2.0, "safety_margin_gb": 1.0, "is_auto_detected": True})
                return
        except ImportError: pass
    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl","-n","hw.memsize"], text=True).strip()
            usable = max(1.0, int(out)/1024**3*0.75 - 2.0)
            memory_config.update({"ram_gb": round(usable,2), "safety_margin_gb": 1.0, "is_auto_detected": True})
            return
        except Exception: pass
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        memory_config.update({"ram_gb": round(max(1.0, avail - 2.0), 2), "safety_margin_gb": 1.0, "is_auto_detected": True})
    except ImportError: pass

def estimate_vram_bytes(precision="bf16", batch_size=1, seq_len=2048, use_8bit=False):
    model_b_bf16 = EXPECTED_MODEL_SIZE * 2

    bytes_per_param = 7 if use_8bit else 16
    optim_b = EXPECTED_MODEL_SIZE * bytes_per_param
    
    cs = 64 # attention chunk size
    act_per_layer = (batch_size * seq_len * DIM * 2) + (batch_size * N_HEADS * cs * seq_len * 4)
    act_b = act_per_layer * N_LAYERS
    
    logits_chunk = 256
    logits_b = batch_size * logits_chunk * VOCAB_SIZE * 4
    
    total = model_b_bf16 + optim_b + act_b + logits_b
    
    return int(total) + int(1.5 * 1024**3)

def recommend_optimal_config(precision="bf16", seq_len=2048, use_8bit=False):
    budget = int(memory_config["ram_gb"] * 1024**3)
    safe = budget - int(0.2 * 1024**3)
    best_bs = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(precision, bs, seq_len, use_8bit) <= safe:
            best_bs = bs
        else:
            break
    return 0, "full_pretrain", best_bs, N_LAYERS

# ============ MODEL ARCHITECTURE ============
def precompute_freqs(dim, sl, dev):
    inv = 1.0/(10000.0**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
    t = torch.arange(sl, dtype=torch.float32)
    f = torch.einsum("i,j->ij", t, inv)
    e = torch.cat((f,f),-1)
    return e.cos()[None,None,:,:].to(dev), e.sin()[None,None,:,:].to(dev)

def rotate_half(x):
    a, b = x[...,:x.shape[-1]//2], x[...,x.shape[-1]//2:]
    return torch.cat((-b,a),-1)

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.nrep = N_HEADS, N_KV_HEADS, N_HEADS//N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS*HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, use_chunked=True):
        B,T,C = x.size()
        q = self.wq(x).view(B,T,self.nh,HEAD_DIM).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,HEAD_DIM).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,HEAD_DIM).transpose(1,2)
        ct, st = cos[:,:,:T,:], sin[:,:,:T,:]
        q = q*ct + rotate_half(q)*st
        k = k*ct + rotate_half(k)*st
        k = k.unsqueeze(2).expand(B,self.nkv,self.nrep,T,HEAD_DIM).reshape(B,self.nh,T,HEAD_DIM)
        v = v.unsqueeze(2).expand(B,self.nkv,self.nrep,T,HEAD_DIM).reshape(B,self.nh,T,HEAD_DIM)
        
        if use_chunked: 
            return self._chunked(q,k,v,B,T,C)
            
        a = (q@k.transpose(-2,-1))*(1.0/math.sqrt(HEAD_DIM))
        m = torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
        a = a.masked_fill(m==0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a@v).transpose(1,2).contiguous().view(B,T,C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        """Process attention in chunks of size 'cs' to drastically reduce peak VRAM."""
        m = torch.tril(torch.ones(T,T,device=q.device)).view(1,1,T,T)
        chunks = []
        for i in range(0, T, cs):
            e = min(i+cs, T)
            aw = (q[:,:,i:e,:] @ k.transpose(-2,-1)) * (1.0/math.sqrt(HEAD_DIM))
            aw = aw.masked_fill(m[:,:,i:e,:]==0, float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v)
        return self.wo(torch.cat(chunks, 2).transpose(1,2).contiguous().view(B,T,C))

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        h = MLP_HIDDEN
        self.w1 = nn.Linear(DIM, h, bias=False)
        self.w2 = nn.Linear(DIM, h, bias=False)
        self.w3 = nn.Linear(h, DIM, bias=False)
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

class SotaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]:
            self.wte.weight = self.lm_head.weight
        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, train_device)
        self.register_buffer("freqs_cos", cm)
        self.register_buffer("freqs_sin", sm)

    def forward(self, idx, use_chunked=True):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, self.freqs_cos, self.freqs_sin, use_chunked)
        return self.lm_head(self.ln_f(x))

    def get_flat_weights(self):
        return np.concatenate([p.detach().float().flatten().cpu().numpy() for p in self.parameters()])

    def load_flat_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for p in self.parameters():
            s = p.numel()
            p.data.copy_(ft[o:o+s].view(p.shape).to(p.device))
            o += s

# ============ DATASET STREAMING ============
def _fetch_with_retry(url, headers=None, timeout=60, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"⚠️ Download failed (attempt {attempt+1}/{retries}), retry in {wait}s: {str(e)[:100]}")
                time.sleep(wait)
            else:
                raise

class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500

    def __init__(self, repo_id, chunk_idx, sub_size=10*1024*1024, tps=65, slot=0, auth_token=None):
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
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._name_fmt.format(self.ci)

    def _discover_chunk_size(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for attempt in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        size = int(r.headers.get('content-length', 0))
                        if size > 0:
                            self._name_fmt = fmt
                            log.info(f"📐 {fmt.format(idx)} size: {size/1024/1024:.1f}MB")
                            return size
                except Exception:
                    time.sleep(1)
        raise RuntimeError(f"Could not find chunk {idx} in {self.repo_id}/chunks/")

    def _fetch_slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        headers = {'Range': f'bytes={off}-{end}'}
        return _fetch_with_retry(self._chunk_url(), headers=headers, timeout=120).content

    def _set_data(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data = tk
        self.n = len(tk) // self.tps
        self.steps_used = 0

    def _load_subchunk(self, off):
        raw = self._fetch_slice(off)
        self.off = off
        self._set_data(raw)
        log.info(f"📦 chunk {self.ci} @ {off/1024/1024:.0f}MB: {self.n:,} sequences ({len(raw)/1024/1024:.1f}MB)")

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
            except Exception as e:
                log.warning(f"⚠️ Prefetch failed: {e}")
                with self._lock:
                    self._pf_result = None
        self._pf_thread = threading.Thread(target=worker, daemon=True)
        self._pf_thread.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, server_url=None, mode="quick", fmt="bf16"):
        next_off = self.off + self.sub_size
        if next_off < self.chunk_size:
            if self._pf_thread is not None:
                self._pf_thread.join(timeout=120)
                self._pf_thread = None
            with self._lock:
                res = self._pf_result
                self._pf_result = None
            if res is not None and res[0] == next_off:
                self.off = next_off
                self._set_data(res[1])
                log.info(f"📦 chunk {self.ci} @ {next_off/1024/1024:.0f}MB (prefetched, {self.n:,} seqs)")
            else:
                self._load_subchunk(next_off)
            self._start_prefetch(next_off + self.sub_size)
            return True

        log.info(f"🔄 End of chunk {self.ci}, requesting next shard from coordinator")
        self.request_new_chunk(server_url, mode, fmt)
        self.chunk_size = self._discover_chunk_size(self.ci)
        self.off = 0
        self._load_subchunk(0)
        self._start_prefetch(self.sub_size)
        return True

    def request_new_chunk(self, server_url, mode="quick", fmt="bf16"):
        if not server_url:
            return False
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            r = requests.get(f"{server_url}/fl/task?mode={mode}&format={fmt}&skip_weights=true",
                             headers=headers, timeout=30)
            if r.status_code == 200:
                raw = r.content
                ml = struct.unpack('<I', raw[:4])[0]
                metadata = json.loads(raw[4:4+ml].decode('utf-8'))
                new_idx = metadata.get("datasetConfig", {}).get("chunkIdx", self.ci)
                if new_idx != self.ci:
                    self.ci = new_idx
                    log.info(f"🔄 Coordinator assigned shard chunk {self.ci}")
                return True
        except Exception as e:
            log.warning(f"⚠️ Failed to request new shard: {e}")
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0:
            self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        starts = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[s:s+self.tps-1] for s in starts])
        tgt = np.stack([self.data[s+1:s+self.tps] for s in starts])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# ============ WEIGHTS ============
def decompress_weights(raw, fmt="bf16"):
    if len(raw) >= 2 and raw[0] == 0x78:
        raw = zlib.decompress(raw)
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()

def parse_safetensors(filepath):
    with open(filepath, 'rb') as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("File too small to be a safetensors file")
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        if header_size > 100 * 1024 * 1024:
            raise ValueError(f"Suspicious header size: {header_size}")
        header = json.loads(f.read(header_size).decode('utf-8'))
        tensor_key = 'weights' if 'weights' in header else next((k for k in header if not k.startswith('__')), None)
        if tensor_key is None:
            raise ValueError("No tensor found in safetensors file")
        info = header[tensor_key]
        start, end = info['data_offsets']
        f.seek(8 + header_size + start)
        raw_bytes = f.read(end - start)
        if info['dtype'] == 'F32':
            weights = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        elif info['dtype'] == 'BF16':
            weights = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()
        elif info['dtype'] == 'F16':
            weights = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).copy()
        else:
            raise ValueError(f"Unsupported safetensors dtype: {info['dtype']}")
        return weights

def download_from_huggingface(precision="bf16"):
    if LOCAL_FP32_CACHE.exists():
        if LOCAL_FP32_CACHE.stat().st_size == EXPECTED_MODEL_SIZE * 4:
            console.print(f"[cyan]📦 Loading cached weights from {LOCAL_FP32_CACHE.name}...[/cyan]")
            try:
                weights = np.fromfile(LOCAL_FP32_CACHE, dtype=np.float32, count=EXPECTED_MODEL_SIZE)
                if len(weights) == EXPECTED_MODEL_SIZE:
                    console.print(f"[green]✅ Loaded {len(weights):,} parameters from cache[/green]")
                    return weights
            except Exception as e:
                console.print(f"[yellow]⚠ Cache read failed: {e}[/yellow]")
                LOCAL_FP32_CACHE.unlink(missing_ok=True)

    url = HF_WEIGHTS_URL_BF16 if precision == "bf16" else HF_WEIGHTS_URL_FP32
    console.print(f"[cyan]🤗 Downloading weights from HuggingFace ({precision})...[/cyan]")
    try:
        r = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(LOCAL_SAFETENSORS_CACHE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(f"\r  ↓ {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB ({downloaded/total*100:.0f}%)", end="", flush=True)
        console.print("")
        weights = parse_safetensors(LOCAL_SAFETENSORS_CACHE)
        if len(weights) != EXPECTED_MODEL_SIZE:
            raise ValueError(f"Weight count mismatch: got {len(weights):,}, expected {EXPECTED_MODEL_SIZE:,}")
        weights.astype(np.float32).tofile(LOCAL_FP32_CACHE)
        console.print(f"[green]✅ Downloaded {len(weights):,} parameters from HuggingFace[/green]")
        LOCAL_SAFETENSORS_CACHE.unlink(missing_ok=True)
        return weights
    except Exception as e:
        console.print(f"[yellow]⚠ HF weight loading failed: {e}[/yellow]")
        LOCAL_SAFETENSORS_CACHE.unlink(missing_ok=True)
        LOCAL_FP32_CACHE.unlink(missing_ok=True)
        return None

# ============ AUTH ============
def authenticate(server_url, username, password):
    if not username or not password: return None
    try:
        r = requests.post(f"{server_url}/auth/login", json={"username": username, "password": password}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token")
        r = requests.post(f"{server_url}/auth/register", json={"username": username, "password": password, "email": f"{username}@crowdgpt.local"}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token")
        console.print(f"[yellow]⚠ Auth failed: {r.text[:100]}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]⚠ Auth request failed: {e}[/yellow]")
        return None

# ============ TRAINING ============
def create_dashboard(step, total_steps, loss, tps, lr, global_step, backend_name, batch_size, seq_len, weight_source="server"):
    table = Table(title=f"🧠 Training Dashboard (Crowd-v1) [{weight_source}]", expand=True, border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right", style="magenta")
    table.add_row("📈 Progress", f"{step}/{total_steps} ({step/max(1,total_steps)*100:.1f}%)")
    table.add_row("📉 Loss", f"{loss:.4f}" if loss > 0 else "—")
    table.add_row("⚡ Tokens/s", f"{tps:.0f}" if tps else "—")
    table.add_row("🎛️ LR", f"{lr:.2e}" if lr else "—")
    table.add_row("🌍 Global Step", str(global_step))
    table.add_row("🖥️ Backend", backend_name)
    table.add_row("📦 Batch / SeqLen", f"{batch_size} / {seq_len}")
    return table

def run_single_contribution(args, session_count, auth_token=None, force_hf=False):
    global train_device, train_backend

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    use_hf = force_hf or (session_count == 1 and not LOCAL_FP32_CACHE.exists())
    hf_weights = None
    weight_source = "server"

    if use_hf:
        hf_weights = download_from_huggingface(args.precision)
        if hf_weights is not None:
            weight_source = "huggingface"

    skip_weights = hf_weights is not None
    task_url = f"{args.server}/fl/task?mode={args.mode}&format={args.precision}"
    if skip_weights:
        task_url += "&skip_weights=true"

    console.print(f"[cyan]📡 Requesting shard{' (weights from HF)' if skip_weights else ''}...[/cyan]")
    task_res = requests.get(task_url, headers=headers, timeout=600)
    if task_res.status_code != 200:
        raise Exception(f"Coordinator rejected: {task_res.text[:200]}")

    raw = task_res.content
    ml = struct.unpack('<I', raw[:4])[0]
    metadata = json.loads(raw[4:4+ml].decode('utf-8'))
    weights_bytes = None if skip_weights else raw[4+ml:]

    train_cfg = metadata.get("trainingConfig", {})
    task_id = metadata['taskId']
    global_step = metadata['globalStep']
    weight_format = metadata.get('weightFormat', 'bf16')
    seq_len = args.seq_len if args.seq_len > 0 else 2048
    steps = args.steps if args.steps > 0 else train_cfg.get("localSteps", 500)

    ds_cfg = metadata.get("datasetConfig", {})
    shard_cfg = metadata.get("shardConfig", {})

    dataset_shard = StreamingShardDataset(
        repo_id=ds_cfg.get("repoId", DATASET_REPO_ID),
        chunk_idx=ds_cfg.get("chunkIdx", 0),
        sub_size=ds_cfg.get("subChunkSize", 10*1024*1024),
        tps=ds_cfg.get("tokensPerSample", seq_len + 1),
        slot=shard_cfg.get("slot", 0),
        auth_token=auth_token,
    )
    console.print(f"[cyan]📚 Streaming chunk {dataset_shard.ci} via Range (10MB slices, prefetched)[/cyan]")

    if hf_weights is not None:
        console.print(f"[cyan]📦 Using weights from HuggingFace[/cyan]")
        initial_weights = hf_weights
    else:
        console.print(f"[cyan]📦 Unpacking weights from coordinator...[/cyan]")
        initial_weights = decompress_weights(weights_bytes, weight_format)
    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        try:
            import bitsandbytes
            use_8bit = True
        except ImportError:
            use_8bit = False
            
        lora_rank, mode, best_bs, gpu_layers = recommend_optimal_config(args.precision, seq_len, use_8bit)
        batch_size = best_bs
        console.print(f"[green]✅ Auto-tuned config:[/green] Mode: [bold]{mode}[/bold] | BS: [bold]{batch_size}[/bold] for SeqLen: [bold]{seq_len}[/bold]")

    console.print(f"[cyan]🧠 Initializing SotaGPT...[/cyan]")
    model = SotaGPT().to(train_device)
    model.load_flat_weights(initial_weights)
    model.train()
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
        console.print("[green]✅ Using 8-bit AdamW (bitsandbytes) to save ~5GB VRAM[/green]")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
        console.print("[yellow]⚠ bitsandbytes not installed. Using standard AdamW.[/yellow]")
        console.print("[yellow]  💡 Tip: Run `pip install bitsandbytes` to prevent OOM on <16GB GPUs![/yellow]")
    
    use_autocast = False
    autocast_dtype = None
    if args.precision == "bf16" and train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported():
        use_autocast = True
        autocast_dtype = torch.bfloat16
    elif args.precision == "fp16" and train_backend in ("CUDA", "ROCM"):
        use_autocast = True
        autocast_dtype = torch.float16

    loss_history = []
    seed = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16) + global_step

    console.print(f"[bold green]🚀 Training ({steps} steps, BS={batch_size}, SeqLen={seq_len})[/bold green]")

    with Live(create_dashboard(0, steps, 0, 0, args.lr, global_step, train_backend, batch_size, seq_len, weight_source),
              console=console, refresh_per_second=4, screen=False) as live:
        t0 = time.time()
        total_tok = 0

        for step in range(1, steps+1):
            if dataset_shard.needs_new_subchunk():
                log.info(f"🔄 Advancing data slice at step {step}")
                dataset_shard.advance(args.server, args.mode, args.precision)

            x, y = dataset_shard.get_batch(batch_size, seed=hash("shard")%10000 + step)
            x, y = x.to(train_device), y.to(train_device)

            try:
                
                def forward_pass():
                    x_emb = model.wte(x)
                    for b in model.blocks:
                        if model.training:
                            # Checkpointing drops activation VRAM by 80%, ain't that cool?
                            x_emb = checkpoint(b, x_emb, model.freqs_cos, model.freqs_sin, True)
                        else:
                            x_emb = b(x_emb, model.freqs_cos, model.freqs_sin, True)
                    return model.ln_f(x_emb)

                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        x_emb = forward_pass()
                        
                        
                        loss = 0.0
                        loss_chunk_size = 256 
                        num_chunks = 0
                        for i in range(0, seq_len, loss_chunk_size):
                            chunk_logits = model.lm_head(x_emb[:, i:i+loss_chunk_size, :])
                            chunk_target = y[:, i:i+loss_chunk_size]
                            loss += F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_target.reshape(-1))
                            num_chunks += 1
                        loss = loss / max(1, num_chunks)
                        loss.backward()
                else:
                    x_emb = forward_pass()
                    
                    loss = 0.0
                    loss_chunk_size = 256
                    num_chunks = 0
                    for i in range(0, seq_len, loss_chunk_size):
                        chunk_logits = model.lm_head(x_emb[:, i:i+loss_chunk_size, :])
                        chunk_target = y[:, i:i+loss_chunk_size]
                        loss += F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_target.reshape(-1))
                        num_chunks += 1
                    loss = loss / max(1, num_chunks)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                lv = float(loss.item())
                if math.isnan(lv) or lv <= 0:
                    lv = 10.0

                total_tok += x.numel()
                elapsed = time.time()-t0
                loss_history.append(lv)
                live.update(create_dashboard(step, steps, lv, int(total_tok/max(elapsed,0.01)), optimizer.param_groups[0]['lr'], global_step, train_backend, batch_size, seq_len, weight_source))
            except Exception as e:
                console.print(f"[bold red]❌ Error: {e}[/bold red]")
                break

    avg_loss = sum(loss_history)/max(1,len(loss_history))
    delta = model.get_flat_weights() - initial_weights

    payload = json.dumps({
        "taskId": task_id,
        "loss": float(avg_loss),
        "localSteps": steps,
        "tokensProcessed": total_tok,
        "loraRank": 0,
        "isDelta": True,
        "weightFormat": "fp32"
    }).encode()

    binary = struct.pack('<I', len(payload)) + payload + np.ascontiguousarray(delta.astype(np.float32)).tobytes()
    compressed = gzip.compress(binary, compresslevel=2)

    console.print(f"[cyan]📤 Seeding delta ({len(compressed)/1024/1024:.2f} MB)...[/cyan]")
    r = requests.post(
        f"{args.server}/fl/submit",
        headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **headers},
        data=compressed,
        timeout=120
    )

    if r.status_code == 200:
        console.print(f"[bold green]✅ Seeded successfully![/bold green]")
    else:
        raise Exception(f"Seed failed: {r.text[:300]}")

    gc.collect()
    if train_backend in ("CUDA", "ROCM"):
        try:
            torch.cuda.empty_cache()
        except:
            pass

def run_swarm_node(args):
    global train_device, train_backend
    print_welcome()

    auth_token = None
    username = args.username or os.environ.get("CROWDGPT_USERNAME")
    password = args.password or os.environ.get("CROWDGPT_PASSWORD")

    if username and password:
        console.print(f"[cyan]🔐 Authenticating as {username}...[/cyan]")
        auth_token = authenticate(args.server, username, password)
        if auth_token:
            console.print(f"[green]✅ Authenticated successfully[/green]")
        else:
            console.print(f"[yellow]⚠ Running anonymously (auth failed)[/yellow]")
    else:
        console.print(f"[dim]Running anonymously (set CROWDGPT_USERNAME/PASSWORD or use --username/--password)[/dim]")

    try:
        train_device, train_backend = detect_training_backend(args.backend)
        console.print(f"[green]✅ Backend:[/green] [bold]{train_backend}[/bold] ({train_device})")
        if train_backend == "CPU":
            console.print("\n[yellow]⚠ Training on CPU will be extremely slow.[/yellow]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Backend error: {e}[/bold red]")
        sys.exit(1)

    auto_detect_vram_budget()
    console.print(f"[green]💾 VRAM/RAM Budget:[/green] [bold]{memory_config['ram_gb']:.1f} GB[/bold] usable")

    session_count = 0
    while True:
        session_count += 1
        console.print(f"\n[bold cyan]🔄 Cycle #{session_count}[/bold cyan]")
        try:
            force_hf = args.from_hf or (session_count == 1)
            run_single_contribution(args, session_count, auth_token, force_hf=force_hf)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[bold red]❌ Cycle failed: {e}[/bold red]")
            time.sleep(10)
            continue

        if args.single:
            console.print("[green]✅ Done.[/green]")
            break
        time.sleep(3)

def main():
    parser = argparse.ArgumentParser(description="CrowdGPT distributed training client")
    parser.add_argument("--server", default="http://api.crowdgpt.net:5006", help="Coordinator URL")
    parser.add_argument("--backend", default="auto", choices=["auto","cuda","rocm","mps","xpu","directml","cpu"], help="Training backend")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0=auto)")
    parser.add_argument("--seq-len", type=int, default=0, help="Override sequence length (0=auto)")
    parser.add_argument("--steps", type=int, default=0, help="Override local steps (0=auto)")
    parser.add_argument("--mode", default="quick", choices=["quick","balanced","deep","ultra"], help="Training mode")
    parser.add_argument("--precision", default="bf16", choices=["fp32","bf16","fp16"], help="Weight precision")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--username", help="Username for leaderboard (or set CROWDGPT_USERNAME)")
    parser.add_argument("--password", help="Password for leaderboard (or set CROWDGPT_PASSWORD)")
    parser.add_argument("--from-hf", action="store_true", help="Force download weights from HuggingFace every cycle")
    parser.add_argument("--single", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()
    run_swarm_node(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Disconnected[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal: {e}[/red]")
