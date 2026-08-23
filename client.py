#!/usr/bin/env python3
"""
CrowdGPT Client - Distributed LLM Training Node

Contributes local GPU compute to train Crowd-v1 (500M parameter transformer).
Fetches model weights from HuggingFace (first run) or coordinator, trains on data
shards, submits weight deltas back to coordinator.
"""

import sys
import io
import os
import gzip
import zlib
import json
import time
import struct
import logging
import hashlib
import math
import gc
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

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
DATASET_URL = "https://huggingface.co/datasets/Vxtzq/CrowdGPT/resolve/main/tinystories_tokens.bin"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace sources (cuts server bandwidth for new clients)
HF_REPO_ID = "Vxtzq/Crowd-v1"
HF_WEIGHTS_URL_BF16 = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/model_bf16.safetensors"
HF_WEIGHTS_URL_FP32 = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/model_fp32.safetensors"
LOCAL_SAFETENSORS_CACHE = CHECKPOINT_DIR / "hf_weights_bf16.safetensors"
LOCAL_FP32_CACHE = CHECKPOINT_DIR / "hf_weights_fp32.bin"

# Crowd-v1: 500M parameter architecture
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

# Calculate expected parameter count for validation
def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2  # ln_1
        size += DIM * (N_HEADS * HEAD_DIM)  # wq
        size += DIM * (N_KV_HEADS * HEAD_DIM)  # wk
        size += DIM * (N_KV_HEADS * HEAD_DIM)  # wv
        size += DIM * DIM  # wo
        size += DIM * 2  # ln_2
        size += DIM * MLP_HIDDEN  # w1
        size += DIM * MLP_HIDDEN  # w2
        size += MLP_HIDDEN * DIM  # w3
    size += DIM * 2  # ln_f
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()

ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
ALLOWED_SEQ_LENS = [8, 16, 32, 64]

memory_config = {"ram_gb": 12, "safety_margin_gb": 1.0, "is_auto_detected": False, "precision": "bf16"}
train_device, train_backend = None, None

BANNER = """
[bold cyan]CrowdGPT[/bold cyan] [dim]::[/dim] [magenta]Distributed LLM Training Node (Crowd-v1)[/magenta]
[dim]═══════════════════════════════════════════════════════════════[/dim]
"""

def print_welcome():
    console.print(BANNER)

# ============ CUDA AUTO-DETECTION & HINTS ============
def check_cuda_installed():
    """Check if CUDA is actually installed on the system (not just if PyTorch sees it)"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

def suggest_cuda_install():
    """Print clear CUDA installation instructions"""
    console.print("\n[bold red]⚠ PyTorch does NOT detect your GPU![/bold red]")
    console.print("[yellow]NVIDIA GPU detected but CUDA-enabled PyTorch not installed.[/yellow]\n")
    console.print("[bold]Fix with one of these commands:[/bold]\n")
    console.print("[cyan]# For CUDA 12.1 (recommended):[/cyan]")
    console.print("[white]pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/white]\n")
    console.print("[cyan]# For CUDA 11.8 (older GPUs):[/cyan]")
    console.print("[white]pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118[/white]\n")
    console.print("[dim]Or check requirements.txt includes the correct torch version for your system.[/dim]\n")

# ============ HARDWARE & MEMORY ============
def check_backend_available(name):
    name = name.lower()
    if name == "cpu":
        return True, torch.device('cpu'), "CPU"
    if name == "cuda":
        if torch.cuda.is_available():
            return True, torch.device('cuda'), f"CUDA: {torch.cuda.get_device_name(0)}"
        return False, None, "CUDA not available"
    return False, None, f"Unknown: {name}"

def detect_training_backend(force=None):
    if force and force != "auto":
        ok, dev, info = check_backend_available(force.lower())
        if not ok:
            raise Exception(f"'{force}': {info}")
        return dev, force.upper()
    
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA"
    
    if check_cuda_installed():
        suggest_cuda_install()
    
    return torch.device('cpu'), "CPU"

def auto_detect_vram_budget():
    global train_backend
    if train_backend == "CUDA" and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            usable = max(0.0, (free_b / 1024**3) - memory_config["safety_margin_gb"])
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True})
            return
        except Exception:
            pass
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        memory_config.update({"ram_gb": round(max(1.0, avail - 2.0), 2), "is_auto_detected": True})
    except ImportError:
        pass

def estimate_vram_bytes(precision="bf16", batch_size=4, seq_len=64):
    """Realistic memory estimation for full pretrain on GPU"""
    w = 2 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else 4
    
    vocab_b = VOCAB_SIZE * DIM * w
    layer_b = (DIM*2 + DIM*(N_HEADS*HEAD_DIM) + DIM*(N_KV_HEADS*HEAD_DIM)*2 + DIM*DIM + DIM*2 + DIM*MLP_HIDDEN*2 + MLP_HIDDEN*DIM) * w
    model_b = vocab_b + (layer_b * N_LAYERS)
    
    optim_grad_b = model_b * 3
    
    act_per_layer = 20 * batch_size * seq_len * DIM * w
    act_b = act_per_layer * N_LAYERS
    
    logits_b = batch_size * seq_len * VOCAB_SIZE * 4
    
    total_raw = model_b + optim_grad_b + act_b + logits_b
    return int(total_raw * 1.1) + int(0.2 * 1024**3)

def recommend_optimal_config(precision="bf16", seq_len=64):
    """Prioritizes full pretrain with all layers on GPU"""
    budget = int(memory_config["ram_gb"] * 1024**3)
    safe = budget - int(0.5 * 1024**3)
    
    best_bs = 4
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(precision, bs, seq_len) <= safe:
            best_bs = bs
        else:
            break
            
    return 0, "full_pretrain", best_bs, N_LAYERS

# ============ MODEL ARCHITECTURE (Crowd-v1) ============
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

    def forward(self, x, cos, sin):
        B,T,C = x.size()
        q = self.wq(x).view(B,T,self.nh,HEAD_DIM).transpose(1,2)
        k = self.wk(x).view(B,T,self.nkv,HEAD_DIM).transpose(1,2)
        v = self.wv(x).view(B,T,self.nkv,HEAD_DIM).transpose(1,2)
        ct, st = cos[:,:,:T,:], sin[:,:,:T,:]
        q = q*ct + rotate_half(q)*st
        k = k*ct + rotate_half(k)*st
        k = k.unsqueeze(2).expand(B,self.nkv,self.nrep,T,HEAD_DIM).reshape(B,self.nh,T,HEAD_DIM)
        v = v.unsqueeze(2).expand(B,self.nkv,self.nrep,T,HEAD_DIM).reshape(B,self.nh,T,HEAD_DIM)
        a = (q@k.transpose(-2,-1))*(1.0/math.sqrt(HEAD_DIM))
        m = torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)
        a = a.masked_fill(m==0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a@v).transpose(1,2).contiguous().view(B,T,C))

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
    
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
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

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, self.freqs_cos, self.freqs_sin)
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

# ============ DATASET & UTILS ============
def _fetch_with_retry(url, headers=None, timeout=60, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise

class StreamingShardDataset:
    """
    Streams dataset chunks sequentially, starting from the coordinator-assigned chunk.
    Downloads the next chunk automatically when the current one runs out.
    Never loops within a chunk; wraps only after a full epoch over all chunks.
    """
    def __init__(self, ds_cfg, seq_len=64):
        self.url = ds_cfg.get("url", DATASET_URL)
        self.chunk_size = ds_cfg.get("chunkSize", 10 * 1024 * 1024)
        self.total_chunks = max(1, ds_cfg.get("totalChunks", 1))
        self.seq_len = seq_len
        self.tps = ds_cfg.get("tokensPerSample", seq_len + 1)
        self.chunk_idx = None
        self.data = None
        self.n = 0
        self.cursor = 0
        self.chunks_consumed = 0
        self.epochs = 0
        self._load_chunk(ds_cfg.get("chunkIdx", 0) % self.total_chunks)

    def _load_chunk(self, idx):
        offset = idx * self.chunk_size
        r = _fetch_with_retry(self.url, headers={'Range': f'bytes={offset}-{offset + self.chunk_size - 1}'}, timeout=120)
        tk = np.frombuffer(r.content, dtype=np.uint16)
        self.data = tk
        self.n = len(tk) // self.tps
        self.cursor = 0
        self.chunk_idx = idx

    def _next_chunk(self):
        nxt = self.chunk_idx + 1
        if nxt >= self.total_chunks:
            nxt = 0
            self.epochs += 1
        self._load_chunk(nxt)
        self.chunks_consumed += 1
        log.info(f"📥 streamed next chunk {nxt}/{self.total_chunks} (epoch {self.epochs})")

    def get_batch(self, bs, seed=None):
        inp, tgt = [], []
        for _ in range(bs):
            guard = 0
            while self.cursor >= self.n:
                self._next_chunk()
                guard += 1
                if guard > self.total_chunks:
                    raise RuntimeError("Dataset stream exhausted (empty chunks)")
            s = self.cursor * self.tps
            inp.append(self.data[s:s + self.seq_len])
            tgt.append(self.data[s + 1:s + self.seq_len + 1])
            self.cursor += 1
        return torch.tensor(np.array(inp), dtype=torch.long), torch.tensor(np.array(tgt), dtype=torch.long)

def decompress_weights(raw, fmt="bf16"):
    if len(raw) >= 2 and raw[0] == 0x78:
        raw = zlib.decompress(raw)
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()

# ============ HUGGINGFACE WEIGHT LOADING ============
def parse_safetensors(filepath):
    """
    Minimal safetensors parser — no external library required.
    Extracts the 'weights' tensor (or first tensor) as a flat fp32 numpy array.
    
    Format: [8 bytes: uint64 header_size][header_size bytes: JSON][raw tensor data]
    JSON contains: {"weights": {"dtype": "BF16", "shape": [N], "data_offsets": [start, end]}, ...}
    """
    with open(filepath, 'rb') as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("File too small to be a safetensors file")
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        if header_size > 100 * 1024 * 1024:  # sanity check: header shouldn't be >100MB
            raise ValueError(f"Suspicious header size: {header_size}")
        
        header_json = f.read(header_size)
        header = json.loads(header_json.decode('utf-8'))
        
        # Find the tensor key (skip __metadata__)
        tensor_key = None
        if 'weights' in header:
            tensor_key = 'weights'
        else:
            for key in header:
                if not key.startswith('__'):
                    tensor_key = key
                    break
        
        if tensor_key is None:
            raise ValueError("No tensor found in safetensors file")
        
        tensor_info = header[tensor_key]
        dtype = tensor_info['dtype']
        shape = tensor_info['shape']
        data_offsets = tensor_info['data_offsets']
        start, end = data_offsets
        
        # Seek to tensor data: 8 (header size field) + header_size + start offset
        f.seek(8 + header_size + start)
        raw_bytes = f.read(end - start)
        
        expected_bytes = end - start
        if len(raw_bytes) != expected_bytes:
            raise ValueError(f"Read {len(raw_bytes)} bytes, expected {expected_bytes}")
        
        total_elements = 1
        for s in shape:
            total_elements *= s
        
        if dtype == 'F32':
            weights = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        elif dtype == 'BF16':
            weights = torch.from_numpy(
                np.frombuffer(raw_bytes, dtype=np.uint16).copy()
            ).view(torch.bfloat16).to(torch.float32).numpy()
        elif dtype == 'F16':
            weights = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).copy()
        else:
            raise ValueError(f"Unsupported safetensors dtype: {dtype}")
        
        if len(weights) != total_elements:
            raise ValueError(f"Element count mismatch: {len(weights)} vs {total_elements}")
        
        return weights

def download_from_huggingface(precision="bf16"):
    """
    Download model weights from HuggingFace, parse safetensors, return fp32 numpy array.
    Caches the parsed fp32 weights locally for fast subsequent loads.
    Returns None on failure.
    """
    # Check for cached fp32 weights first (fast path)
    if LOCAL_FP32_CACHE.exists():
        file_size = LOCAL_FP32_CACHE.stat().st_size
        expected_size = EXPECTED_MODEL_SIZE * 4  # fp32 = 4 bytes per param
        if file_size == expected_size:
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
    console.print(f"[dim]   Source: {HF_REPO_ID}[/dim]")
    
    try:
        r = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        r.raise_for_status()
        
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(LOCAL_SAFETENSORS_CACHE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total / 1024 / 1024
                        pct = downloaded / total * 100
                        # Use rich's print for in-place update
                        print(f"\r  ↓ {mb_done:.1f}MB / {mb_total:.1f}MB ({pct:.0f}%)", end="", flush=True)
        
        console.print("")  # newline after progress
        
        console.print(f"[cyan]📦 Parsing safetensors...[/cyan]")
        weights = parse_safetensors(LOCAL_SAFETENSORS_CACHE)
        
        if len(weights) != EXPECTED_MODEL_SIZE:
            raise ValueError(f"Weight count mismatch: got {len(weights):,}, expected {EXPECTED_MODEL_SIZE:,}")
        
        # Cache as raw fp32 for instant loading next time
        console.print(f"[cyan]💾 Caching fp32 weights for fast reload...[/cyan]")
        weights.astype(np.float32).tofile(LOCAL_FP32_CACHE)
        
        console.print(f"[green]✅ Downloaded {len(weights):,} parameters from HuggingFace[/green]")
        
        # Cleanup the safetensors file (we have the fp32 cache now)
        LOCAL_SAFETENSORS_CACHE.unlink(missing_ok=True)
        
        return weights
        
    except requests.exceptions.RequestException as e:
        console.print(f"[yellow]⚠ HF download failed: {e}[/yellow]")
        LOCAL_SAFETENSORS_CACHE.unlink(missing_ok=True)
        return None
    except Exception as e:
        console.print(f"[yellow]⚠ HF weight loading failed: {e}[/yellow]")
        LOCAL_SAFETENSORS_CACHE.unlink(missing_ok=True)
        LOCAL_FP32_CACHE.unlink(missing_ok=True)
        return None

# ============ AUTHENTICATION ============
def authenticate(server_url, username, password):
    """
    Single auth endpoint: registers if new, logs in if existing.
    Returns auth token or None on failure.
    """
    if not username or not password:
        return None
    
    try:
        r = requests.post(
            f"{server_url}/auth/login",
            json={"username": username, "password": password},
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            return data.get("token")
        
        # If login failed, try register
        r = requests.post(
            f"{server_url}/auth/register",
            json={"username": username, "password": password, "email": f"{username}@crowdgpt.local"},
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            return data.get("token")
        
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
    """
    Run one training cycle.
    
    Weight source priority:
      1. If force_hf or first run: try HuggingFace → call /fl/task?skip_weights=true
      2. Fallback: fetch weights from coordinator via /fl/task (full payload)
    """
    global train_device, train_backend

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # Determine if we should try HF weights
    use_hf = force_hf or (session_count == 1 and not LOCAL_FP32_CACHE.exists())
    hf_weights = None
    weight_source = "server"
    
    if use_hf:
        hf_weights = download_from_huggingface(args.precision)
        if hf_weights is not None:
            weight_source = "huggingface"

    # Request task from coordinator
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
    
    if skip_weights:
        # Weights came from HF, no weight bytes in response
        weights_bytes = None
    else:
        weights_bytes = raw[4+ml:]

    train_cfg = metadata.get("trainingConfig", {})
    task_id = metadata['taskId']
    global_step = metadata['globalStep']
    weight_format = metadata.get('weightFormat', 'bf16')

    batch_size = args.batch_size if args.batch_size > 0 else train_cfg.get("batchSize", 0)
    seq_len = args.seq_len if args.seq_len > 0 else train_cfg.get("seqLen", 64)
    steps = args.steps if args.steps > 0 else train_cfg.get("localSteps", 500)

    ds_cfg = metadata.get("datasetConfig", {})
    dataset_shard = StreamingShardDataset(ds_cfg, seq_len)
    console.print(f"[cyan]📚 Streaming from chunk {dataset_shard.chunk_idx}/{dataset_shard.total_chunks}[/cyan]")

    # Get initial weights
    if hf_weights is not None:
        console.print(f"[cyan]📦 Using weights from HuggingFace[/cyan]")
        initial_weights = hf_weights
    else:
        console.print(f"[cyan]📦 Unpacking weights from coordinator...[/cyan]")
        initial_weights = decompress_weights(weights_bytes, weight_format)

    # Auto-tuning logic
    if batch_size == 0:
        lora_rank, mode, best_bs, gpu_layers = recommend_optimal_config(args.precision, seq_len)
        batch_size = best_bs
        console.print(f"[green]✅ Optimized config:[/green] Mode: [bold]{mode}[/bold] | BS: [bold]{batch_size}[/bold] | GPU Layers: [bold]{gpu_layers}/{N_LAYERS}[/bold]")
    else:
        lora_rank = 0

    console.print(f"[cyan]🧠 Initializing SotaGPT...[/cyan]")
    model = SotaGPT().to(train_device)
    model.load_flat_weights(initial_weights)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    
    use_autocast = (args.precision == "bf16" and train_backend == "CUDA" and torch.cuda.is_bf16_supported())
    autocast_dtype = torch.bfloat16 if use_autocast else None

    loss_history = []
    seed = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16) + global_step

    console.print(f"[bold green]🚀 Training ({steps} steps, BS={batch_size}, SeqLen={seq_len})[/bold green]")

    with Live(create_dashboard(0, steps, 0, 0, args.lr, global_step, train_backend, batch_size, seq_len, weight_source),
              console=console, refresh_per_second=4, screen=False) as live:
        t0 = time.time()
        total_tok = 0

        for step in range(1, steps+1):
            x, y = dataset_shard.get_batch(batch_size, seed=hash("shard")%10000 + step)
            x, y = x.to(train_device), y.to(train_device)

            try:
                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
                        loss.backward()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                lv = float(loss.item())
                if math.isnan(lv) or lv <= 0:
                    lv = 10.0
                
                total_tok += x.numel()
                elapsed = time.time()-t0
                current_tps = int(total_tok/max(elapsed,0.01))
                loss_history.append(lv)

                live.update(create_dashboard(step, steps, lv, current_tps, optimizer.param_groups[0]['lr'], global_step, train_backend, batch_size, seq_len, weight_source))
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
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "gzip",
            **headers
        },
        data=compressed,
        timeout=120
    )
    
    if r.status_code == 200:
        console.print(f"[bold green]✅ Seeded successfully![/bold green]")
    else:
        raise Exception(f"Seed failed: {r.text[:300]}")

    gc.collect()
    if train_backend == "CUDA":
        torch.cuda.empty_cache()

def run_swarm_node(args):
    global train_device, train_backend
    print_welcome()

    # Authentication
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
            console.print("\n[yellow]⚠ Training on CPU will be extremely slow.[/yellow]")
            console.print("[dim]Consider installing CUDA-enabled PyTorch for GPU acceleration.[/dim]\n")
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
            # Force HF on first cycle or when --from-hf flag is set
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
    parser.add_argument("--server", default="https://api.crowdgpt.net", help="Coordinator URL")
    parser.add_argument("--backend", default="auto", choices=["auto","cuda","cpu"], help="Training backend")
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
