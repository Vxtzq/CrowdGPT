#!/usr/bin/env python3
"""
CrowdGPT GUI Client — dual-logo support (light + dark mode SVGs).
"""
import os, sys, io, json, time, struct, math, gc, threading, base64, logging, tempfile, atexit
from pathlib import Path
import webview
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.checkpoint import checkpoint
import http.client
from requests.exceptions import ChunkedEncodingError
import gzip

logging.getLogger("pywebview").setLevel(logging.CRITICAL)

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ============ CONFIG ============
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_REPO_ID = "Vxtzq/CrowdGPT"

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]; DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]; N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]; HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]; MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865
LOSS_CHUNK = 256; UPLOAD_BUFFER_MIN = 25; TPS_DEGRADATION = 0.85
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM*2 + DIM*(N_HEADS*HEAD_DIM) + DIM*(N_KV_HEADS*HEAD_DIM)*2 + DIM*DIM + DIM*2
        size += DIM*MLP_HIDDEN*2 + MLP_HIDDEN*DIM
    size += DIM*2
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()
EXPECTED_ENGRAM_SIZE = ENG_NUM_BUCKETS * DIM
EXPECTED_WEIGHT_BYTES = (EXPECTED_MODEL_SIZE + EXPECTED_ENGRAM_SIZE) * 2
memory_config = {"ram_gb": 12, "is_auto_detected": False}
train_device, train_backend = None, None

# ============ DUAL LOGOS ============
LOGO_URI_LIGHT = ""
LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_app = Path(__file__).parent / "docs/logo-app.svg"
_lp_dark = Path(__file__).parent / "docs/logo-white.svg"

if _lp_light.exists():
    try: LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except: pass
if _lp_dark.exists():
    try: LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except: pass
# Fallback: if no dark SVG exists, reuse the light one
if not LOGO_URI_DARK:
    LOGO_URI_DARK = LOGO_URI_LIGHT

# ============ APP ICON ============
ICON_PATH = None
_cleanup_icon = []
if _lp_light.exists():
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(_lp_app), output_width=256, output_height=256)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(png_bytes)
        tmp.close()
        ICON_PATH = os.path.abspath(tmp.name)
        _cleanup_icon.append(ICON_PATH)
        def _clean():
            for p in _cleanup_icon:
                try: os.unlink(p)
                except: pass
        atexit.register(_clean)
    except ImportError:
        ICON_PATH = os.path.abspath(str(_lp_app))
    except Exception:
        ICON_PATH = os.path.abspath(str(_lp_app))

# ============ HARDWARE ============
def get_available_backends():
    backends = {}
    backends['cuda'] = torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip)
    backends['rocm'] = hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available()
    backends['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    try:
        import intel_extension_for_pytorch
        backends['xpu'] = hasattr(torch, 'xpu') and torch.xpu.is_available()
    except ImportError:
        backends['xpu'] = False
    try:
        import torch_directml
        backends['directml'] = torch_directml.is_available()
    except ImportError:
        backends['directml'] = False
    backends['cpu'] = True
    return backends

def get_best_default_backend():
    avail = get_available_backends()
    for bk in ['cuda', 'rocm', 'mps', 'xpu', 'directml', 'cpu']:
        if avail.get(bk): return bk
    return 'cpu'

def detect_training_backend(force=None):
    if force and force != "auto":
        n = force.lower()
        if n == "cpu": return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return torch.device('cuda'), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
            except ImportError: pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
            except ImportError: pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip): return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return torch.device('cuda'), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
    except ImportError: pass
    try:
        import torch_directml
        if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
    except ImportError: pass
    return torch.device('cpu'), "CPU"

def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            fb, tb = torch.cuda.mem_get_info(0)
            memory_config.update({"ram_gb": round(max(1.0, min(fb/1024**3, tb/1024**3 - 2.0) - 1.0), 2), "is_auto_detected": True}); return
        except: pass
    if train_backend == "MPS":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024**3
            memory_config.update({"ram_gb": round(max(1.0, total_gb * 0.75 - 2.0), 2), "is_auto_detected": True}); return
        except: pass
    try:
        import psutil
        memory_config.update({"ram_gb": round(max(1.0, psutil.virtual_memory().available/1024**3 - 2.0), 2), "is_auto_detected": True})
    except: pass

def has_bitsandbytes():
    try: import bitsandbytes; return True
    except: return False

def estimate_vram_bytes(bs, sl, u8):
    mb = EXPECTED_MODEL_SIZE*2; ob = EXPECTED_MODEL_SIZE*(4 if u8 else 12); gb = EXPECTED_MODEL_SIZE*4
    ab = N_LAYERS*(bs*sl*DIM*2 + bs*N_HEADS*64*sl*4)
    lb = bs*LOSS_CHUNK*VOCAB_SIZE*4; eb = bs*sl*DIM*2
    return int(mb+ob+gb+ab+lb+eb+0.8*1024**3)

def recommend_batch_size(sl=2048):
    u8 = has_bitsandbytes(); budget = int(memory_config["ram_gb"]*1024**3); best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget: best = bs
        else: break
    return best

# ============ MODEL ============
def precompute_freqs(dim, sl, dev):
    inv = 1.0/(10000.0**(torch.arange(0, dim, 2, dtype=torch.float32)/dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)

def rotate_half(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.nrep = N_HEADS, N_KV_HEADS, N_HEADS//N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS*HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
    def forward(self, x, cos, sin, use_chunked=True):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        ct, st = cos[:, :, :T, :], sin[:, :, :T, :]
        q = q*ct + rotate_half(q)*st; k = k*ct + rotate_half(k)*st
        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        if use_chunked: return self._chunked(q, k, v, B, T, C)
        a = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(HEAD_DIM))
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))
    def _chunked(self, q, k, v, B, T, C, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i+cs, T)
            aw = (q[:, :, i:e, :] @ k[:, :, :e, :].transpose(-2, -1))*(1.0/math.sqrt(HEAD_DIM))
            ri = torch.arange(i, e, device=q.device).unsqueeze(1)
            ci = torch.arange(e, device=q.device).unsqueeze(0)
            aw = aw.masked_fill(~(ci <= ri).unsqueeze(0).unsqueeze(0), float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v[:, :, :e, :])
        return self.wo(torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x))*self.w2(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1, self.attn, self.ln_2, self.mlp = nn.LayerNorm(DIM), GroupedQueryAttention(), nn.LayerNorm(DIM), SwiGLU()
    def forward(self, x, cos, sin, use_chunked=True):
        x = x + self.attn(self.ln_1(x), cos, sin, use_chunked)
        return x + self.mlp(self.ln_2(x))

class EngramMemory(nn.Module):
    def __init__(self, bn, device):
        super().__init__()
        self.device = device
        self.table = nn.Embedding(ENG_NUM_BUCKETS, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.use_async = device.type in ('cuda',) and torch.cuda.is_available()
        self.ts = torch.cuda.Stream(device=device) if self.use_async else None
    def forward(self, idx):
        px = torch.cat([torch.zeros_like(idx[:, :1]), idx[:, :-1]], dim=1)
        hi = (px*1000003 + idx) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.flatten(), return_inverse=True)
        cr = self.table(uq.cpu())
        if self.use_async:
            with torch.cuda.stream(self.ts): cg = cr.to(self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.ts)
        else: cg = cr.to(self.device)
        return cg[inv].view(idx.shape[0], idx.shape[1], DIM)

class SotaGPT(nn.Module):
    def __init__(self, bn, device):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory(bn, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]: self.wte.weight = self.lm_head.weight
        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos", cm); self.register_buffer("freqs_sin", sm)
    def get_base_weights(self):
        return np.concatenate([p.detach().float().flatten().cpu().numpy() for n, p in self.named_parameters() if not n.startswith('engram.')])
    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = p.numel(); p.data.copy_(ft[o:o+s].view(p.shape).to(p.device)); o += s

def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True); r.raise_for_status(); return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries-1: time.sleep(2**(a+1))
            else: raise

class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500
    def __init__(self, repo_id, ci, ss=10*1024*1024, tps=65, slot=0, at=None):
        self.repo_id, self.ci, self.sub_size, self.tps, self.auth_token = repo_id, ci, ss, tps, at
        self._fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover(ci)
        self.off = (slot*ss) % self.chunk_size
        self.data, self.n, self.steps_used = None, 0, 0
        self._pft, self._pfr, self._lock = None, None, threading.Lock()
        self._load(self.off); self._prefetch(self.off+ss)
    def _url(self): return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._fmt.format(self.ci)
    def _discover(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        sz = int(r.headers.get('content-length', 0))
                        if sz > 0: self._fmt = fmt; return sz
                except: time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")
    def _slice(self, off):
        end = min(off+self.sub_size, self.chunk_size)-1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content
    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32); self.data, self.n, self.steps_used = tk, len(tk)//self.tps, 0
    def _load(self, off): self._set(self._slice(off)); self.off = off
    def _prefetch(self, off):
        if off >= self.chunk_size: return
        with self._lock: self._pfr = None
        def w():
            try:
                r = self._slice(off)
                with self._lock: self._pfr = (off, r)
            except: pass
        self._pft = threading.Thread(target=w, daemon=True); self._pft.start()
    def needs_new_subchunk(self): return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK
    def advance(self, srv=None, fmt="bf16"):
        no = self.off+self.sub_size
        if no < self.chunk_size:
            if self._pft: self._pft.join(timeout=120); self._pft = None
            with self._lock: r = self._pfr; self._pfr = None
            if r and r[0] == no: self.off = no; self._set(r[1])
            else: self._load(no)
            self._prefetch(no+self.sub_size); return True
        self._req(srv, fmt)
        self.chunk_size = self._discover(self.ci); self.off = 0
        self._load(0); self._prefetch(self.sub_size); return True
    def _req(self, srv, fmt="bf16"):
        if not srv: return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers=h, timeout=30)
            if r.status_code == 200:
                raw = r.content; ml = struct.unpack('<I', raw[:4])[0]
                ni = json.loads(raw[4:4+ml].decode()).get("datasetConfig", {}).get("chunkIdx", self.ci)
                if ni != self.ci: self.ci = ni; return True
        except: pass
        return False
    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0: self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        s = rng.randint(0, self.n, size=bs)*self.tps
        inp = np.stack([self.data[x:x+self.tps-1] for x in s])
        tgt = np.stack([self.data[x+1:x+self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16": return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()

def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        try: msg = r.json().get("detail", "")
        except: msg = r.text
        if "invalid" in msg.lower() or "credentials" in msg.lower(): return None, "invalid_credentials"
        return None, "login_failed"
    except requests.exceptions.ConnectionError: return None, "network"
    except: return None, "network"

def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        try: msg = r.json().get("detail", "")
        except: msg = r.text
        msg_l = msg.lower()
        if "username" in msg_l and "taken" in msg_l: return None, "username_taken"
        if "email" in msg_l and "registered" in msg_l: return None, "email_taken"
        if "password" in msg_l: return None, "weak_password"
        if "invalid" in msg_l: return None, "invalid_input"
        return None, "register_failed"
    except requests.exceptions.ConnectionError: return None, "network"
    except: return None, "network"

class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv, self.h, self.cr = srv, h, cr
        self.stop_training = threading.Event(); self.shutdown = threading.Event()
        self.se, self.emit = se, emit
        self._t = threading.Thread(target=self._r, daemon=True)
    def start(self): self._t.start()
    def stop(self): self.shutdown.set()
    def should_stop(self): return self.stop_training.is_set()
    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10)
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr:
                        self.emit('log', "Next round. Stopping."); self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0))*60
                    if rm <= 3:
                        self.emit('log', f"ULTIMATUM: {rm:.0f} min left!"); self.stop_training.set()
            except: pass
            self.shutdown.wait(timeout=15)

def wait_for_round(srv, h, se, emit):
    overlay_on = False
    def show_wait():
        nonlocal overlay_on
        if not overlay_on:
            emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True}); overlay_on = True
    def hide():
        nonlocal overlay_on
        if overlay_on:
            emit('overlay_hide', {}); overlay_on = False
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting'); show_wait()
                    for _ in range(30):
                        if se.is_set(): hide(); return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False):
                    hide(); return st
                emit('status', 'waiting'); show_wait()
                for _ in range(30):
                    if se.is_set(): hide(); return None
                    time.sleep(1)
            else: time.sleep(10)
        except: time.sleep(10)
    hide(); return None

def fetch_task(srv, h, prec, se, emit):
    while not se.is_set():
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})
            r = requests.get(f"{srv}/fl/task?format={prec}", headers=h, timeout=(15, 3600), stream=True)
            if r.headers.get("X-Status") == "wait":
                r.close(); emit('overlay_hide', {})
                for _ in range(10):
                    if se.is_set(): return None, None
                    time.sleep(1)
                continue
            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []; done = 0; last = 0.0
            for chunk in r.iter_content(chunk_size=512*1024):
                if not chunk: continue
                chunks.append(chunk); done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total}); last = now
            r.close()
            emit('overlay_progress', {'done': done, 'total': total})
            emit('overlay_hide', {})
            emit('status', 'preparing')
            raw = b"".join(chunks); del chunks
            ml = struct.unpack('<I', raw[:4])[0]
            return json.loads(raw[4:4+ml].decode()), raw[4+ml:]
        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set(): return None, None
            emit('log', f"Task fetch failed: {e}"); time.sleep(15)

def _chunk_ce(m, xs, ys): return F.cross_entropy(m.lm_head(xs).reshape(-1, VOCAB_SIZE), ys.reshape(-1))

def _fwl(m, x, y, sl, ua, ad, ls=1.0):
    def fwd():
        xe = m.wte(x)
        if ua:
            with torch.autocast(device_type='cuda', enabled=False): eo = m.engram(x)
        else: eo = m.engram(x)
        xe = xe + eo.to(xe.dtype)
        for b in m.blocks: xe = checkpoint(b, xe, m.freqs_cos, m.freqs_sin, True, use_reentrant=False)
        return m.ln_f(xe)
    nc = max(1, math.ceil(sl/LOSS_CHUNK))
    if ua:
        with torch.autocast(device_type='cuda', dtype=ad):
            xe = fwd()
            lo = sum(checkpoint(_chunk_ce, m, xe[:, i:i+LOSS_CHUNK, :], y[:, i:i+LOSS_CHUNK], use_reentrant=False) for i in range(0, sl, LOSS_CHUNK))/nc
    else:
        xe = fwd()
        lo = sum(checkpoint(_chunk_ce, m, xe[:, i:i+LOSS_CHUNK, :], y[:, i:i+LOSS_CHUNK], use_reentrant=False) for i in range(0, sl, LOSS_CHUNK))/nc
    return lo*ls

def run_single_round(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting')
    rs = wait_for_round(srv, h, se, emit)
    if not rs: return
    cr = rs["current_round"]
    hb = HeartbeatManager(srv, h, cr, se, emit); hb.start()
    try:
        emit('log', "Downloading weights...")
        md, wb = fetch_task(srv, h, "bf16", se, emit)
        if md is None: return
        
        emit('status', 'preparing')
        emit('log', "Preparing model...")
        
        tid, gs = md['taskId'], md['globalStep']
        sl = 2048
        bbl = EXPECTED_MODEL_SIZE*2
        iw = decompress_weights(wb[:bbl], "bf16"); ie = decompress_weights(wb[bbl:], "bf16")
        del wb; gc.collect()
        dc = md.get("datasetConfig", {}); sc = md.get("shardConfig", {})
        ds = StreamingShardDataset(dc.get("repoId", DATASET_REPO_ID), dc.get("chunkIdx", 0),
            dc.get("subChunkSize", 10*1024*1024), dc.get("tokensPerSample", sl+1), sc.get("slot", 0), at)
        bs = recommend_batch_size(sl)
        model = None
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except: pass
                gc.collect()
                model = SotaGPT(train_backend, train_device).to(train_device)
                model.engram.table.to('cpu'); model.load_base_weights(iw)
                model.engram.table.weight.data.copy_(torch.from_numpy(ie).view(model.engram.table.weight.shape))
                model.train(); break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if model: del model; model = None
                    if train_backend in ("CUDA", "ROCM"):
                        try: torch.cuda.empty_cache()
                        except: pass
                    gc.collect(); bs = max(1, bs//2)
                else: raise
        mbt = bs*sl; as_ = max(1, round(131072/mbt)); ls = 1.0/as_
        def ha():
            nonlocal as_, ls
            as_ = max(1, as_//2); ls = 1.0/as_
            ob.zero_grad(set_to_none=True); gc.collect()
            if train_backend in ("CUDA", "ROCM"):
                try: torch.cuda.empty_cache()
                except: pass
            emit('log', f"OOM -> accum now {as_}x")
        bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
        ep = list(model.engram.parameters())
        try:
            import bitsandbytes as bnb
            ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        except: ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        try: oe = torch.optim.SparseAdam(ep, lr=1e-4)
        except: oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)
        iew = model.engram.table.weight.data.cpu().clone()
        ua = train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported()
        ad = torch.bfloat16 if ua else None
        emit('log', "Calibrating speed...")
        tt = 0; cs_ = time.time()
        micro_step = 0
        cal_global_steps = 3
        target_micro_steps = 100
        emit('status', 'calibrating')
        cal_done = 0
        while cal_done < cal_global_steps:
            if hb.should_stop() or se.is_set(): return
            ob.zero_grad(set_to_none=True)
            try:
                for mi in range(as_):
                    if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                    sd = (hash("cal") % 10000) + cal_done*1000 + mi
                    x, y = ds.get_batch(bs, seed=sd); x, y = x.to(train_device), y.to(train_device)
                    lo = _fwl(model, x, y, sl, ua, ad, ls)
                    lval = float(lo.item())
                    micro_step += 1
                    if cal_done >= 1 and mi == 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh*3600) - elapsed - (UPLOAD_BUFFER_MIN*60))
                        target_micro_steps = max(micro_step + 100, int(remaining / sps_micro))
                    
                    elapsed_cal = time.time() - cs_
                    current_tps = tt / max(elapsed_cal, 1.0)
                    
                    emit('cal_stats', {
                        'step': micro_step, 'total': target_micro_steps,
                        'loss': lval * as_, 'microbatch': micro_step,
                        'tps': current_tps
                    })
                    lo.backward()
                    oe.step(); oe.zero_grad(set_to_none=True)
                    tt += x.numel(); del lo, x, y
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                ha(); continue
            torch.nn.utils.clip_grad_norm_(bp, 1.0); ob.step(); ob.zero_grad(set_to_none=True)
            gc.collect()
            cal_done += 1
        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
        remaining = max(60, (rh*3600) - (time.time() - cs_) - (UPLOAD_BUFFER_MIN*60))
        dl = time.time() + remaining
        emit('status', 'training')
        emit('log', f"Target: ~{target_micro_steps} micro-steps")
        last_stats_emit = 0.0
        while micro_step < target_micro_steps:
            if hb.should_stop() or se.is_set(): break
            if time.time() >= dl: break
            ob.zero_grad(set_to_none=True)
            als = 0.0; cc = 0
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set() or time.time() >= dl: break
                    if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                    sd = (hash("t") % 10000) + micro_step*1000 + mi
                    x, y = ds.get_batch(bs, seed=sd); x, y = x.to(train_device), y.to(train_device)
                    lo = _fwl(model, x, y, sl, ua, ad, ls)
                    lval = float(lo.item())*as_
                    emit('preview_loss', lval)
                    lo.backward()
                    oe.step(); oe.zero_grad(set_to_none=True)
                    als += lval; tt += x.numel(); cc += 1
                    micro_step += 1
                    if time.time() - last_stats_emit > 0.25 or micro_step >= target_micro_steps:
                        last_stats_emit = time.time()
                        lv = als/cc
                        ctps = tt/max(time.time()-cs_, 1)
                        elapsed_total = time.time() - cs_
                        if micro_step > 50 and elapsed_total > 30:
                            sps_micro = elapsed_total / micro_step
                            rem_time = max(0, dl - time.time())
                            target_micro_steps = max(micro_step + 50, int(rem_time / sps_micro))
                        emit('stats', {
                            'step': micro_step, 'target': target_micro_steps, 'loss': lval, 
                            'tps': ctps, 'global_step': gs, 'round': cr, 
                            'time_left': max(0, (dl-time.time())/60)
                        })
                    del lo, x, y
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                ha(); continue
            if cc == 0: break
            torch.nn.utils.clip_grad_norm_(bp, 1.0); ob.step(); ob.zero_grad(set_to_none=True)
    finally:
        hb.stop()
        if 'ds' in locals() and ds: del ds
        gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try: torch.cuda.empty_cache()
            except: pass
    if 'micro_step' not in locals() or micro_step <= 0: return
    fl = float(lval) if 'lval' in locals() else 10.0
    emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}")
    emit('status', 'uploading')
    dbf = torch.from_numpy(model.get_base_weights()-iw).to(torch.bfloat16).view(torch.uint16).numpy()
    ed = model.engram.table.weight.data.cpu()-iew
    rn = ed.abs().sum(dim=1); k = max(1, int(len(rn)*0.10))
    tv, ti = torch.topk(rn, k); ai = ti[tv > 1e-8]
    si = ai.cpu().numpy().astype(np.uint32) if len(ai) else np.array([], dtype=np.uint32)
    sv = ed[ai].to(torch.bfloat16).view(torch.uint16).numpy() if len(ai) else np.array([], dtype=np.uint16)
    pl = json.dumps({"taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt, "loraRank": 0, "isDelta": True, "weightFormat": "bf16", "hasEngram": True, "engramSparseCount": len(si)}).encode()
    bi = struct.pack('<I', len(pl)) + pl + np.ascontiguousarray(dbf).tobytes() + np.ascontiguousarray(si).tobytes() + np.ascontiguousarray(sv).tobytes()
    try:
        r = requests.post(f"{srv}/fl/submit", headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h}, data=gzip.compress(bi, compresslevel=2), timeout=600)
        emit('log', "Submitted!" if r.status_code == 200 else f"Submit failed: {r.text[:100]}")
    except Exception as e: emit('log', f"Upload failed: {e}")
    del model; gc.collect()


HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#fafaf9; --bg-soft:#ffffff; --bg-softer:#f0f0ee;
  --text:#111113; --text-dim:#3a3a40; --text-muted:#6b6b73;
  --border:#e4e4e1; --border-strong:#d0d0cb;
  --accent:#16a34a; --accent-dim:#15803d;
  --err:#dc2626; --err-soft:rgba(220,38,38,.06); --err-border:rgba(220,38,38,.28);
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --r:6px;
}
body.dark{
  --bg:#1a1a1e; --bg-soft:#242428; --bg-softer:#2e2e33;
  --text:#e8e8ec; --text-dim:#b8b8be; --text-muted:#888890;
  --border:#3a3a40; --border-strong:#505058;
  --accent:#22c55e; --accent-dim:#4ade80;
  --err:#f87171; --err-soft:rgba(248,113,113,.08); --err-border:rgba(248,113,113,.35);
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;width:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);display:flex;flex-direction:column;overflow:hidden;font-size:14px;line-height:1.5}
.mono{font-family:var(--mono)}

header{flex:0 0 auto;height:54px;border-bottom:1px solid var(--border);background:var(--bg);display:flex;align-items:center;justify-content:space-between;padding:0 clamp(12px,2vw,20px);gap:10px}
.logo{display:flex;align-items:center;gap:9px;min-width:0}
.logo-img{height:22px;width:auto}
.logo-fallback{width:24px;height:24px;border:1px solid var(--border-strong);display:none;place-items:center;font-size:13px;font-weight:600}
.logo-text{font-size:15px;font-weight:600;white-space:nowrap}
.logo-ver{font-family:var(--mono);font-size:10px;color:var(--text-muted);border:1px solid var(--border);padding:1px 5px;border-radius:3px}
.header-right{display:flex;align-items:center;gap:8px}
#user-info{font-family:var(--mono);font-size:11px;color:var(--text-muted);max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.theme-toggle{
  width:30px;height:30px;flex:0 0 auto;
  display:flex;align-items:center;justify-content:center;
  background:var(--bg-soft);color:var(--text);
  border:1px solid var(--border);border-radius:4px;
  cursor:pointer;outline:none;font-size:14px;line-height:1;padding:0;
}
.theme-toggle:hover{border-color:var(--border-strong)}
.theme-toggle .icon-sun{display:none}
.theme-toggle .icon-moon{display:inline}
body.dark .theme-toggle .icon-sun{display:inline}
body.dark .theme-toggle .icon-moon{display:none}

body:not(.dashboard-active) #backend-select { display: none; }

.backend-select, .lang-select{
  font-family:var(--sans);font-size:12px;
  background:var(--bg-soft);color:var(--text);
  border:1px solid var(--border);
  padding:5px 22px 5px 8px;border-radius:4px;
  cursor:pointer;outline:none;appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'><path d='M0 0l4 5 4-5z' fill='%236b6b73'/></svg>");
  background-repeat:no-repeat;background-position:right 7px center;background-size:8px;
}
body.dark .backend-select, body.dark .lang-select{
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'><path d='M0 0l4 5 4-5z' fill='%23888890'/></svg>");
}
.backend-select:hover, .lang-select:hover{border-color:var(--border-strong)}
.backend-select:focus, .lang-select:focus{border-color:var(--accent)}
.backend-select:disabled{opacity:.5;cursor:not-allowed}
.backend-select option:disabled{color:var(--text-muted);opacity:.5}

main{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;padding:clamp(10px,2vw,18px) clamp(12px,2vw,20px)}
.view{display:none;height:100%;min-height:0}
.view.active{display:flex;flex-direction:column}

#view-login.active{flex:1 1 auto;align-items:center;justify-content:center;padding:10px 0;min-height:0}
.login-box{
  width: min(380px, 100%); max-height: 100%; overflow-y: auto;
  background: var(--bg-soft); border: 1px solid var(--border);
  padding: clamp(14px, 2.5vw, 24px); border-radius: 8px;
  display: flex; flex-direction: column;
  scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent;
}
.login-box::-webkit-scrollbar { width: 4px; }
.login-box::-webkit-scrollbar-track { background: transparent; }
.login-box::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 2px; }

.auth-screen{display:none;flex-direction:column;gap:0}
.auth-screen.active{display:flex}

.login-box h2{font-size:clamp(17px,3vw,21px);font-weight:600;text-align:center;margin-bottom:3px}
.login-box h2 .em{color:var(--accent);font-weight:700}
.login-sub{font-size:12px;color:var(--text-dim);text-align:center;margin-bottom:14px}
.fg{margin-bottom:9px}
.fg label{display:block;font-size:9px;color:var(--text-muted);margin-bottom:3px;font-family:var(--mono);text-transform:uppercase;letter-spacing:.07em}
input{width:100%;font-family:inherit;font-size:12.5px;background:var(--bg-softer);color:var(--text);border:1px solid var(--border);padding:8px 10px;border-radius:4px;outline:none;transition:border-color .15s}
input:focus{border-color:var(--accent)}
input.err{border-color:var(--err)}

button{font-family:inherit;font-size:12.5px;font-weight:500;cursor:pointer;border-radius:4px;transition:all .15s}
.btn-primary{background:var(--text);color:var(--bg);border:1px solid var(--text);padding:9px 14px}
.btn-primary:hover:not(:disabled){opacity:.85}
.btn-danger{background:transparent;color:var(--err);border:1px solid var(--err);padding:9px 14px}
.btn-danger:hover:not(:disabled){background:var(--err);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
.auth-btn{width:100%;margin-top:2px}

.error-box{
  display:none;margin-top:8px;padding:8px 10px 8px 12px;
  background:var(--err-soft);border:1px solid var(--err-border);border-left:3px solid var(--err);
  border-radius:4px;font-size:12px;color:var(--err);line-height:1.4;
}
.error-box.show{display:block}

.auth-switch{margin-top:12px;font-size:12px;color:var(--text-muted);text-align:center}
.auth-switch a{color:var(--accent-dim);font-weight:500;cursor:pointer;text-decoration:none;border-bottom:1px solid transparent;transition:border-color .15s}
.auth-switch a:hover{border-bottom-color:var(--accent-dim)}

.anon-hint{margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--text-muted);text-align:center;line-height:1.4}
.anon-hint a{color:var(--text-dim);cursor:pointer;text-decoration:none;border-bottom:1px dotted var(--border-strong)}
.anon-hint a:hover{color:var(--text);border-bottom-color:var(--text)}

.dash{display:flex;flex-direction:column;gap:10px;flex:1 1 auto;min-height:0}
.dash-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex:0 0 auto}
.dash-title{font-size:17px;font-weight:600}
.status-badge{font-family:var(--mono);font-size:10px;padding:3px 10px;border-radius:100px;background:rgba(22,163,74,.08);color:var(--accent-dim);border:1px solid rgba(22,163,74,.35);white-space:nowrap}
body.dark .status-badge{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4)}

.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;flex:0 0 auto}
.stat-card{background:var(--bg-soft);border:1px solid var(--border);padding:12px 14px;border-radius:var(--r);min-width:0}
.stat-val{font-family:var(--mono);font-size:clamp(17px,2.4vw,22px);font-weight:500;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat-val.accent{color:var(--accent-dim)}
.stat-lbl{font-size:11px;color:var(--text-muted);margin-top:2px}

.panel{background:var(--bg-soft);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px;flex:0 0 auto}
.panel-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;font-family:var(--mono);font-size:11px;color:var(--text-muted);flex-wrap:wrap}
.panel-head .pv{color:var(--text)}
.progress-track{height:6px;background:var(--bg-softer);border:1px solid var(--border);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;width:0%;background:var(--accent);transition:width .4s ease}

#loss-graph{display:block;width:100%;height:64px}
.graph-empty{color:var(--text-muted);font-family:var(--mono);font-size:11px;text-align:center;padding:20px 0}

.status-strip{
  flex:0 0 auto;
  font-family:var(--mono);font-size:11px;color:var(--text-muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  padding: 4px 0;
}
.ss-event{display:block;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.controls{flex:0 0 auto;display:flex;gap:8px;justify-content:flex-end;align-items:center;flex-wrap:wrap}

.overlay{position:fixed;inset:0;background:rgba(250,250,249,.86);backdrop-filter:blur(3px);display:flex;align-items:center;justify-content:center;z-index:50;opacity:0;pointer-events:none;transition:opacity .25s}
body.dark .overlay{background:rgba(26,26,30,.86)}
.overlay.show{opacity:1;pointer-events:auto}
.overlay-card{width:min(420px,88vw);background:var(--bg-soft);border:1px solid var(--border);border-radius:8px;padding:22px 24px;box-shadow:0 8px 28px rgba(17,17,19,.08)}
body.dark .overlay-card{box-shadow:0 8px 28px rgba(0,0,0,.35)}
.overlay-title{font-size:15px;font-weight:600;margin-bottom:4px}
.overlay-subtitle{font-size:12.5px;color:var(--text-muted);margin-bottom:14px;line-height:1.5;display:none}
.overlay-subtitle.show{display:block}
.overlay-bar{height:10px;background:var(--bg-softer);border:1px solid var(--border);border-radius:5px;overflow:hidden;position:relative}
.overlay-bar.hidden{display:none}
.overlay-fill{height:100%;width:0%;background:linear-gradient(90deg,#15803d,#16a34a 55%,#4ade80);position:relative;overflow:hidden;transition:width .25s ease}
body.dark .overlay-fill{background:linear-gradient(90deg,#15803d,#22c55e 55%,#86efac)}
.overlay-fill::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);animation:shimmer 1.6s linear infinite}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.overlay-bar.indet .overlay-fill{width:35%;transition:none;animation:indet 1.15s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.overlay-sub{margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--text-muted);display:flex;justify-content:space-between;gap:8px}

.overlay-card .overlay-msg{font-size:13px;color:var(--text-dim);line-height:1.5;margin-bottom:16px}
.overlay-card .overlay-actions{display:flex;gap:8px;justify-content:flex-end}
.overlay-card .overlay-actions button{min-width:90px;justify-content:center}

@media (max-width:520px){
  header{height:auto;flex-wrap:wrap;padding:8px 12px}
  #user-info{display:none}
  .controls button{flex:1}
}

@media (max-height: 560px){
  main { padding: 8px 12px; }
  .login-box { padding: 12px; }
  .login-box h2 { font-size: 16px; margin-bottom: 2px; }
  .login-sub { font-size: 11px; margin-bottom: 10px; }
  .fg { margin-bottom: 7px; }
  .fg label { margin-bottom: 2px; font-size: 8px; }
  input { padding: 7px 9px; font-size: 12px; }
  .auth-btn { padding: 8px 12px; margin-top: 1px; font-size: 12px; }
  .auth-switch { margin-top: 8px; font-size: 11px; }
  .anon-hint { margin-top: 8px; padding-top: 8px; font-size: 10px; }
  .error-box { font-size: 11px; padding: 6px 8px 6px 10px; }
  #loss-graph{height:48px}
  .stat-card{padding:8px 10px}
  .stat-val{font-size:15px}
  .panel{padding:8px 10px}
  .graph-empty{padding:12px 0}
}
</style></head>
<body>
<header>
  <div class="logo">
    <img src="__LOGO__" id="logo-img" data-light="__LOGO__" data-dark="__LOGO_DARK__" class="logo-img" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='grid';">
    <div class="logo-fallback">C</div>
    <span class="logo-text">CrowdGPT</span>
    <span class="logo-ver">v0.4</span>
  </div>
  <div class="header-right">
    <span id="user-info"></span>
    <select class="backend-select" id="backend-select" disabled></select>
    <select class="lang-select" id="lang-select">
      <option value="en">🇬🇧 English</option>
      <option value="fr">🇫🇷 Français</option>
      <option value="es">🇪🇸 Español</option>
      <option value="de">🇩🇪 Deutsch</option>
      <option value="tr">🇹🇷 Türkçe</option>
    </select>
    <button class="theme-toggle" id="theme-toggle" title="Toggle dark mode"><span class="icon-moon">🌙</span><span class="icon-sun">☀️</span></button>
  </div>
</header>

<main>
  <div id="view-login" class="view active">
    <div class="login-box">
      <div class="auth-screen active" id="screen-login">
        <h2><span data-i18n="login_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="login_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="login-server" value="http://api.crowdgpt.net:5006"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="login-user" data-i18n-ph="ph_user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="login-pass" autocomplete="current-password"></div>
        <div class="error-box" id="login-error"></div>
        <button id="btn-login" class="btn-primary auth-btn" data-i18n="login_btn"></button>
        <div class="auth-switch">
          <span data-i18n="no_account"></span>
          <a id="go-to-register" data-i18n="register_link"></a>
        </div>
        <div class="anon-hint">
          <span data-i18n="anon_preface"></span>
          <a id="btn-anon" data-i18n="anon_link"></a>
        </div>
      </div>

      <div class="auth-screen" id="screen-register">
        <h2><span data-i18n="register_title_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="register_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="reg-server" value="http://api.crowdgpt.net:5006"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="reg-user" data-i18n-ph="ph_user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="email"></label><input type="email" id="reg-email" autocomplete="email"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="reg-pass" autocomplete="new-password"></div>
        <div class="error-box" id="reg-error"></div>
        <button id="btn-register" class="btn-primary auth-btn" data-i18n="register_btn"></button>
        <div class="auth-switch">
          <span data-i18n="have_account"></span>
          <a id="go-to-login" data-i18n="login_link"></a>
        </div>
      </div>
    </div>
  </div>

  <div id="view-dashboard" class="view">
    <div class="dash">
      <div class="dash-head">
        <span class="dash-title" data-i18n="dash_title"></span>
        <span class="status-badge" id="status-indicator"></span>
      </div>

      <div class="stats-grid">
        <div class="stat-card"><div class="stat-val accent" id="stat-loss">—</div><div class="stat-lbl" data-i18n="loss"></div></div>
        <div class="stat-card"><div class="stat-val" id="stat-tps">—</div><div class="stat-lbl" data-i18n="tokens_sec"></div></div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <span><span data-i18n="round"></span> <span class="pv" id="stat-round">—</span> · <span data-i18n="step"></span> <span class="pv" id="stat-step-cur">0</span>/<span class="pv" id="stat-step-tot">0</span></span>
          <span class="pv" id="progress-text">0%</span>
        </div>
        <div class="progress-track"><div class="progress-fill" id="progress-bar"></div></div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <span data-i18n="loss_history"></span>
          <span class="pv" id="graph-last"></span>
        </div>
        <canvas id="loss-graph"></canvas>
        <div class="graph-empty" id="graph-empty" data-i18n="waiting_data"></div>
      </div>

      <div class="status-strip">
        <span class="ss-event" id="status-event"></span>
      </div>

      <div class="controls">
        <button id="btn-start" class="btn-primary api-btn" data-i18n="start"></button>
        <button id="btn-stop" class="btn-danger api-btn" data-i18n="stop" disabled></button>
      </div>
    </div>
  </div>
</main>

<div class="overlay" id="overlay">
  <div class="overlay-card">
    <div class="overlay-title" id="overlay-title"></div>
    <div class="overlay-subtitle" id="overlay-subtitle"></div>
    <div class="overlay-bar" id="overlay-bar"><div class="overlay-fill" id="overlay-fill"></div></div>
    <div class="overlay-sub"><span id="overlay-pct"></span><span id="overlay-mb"></span></div>
  </div>
</div>

<div class="overlay" id="backend-overlay">
  <div class="overlay-card">
    <div class="overlay-title" data-i18n="backend_change_title"></div>
    <div class="overlay-msg" data-i18n="backend_change_msg"></div>
    <div class="overlay-actions">
      <button id="btn-backend-cancel" class="btn-danger" data-i18n="cancel"></button>
      <button id="btn-backend-confirm" class="btn-primary" data-i18n="confirm"></button>
    </div>
  </div>
</div>

<script>
const I18N={
 en:{
   login_pre:"Sign in to ",login_sub:"Welcome back to the swarm.",
   register_title_pre:"Join ",register_sub:"Create your account and start contributing.",
   server_url:"Server address",username:"Username",password:"Password",email:"Email",
   ph_user:" ",
   login_btn:"Log in",register_btn:"Create account",
   no_account:"No account yet?",register_link:"Create one",
   have_account:"Already registered?",login_link:"Log in",
   anon_preface:"Prefer to stay anonymous?",anon_link:"Skip and contribute anonymously",
   dash_title:"Training",loss:"Loss",tokens_sec:"Tokens/sec",round:"Round",step:"Step",
   loss_history:"Loss history",waiting_data:"Waiting for training data…",time_left:"Left",
   idle:"Idle",training:"Training…",waiting:"Waiting…",uploading:"Uploading…",
   downloading:"Downloading…",preparing:"Preparing…",calibrating:"Calibrating…",
   dl_weights:"Downloading weights…",wait_coord:"Waiting for coordinator…",
   wait_coord_sub:"This can take a few minutes. The coordinator aggregates updates from all clients between rounds.",
   start:"Start training",stop:"Stop",
   backend_change_title:"Switch backend?",
   backend_change_msg:"Switching backend will reset your current training session and restart the round. Your contribution so far will be discarded. Continue?",
   cancel:"Cancel",confirm:"Switch & restart",
   err_invalid_credentials:"Invalid username or password.",
   err_username_taken:"This username is already taken. Please choose another.",
   err_email_taken:"An account with this email already exists.",
   err_weak_password:"Password must be at least 6 characters.",
   err_invalid_input:"Please check your input and try again.",
   err_login_failed:"Could not log in. Please try again.",
   err_register_failed:"Could not create account. Please try again.",
   err_network:"Cannot reach the server."
 },
 fr:{
   login_pre:"Connexion à ",login_sub:"Bon retour parmi nous.",
   register_title_pre:"Rejoignez ",register_sub:"Créez votre compte et commencez à contribuer.",
   server_url:"Adresse du serveur",username:"Nom d'utilisateur",password:"Mot de passe",email:"E-mail",
   ph_user:" ",
   login_btn:"Se connecter",register_btn:"Créer le compte",
   no_account:"Pas encore de compte ?",register_link:"Créer un compte",
   have_account:"Déjà inscrit ?",login_link:"Se connecter",
   anon_preface:"Vous préférez rester anonyme ?",anon_link:"Continuer anonymement",
   dash_title:"Entraînement",loss:"Perte",tokens_sec:"Tokens/sec",round:"Manche",step:"Étape",
   loss_history:"Historique de perte",waiting_data:"En attente de données…",time_left:"Reste",
   idle:"Inactif",training:"Entraînement…",waiting:"Attente…",uploading:"Envoi…",
   downloading:"Téléchargement…",preparing:"Préparation…",calibrating:"Calibration…",
   dl_weights:"Téléchargement des poids…",wait_coord:"En attente du coordinateur…",
   wait_coord_sub:"Cela peut prendre quelques minutes. Le coordinateur agrège les mises à jour de tous les clients entre les manches.",
   start:"Démarrer l'entraînement",stop:"Arrêter",
   backend_change_title:"Changer de backend ?",
   backend_change_msg:"Changer de backend réinitialisera votre session d'entraînement actuelle. Votre contribution en cours sera perdue. Continuer ?",
   cancel:"Annuler",confirm:"Changer & redémarrer",
   err_invalid_credentials:"Nom d'utilisateur ou mot de passe invalide.",
   err_username_taken:"Ce nom d'utilisateur est déjà pris.",
   err_email_taken:"Un compte avec cet e-mail existe déjà.",
   err_weak_password:"Le mot de passe doit contenir au moins 6 caractères.",
   err_invalid_input:"Veuillez vérifier vos informations.",
   err_login_failed:"Connexion impossible. Réessayez.",
   err_register_failed:"Création du compte impossible. Réessayez.",
   err_network:"Serveur injoignable."
 },
 es:{
   login_pre:"Inicia sesión en ",login_sub:"Bienvenido de vuelta al enjambre.",
   register_title_pre:"Únete a ",register_sub:"Crea tu cuenta y empieza a contribuir.",
   server_url:"Dirección del servidor",username:"Nombre de usuario",password:"Contraseña",email:"Correo",
   ph_user:" ",
   login_btn:"Iniciar sesión",register_btn:"Crear cuenta",
   no_account:"¿Aún no tienes cuenta?",register_link:"Crear una",
   have_account:"¿Ya estás registrado?",login_link:"Iniciar sesión",
   anon_preface:"¿Prefieres seguir anónimo?",anon_link:"Continuar anónimamente",
   dash_title:"Entrenamiento",loss:"Pérdida",tokens_sec:"Tokens/seg",round:"Ronda",step:"Paso",
   loss_history:"Historial de pérdida",waiting_data:"Esperando datos…",time_left:"Queda",
   idle:"Inactivo",training:"Entrenando…",waiting:"Esperando…",uploading:"Subiendo…",
   downloading:"Descargando…",preparing:"Preparando…",calibrating:"Calibrando…",
   dl_weights:"Descargando pesos…",wait_coord:"Esperando al coordinador…",
   wait_coord_sub:"Esto puede tardar unos minutos. El coordinador agrega las actualizaciones de todos los clientes entre rondas.",
   start:"Iniciar entrenamiento",stop:"Detener",
   backend_change_title:"¿Cambiar de backend?",
   backend_change_msg:"Cambiar de backend reiniciará tu sesión de entrenamiento actual. Tu contribución actual se descartará. ¿Continuar?",
   cancel:"Cancelar",confirm:"Cambiar y reiniciar",
   err_invalid_credentials:"Usuario o contraseña incorrectos.",
   err_username_taken:"Este nombre de usuario ya está en uso.",
   err_email_taken:"Ya existe una cuenta con este correo.",
   err_weak_password:"La contraseña debe tener al menos 6 caracteres.",
   err_invalid_input:"Revisa los datos e inténtalo de nuevo.",
   err_login_failed:"No se pudo iniciar sesión. Inténtalo de nuevo.",
   err_register_failed:"No se pudo crear la cuenta. Inténtalo de nuevo.",
   err_network:"No se puede contactar el servidor."
 },
 de:{
   login_pre:"Anmelden bei ",login_sub:"Willkommen zurück im Schwarm.",
   register_title_pre:"Tritt ",register_sub:"Erstelle dein Konto und trage bei.",
   server_url:"Serveradresse",username:"Benutzername",password:"Passwort",email:"E-Mail",
   ph_user:" ",
   login_btn:"Anmelden",register_btn:"Konto erstellen",
   no_account:"Noch kein Konto?",register_link:"Jetzt erstellen",
   have_account:"Bereits registriert?",login_link:"Anmelden",
   anon_preface:"Lieber anonym bleiben?",anon_link:"Anonym weitermachen",
   dash_title:"Training",loss:"Verlust",tokens_sec:"Tokens/Sek",round:"Runde",step:"Schritt",
   loss_history:"Verlustverlauf",waiting_data:"Warte auf Trainingsdaten…",time_left:"Rest",
   idle:"Bereit",training:"Training…",waiting:"Warten…",uploading:"Upload…",
   downloading:"Wird heruntergeladen…",preparing:"Vorbereitung…",calibrating:"Kalibrierung…",
   dl_weights:"Gewichte werden heruntergeladen…",wait_coord:"Warte auf Koordinator…",
   wait_coord_sub:"Das kann ein paar Minuten dauern. Der Koordinator aggregiert die Updates aller Clients zwischen den Runden.",
   start:"Training starten",stop:"Stopp",
   backend_change_title:"Backend wechseln?",
   backend_change_msg:"Das Wechseln des Backends setzt deine aktuelle Trainingssitzung zurück. Dein bisheriger Beitrag wird verworfen. Fortfahren?",
   cancel:"Abbrechen",confirm:"Wechseln & neu starten",
   err_invalid_credentials:"Benutzername oder Passwort ungültig.",
   err_username_taken:"Dieser Benutzername ist bereits vergeben.",
   err_email_taken:"Ein Konto mit dieser E-Mail existiert bereits.",
   err_weak_password:"Das Passwort muss mindestens 6 Zeichen lang sein.",
   err_invalid_input:"Bitte Eingabe prüfen und erneut versuchen.",
   err_login_failed:"Anmeldung fehlgeschlagen. Bitte erneut versuchen.",
   err_register_failed:"Konto konnte nicht erstellt werden. Bitte erneut versuchen.",
   err_network:"Server nicht erreichbar."
 },
 tr:{
   login_pre:"Giriş yap: ",login_sub:"Sürüye tekrar hoş geldin.",
   register_title_pre:"Katıl: ",register_sub:"Hesabını oluştur ve katkıda bulunmaya başla.",
   server_url:"Sunucu adresi",username:"Kullanıcı adı",password:"Şifre",email:"E-posta",
   ph_user:" ",
   login_btn:"Giriş yap",register_btn:"Hesap oluştur",
   no_account:"Hesabın yok mu?",register_link:"Bir tane oluştur",
   have_account:"Zaten kayıtlı mısın?",login_link:"Giriş yap",
   anon_preface:"Anonim kalmayı mı tercih edersin?",anon_link:"Atla ve anonim olarak katkıda bulun",
   dash_title:"Eğitim",loss:"Kayıp",tokens_sec:"Token/sn",round:"Tur",step:"Adım",
   loss_history:"Kayıp geçmişi",waiting_data:"Eğitim verisi bekleniyor…",time_left:"Kalan",
   idle:"Boşta",training:"Eğitiliyor…",waiting:"Bekleniyor…",uploading:"Yükleniyor…",
   downloading:"İndiriliyor…",preparing:"Hazırlanıyor…",calibrating:"Kalibre ediliyor…",
   dl_weights:"Ağırlıklar indiriliyor…",wait_coord:"Koordinatör bekleniyor…",
   wait_coord_sub:"Bu birkaç dakika sürebilir. Koordinatör, turlar arasında tüm istemcilerden gelen güncellemeleri toplar.",
   start:"Eğitimi başlat",stop:"Durdur",
   backend_change_title:"Backend değiştir?",
   backend_change_msg:"Backend değiştirmek mevcut eğitim oturumunu sıfırlayacak ve turu yeniden başlatacak. Şu ana kadarki katkınız silinecek. Devam edilsin mi?",
   cancel:"İptal",confirm:"Değiştir & yeniden başlat",
   err_invalid_credentials:"Geçersiz kullanıcı adı veya şifre.",
   err_username_taken:"Bu kullanıcı adı zaten alınmış. Lütfen başka bir tane seçin.",
   err_email_taken:"Bu e-posta ile zaten bir hesap mevcut.",
   err_weak_password:"Şifre en az 6 karakter olmalıdır.",
   err_invalid_input:"Lütfen girdinizi kontrol edip tekrar deneyin.",
   err_login_failed:"Giriş yapılamadı. Lütfen tekrar deneyin.",
   err_register_failed:"Hesap oluşturulamadı. Lütfen tekrar deneyin.",
   err_network:"Sunucuya ulaşılamıyor."
 }
};

let lang='en',statusKey='idle';
let pendingBackend=null;
const lossHistory=[];const MAXP=80;
function t(k){return (I18N[lang]&&I18N[lang][k])||I18N.en[k]||k}
function applyT(){
  document.querySelectorAll('[data-i18n]').forEach(e=>e.textContent=t(e.getAttribute('data-i18n')));
  document.querySelectorAll('[data-i18n-ph]').forEach(e=>e.placeholder=t(e.getAttribute('data-i18n-ph')));
  document.getElementById('status-indicator').textContent=t(statusKey);
}
document.getElementById('lang-select').addEventListener('change',e=>{lang=e.target.value;applyT()});

function showAuthScreen(name){
  document.querySelectorAll('.auth-screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('screen-'+name).classList.add('active');
  const loginSrv = document.getElementById('login-server');
  const regSrv = document.getElementById('reg-server');
  if(name==='register' && loginSrv.value) regSrv.value = loginSrv.value;
  if(name==='login' && regSrv.value) loginSrv.value = regSrv.value;
  document.getElementById('login-error').classList.remove('show');
  document.getElementById('reg-error').classList.remove('show');
}
document.getElementById('go-to-register').addEventListener('click',e=>{e.preventDefault();showAuthScreen('register')});
document.getElementById('go-to-login').addEventListener('click',e=>{e.preventDefault();showAuthScreen('login')});

function callApi(n,...a){if(window.pywebview&&window.pywebview.api&&window.pywebview.api[n]){window.pywebview.api[n](...a);return true}return false}
window.addEventListener('pywebviewready',()=>{
  document.body.classList.add('ready');
  callApi('get_backends');
});

document.getElementById('btn-login').addEventListener('click',()=>{
  const srv=document.getElementById('login-server').value.trim();
  const u=document.getElementById('login-user').value.trim();
  const p=document.getElementById('login-pass').value;
  if(!u||!p){ showLoginError('err_invalid_credentials'); return; }
  document.getElementById('login-error').classList.remove('show');
  document.getElementById('btn-login').disabled=true;
  callApi('login',srv,u,p);
});

document.getElementById('btn-register').addEventListener('click',()=>{
  const srv=document.getElementById('reg-server').value.trim();
  const u=document.getElementById('reg-user').value.trim();
  const email=document.getElementById('reg-email').value.trim();
  const p=document.getElementById('reg-pass').value;
  if(!u||!email||!p){ showRegError('err_invalid_input'); return; }
  document.getElementById('reg-error').classList.remove('show');
  document.getElementById('btn-register').disabled=true;
  callApi('register',srv,u,email,p);
});

document.getElementById('btn-anon').addEventListener('click',e=>{
  e.preventDefault();
  callApi('login_anon');
});

document.getElementById('btn-start').addEventListener('click',()=>{if(callApi('start')){document.getElementById('btn-start').disabled=true;document.getElementById('btn-stop').disabled=false}});
document.getElementById('btn-stop').addEventListener('click',()=>{callApi('stop');document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true});

document.getElementById('backend-select').addEventListener('change',e=>{
  const newBackend = e.target.value;
  if(newBackend === window._currentBackend){ return; }
  if(statusKey === 'idle'){
    callApi('set_backend', newBackend);
  } else {
    pendingBackend = newBackend;
    document.getElementById('backend-overlay').classList.add('show');
  }
});

document.getElementById('btn-backend-cancel').addEventListener('click',()=>{
  document.getElementById('backend-overlay').classList.remove('show');
  const sel = document.getElementById('backend-select');
  sel.value = window._currentBackend || sel.value;
  pendingBackend = null;
});

document.getElementById('btn-backend-confirm').addEventListener('click',()=>{
  document.getElementById('backend-overlay').classList.remove('show');
  if(pendingBackend){
    callApi('set_backend', pendingBackend);
    pendingBackend = null;
  }
});

function showLoginError(code){
  const el=document.getElementById('login-error');
  el.textContent=t(code);
  el.classList.add('show');
  document.getElementById('btn-login').disabled=false;
}
function showRegError(code){
  const el=document.getElementById('reg-error');
  el.textContent=t(code);
  el.classList.add('show');
  document.getElementById('btn-register').disabled=false;
}

function mb(b){return (b/1048576).toFixed(1)}
function overlayShow(data){
  document.getElementById('overlay').classList.add('show');
  document.getElementById('overlay-title').textContent=t(data.title_key);
  
  const sub = document.getElementById('overlay-subtitle');
  if(data.subtitle_key){
    sub.textContent = t(data.subtitle_key);
    sub.classList.add('show');
  } else {
    sub.classList.remove('show');
  }
  
  const bar = document.getElementById('overlay-bar');
  const subBox = document.querySelector('#overlay .overlay-sub');
  if(data.hide_bar){
    bar.classList.add('hidden');
    subBox.style.display = 'none';
  } else {
    bar.classList.remove('hidden');
    subBox.style.display = 'flex';
    bar.classList.toggle('indet',!!data.indeterminate);
    document.getElementById('overlay-fill').style.width=data.indeterminate?'':'0%';
    document.getElementById('overlay-pct').textContent='';
    document.getElementById('overlay-mb').textContent='';
  }
}
function overlayHide(){document.getElementById('overlay').classList.remove('show')}

function drawGraph(){
  const c=document.getElementById('loss-graph');if(!c)return;
  const ctx=c.getContext('2d');
  const rect=c.getBoundingClientRect();
  if(rect.width<10||rect.height<10)return;
  const dpr=window.devicePixelRatio||1;
  c.width=rect.width*dpr;c.height=rect.height*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=rect.width,H=rect.height;
  ctx.clearRect(0,0,W,H);
  const empty=document.getElementById('graph-empty');
  if(lossHistory.length<2){empty.style.display='block';c.style.display='none';return}
  empty.style.display='none';c.style.display='block';
  const data=lossHistory.slice(-MAXP);
  const mn=Math.min(...data),mx=Math.max(...data),range=(mx-mn)||1;
  const pad=5,pH=H-pad*2,pW=W-pad*2,st=pW/(data.length-1);
  const isDark=document.body.classList.contains('dark');
  const gridColor=isDark?'rgba(255,255,255,.06)':'rgba(17,17,19,.07)';
  const lineColor=isDark?'#22c55e':'#16a34a';
  ctx.strokeStyle=gridColor;ctx.lineWidth=1;
  for(let g=1;g<4;g++){const y=pad+pH*g/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+pW,y);ctx.stroke()}
  const pt=i=>[pad+i*st,pad+pH-(data[i]-mn)/range*pH];
  ctx.beginPath();
  data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.lineTo(pad+pW,pad+pH);ctx.lineTo(pad,pad+pH);ctx.closePath();
  const gr=ctx.createLinearGradient(0,0,0,H);
  gr.addColorStop(0,isDark?'rgba(34,197,94,.20)':'rgba(22,163,74,.16)');gr.addColorStop(1,'rgba(22,163,74,0)');
  ctx.fillStyle=gr;ctx.fill();
  ctx.beginPath();
  data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.strokeStyle=lineColor;ctx.lineWidth=1.5;ctx.stroke();
  const[lx,ly]=pt(data.length-1);
  ctx.fillStyle=lineColor;ctx.beginPath();ctx.arc(lx,ly,2.5,0,Math.PI*2);ctx.fill();
}

window.handleEvent=function(ev,data){
  if(ev==='login_success'){
    document.getElementById('view-login').classList.remove('active');
    document.getElementById('view-dashboard').classList.add('active');
    document.body.classList.add('dashboard-active');
    document.getElementById('user-info').textContent=data.username;
    document.getElementById('btn-login').disabled=false;
    document.getElementById('btn-register').disabled=false;
    requestAnimationFrame(drawGraph);
  }
  else if(ev==='login_error'){
    const code = data.code || 'err_invalid_credentials';
    const errMap = {
      invalid_credentials:'err_invalid_credentials',
      login_failed:'err_login_failed',
      username_taken:'err_username_taken',
      email_taken:'err_email_taken',
      weak_password:'err_weak_password',
      invalid_input:'err_invalid_input',
      register_failed:'err_register_failed',
      network:'err_network'
    };
    const tKey = errMap[code] || 'err_login_failed';
    const loginActive = document.getElementById('screen-login').classList.contains('active');
    if(loginActive) showLoginError(tKey);
    else showRegError(tKey);
  }
  else if(ev==='meta'){}
  else if(ev==='log'){document.getElementById('status-event').textContent=data}
  else if(ev==='backends'){
    const sel = document.getElementById('backend-select');
    sel.innerHTML = '';
    window._currentBackend = data.current;
    const order = [
      {key:'cuda', label:'CUDA (NVIDIA)'},
      {key:'rocm', label:'ROCm (AMD)'},
      {key:'mps', label:'MPS (Apple)'},
      {key:'xpu', label:'XPU (Intel)'},
      {key:'directml', label:'DirectML (Windows)'},
      {key:'cpu', label:'CPU'}
    ];
    order.forEach(bk=>{
      const opt = document.createElement('option');
      opt.value = bk.key;
      opt.textContent = bk.label;
      opt.disabled = !data.available[bk.key];
      if(bk.key === data.current) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.disabled = false;
  }
  else if(ev==='overlay_show'){overlayShow(data)}
  else if(ev==='overlay_progress'){
    const bar=document.getElementById('overlay-bar');
    if(data.total>0){
      bar.classList.remove('indet');
      const pct=Math.min(100,data.done/data.total*100);
      document.getElementById('overlay-fill').style.width=pct+'%';
      document.getElementById('overlay-pct').textContent=pct.toFixed(1)+'%';
      document.getElementById('overlay-mb').textContent=mb(data.done)+' / '+mb(data.total)+' MB';
    }else{bar.classList.add('indet');document.getElementById('overlay-mb').textContent=mb(data.done)+' MB'}
  }
  else if(ev==='overlay_hide'){overlayHide()}
  else if(ev==='cal_stats'){
    document.getElementById('stat-loss').textContent=data.loss.toFixed(4);
    if(data.tps !== undefined) document.getElementById('stat-tps').textContent=Math.round(data.tps);
    document.getElementById('stat-round').textContent='—';
    document.getElementById('stat-step-cur').textContent=data.step;
    document.getElementById('stat-step-tot').textContent=data.total;
    const pct=Math.min(100,(data.step/Math.max(1,data.total))*100);
    document.getElementById('progress-bar').style.width=pct+'%';
    document.getElementById('progress-text').textContent=pct.toFixed(1)+'%';
    document.getElementById('graph-last').textContent=data.loss.toFixed(4);
    lossHistory.push(data.loss);
    if(lossHistory.length>MAXP*2)lossHistory.splice(0,lossHistory.length-MAXP*2);
    drawGraph();
  }
  else if(ev==='stats'){
    document.getElementById('stat-loss').textContent=data.loss.toFixed(4);
    document.getElementById('stat-tps').textContent=Math.round(data.tps);
    document.getElementById('stat-round').textContent=data.round;
    document.getElementById('stat-step-cur').textContent=data.step;
    document.getElementById('stat-step-tot').textContent=data.target;
    const pct=Math.min(100,(data.step/Math.max(1,data.target))*100);
    document.getElementById('progress-bar').style.width=pct+'%';
    document.getElementById('progress-text').textContent=pct.toFixed(1)+'%';
    document.getElementById('graph-last').textContent=data.loss.toFixed(4);
    lossHistory.push(data.loss);
    if(lossHistory.length>MAXP*2)lossHistory.splice(0,lossHistory.length-MAXP*2);
    drawGraph();
  }else if(ev==='preview_loss'){}
  else if(ev==='status'){
    statusKey=data;
    document.getElementById('status-indicator').textContent=t(statusKey);
    if(data==='training'||data==='idle'||data==='calibrating'||data==='preparing')overlayHide();
    if(data==='idle'){document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true}
  }
};
window.addEventListener('resize',()=>requestAnimationFrame(drawGraph));
if(window.ResizeObserver){new ResizeObserver(()=>requestAnimationFrame(drawGraph)).observe(document.getElementById('loss-graph'))}

/* ===== DARK MODE + LOGO SWAP (fully guarded) ===== */
function updateLogo(){
  try {
    var img = document.getElementById('logo-img');
    if(!img) return;
    var isDark = document.body.classList.contains('dark');
    var src = isDark ? img.getAttribute('data-dark') : img.getAttribute('data-light');
    if(src && src.length > 30) img.src = src;
  } catch(e){}
}

(function(){
  try{
    var saved=null;
    try{ saved=localStorage.getItem('crowdgpt-theme'); }catch(e){}
    var prefersDark=false;
    try{ prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; }catch(e){}
    if(saved==='dark' || (!saved && prefersDark)){ document.body.classList.add('dark'); }
    updateLogo();
  }catch(e){}
  var tbtn=document.getElementById('theme-toggle');
  if(tbtn){
    tbtn.addEventListener('click',function(){
      document.body.classList.toggle('dark');
      try{ localStorage.setItem('crowdgpt-theme', document.body.classList.contains('dark')?'dark':'light'); }catch(e){}
      updateLogo();
      requestAnimationFrame(drawGraph);
    });
  }
})();

applyT();
</script></body></html>"""


class Api:
    def __init__(self):
        self.window=None; self.stop_event=threading.Event()
        self.thread=None; self.auth_token=None; self.username=None
        self.server_url="http://api.crowdgpt.net:5006"; self._lp=0.0
        self.selected_backend = get_best_default_backend()

    def emit(self, ev, data):
        if not self.window: return
        if ev=='preview_loss':
            n=time.time()
            if n-self._lp<0.4: return
            self._lp=n
        try: self.window.evaluate_js(f"window.handleEvent({json.dumps(ev)},{json.dumps(data)});")
        except: pass

    def get_backends(self):
        available = get_available_backends()
        self.emit('backends', {'available': available, 'current': self.selected_backend})

    def set_backend(self, backend_name):
        self.selected_backend = backend_name
        if self.thread and self.thread.is_alive():
            self.stop_event.set()

    def login(self, s, u, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p), daemon=True).start()

    def _do_login(self, u, p):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token, self.username = tok, u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p), daemon=True).start()

    def _do_register(self, u, email, p):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token, self.username = tok, u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'register_failed'})

    def login_anon(self):
        self.auth_token, self.username = None, "anonymous"
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear(); self.thread=threading.Thread(target=self._rs, daemon=True); self.thread.start()
        
    def stop(self): self.stop_event.set()
    
    def _rs(self):
        global train_device, train_backend
        try:
            train_device, train_backend = detect_training_backend(self.selected_backend)
            self.emit('meta',{'backend':train_backend})
            self.emit('log',f"Backend: {train_backend} ({train_device})")
            auto_detect_vram_budget()
            self.emit('log',f"VRAM: {memory_config['ram_gb']:.1f} GB")
        except Exception as e:
            self.emit('log',f"Error: {e}"); self.emit('status','idle'); return
        while not self.stop_event.is_set():
            try: run_single_round(self.server_url, self.auth_token, self.stop_event, self.emit)
            except Exception as e: self.emit('log',f"Failed: {e}"); time.sleep(5)
            if self.stop_event.is_set(): break
            for _ in range(10):
                if self.stop_event.is_set(): break
                time.sleep(1)
        self.emit('status','idle')


def startup():
    try:
        scr = webview.screens[0]
        aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80)))
        h = max(360, min(800, int(ah * 0.80)))
        w = min(w, aw - 16); h = min(h, ah - 48)
        w = max(320, w); h = max(240, h)
        window.resize(w, h)
        window.move(max(0, (aw - w)//2), max(0, (ah - h)//2))
    except Exception:
        pass
    
    def set_window_icon():
        try:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import Gtk, GdkPixbuf
            
            icon_path = None
            if ICON_PATH and os.path.exists(ICON_PATH):
                icon_path = ICON_PATH
            elif _lp_light.exists():
                icon_path = str(_lp_light.absolute())
                
            if not icon_path:
                return
                
            for w in Gtk.Window.list_toplevels():
                if w.get_title() == "CrowdGPT":
                    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, 64, 64, True)
                    w.set_icon(pixbuf)
                    break
        except Exception:
            pass

    threading.Timer(0.5, set_window_icon).start()


if __name__ == "__main__":
    api = Api()
    kwargs = {
        "js_api": api,
        "width": 960,
        "height": 680,
        "min_size": (320, 240),
    }
    
    html_out = HTML.replace("__LOGO__", LOGO_URI_LIGHT).replace("__LOGO_DARK__", LOGO_URI_DARK)
    
    try:
        if ICON_PATH and os.path.exists(ICON_PATH):
            window = webview.create_window(
                "CrowdGPT", html=html_out,
                icon=ICON_PATH, **kwargs
            )
        else:
            window = webview.create_window(
                "CrowdGPT", html=html_out,
                **kwargs
            )
    except TypeError:
        window = webview.create_window(
            "CrowdGPT", html=html_out,
            **kwargs
        )
    
    api.window = window
    webview.start(startup, debug=False)
