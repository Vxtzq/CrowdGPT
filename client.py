#!/usr/bin/env python3
"""
CrowdGPT GUI Client v3.2
v3.2 changes
  - Merged pull requests
  - Many bugfixes
v3.1 changes
  - Fixed register bugs
  - Background upload: training continues while the previous delta uploads (max 1 upload in flight;
    a finished delta waits for the running upload before starting its own)
v3.0 changes
  - Enhanced CLI client
  - better Engram training
  - Merged pull requests

v2.9 changes
  - Crisp app icon: multi-size icon (16..256 px) rendered natively from the logo SVG and embedded in the client.
    Windows picks the right size for the window's DPI (title bar, taskbar, Alt-Tab); Linux/Qt uses the real
    16/24/32/48/... images instead of down-scaling one 256 px PNG; GTK gets a full icon list.
  - Weights are cached by FILE CONTENT (Hugging Face sha256), not by commit. If the weights file on Hugging Face
    did not change (new commit, same bytes) nothing is downloaded again. Only the changed file is fetched.

v2.8 changes
  - Fixed buttton bugs again!

v2.7 changes
  - fix UI button bugs
  - Same user on multiple devices
  - Fixed missing import
  - Clarified auto-login
  - Less false-flags
  - fixed XPU backend
  - fixed random OOMs
  - Smoothed out loss reported by clients
v2.0 changes (dynamic optimizer adaptation):
  - Reports optimizer capabilities to server: X-Client-Optim-Can-Fp32, X-Client-Optim-Can-8bit
  - Honors crossCheck.targetOptimizer from the pool (asymmetric: fp32 accepts any, 8bit requires native)
  - recommend_batch_size() and _build_optimizers() accept force_mode so a native-8bit client
    can be downgraded to fp32 without OOMing on the wrong bs estimate
  - _on_step_down preserves force_mode across OOM recovery (otherwise pool mismatch mid-round)
  - Round aborts if server forces adamw8bit and client cannot (defensive; v2 server filters this)
  - X-Client-Optim continues to report native preference; forced mode goes into the new headers

v1.9 changes (retained):
  - Engram training ENABLED by default (CROWDGPT_ENGRAM=0 to disable)
  - Engram wired into forward pass (x = x + self.engram(idx))
  - Reports X-Client-Autocast and X-Client-Optim to the coordinator
  - Honors targetAutocast=fp32 from the pool (disables AMP)
  - CPU batch-size capped at 2 (prevents "bs=128 on CPU" advertisement)
  - Reports X-Client-Engram for forward-compat server matching
  - Fix a freeze on certain computers on startup
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import re, sys, io, json, time, struct, math, gc, threading, queue, base64, logging, tempfile, atexit, subprocess, hashlib, shutil, ctypes, signal, argparse, datetime, uuid, getpass
import platform
import importlib.metadata as importlib_metadata
from pathlib import Path

try:
    import termios, tty, select
    _POSIX_KEYS = True
except Exception:
    termios = tty = select = None
    _POSIX_KEYS = False
try:
    import msvcrt
except Exception:
    msvcrt = None

try:
    import webview
    _WEBVIEW_OK = True
    _WEBVIEW_ERR = None
except Exception as _e:
    webview = None
    _WEBVIEW_OK = False
    _WEBVIEW_ERR = _e

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.checkpoint import checkpoint
import http.client
from requests.exceptions import ChunkedEncodingError
import gzip

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box as rich_box
    _RICH_OK = True
except Exception:
    # Rich is optional; the client still runs with plain line logs.
    Console = Group = Layout = Live = Panel = Table = Text = None
    rich_box = None
    _RICH_OK = False

from collections import deque
if _WEBVIEW_OK:
    logging.getLogger("pywebview").setLevel(logging.CRITICAL)
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if _stream is None:
        _stream = open(os.devnull, "w", encoding="utf-8")
    elif getattr(_stream, "buffer", None) is not None:
        _stream = io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace")
    setattr(sys, _stream_name, _stream)
del _stream_name, _stream
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

CLIENT_VERSION = "3.2"
VERSION_URL = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/version.txt"
CLIENT_URL  = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client.py"
HTML_URL    = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client.html"
VERSION_SIG_URL = VERSION_URL + ".sig"

UPDATE_PUBKEYS = [
    "rVmzV5syTbiv3okEACObqUyZDNa1S5iWZHKpTJJs8kc=",
    "ZF5A5Wt9+Gwib8hlPYH0Uqxk+gdhmMadETKy+60NT5A=",
]

_ED_P = 2 ** 255 - 19
_ED_L = 2 ** 252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_P - 2, _ED_P)) % _ED_P
_ED_I = pow(2, (_ED_P - 1) // 4, _ED_P)

def _ed_recover_x(y, sign):
    if y >= _ED_P: return None
    x2 = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_P - 2, _ED_P) % _ED_P
    if x2 == 0: return None if sign else 0
    x = pow(x2, (_ED_P + 3) // 8, _ED_P)
    if (x * x - x2) % _ED_P: x = x * _ED_I % _ED_P
    if (x * x - x2) % _ED_P: return None
    return _ED_P - x if (x & 1) != sign else x

def _ed_add(p, q):
    a = (p[1] - p[0]) * (q[1] - q[0]) % _ED_P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _ED_P
    c = 2 * p[3] * q[3] * _ED_D % _ED_P
    d = 2 * p[2] * q[2] % _ED_P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _ED_P, g * h % _ED_P, f * g % _ED_P, e * h % _ED_P)

def _ed_mul(s, p):
    r = (0, 1, 1, 0)
    while s > 0:
        if s & 1: r = _ed_add(r, p)
        p = _ed_add(p, p)
        s >>= 1
    return r

def _ed_decode(b):
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _ed_recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _ED_P)

_ED_GY = 4 * pow(5, _ED_P - 2, _ED_P) % _ED_P
_ED_G = (_ed_recover_x(_ED_GY, 0), _ED_GY, 1, _ed_recover_x(_ED_GY, 0) * _ED_GY % _ED_P)

def ed25519_verify(public_key, message, signature):
    try:
        if len(public_key) != 32 or len(signature) != 64: return False
        a, r = _ed_decode(public_key), _ed_decode(signature[:32])
        s = int.from_bytes(signature[32:], "little")
        if a is None or r is None or s >= _ED_L: return False
        h = int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little") % _ED_L
        lhs, rhs = _ed_mul(s, _ED_G), _ed_add(r, _ed_mul(h, a))
        return (lhs[0] * rhs[2] - rhs[0] * lhs[2]) % _ED_P == 0 and (lhs[1] * rhs[2] - rhs[1] * lhs[2]) % _ED_P == 0
    except Exception: return False

class AuthRejected(Exception): pass
class CohortAbort(Exception): pass
class ArchIncompatible(Exception): pass
class CoordinatorRejected(Exception):
    def __init__(self, message, status=None): super().__init__(message); self.status = status   # HTTP 403 or 426

def _parse_version(v):
    try: return tuple(int(x) for x in str(v).strip().split("-")[0].split(".")[:4])
    except Exception: return (0,)

def _verify_manifest(manifest_bytes, sig_text, pubkeys=None):
    try: sig = base64.b64decode(str(sig_text).strip(), validate=True)
    except Exception: return False
    for k in (UPDATE_PUBKEYS if pubkeys is None else pubkeys):
        try:
            if ed25519_verify(base64.b64decode(k), manifest_bytes, sig): return True
        except Exception: continue
    return False

def _parse_manifest(manifest_bytes):
    lines = [l.strip() for l in manifest_bytes.decode("utf-8").replace("\r\n", "\n").split("\n") if l.strip()]
    m = {"version": lines[0]}
    for l in lines[1:]:
        if ":" in l:
            k, v = l.split(":", 1)
            m[k.strip().lower()] = v.strip()
    return m

def _is_sha256(h): return isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)

def _fetch_remote_version():
    if not UPDATE_PUBKEYS: return None, None, None
    try:
        rv = requests.get(VERSION_URL, timeout=5); rv.raise_for_status()
        rs = requests.get(VERSION_SIG_URL, timeout=5); rs.raise_for_status()
        manifest = rv.content
        if not _verify_manifest(manifest, rs.text): return None, None, None
        m = _parse_manifest(manifest)
        exp = m.get("expires")
        if exp and datetime.datetime.now(datetime.timezone.utc).date() > datetime.date.fromisoformat(exp): return None, None, None
        hash_py, hash_html = m.get("sha256_clientpy", "").lower(), m.get("sha256_clienthtml", "").lower()
        if not (_is_sha256(hash_py) and _is_sha256(hash_html)): return None, None, None
        return m["version"], hash_py, hash_html
    except Exception: return None, None, None

def _download_new_client(expected_version, expected_hash_py=None, expected_hash_html=None):
    try:
        response = requests.get(CLIENT_URL, timeout=60); response.raise_for_status(); code = response.text
        response = requests.get(HTML_URL, timeout=60); response.raise_for_status(); html = response.text
    except Exception as e:
        log.warning(f"Could not download client update: {e}"); return None
    if not (_is_sha256(expected_hash_py) and _is_sha256(expected_hash_html)): return None
    if hashlib.sha256(code.encode('utf-8')).hexdigest() != expected_hash_py: return None
    if hashlib.sha256(html.encode('utf-8')).hexdigest() != expected_hash_html: return None
    try: compile(code, "client.py", "exec")
    except Exception: return None
    if not any(m in code for m in (f'CLIENT_VERSION = "{expected_version}"', f"CLIENT_VERSION = '{expected_version}'")): return None
    if not html.lstrip().lower().startswith("<!doctype html") and "<html" not in html[:500].lower(): return None
    if not all(ph in html for ph in ("__LOGO__", "__LOGO_DARK__", "__VERSION__")): return None
    return code, html

def _apply_update(new_files):
    if (not isinstance(new_files, tuple) or len(new_files) != 2 or not all(isinstance(x, str) for x in new_files)): return False
    code, html = new_files
    base = Path(__file__).resolve().parent
    targets = {base / "client.py": code, base / "client.html": html}
    backups, temps, replaced = {}, {}, []
    try:
        for target, content in targets.items():
            if target.exists():
                backup = target.with_name(target.name + ".bak"); shutil.copy2(str(target), str(backup)); backups[target] = backup
            tmp = target.with_name(target.name + ".new")
            tmp.write_text(content, encoding="utf-8")
            if target.exists():
                try: os.chmod(str(tmp), target.stat().st_mode)
                except OSError: pass
            temps[target] = tmp
        for target, tmp in temps.items():
            os.replace(str(tmp), str(target)); replaced.append(target)
        return True
    except Exception as e:
        log.error(f"Could not apply update: {e}; attempting rollback")
        for target in reversed(replaced):
            try:
                b = backups.get(target)
                if b is not None and b.exists(): os.replace(str(b), str(target))
                elif target.exists(): target.unlink()
            except Exception: pass
        return False
    finally:
        for tmp in temps.values():
            try: tmp.unlink(missing_ok=True)
            except Exception: pass

def check_for_update():
    if getattr(torch.version, "hip", None) and os.environ.get("CROWDGPT_ROCM_AUTOUPDATE") != "1": return False
    remote, hash_py, hash_html = _fetch_remote_version()
    if not remote or _parse_version(remote) <= _parse_version(CLIENT_VERSION): return False
    downloaded = _download_new_client(remote, hash_py, hash_html)
    if downloaded and _apply_update(downloaded): return True
    return False

def _restart_self(resume_training=False):
    # resume_training=True only when an update lands WHILE training: the new process then logs back in and resumes by itself
    if resume_training: os.environ["CROWDGPT_RESUME_TRAINING"] = "1"
    else: os.environ.pop("CROWDGPT_RESUME_TRAINING", None)
    python = sys.executable; script = str(Path(__file__).resolve()); args = [python, script] + sys.argv[1:]
    if os.name == "nt":
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008); sys.exit(0)
    else: os.execv(python, args)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
AUTH_TOKEN_FILE = CHECKPOINT_DIR / "auth_token.json"
GPU_SELECTION_FILE = CHECKPOINT_DIR / "selected_gpu.json"
DEVICE_ID_FILE = CHECKPOINT_DIR / "device_id.json"

def _write_private(path, text):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(text)
    try: os.chmod(str(path), 0o600)
    except OSError: pass

def _save_auth_token(server, username, token, remember=True):
    _write_private(AUTH_TOKEN_FILE, json.dumps({
        "token": token,
        "username": username,
        "server": server,
        "remember": bool(remember),
    }))

def _migrate_cwd_checkpoints():
    old = Path.cwd() / "checkpoints"
    try:
        if not old.is_dir() or old.resolve() == CHECKPOINT_DIR: return
    except OSError: return
    for name in ("hf_weights", "auth_token.json", "selected_backend.json", "selected_vram.json"):
        src, dst = old / name, CHECKPOINT_DIR / name
        if src.exists() and not dst.exists():
            try: shutil.move(str(src), str(dst)); log.info(f"[cache] moved {src} -> {dst}")
            except Exception as e: log.warning(f"[cache] could not move {src}: {e}")
    try: old.rmdir()
    except OSError: pass

try: _migrate_cwd_checkpoints()
except Exception: pass
if AUTH_TOKEN_FILE.exists():
    try: os.chmod(str(AUTH_TOKEN_FILE), 0o600)
    except OSError: pass

def _load_or_create_device_id() -> str:
    """Stable per-install ID used to distinguish multiple devices on one account.
    The coordinator keys per-client dicts on user:<uuid>:dev:<device>, so two
    devices running the same account are treated as distinct training clients
    (separate pools, delta slots, quotas). The UUID part is still used for
    user-level bans / trust / token accounting."""
    def _read_id():
        try:
            v = json.loads(DEVICE_ID_FILE.read_text()).get("id")
            if isinstance(v, str) and 8 <= len(v) <= 32: return v
        except Exception:
            pass
        return None

    winner = _read_id()
    if winner: return winner
    new_id = uuid.uuid4().hex[:16]
    payload = json.dumps({"id": new_id, "created": time.time()})
    for _ in range(3):
        try:
            fd = os.open(str(DEVICE_ID_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Another process won the create race. Give it a moment to finish writing,
            # then use its ID so both instances agree on one device identity.
            for _ in range(20):
                winner = _read_id()
                if winner: return winner
                time.sleep(0.05)
            # A corrupt/empty file would otherwise make every launch generate a new
            # ID; clear it and retry the exclusive create (bounded).
            try: os.unlink(str(DEVICE_ID_FILE))
            except OSError: break
            continue
        except OSError:
            return new_id
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload); f.flush(); os.fsync(f.fileno())
            return new_id
        except Exception:
            try: os.unlink(str(DEVICE_ID_FILE))
            except OSError: pass
            return new_id
    return new_id

DEVICE_ID = _load_or_create_device_id()
log.info(f"[device] id={DEVICE_ID[:8]}… ({DEVICE_ID_FILE.name})")

MAX_CACHED_REVISIONS = 1; MAX_HF_CACHE_GB = 6.0; STALE_PART_AGE_SEC = 60 * 60; STALE_BAK_AGE_SEC = 24 * 60 * 60

def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try: total += p.stat().st_size
                except Exception: pass
    except Exception: pass
    return total

def _safe_rmtree(path: Path):
    try:
        if path.is_dir(): shutil.rmtree(path, ignore_errors=True)
    except Exception: pass

def prune_hf_cache(keep_revision: str = None, log_fn=None):
    """v2.9: weights live in hf_weights/blobs/<role>__<sha256>__<size>.safetensors (content-addressed).
    Keep the newest MAX_CACHED_REVISIONS file per role; legacy per-revision folders (<= v2.8) are removed."""
    def _log(msg):
        if log_fn is not None:
            try: log_fn(msg)
            except Exception: pass
        log.info(msg)
    try:
        hf_root = CHECKPOINT_DIR / "hf_weights"
        if hf_root.is_dir():
            for p in list(hf_root.iterdir()):
                if p.is_dir() and p.name != "blobs":
                    sz = _dir_size_bytes(p); _safe_rmtree(p); _log(f"[cache] removed legacy revision folder {p.name} ({sz/1024**2:.0f} MB)")
            blobs = hf_root / "blobs"
            if blobs.is_dir():
                groups = {}
                for f in blobs.glob("*.safetensors"): groups.setdefault(f.name.split("__", 1)[0], []).append(f)
                for role, fs in groups.items():
                    fs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    for f in fs[MAX_CACHED_REVISIONS:]:
                        try: sz = f.stat().st_size; f.unlink(); _log(f"[cache] pruned old {role} weights ({sz/1024**2:.0f} MB)")
                        except Exception: pass
                files = sorted(blobs.glob("*.safetensors"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
                total = sum(f.stat().st_size for f in files if f.exists()); cap = int(MAX_HF_CACHE_GB * 1024**3)
                while total > cap and len(files) > 1:
                    victim = files.pop(0)
                    try: sz = victim.stat().st_size; victim.unlink(); total -= sz; _log(f"[cache] pruned {victim.name} for size cap ({sz/1024**2:.0f} MB)")
                    except Exception: pass
            try:
                if hf_root.is_dir() and not any(hf_root.iterdir()): hf_root.rmdir()
            except Exception: pass
        now = time.time()
        try:
            for part in CHECKPOINT_DIR.rglob("*.part"):
                try:
                    if now - part.stat().st_mtime > STALE_PART_AGE_SEC: part.unlink(); _log(f"[cache] removed stale {part.name}")
                except Exception: pass
        except Exception: pass
        try:
            upload_active = globals().get("_upload_active")   # not defined yet when this runs at import
            if not (upload_active and upload_active()):        # a running (slow) upload owns its delta file, however old
                for f in CHECKPOINT_DIR.glob("delta-*.gz"):
                    try:
                        if now - f.stat().st_mtime > STALE_PART_AGE_SEC: f.unlink(); _log(f"[cache] removed stale {f.name}")
                    except Exception: pass
        except Exception: pass
        try:
            here = Path(__file__).resolve().parent
            for pat in ("*.bak", "*.new"):
                for f in here.glob(pat):
                    try:
                        if now - f.stat().st_mtime > STALE_BAK_AGE_SEC: f.unlink(); _log(f"[cache] removed stale {f.name}")
                    except Exception: pass
        except Exception: pass
    except Exception as e: log.warning(f"[cache] prune failed: {e}")

try: prune_hf_cache()
except Exception: pass

DEFAULT_SERVER = "https://server.crowdgpt.net"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
HF_MODEL_REPO_ID = "Vxtzq/Crowd-v1"
USE_HF_WEIGHTS = True
UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
MAX_DIRECT_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_CHUNK_RETRY = 4

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]; DIM = MODEL_CONFIG["dim"]; N_LAYERS = MODEL_CONFIG["nLayers"]
N_HEADS = MODEL_CONFIG["nHeads"]; N_KV_HEADS = MODEL_CONFIG["nKvHeads"]; HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]; MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865

ENGRAM_ACTIVE = os.environ.get("CROWDGPT_ENGRAM", "1") == "1"

LOSS_CHUNK = 256; UPLOAD_BUFFER_MIN = 25; ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SAFETY_MARGIN_GB = 1.0; ALLOCATOR_RESERVE_GB = 0.5; MIN_HEADROOM_GB = 0.25
CAL_TIME_BUDGET = 90.0; CAL_MIN_MICRO = 3; CAL_MAX_MICRO_HARD = 20

def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM) * 2 + DIM * DIM + DIM * 2
        size += DIM * MLP_HIDDEN * 2 + MLP_HIDDEN * DIM
    size += DIM * 2
    return size

EXPECTED_MODEL_SIZE = _calc_model_size()
EXPECTED_ENGRAM_SIZE = ENG_NUM_BUCKETS * DIM
EXPECTED_MODEL_BYTES = EXPECTED_MODEL_SIZE * 2
EXPECTED_ENGRAM_BYTES = EXPECTED_ENGRAM_SIZE * 2
EXPECTED_WEIGHT_BYTES = EXPECTED_MODEL_BYTES + EXPECTED_ENGRAM_BYTES
memory_config = {"ram_gb": 12, "is_auto_detected": False}
train_device, train_backend = None, None

def safe_float(v, default=0.0):
    try:
        if v is None or math.isnan(v) or math.isinf(v): return default
        return float(v)
    except Exception: return default

def sanitize_for_json(obj):
    if isinstance(obj, dict): return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return 0.0
        return obj
    try:
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj); return 0.0 if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
    except ImportError: pass
    return str(obj)

TOKENIZER = None; _tokenizer_started = False; PREVIEW_EVERY = 10

def start_tokenizer_loader():
    global _tokenizer_started
    if _tokenizer_started: return
    _tokenizer_started = True
    threading.Thread(target=_load_tokenizer_worker, daemon=True).start()

def _load_tokenizer_worker():
    global TOKENIZER
    try:
        from transformers import AutoTokenizer
        candidates = [TOKENIZER_REPO, "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "Vxtzq/Crowd-v1"]
        for repo in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                if len(tok) >= 151000:
                    TOKENIZER = tok; log.info(f"Tokenizer loaded: {repo} ({len(tok)} tokens)"); return
            except Exception: continue
    except ImportError: pass
    TOKENIZER = None

def decode_ids(ids, max_chars=220):
    try:
        ids = [int(i) for i in ids]
        if TOKENIZER is not None:
            s = TOKENIZER.decode(ids, skip_special_tokens=True); s = ' '.join(s.split())
            if len(s) > max_chars: s = s[:max_chars] + "…"
            return s or "…"
    except Exception: pass
    s = ' '.join(f"#{int(i)}" for i in ids)
    if len(s) > max_chars: s = s[:max_chars] + "…"
    return s

def decode_one(tok_id): return decode_ids([tok_id], max_chars=32) or "␠"

# ---- embedded assets (v2.8): used when docs/*.svg / icon.ico are missing or no SVG renderer is installed ----
_EMBEDDED_ICO_B64 = "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"
_EMBEDDED_LOGO_BLACK_B64 = "PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iMTM3LjAwMDAwMHB0IiBoZWlnaHQ9IjEzOS4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDEzNy4wMDAwMDAgMTM5LjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsMTM5LjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTM0MSAxMTg4IGMtMTkgLTM1IC03MSAtMTI4IC0xMTYgLTIwOCAtNDUgLTgwIC0xMDAgLTE3OSAtMTIzIC0yMjEKbC00MSAtNzUgODMgLTE1MCBjNDUgLTgyIDk0IC0xNjkgMTA3IC0xOTQgMTQgLTI1IDQ3IC04NCA3NCAtMTMzIGw0OSAtODcgMzE2CjAgMzE3IDAgMTUzIDI3OCBjODQgMTUyIDE1NSAyNzkgMTU2IDI4MiAyIDQgLTM3IDc4IC04NyAxNjUgLTQ5IDg4IC0xMTkgMjE1Ci0xNTYgMjgzIGwtNjcgMTIyIC0zMTYgMCAtMzE1IDAgLTM0IC02MnogbTczMiAtMjY3IGM3MCAtMTI3IDEyNyAtMjMzIDEyNwotMjM4IDAgLTQgLTU3IC0xMTAgLTEyNyAtMjM1IGwtMTI2IC0yMjggLTI1NyAwIC0yNTcgMCAtMTI2IDIyOCBjLTcwIDEyNQotMTI3IDIzMSAtMTI3IDIzNiAwIDUgNTcgMTExIDEyOCAyMzcgbDEyNyAyMjkgMjU1IDAgMjU1IDAgMTI4IC0yMjl6Ii8+CjwvZz4KPC9zdmc+Cg=="
_EMBEDDED_LOGO_WHITE_B64 = "PHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgd2lkdGg9IjEwMCUiIHZpZXdCb3g9IjAgMCAxMDAwIDEwMTgiIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDEwMDAgMTAxOCIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+Cgo8cGF0aCBmaWxsPSIjRkZGRkZGIiBvcGFjaXR5PSIxLjAwMDAwMCIgc3Ryb2tlPSJub25lIiBkPSIKTTI4MC4xNDMwOTcsOTMwLjAwMDAwMCAKCUMyNzQuMzg0NDkxLDkzMS4xMTg4MzUgMjcyLjIwNzEyMyw5MjguMTYxMTk0IDI2OS44NTA1MjUsOTIzLjg0OTU0OCAKCUMyNjAuMjkxNTA0LDkwNi4zNjA0NzQgMjUwLjE3ODMxNCw4ODkuMTc1MjkzIDI0MC4zODIwMDQsODcxLjgxNDUxNCAKCUMyMzMuMTkyNTY2LDg1OS4wNzM1NDcgMjI2LjIwMzA3OSw4NDYuMjE5OTcxIDIxOS4wODAxODUsODMzLjQ0MTE2MiAKCUMyMDguMTI2MTc1LDgxMy43ODkyNDYgMTk3LjA4Nzc4NCw3OTQuMTg0MTQzIDE4Ni4xNzU4MTIsNzc0LjUwODk3MiAKCUMxNzkuNzA0ODgwLDc2Mi44NDEzNzAgMTczLjQ2NzY5Nyw3NTEuMDQ0MTg5IDE2Ny4wMTk0MjQsNzM5LjM2MzgzMSAKCUMxNTUuNDQ3MTQ0LDcxOC40MDE3MzMgMTQzLjc1MDc3OCw2OTcuNTA4MTE4IDEzMi4xOTIwNzgsNjc2LjUzODYzNSAKCUMxMjQuNjcwNzIzLDY2Mi44OTM1NTUgMTE3LjM5MDA4Myw2NDkuMTE1NTQwIDEwOS44NDI5NzksNjM1LjQ4NDg2MyAKCUM5OC4wMDQ5NjcsNjE0LjEwNDQzMSA4NS45Nzc2MzgsNTkyLjgyODg1NyA3NC4xMzc2ODgsNTcxLjQ0OTQ2MyAKCUM2Ni41ODkzOTQsNTU3LjgxOTU4MCA2MC4wOTM4NjgsNTQzLjUxMTU5NyA1MS41MDUxMzgsNTMwLjU4NDc3OCAKCUM0NS4yOTYyOTEsNTIxLjIzOTg2OCA0Ni4xODI1OTgsNTE0LjUyODU2NCA1MS42NTYwODIsNTA1LjI2NDM3NCAKCUM2Ni43NzQwNDAsNDc5LjY3NjUxNCA4MC42NDYwNDksNDUzLjM1MTUwMSA5NC45NjI5OTcsNDI3LjI5MTc0OCAKCUMxMDQuMDg5ODc0LDQxMC42Nzg5ODYgMTEzLjAwMDAzOCwzOTMuOTQ1OTg0IDEyMi4yNDMyNjMsMzc3LjM5ODYyMSAKCUMxMzUuNjQ4ODQ5LDM1My4zOTk3ODAgMTQ5LjMzMzM4OSwzMjkuNTU2ODU0IDE2Mi43NzE2MjIsMzA1LjU3NjA4MCAKCUMxNzEuMzMwMzY4LDI5MC4zMDI3OTUgMTc5LjU1Mzk1NSwyNzQuODQxNjc1IDE4OC4xMTU3ODQsMjU5LjU3MDEyOSAKCUMxOTkuOTA5MTQ5LDIzOC41MzQ1NjEgMjExLjk2MjM4NywyMTcuNjQ0NjIzIDIyMy43NDU1NzUsMTk2LjYwMzQyNCAKCUMyMzEuNDM4NjI5LDE4Mi44NjU5ODIgMjM4LjgyNTAyNywxNjguOTU3MTIzIDI0Ni4zOTQwNzMsMTU1LjE0OTg3MiAKCUMyNTUuNjcwNTQ3LDEzOC4yMjc5NjYgMjY0Ljk5NDI5MywxMjEuMzMxOTc4IDI3NC40MjM0OTIsMTA0LjE5NTk4NCAKCUM0MjcuOTg5OTkwLDEwNC4xOTU5ODQgNTgxLjc0MjM3MSwxMDQuMTk1OTg0IDczNS41MjExNzksMTA0LjE5NTk4NCAKCUM3NDkuNjA4NzA0LDEyOS41OTAzNzggNzYzLjc4MTAwNiwxNTQuOTg5ODM4IDc3Ny44MDQzODIsMTgwLjQ3MTI4MyAKCUM3ODYuMzI1Mzc4LDE5NS45NTQ2MzYgNzk0LjUyMjIxNywyMTEuNjE2MzAyIDgwMy4wMzMzODYsMjI3LjEwNTEzMyAKCUM4MTUuMTg3NjIyLDI0OS4yMjM1MjYgODI3LjU3OTk1NiwyNzEuMjExMjEyIDgzOS42OTE4MzMsMjkzLjM1MjQ3OCAKCUM4NDMuNzU0MTUwLDMwMC43Nzg2MjUgODQ3LjAxOTUzMSwzMDguNjQxMjM1IDg1MS4wOTQyMzgsMzE2LjA1OTg0NSAKCUM4NTYuOTg1NjU3LDMyNi43ODU5NTAgODYzLjM0MDMzMiwzMzcuMjU2NDcwIDg2OS4zODk4OTMsMzQ3Ljg5NzIxNyAKCUM4NzUuNjYzNTc0LDM1OC45MzIwOTggODgxLjc2Njg0NiwzNzAuMDYzODEyIDg4OC4wMjA1MDgsMzgxLjExMDE5OSAKCUM4OTcuODM5NDc4LDM5OC40NTQ0MzcgOTA3LjgxNzI2MSw0MTUuNzA5MzgxIDkxNy41NDM3NjIsNDMzLjEwNTA0MiAKCUM5MjMuODk0Mjg3LDQ0NC40NjI4NjAgOTI5LjkzODQxNiw0NTUuOTkyNDkzIDkzNi4wODQ0MTIsNDY3LjQ2NDA4MSAKCUM5NDIuMjM2MzI4LDQ3OC45NDY1OTQgOTQ4LjYyMzcxOCw0OTAuMzE1NzA0IDk1NC4zNTM0NTUsNTAyLjAwNjIyNiAKCUM5NTcuMjA4NDk2LDUwNy44MzEyNjggOTU4Ljk5MTgyMSw1MTQuMjEzMjU3IDk2MC44NzUzNjYsNTIwLjQ1OTI5MCAKCUM5NjEuMzE1NTUyLDUyMS45MTg4ODQgOTYwLjU1NDYyNiw1MjQuMDQ4MjE4IDk1OS43MzY1NzIsNTI1LjUzMzU2OSAKCUM5NDcuNjQwNjI1LDU0Ny40OTQ5OTUgOTM1LjQzMjQ5NSw1NjkuMzk0NTMxIDkyMy4yNzA2OTEsNTkxLjMxOTcwMiAKCUM5MTYuMjU0NjM5LDYwMy45NjgwNzkgOTA5LjI2MTEwOCw2MTYuNjI5MDI4IDkwMi4yODAyNzMsNjI5LjI5NjgxNCAKCUM4OTEuMTg2NTIzLDY0OS40Mjc4NTYgODgwLjExMTAyMyw2NjkuNTY4OTA5IDg2OS4wMjk3MjQsNjg5LjcwNjc4NyAKCUM4NjAuNTk1Mzk4LDcwNS4wMzQzNjMgODUyLjIxMzMxOCw3MjAuMzkwOTMwIDg0My43MjAzOTgsNzM1LjY4NTk3NCAKCUM4MzAuMjE4NzUwLDc2MC4wMDA5NzcgODE2LjU3MDEyOSw3ODQuMjM0NzQxIDgwMy4xMjc1MDIsODA4LjU4MjE1MyAKCUM3OTQuODk3OTQ5LDgyMy40ODc2MTAgNzg3LjA1MjU1MSw4MzguNjA1MDQyIDc3OC44Mjc4ODEsODUzLjUxMzI0NSAKCUM3NjcuMDIyNzA1LDg3NC45MTE2MjEgNzUzLjk2NDY2MSw4OTUuNzA0MTAyIDc0My41NjcwMTcsOTE3Ljc1NzMyNCAKCUM3MzguNDA4ODc1LDkyOC42OTc2MzIgNzMyLjAyMjI3OCw5MzAuMjE4NTA2IDcyMS41NDgyMTgsOTMwLjE5Nzc1NCAKCUM1NzQuNTczNDI1LDkyOS45MDY0OTQgNDI3LjU5NzkzMSw5MzAuMDAwMDAwIDI4MC4xNDMwOTcsOTMwLjAwMDAwMCAKTTI1Mi40Nzg5NTgsMjk4LjAxNTM4MSAKCUMyMzkuMzk5OTQ4LDMyMS40NjEyNzMgMjI2LjI5MjIwNiwzNDQuODkxMjM1IDIxMy4yNTA0NDMsMzY4LjM1NzgxOSAKCUMyMDEuNDYxNDcyLDM4OS41NzAyNTEgMTg5LjU2ODQyMCw0MTAuNzI3NjMxIDE3OC4wNDA2NjUsNDMyLjA4MTY5NiAKCUMxNjMuNzU5MDE4LDQ1OC41MzcwNDggMTQ5Ljg3NTE2OCw0ODUuMjA2OTQwIDEzNS42NDM1NTUsNTExLjY4OTYwNiAKCUMxMzMuMzAxMjM5LDUxNi4wNDgyNzkgMTMzLjEyODUyNSw1MTkuNzM1NTM1IDEzNS41MDg1NDUsNTI0LjE0Nzk0OSAKCUMxNDYuNDk3NjUwLDU0NC41MjA5OTYgMTU3LjExNzY0NSw1NjUuMDkzMjAxIDE2OC4xMTQ3MTYsNTg1LjQ2MTkxNCAKCUMxNzkuMjY1NDI3LDYwNi4xMTUxNzMgMTkwLjY4NDY2Miw2MjYuNjIzNjU3IDIwMi4wMTk4NjcsNjQ3LjE3NzAwMiAKCUMyMTUuNjc5NTY1LDY3MS45NDUyNTEgMjI5LjM2MjA2MSw2OTYuNzAwOTI4IDI0My4wNTgyODksNzIxLjQ0ODk3NSAKCUMyNTUuMzU4MTI0LDc0My42NzM3NjcgMjY3LjcyMDU4MSw3NjUuODYzOTUzIDI3OS45OTg1NjYsNzg4LjEwMDc2OSAKCUMyOTEuNjU1NjQwLDgwOS4yMTMwNzQgMzAzLjMxNzcxOSw4MzAuMzIzNDg2IDMxNC43NzE0NTQsODUxLjU0NjAyMSAKCUMzMTYuNTI2ODU1LDg1NC43OTg1MjMgMzE4LjQwMDQyMSw4NTYuMDg1NDQ5IDMyMi4xNjI1MDYsODU2LjA4MjMzNiAKCUM0NDMuNjU3MTA0LDg1NS45ODIzNjEgNTY1LjE1MTczMyw4NTYuMDAwMDAwIDY4Ni42NDY0MjMsODU2LjAwMDAwMCAKCUM2ODcuNjQ2MzYyLDg1Ni4wMDAwMDAgNjg4Ljc0Mzg5Niw4NTYuMjYyMzkwIDY4OS42MjI2MjAsODU1LjkzNTk3NCAKCUM2OTAuODM2NDg3LDg1NS40ODQ5ODUgNjkyLjM2ODM0Nyw4NTQuODU5NTU4IDY5Mi45MzQwODIsODUzLjg0NTY0MiAKCUM3MDUuMDc0NDYzLDgzMi4wOTAwMjcgNzE3LjExNjQ1NSw4MTAuMjc5MzU4IDcyOS4xMzkxNjAsNzg4LjQ1ODI1MiAKCUM3MzYuODE3Njg4LDc3NC41MjE5MTIgNzQ0LjM2NTQ3OSw3NjAuNTEzMzY3IDc1Mi4wOTM5MzMsNzQ2LjYwNDkxOSAKCUM3NjYuODUxNTAxLDcyMC4wNDYxNDMgNzgxLjcwNjY2NSw2OTMuNTQxNTA0IDc5Ni41MDEyMjEsNjY3LjAwMzE3NCAKCUM4MDYuNjc4NTI4LDY0OC43NDcyNTMgODE3LjAwNTYxNSw2MzAuNTcxNTk0IDgyNi45Mzc1MDAsNjEyLjE4MjgwMCAKCUM4NDIuNDQ2MTA2LDU4My40Njg5OTQgODU3LjYzMDg1OSw1NTQuNTgwNTA1IDg3My4wNjc5OTMsNTI1LjgyNzg4MSAKCUM4NzUuNzk0MDY3LDUyMC43NTAzNjYgODc1LjczNDY4MCw1MTYuMDE1MzgxIDg3My4wMjk0ODAsNTExLjA1ODE5NyAKCUM4NjkuOTQ2Nzc3LDUwNS40MDkyNDEgODY2Ljk4MTI2Miw0OTkuNjk1ODMxIDg2NC4wMDM3ODQsNDkzLjk5MDExMiAKCUM4NTYuNjQwODY5LDQ3OS44ODA0MDIgODQ5LjQ5MzI4Niw0NjUuNjU0NDE5IDg0MS45MDgwODEsNDUxLjY2NTUyNyAKCUM4MjguNDk0ODczLDQyNi45Mjg0NjcgODE0Ljc4OTc5NSw0MDIuMzQ5ODU0IDgwMS4yNjc5NDQsMzc3LjY3MTUwOSAKCUM3OTMuMzkyODIyLDM2My4yOTg5MjAgNzg1Ljc2MTIzMCwzNDguNzkxODcwIDc3Ny43OTU3NzYsMzM0LjQ3MDA2MiAKCUM3NjcuMDEwMDEwLDMxNS4wNzcxNzkgNzU1Ljk1Mzk3OSwyOTUuODM0Nzc4IDc0NS4xMjAxMTcsMjc2LjQ2ODQxNCAKCUM3MzguMjE4NDQ1LDI2NC4xMzExMzQgNzMxLjU4ODAxMywyNTEuNjQyMjI3IDcyNC42OTU2MTgsMjM5LjI5OTcxMyAKCUM3MTQuMTkzMTE1LDIyMC40OTI2NjEgNzAzLjQ0ODI0MiwyMDEuODE5Njg3IDY5My4xMDA3NjksMTgyLjkyODQ2NyAKCUM2OTEuMDg2MjQzLDE3OS4yNTA1MTkgNjg4Ljk1MjI3MSwxNzcuOTAyMjUyIDY4NC43NTAxMjIsMTc3LjkwNjM0MiAKCUM1NjUuNTg4NzQ1LDE3OC4wMjI2OTAgNDQ2LjQyNzMwNywxNzguMDIzNjgyIDMyNy4yNjU5NjEsMTc3LjkyMzExMSAKCUMzMjEuODY4NTMwLDE3Ny45MTg1NjQgMzE4LjU4ODQ0MCwxNzguNjI3MTIxIDMxNS41NjQzMzEsMTg0LjM4MzE2MyAKCUMzMDMuMzA0NTA0LDIwNy43MTg1MDYgMjg5Ljk4MDQzOCwyMzAuNDkzMzMyIDI3Ny4xMzU4MDMsMjUzLjUyMzM2MSAKCUMyNjguOTk4ODQwLDI2OC4xMTI3MDEgMjYxLjAxNTE5OCwyODIuNzg3NTM3IDI1Mi40Nzg5NTgsMjk4LjAxNTM4MSAKeiIgZGF0YS1pbmRleD0iMSIgc3R5bGU9Im9wYWNpdHk6IDE7Ij48L3BhdGg+Cgo8L3N2Zz4="
LOGO_URI_LIGHT = ""; LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_dark  = Path(__file__).parent / "docs/logo-white.svg"
if _lp_light.exists():
    try: LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except Exception: pass
if _lp_dark.exists():
    try: LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except Exception: pass
if not LOGO_URI_LIGHT: LOGO_URI_LIGHT = "data:image/svg+xml;base64," + _EMBEDDED_LOGO_BLACK_B64
if not LOGO_URI_DARK: LOGO_URI_DARK = "data:image/svg+xml;base64," + _EMBEDDED_LOGO_WHITE_B64

ICON_PATH = None; WINDOWS_ICO_PATH = None; WINDOW_TITLE = "CrowdGPT"; _cleanup_icon = []

def _render_svg_to_png(svg_path, size: int = 256):
    svg_path = Path(svg_path)
    if not svg_path.exists(): return None
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False); tmp.close(); out = tmp.name
    def _ok():
        try: return os.path.getsize(out) > 100
        except Exception: return False
    try:
        import cairosvg; cairosvg.svg2png(url=str(svg_path), write_to=out, output_width=size, output_height=size)
        if _ok(): return out
    except Exception: pass
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            sw, sh = float(drawing.width or 0), float(drawing.height or 0)
            if sw > 0 and sh > 0:
                scale = size / max(sw, sh); drawing.scale(scale, scale)
                drawing.width = sw * scale; drawing.height = sh * scale
            renderPM.drawToFile(drawing, out, fmt='PNG')
            if _ok(): return out
    except Exception: pass
    for mod in ("PySide6", "PyQt5"):
        try:
            if mod == "PySide6":
                from PySide6.QtSvg import QSvgRenderer
                from PySide6.QtGui import QImage, QPainter
                from PySide6.QtCore import QByteArray
            else:
                from PyQt5.QtSvg import QSvgRenderer
                from PyQt5.QtGui import QImage, QPainter
                from PyQt5.QtCore import QByteArray
            renderer = QSvgRenderer(QByteArray(svg_path.read_bytes()))
            if not renderer.isValid(): continue
            img = QImage(size, size, QImage.Format_ARGB32); img.fill(0)
            painter = QPainter(img); renderer.render(painter); painter.end()
            if img.save(out, "PNG") and _ok(): return out
        except Exception: pass
    try: os.unlink(out)
    except Exception: pass
    return None

def _png_to_ico(png_path: str, sizes=(256, 128, 64, 48, 32, 16)):
    try: from PIL import Image
    except ImportError: return None
    try:
        img = Image.open(png_path).convert("RGBA")
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False); tmp.close()
        img.save(tmp.name, format="ICO", sizes=[(s, s) for s in sizes])
        return tmp.name
    except Exception: return None

def _generate_fallback_ico():
    try:
        from PIL import Image, ImageDraw
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False); tmp.close()
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((20, 20, 236, 236), radius=40, fill=(245, 158, 11, 255))
        d.ellipse((78, 78, 178, 178), fill=(255, 255, 255, 255))
        img.save(tmp.name, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
        return tmp.name
    except Exception:
        return None

def _embedded_ico():
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False); tmp.write(base64.b64decode(_EMBEDDED_ICO_B64)); tmp.close()
        return tmp.name
    except Exception: return None

def _ico_pngs(data: bytes) -> dict:
    """{size: png_bytes} for every PNG-compressed image inside an ICO."""
    out = {}
    try:
        n = struct.unpack_from("<H", data, 4)[0]
        for i in range(n):
            w, h, _c, _r, _pl, _bpp, size, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
            blob = data[off:off + size]
            if blob[:8] == b"\x89PNG\r\n\x1a\n": out[w or 256] = blob
    except Exception: pass
    return out

def _embedded_png(size: int = 256):
    try:
        blob = _ico_pngs(base64.b64decode(_EMBEDDED_ICO_B64)).get(size)
        if not blob: return None
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False); tmp.write(blob); tmp.close()
        _cleanup_icon.append(tmp.name); return tmp.name
    except Exception: return None

def _cleanup_icon_files():
    for p in _cleanup_icon:
        try: os.unlink(p)
        except Exception: pass

atexit.register(_cleanup_icon_files)

def _find_best_icon():
    base = Path(__file__).parent
    for ico in [base / "icon.ico", base / "docs" / "icon.ico"]:
        if ico.is_file(): return str(ico), str(ico)
    emb = _embedded_ico()          # v2.9: multi-size (16..256) icon rendered natively from the SVG -> crisp everywhere
    if emb:
        _cleanup_icon.append(emb); return emb, emb
    for svg in [base / "docs" / "logo-app.svg", base / "logo-app.svg"]:
        if svg.is_file():
            rendered = _render_svg_to_png(svg, 256)
            if rendered:
                png_path = Path(rendered); _cleanup_icon.append(rendered); raster = str(png_path); ico_path = None
                if sys.platform == "win32":
                    ico_path = _png_to_ico(raster)
                    if ico_path: _cleanup_icon.append(ico_path)
                return raster, ico_path
            if sys.platform != "win32": return str(svg), None
            break
    for png in [base / "icon.png", base / "docs" / "logo-app.png"]:
        if png.is_file():
            raster = str(png); ico_path = None
            if sys.platform == "win32":
                ico_path = _png_to_ico(raster)
                if ico_path: _cleanup_icon.append(ico_path)
            return raster, ico_path
    if sys.platform == "win32":
        fallback = _embedded_ico() or _generate_fallback_ico()
        if fallback:
            _cleanup_icon.append(fallback)
            return fallback, fallback
    return None, None

ICON_PATH, WINDOWS_ICO_PATH = _find_best_icon()

def set_windows_icon(window_title: str, ico_path: str):
    if sys.platform != "win32" or not ico_path: return
    ico_path = os.path.abspath(ico_path)
    if not os.path.exists(ico_path) or not ico_path.lower().endswith(".ico"): return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes   = [ctypes.c_wchar_p, ctypes.c_wchar_p]; user32.FindWindowW.restype = ctypes.c_void_p
    user32.LoadImageW.argtypes    = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.LoadImageW.restype     = ctypes.c_void_p
    user32.SendMessageW.argtypes  = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    user32.SendMessageW.restype   = ctypes.c_void_p
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]; user32.GetSystemMetrics.restype = ctypes.c_int
    hwnd = user32.FindWindowW(None, window_title)
    if not hwnd: return
    IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON, ICON_SMALL, ICON_BIG = 1, 0x0010, 0x0080, 0, 1
    SM_CXSMICON, SM_CYSMICON = 49, 50
    cx = user32.GetSystemMetrics(SM_CXSMICON) or 16
    cy = user32.GetSystemMetrics(SM_CYSMICON) or 16
    bx = user32.GetSystemMetrics(11) or 32          # SM_CXICON
    by = user32.GetSystemMetrics(12) or 32          # SM_CYICON
    try:                                            # per-monitor DPI (Win10 1607+): ask for the size this window really needs
        user32.GetDpiForWindow.argtypes = [ctypes.c_void_p]; user32.GetDpiForWindow.restype = ctypes.c_uint
        user32.GetSystemMetricsForDpi.argtypes = [ctypes.c_int, ctypes.c_uint]; user32.GetSystemMetricsForDpi.restype = ctypes.c_int
        dpi = user32.GetDpiForWindow(hwnd) or 96
        cx = cy = user32.GetSystemMetricsForDpi(SM_CXSMICON, dpi) or cx
        bx = by = user32.GetSystemMetricsForDpi(11, dpi) or bx
    except Exception: pass
    hicon_small = user32.LoadImageW(None, ico_path, IMAGE_ICON, cx, cy, LR_LOADFROMFILE)
    hicon_big   = user32.LoadImageW(None, ico_path, IMAGE_ICON, bx, by, LR_LOADFROMFILE)
    if hicon_small: user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
    if hicon_big: user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)

def _wait_for_window(title: str, timeout: float = 5.0) -> bool:
    if sys.platform != "win32": return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]; user32.FindWindowW.restype = ctypes.c_void_p
    deadline = time.time() + timeout
    while time.time() < deadline:
        if user32.FindWindowW(None, title): return True
        time.sleep(0.1)
    return False

def get_available_backends():
    backends = {}
    backends['cuda'] = torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip)
    backends['rocm'] = hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available()
    backends['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    backends['xpu'] = _xpu_available()
    try:
        import torch_directml
        backends['directml'] = torch_directml.is_available()
    except ImportError: backends['directml'] = False
    backends['cpu'] = True
    return backends

def _xpu_available():
    """Intel Arc / DC GPU. Modern PyTorch (>=2.4) supports XPU natively via
    torch.xpu; IPEX (`intel_extension_for_pytorch`) is deprecated and NOT
    installed by the current XPU wheel. The old import-gate silently disabled
    XPU on every modern Arc install."""
    try:
        return hasattr(torch, "xpu") and bool(torch.xpu.is_available())
    except Exception:
        return False

def get_best_default_backend():
    avail = get_available_backends()
    for bk in ['cuda', 'rocm', 'mps', 'xpu', 'directml', 'cpu']:
        if avail.get(bk): return bk
    return 'cpu'

def _pick_rocm_device(preferred_idx=None):
    candidates = []
    for idx in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(idx)
            candidates.append((idx, str(props.name), int(props.total_memory)))
        except Exception: pass
    if not candidates: raise RuntimeError("ROCm reports no usable GPU")
    if preferred_idx is not None:
        for idx, name, total in candidates:
            if idx == preferred_idx:
                torch.cuda.set_device(idx)
                log.info(f"[ROCM-FIX] Selected cuda:{idx} (user choice): {name} ({total / 1024**3:.1f} GB VRAM)")
                return torch.device(f"cuda:{idx}")
    preferred = [g for g in candidates if "radeon rx" in g[1].lower()]
    idx, name, total = max(preferred or candidates, key=lambda g: g[2])
    torch.cuda.set_device(idx)
    log.info(f"[ROCM-FIX] Selected cuda:{idx}: {name} ({total / 1024**3:.1f} GB VRAM)")
    return torch.device(f"cuda:{idx}")

def _check_arch_compat(device_idx):
    """Returns (compatible, arch_needed, arch_have). Only meaningful for CUDA.
    ROCm uses gfx names; always report compatible.

    Compatibility rules:
      1. Exact sm_XY match → runs natively.
      2. Any sm_X* with same major X in the wheel → runs natively (minor version fwd-compat
         within same major: sm_89 GPU runs sm_90 kernels? No — it runs sm_8X kernels.
         Actually a sm_89 GPU runs any sm_8X wheel binary, and JIT-compiles any compute_8X PTX.)
      3. Any compute_X* PTX in the wheel → runs via JIT (small startup cost).
      4. Otherwise: no path from this wheel → incompatible (this is the RTX 50-series case:
         sm_120 GPU, wheel has only sm_75..sm_90, no compute_12X PTX)."""
    if getattr(torch.version, "hip", None):
        return True, "rocm", []
    try:
        major, minor = torch.cuda.get_device_capability(device_idx)
    except Exception:
        return True, "?", []
    arch_needed = f"sm_{major}{minor}"
    try:
        arch_have = list(torch.cuda.get_arch_list())
    except Exception:
        arch_have = []
    if not arch_have:
        # Cannot determine — fail open (don't block training on older torch builds
        # that don't expose get_arch_list). Worst case: user sees the runtime
        # cudaErrorNoKernelImageForDevice instead of a pre-flight message.
        return True, arch_needed, []

    # 1. Exact match
    if arch_needed in arch_have or f"{arch_needed}a" in arch_have:
        return True, arch_needed, arch_have

    # 2. Any sm_X* in wheel with SAME MAJOR (Ada sm_89 gets sm_80/sm_86/sm_89, not sm_90)
    same_major_sm = any(
        h.startswith(f"sm_{major}") and h[3:].isdigit() for h in arch_have
    )
    if same_major_sm:
        return True, arch_needed, arch_have

    # 3. Any compute_X* PTX in wheel with SAME MAJOR (JIT path)
    same_major_ptx = any(
        h.startswith(f"compute_{major}") and h[8:].isdigit() for h in arch_have
    )
    if same_major_ptx:
        return True, arch_needed, arch_have

    # 4. No compatible path — this is the real "wrong wheel" case
    return False, arch_needed, arch_have


def get_gpu_list():
    """Returns [{idx, name, vram_gb, arch_ok, arch_needed}, ...]. Empty if no CUDA/ROCm devices."""
    try:
        if not torch.cuda.is_available():
            return []
        n = torch.cuda.device_count()
    except Exception:
        return []
    out = []
    for idx in range(n):
        try:
            props = torch.cuda.get_device_properties(idx)
            arch_ok, arch_needed, _ = _check_arch_compat(idx)
            out.append({
                "idx": idx,
                "name": str(props.name),
                "vram_gb": round(int(props.total_memory) / (1024 ** 3), 1),
                "arch_ok": arch_ok,
                "arch_needed": arch_needed,
            })
        except Exception as e:
            log.warning(f"[gpu] probe {idx} failed: {e}")
    return out


def _load_saved_gpu_idx():
    try:
        if GPU_SELECTION_FILE.exists():
            v = json.loads(GPU_SELECTION_FILE.read_text()).get("index")
            if v is not None:
                return int(v)
    except Exception:
        pass
    return None


def _save_saved_gpu_idx(idx):
    try:
        GPU_SELECTION_FILE.write_text(json.dumps({"index": int(idx)}))
    except Exception:
        pass


def _pick_cuda_device(preferred_idx=None):
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n == 0:
        raise RuntimeError("CUDA reports no devices")
    idx = preferred_idx if (preferred_idx is not None and 0 <= preferred_idx < n) else 0
    try:
        torch.cuda.set_device(idx)
        name = torch.cuda.get_device_name(idx)
        log.info(f"[CUDA] Selected cuda:{idx}: {name}")
    except Exception:
        log.info(f"[CUDA] Selected cuda:{idx}")
    return torch.device(f"cuda:{idx}")


_arch_warned = set()
def _check_arch_and_warn(emit, device_idx):
    """Returns True if compatible. Logs a once-per-session warning with the fix if not."""
    compat, needed, have = _check_arch_compat(device_idx)
    if compat:
        return True
    if device_idx not in _arch_warned:
        _arch_warned.add(device_idx)
        try:
            name = torch.cuda.get_device_name(device_idx)
        except Exception:
            name = f"cuda:{device_idx}"
        have_s = ", ".join(have) if have else "(none)"
        emit('log', f"❌ GPU {device_idx} ({name}) requires {needed} but installed PyTorch only has: {have_s}")
        emit('log', f"   This is the classic RTX 50-series / Blackwell issue — PyTorch wheel was built for older GPUs.")
        emit('log', f"   Windows fix (cmd): \"%LOCALAPPDATA%\\CrowdGPT\\.venv\\Scripts\\pip.exe\" install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        emit('log', f"   Linux fix: pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        emit('log', f"   Or: pick a different GPU from the selector, or switch backend to CPU (slow).")
    return False



def _pick_directml_device(dml):
    try:
        devices = [(i, str(dml.device_name(i))) for i in range(dml.device_count())]
        if devices:
            preferred = [gpu for gpu in devices if "radeon rx" in gpu[1].lower()]
            idx, name = (preferred or devices)[0]
            log.info(f"[AMD-AUTO] DirectML device {idx}: {name}")
            return dml.device(idx)
    except Exception: pass
    return dml.device(0)

def detect_training_backend(force=None):
    saved_gpu = _load_saved_gpu_idx()
    if force and force != "auto":
        n = force.lower()
        if n == "cpu": return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available() and not getattr(torch.version, "hip", None):
            return _pick_cuda_device(saved_gpu), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return _pick_rocm_device(saved_gpu), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
        if n == "xpu" and _xpu_available():
            return torch.device('xpu'), "XPU"
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available(): return _pick_directml_device(torch_directml), "DIRECTML"
            except ImportError: pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return _pick_cuda_device(saved_gpu), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return _pick_rocm_device(saved_gpu), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    if _xpu_available():
        return torch.device('xpu'), "XPU"
    try:
        import torch_directml
        if torch_directml.is_available(): return _pick_directml_device(torch_directml), "DIRECTML"
    except ImportError: pass
    return torch.device('cpu'), "CPU"

def install_hard_vram_cap(device, safety_gb=SAFETY_MARGIN_GB):
    if device is None or device.type != "cuda": return None
    try:
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx); total = int(props.total_memory)
        cap_bytes = max(int(MIN_HEADROOM_GB * 1024**3), total - int(safety_gb * 1024**3))
        frac = max(0.05, min(1.0, cap_bytes / total))
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        return cap_bytes / 1024**3
    except Exception: return None

def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            idx = train_device.index if train_device is not None and train_device.index is not None else 0
            fb, tb = torch.cuda.mem_get_info(idx)
            total_gb = tb / 1024**3; free_gb = fb / 1024**3
            usable = max(0.25, min(total_gb, free_gb) - SAFETY_MARGIN_GB - ALLOCATOR_RESERVE_GB)
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True}); return
        except Exception: pass
    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024**3
            usable = max(0.5, total_gb * 0.75 - SAFETY_MARGIN_GB)
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True}); return
        except Exception: pass
    if train_backend == "XPU":
        try:
            total = int(torch.xpu.get_device_properties(0).total_memory); free = total
            try: free, _t = torch.xpu.mem_get_info(0)
            except Exception: pass
            total_gb, free_gb = total / 1024**3, int(free) / 1024**3
            usable = max(0.25, min(total_gb, free_gb) - SAFETY_MARGIN_GB - ALLOCATOR_RESERVE_GB)
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True}); return
        except Exception: pass
    if train_backend == "DIRECTML": memory_config.update({"ram_gb": 1.5, "is_auto_detected": True}); return
    try:
        import psutil
        usable = max(0.5, psutil.virtual_memory().available / 1024**3 - SAFETY_MARGIN_GB)
        memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True})
    except Exception: pass

def apply_vram_override(user_gb=0.0):
    if "auto_gb" not in memory_config: return
    auto = float(memory_config["auto_gb"]); cap, src = 0.0, ""
    try:
        if user_gb and float(user_gb) > 0: cap, src = float(user_gb), "UI"
        else:
            v = os.environ.get("CROWDGPT_VRAM_GB", "").strip()
            if v: src = "CROWDGPT_VRAM_GB"
            else:
                v = os.environ.get("CROWDBPT_VRAM_GB", "").strip()
                if v: src = "CROWDBPT_VRAM_GB (deprecated)"
            if v and float(v) > 0: cap = float(v)
    except ValueError: pass
    new = min(auto, cap) if cap > 0 else auto
    memory_config["ram_gb"] = round(new, 2)
    if cap > 0: memory_config["is_auto_detected"] = False

def has_bitsandbytes():
    try:
        import bitsandbytes; return True
    except Exception: return False

def _bytes_per_block_params():
    return (DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM)
            + DIM * (N_KV_HEADS * HEAD_DIM) + DIM * DIM + DIM * 2
            + DIM * MLP_HIDDEN + DIM * MLP_HIDDEN + MLP_HIDDEN * DIM)

def estimate_vram_bytes(bs, sl, u8):
    per_block = _bytes_per_block_params(); head_params = VOCAB_SIZE * DIM
    head_bytes  = head_params * (4 + 4 + (2 if u8 else 8))
    blocks = N_LAYERS * per_block; opt_state_per_param = 2 if u8 else 8
    block_bytes = blocks * (4 + 4 + opt_state_per_param)
    act = N_LAYERS * bs * sl * DIM * 4 + bs * sl * (DIM * 12 + MLP_HIDDEN * 3) * 4; loss = bs * LOSS_CHUNK * VOCAB_SIZE * 4
    workspace = int(ALLOCATOR_RESERVE_GB * 1024**3)
    return int(head_bytes + block_bytes + act + loss + workspace)

def recommend_batch_size(sl=MAX_SEQ_LEN, force_mode=None):
    """force_mode: None = native (8bit if available, else 32); "adamw8bit" = force 8bit (2 bytes/param);
    "adamw32" = force fp32 (8 bytes/param). Forcing matters because 8bit<->fp32 changes the VRAM
    estimate by ~6 bytes per param on the optimizer state, which can be the difference between
    fitting and OOMing at model build."""
    if force_mode == "adamw8bit":
        u8 = True
    elif force_mode == "adamw32":
        u8 = False
    else:
        u8 = has_bitsandbytes() and train_backend in ("CUDA", "ROCM")
    budget = int(memory_config["ram_gb"] * 1024**3); best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget: best = bs
        else: break
    if train_backend == "ROCM":
        rocm_bs_cap = max(1, min(8, int(os.environ.get("CROWDGPT_ROCM_BATCH", 1)))); best = min(best, rocm_bs_cap)
    if train_backend == "CPU": best = min(best, 2)
    return best

def _predict_optim(backend, force_mode=None):
    """Returns the effective optimizer name. force_mode overrides native preference.
    - force_mode=None → native (8bit if available, else 32)
    - force_mode="adamw32" → fp32 always
    - force_mode="adamw8bit" → 8bit if available, else gracefully 32 (caller is expected to have already aborted)"""
    if force_mode == "adamw32": return "adamw32"
    if force_mode == "adamw8bit":
        try:
            import bitsandbytes as bnb
            if hasattr(bnb, "optim") and hasattr(bnb.optim, "AdamW8bit") and backend in ("CUDA", "ROCM"):
                return "adamw8bit"
        except Exception: pass
        return "adamw32"
    # native
    if backend not in ("CUDA", "ROCM"): return "adamw32"
    try:
        import bitsandbytes as bnb
        if not hasattr(bnb, "optim") or not hasattr(bnb.optim, "AdamW8bit"): return "adamw32"
        return "adamw8bit"
    except Exception: return "adamw32"

def _detect_optim_capability():
    """Returns (native, can_fp32, can_8bit). can_fp32 is always True (every backend supports fp32 AdamW).
    can_8bit is True only on CUDA/ROCm with a bitsandbytes that ships AdamW8bit."""
    native = _predict_optim(train_backend)
    can_8bit = (native == "adamw8bit")
    return native, True, can_8bit

def _http_detail(response, limit=180):
    """Return a short, redacted error detail for a coordinator response."""
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(body.get("detail") or body.get("message") or body.get("status") or "")
        else:
            detail = str(body)
    except Exception:
        try: detail = response.text or ""
        except Exception: detail = ""
    detail = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", "<id>", detail)
    detail = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", detail)
    detail = " ".join(detail.split())
    return detail[:limit]

def _clear_saved_token_if_matches(h):
    """Delete the saved session only when this request actually used it."""
    auth = (h or {}).get("Authorization", "")
    if not auth.startswith("Bearer "): return
    token = auth[7:].strip()
    if not token: return
    try:
        saved = json.loads(AUTH_TOKEN_FILE.read_text()).get("token")
    except Exception:
        return
    if saved and saved == token:
        try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
        except OSError: pass

def _hardware_headers(force_mode=None):
    try: bs = recommend_batch_size(MAX_SEQ_LEN, force_mode=force_mode)
    except Exception: bs = 1
    gpu = "UNKNOWN"
    try:
        if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
            dev_idx = train_device.index if (train_device is not None and train_device.index is not None) else 0
            gpu = torch.cuda.get_device_name(dev_idx)
        elif train_backend: gpu = str(train_backend)
    except Exception: pass
    try:
        _ua, _ad = _pick_autocast(train_backend); autocast_str = "bf16" if _ua else "fp32"
    except Exception: autocast_str = "fp32"
    native, can_fp32, can_8bit = _detect_optim_capability()

    # NEW in v2.5: report CUDA compute capability so the coordinator can pool
    # same-arch GPUs together in bf16 and fall back to fp32 only for mixed pools.
    # Only populated for CUDA — ROCm/XPU/MPS/DirectML/CPU leave this empty so the
    # server's name-based inference produces consistent values across old and new
    # clients (all ROCm clients resolve to "gfx_unknown", etc.).
    _cc = ""
    if train_backend == "CUDA" and torch.cuda.is_available():
        try:
            _dev_idx = train_device.index if (train_device is not None and train_device.index is not None) else 0
            _cap = torch.cuda.get_device_capability(_dev_idx)
            _cc = f"sm_{_cap[0]}{_cap[1]}"
        except Exception:
            _cc = ""

    return {
        "X-Client-Backend":   (train_backend or "UNKNOWN").upper(),
        "X-Client-Batch-Size": str(bs),
        "X-Client-Vram-Gb":   f"{memory_config.get('ram_gb', 0.0):.2f}",
        "X-Client-Gpu-Name":  gpu,
        "X-Client-Autocast":  autocast_str,
        "X-Client-Optim":     native,
        "X-Client-Optim-Can-Fp32": "1" if can_fp32 else "0",
        "X-Client-Optim-Can-8bit": "1" if can_8bit else "0",
        "X-Client-Engram":    "1" if ENGRAM_ACTIVE else "0",
        "X-Client-Device":    DEVICE_ID,
        "X-Client-Compute-Cap": _cc,   # NEW
        "X-Client-Version":   CLIENT_VERSION,
    }

def _doctor_package_version(name):
    """Return an installed distribution version, or None. Optional packages are probed lazily."""
    try:
        return importlib_metadata.version(name)
    except Exception:
        try:
            mod = __import__(name.replace("-", "_"))
            value = getattr(mod, "__version__", None)
            return str(value) if value else "installed"
        except Exception:
            return None

def _doctor_sha256_local(path):
    """Hash a shipped file after normalizing CRLF, matching the release manifest convention."""
    try:
        data = Path(path).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()
    except OSError:
        return None

def _doctor_session_probe():
    """Read the saved session without ever returning the token for display."""
    result = {"present": False, "username": None, "server": None, "remember": None, "error": None}
    try:
        data = json.loads(AUTH_TOKEN_FILE.read_text(encoding="utf-8"))
        token = data.get("token") if isinstance(data, dict) else None
        if isinstance(token, str) and token.strip():
            result.update({
                "present": True,
                "username": data.get("username"),
                "server": data.get("server"),
                "remember": data.get("remember"),
            })
            return result, token
        result["error"] = "auth file has no usable token"
    except FileNotFoundError:
        pass
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result, None

def _doctor_update_probe():
    """Fetch and verify the signed update manifest without applying an update."""
    result = {
        "checked": False,
        "signature": None,
        "remote_version": None,
        "expires": None,
        "client_present": None,
        "html_present": None,
        "client_matches": None,
        "html_matches": None,
        "error": None,
    }
    try:
        response = requests.get(VERSION_URL, timeout=5)
        response.raise_for_status()
        signature = requests.get(VERSION_SIG_URL, timeout=5)
        signature.raise_for_status()
        manifest = _parse_manifest(response.content)
        result["checked"] = True
        result["remote_version"] = manifest.get("version")
        result["expires"] = manifest.get("expires")
        result["signature"] = bool(_verify_manifest(response.content, signature.text.strip()))
        local_py = _doctor_sha256_local(Path(__file__).resolve())
        local_html = _doctor_sha256_local(Path(__file__).resolve().parent / "client.html")
        result["client_present"] = local_py is not None
        result["html_present"] = local_html is not None
        result["client_matches"] = bool(local_py and local_py == manifest.get("sha256_clientpy"))
        result["html_matches"] = bool(local_html and local_html == manifest.get("sha256_clienthtml"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

def _doctor_http_probe(url, headers=None, timeout=3.0):
    started = time.time()
    try:
        with requests.get(url, headers=headers or {}, timeout=timeout, stream=True, allow_redirects=True) as response:
            status = response.status_code
        return {
            "ok": 200 <= status < 400,
            "status": status,
            "ms": int((time.time() - started) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {"ok": False, "status": None, "ms": int((time.time() - started) * 1000), "error": f"{type(exc).__name__}: {exc}"}

def _doctor_self_test(device, backend):
    if device is None or backend == "DIRECTML":
        return {"status": "info", "detail": "no supported torch device for this backend", "ms": None}
    started = time.time()
    try:
        with torch.no_grad():
            left = torch.ones((8, 8), device=device)
            right = left @ left
        if device.type == "cuda": torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"): torch.mps.synchronize()
        elif device.type == "xpu" and hasattr(torch, "xpu"): torch.xpu.synchronize()
        elapsed = int((time.time() - started) * 1000)
        del left, right
        return {"status": "ok", "detail": f"8x8 matmul on {device} ({elapsed} ms)", "ms": elapsed}
    except Exception as exc:
        return {"status": "fail", "detail": f"{type(exc).__name__}: {exc}", "ms": None}

def _doctor_add(checks, category, name, status, detail):
    checks.append({"category": category, "name": name, "status": status, "detail": str(detail)})

def _doctor_summary(checks):
    counts = {"ok": 0, "warn": 0, "fail": 0, "info": 0}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    return counts

def _doctor_render_text(report):
    lines = ["CrowdGPT doctor", ""]
    category = None
    for check in report["checks"]:
        if check["category"] != category:
            category = check["category"]
            if len(lines) > 2: lines.append("")
            lines.append(f"{category}:")
        lines.append(f"  [{check['status']}] {check['name']}: {check['detail']}")
    summary = report["summary"]
    lines.extend([
        "",
        f"Summary: {summary['ok']} ok, {summary['warn']} warning(s), {summary['fail']} failure(s), {summary['info']} info",
    ])
    lines.append("No blocking problems found." if summary["fail"] == 0 else "Doctor found blocking problems. See [fail] entries above.")
    return "\n".join(lines)

def _doctor_collect_report(args):
    checks = []
    client_path = Path(__file__).resolve()
    _doctor_add(checks, "Client", "version", "ok", f"{CLIENT_VERSION} at {client_path}")
    _doctor_add(checks, "Client", "python", "ok", f"{platform.python_version()} ({platform.python_implementation()})")
    _doctor_add(checks, "Client", "platform", "ok", f"{platform.system()} {platform.release()} {platform.machine()}")

    cpu_count = os.cpu_count() or 0
    ram_gb = None
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
    except Exception:
        try:
            if hasattr(os, "sysconf"):
                ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except Exception:
            ram_gb = None
    if cpu_count:
        _doctor_add(checks, "System", "cpu", "ok", f"{cpu_count} logical core(s)")
    else:
        _doctor_add(checks, "System", "cpu", "warn", "could not determine logical CPU count")
    _doctor_add(checks, "System", "ram", "ok" if ram_gb else "warn", f"{ram_gb:.1f} GB" if ram_gb else "could not determine total RAM")

    checkpoint_writable = False
    checkpoint_error = "unknown error"
    try:
        with tempfile.NamedTemporaryFile(dir=CHECKPOINT_DIR, prefix=".doctor-", delete=True) as probe:
            probe.write(b"ok")
        checkpoint_writable = True
    except Exception as exc:
        checkpoint_error = f"{type(exc).__name__}: {exc}"
    _doctor_add(checks, "Storage", "checkpoint", "ok" if checkpoint_writable else "fail",
                f"{CHECKPOINT_DIR} is writable" if checkpoint_writable else f"{CHECKPOINT_DIR} is not writable ({checkpoint_error})")
    try:
        usage = shutil.disk_usage(str(CHECKPOINT_DIR))
        _doctor_add(checks, "Storage", "disk", "ok", f"{usage.free / 1024**3:.1f} GB free of {usage.total / 1024**3:.1f} GB")
    except Exception as exc:
        _doctor_add(checks, "Storage", "disk", "warn", f"could not read disk usage: {type(exc).__name__}: {exc}")

    required_packages = ["torch", "numpy", "requests", "transformers", "huggingface_hub", "safetensors"]
    optional_packages = ["rich", "psutil", "pywebview", "bitsandbytes", "sentencepiece", "tiktoken", "torch-directml"]
    package_versions = {}
    for name in required_packages:
        version = _doctor_package_version(name)
        package_versions[name] = version
        _doctor_add(checks, "Packages", name, "ok" if version else "fail", version or "not installed")
    for name in optional_packages:
        version = _doctor_package_version(name)
        package_versions[name] = version
        _doctor_add(checks, "Packages", name, "ok" if version else "info", version or "not installed (optional)")

    available = get_available_backends()
    available_names = [name for name, present in available.items() if present]
    _doctor_add(checks, "Backend", "available", "ok" if available_names else "fail", ", ".join(available_names) or "none")
    requested = (getattr(args, "backend", None) or "auto").lower()
    if requested == "auto":
        selected = get_best_default_backend()
        _doctor_add(checks, "Backend", "requested", "ok", f"auto -> {selected}")
    elif available.get(requested):
        selected = requested
        _doctor_add(checks, "Backend", "requested", "ok", f"{requested} is available")
    else:
        selected = get_best_default_backend()
        _doctor_add(checks, "Backend", "requested", "fail", f"{requested} requested but not available; best fallback is {selected}")

    device = None
    gpu_index = 0
    gpus = []
    if selected in ("cuda", "rocm"):
        gpus = get_gpu_list()
        if gpus:
            saved = _load_saved_gpu_idx()
            gpu_index = saved if any(g["idx"] == saved for g in gpus) else gpus[0]["idx"]
            chosen = next(g for g in gpus if g["idx"] == gpu_index)
            if not chosen["arch_ok"]:
                compatible = [g for g in gpus if g["arch_ok"]]
                _doctor_add(checks, "Backend", "gpu-arch", "fail",
                            f"{chosen['name']} needs {chosen['arch_needed']} but this build supports a different architecture")
                if compatible: gpu_index = compatible[0]["idx"]
            else:
                _doctor_add(checks, "Backend", "gpu-arch", "ok", f"{chosen['name']} is supported")
            for gpu in gpus:
                _doctor_add(checks, "Backend", f"gpu-{gpu['idx']}", "ok" if gpu["arch_ok"] else "warn",
                            f"{gpu['name']} ({gpu['vram_gb']:.1f} GB, {gpu['arch_needed']})")
        else:
            _doctor_add(checks, "Backend", "gpu", "warn", "no CUDA/ROCm GPU details reported")
        device = torch.device(f"cuda:{gpu_index}")
    elif selected == "mps":
        device = torch.device("mps")
    elif selected == "xpu":
        device = torch.device("xpu")
    elif selected == "cpu":
        device = torch.device("cpu")

    global train_backend, train_device
    old_backend, old_device = train_backend, train_device
    try:
        train_backend, train_device = selected.upper(), device
        auto_detect_vram_budget()
        memory_config["auto_gb"] = memory_config["ram_gb"]
        apply_vram_override(getattr(args, "vram_gb", 0.0) or 0.0)
    except Exception as exc:
        _doctor_add(checks, "Backend", "memory", "warn", f"could not estimate memory budget: {type(exc).__name__}: {exc}")
    finally:
        train_backend, train_device = old_backend, old_device

    _doctor_add(checks, "Backend", "selected", "ok" if selected in available_names else "fail", f"{selected} on {device or 'directml'}")
    _doctor_add(checks, "Backend", "budget", "ok", f"{memory_config.get('ram_gb', 0.0):.2f} GB")
    batch_size = None
    try:
        batch_size = recommend_batch_size(MAX_SEQ_LEN)
        _doctor_add(checks, "Backend", "batch-size", "ok" if batch_size else "warn", f"{batch_size} sequences of {MAX_SEQ_LEN} tokens")
    except Exception as exc:
        _doctor_add(checks, "Backend", "batch-size", "warn", f"could not estimate: {type(exc).__name__}: {exc}")
    _doctor_add(checks, "Backend", "optimizer", "ok", _predict_optim(selected.upper()))

    session, token = _doctor_session_probe()
    if session["present"]:
        _doctor_add(checks, "Session", "saved", "ok", f"{session['username'] or 'unknown user'} on {session['server'] or 'unknown server'}")
    else:
        _doctor_add(checks, "Session", "saved", "info", session["error"] or "no saved session; use --register or --user/--password")

    update = _doctor_update_probe()
    if not update["checked"]:
        _doctor_add(checks, "Updates", "manifest", "warn", f"could not fetch update manifest: {update['error']}")
    else:
        signature_status = "ok" if update["signature"] else "fail"
        _doctor_add(checks, "Updates", "signature", signature_status, "signed manifest verified" if update["signature"] else "signature did not verify")
        _doctor_add(checks, "Updates", "version", "ok", f"local {CLIENT_VERSION}, remote {update['remote_version'] or 'unknown'}")
        if update["expires"]:
            try:
                expired = datetime.date.fromisoformat(update["expires"]) < datetime.date.today()
                _doctor_add(checks, "Updates", "expiry", "warn" if expired else "ok",
                            f"manifest expires {update['expires']}" + (" (expired)" if expired else ""))
            except ValueError:
                _doctor_add(checks, "Updates", "expiry", "warn", f"unparseable expiry {update['expires']}")
        if update["client_present"] is False:
            _doctor_add(checks, "Updates", "client.py", "fail", "working file is missing")
        elif update["client_matches"] is False:
            _doctor_add(checks, "Updates", "client.py", "warn", "working file differs from the signed release manifest")
        elif update["client_matches"] is True:
            _doctor_add(checks, "Updates", "client.py", "ok", "matches the signed release manifest")
        if update["html_present"] is False:
            _doctor_add(checks, "Updates", "client.html", "warn", "working file is missing")
        elif update["html_matches"] is False:
            _doctor_add(checks, "Updates", "client.html", "warn", "working file differs from the signed release manifest")
        elif update["html_matches"] is True:
            _doctor_add(checks, "Updates", "client.html", "ok", "matches the signed release manifest")

    hf = _doctor_http_probe(f"https://huggingface.co/api/models/{HF_MODEL_REPO_ID}")
    _doctor_add(checks, "Network", "hugging-face", "ok" if hf["ok"] else "warn",
                f"HTTP {hf['status']} in {hf['ms']} ms" if hf["status"] else hf["error"])
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    coordinator = _doctor_http_probe(args.server.rstrip("/") + "/fl/round_status", headers, timeout=5.0)
    if coordinator["ok"]:
        coordinator_detail = f"HTTP {coordinator['status']} in {coordinator['ms']} ms"
    elif coordinator["status"] in (401, 403):
        coordinator_detail = f"reachable, but HTTP {coordinator['status']}; saved session is not authorized"
    else:
        coordinator_detail = f"HTTP {coordinator['status']} in {coordinator['ms']} ms" if coordinator["status"] else coordinator["error"]
    _doctor_add(checks, "Network", "coordinator", "ok" if coordinator["ok"] else "warn", coordinator_detail)

    self_test = _doctor_self_test(device, selected.upper())
    _doctor_add(checks, "Self-test", "matmul", self_test["status"], self_test["detail"])

    report = {
        "schema": "crowdgpt.doctor.v1",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "client": {
            "version": CLIENT_VERSION,
            "path": str(client_path),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "system": {"cpu_count": cpu_count, "ram_gb": round(ram_gb, 1) if ram_gb else None},
        "packages": package_versions,
        "backend": {
            "available": available,
            "requested": requested,
            "selected": selected,
            "device": str(device) if device is not None else "directml",
            "gpus": gpus,
            "memory_budget_gb": memory_config.get("ram_gb"),
            "batch_size": batch_size,
            "optimizer": _predict_optim(selected.upper()),
        },
        "session": session,
        "update": update,
        "network": {"hugging_face": hf, "coordinator": coordinator},
        "self_test": self_test,
        "checks": checks,
    }
    report["summary"] = _doctor_summary(checks)
    return report

def run_doctor(args):
    report = _doctor_collect_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_doctor_render_text(report))
    return 0 if report["summary"]["fail"] == 0 else 1

def _plan_placement(device, backend_name, budget_gb, bs=1, sl=MAX_SEQ_LEN,
                    u8=False, loss_chunk=LOSS_CHUNK, act_bytes=None):
    if backend_name == "ROCM": loss_chunk = min(loss_chunk, 128)
    if device is None or device.type == "cpu": return False, 0, loss_chunk
    GB = 1024 ** 3
    cd = act_bytes if act_bytes is not None else (2 if device.type in ("cuda", "xpu") else 4)
    budget = int(max(0.0, budget_gb) * GB); per_block = _bytes_per_block_params(); head_params = VOCAB_SIZE * DIM
    opt_b = 2 if u8 else 8
    def fixed_cost(lc):
        return (N_LAYERS * bs * sl * DIM * 4 + bs * sl * (DIM * 12 + MLP_HIDDEN * 3) * 4
                + per_block * cd * 2 + bs * lc * VOCAB_SIZE * 4 * 3 + int(ALLOCATOR_RESERVE_GB * GB))
    blk_res = per_block * (4 + 4 + opt_b); head_res = head_params * (4 + 4 + opt_b); head_stream = head_params * cd * 2
    lc = loss_chunk
    while lc > 32 and budget - fixed_cost(lc) <= head_stream: lc //= 2
    fixed = fixed_cost(lc); remaining = budget - fixed; head_gpu, n = False, 0
    if remaining > head_stream:
        n = int((remaining - head_stream) // blk_res); n = max(0, min(N_LAYERS, n))
        if n == N_LAYERS and remaining >= N_LAYERS * blk_res + head_res:
            head_gpu = True
        elif remaining >= head_res + blk_res:
            # Keeping the head on the GPU is cheaper than offloading it and
            # copying it back every step; stream the blocks that do not fit.
            n = max(0, min(N_LAYERS - 1, int((remaining - head_res) // blk_res)))
            head_gpu = True
    if backend_name == "DIRECTML": head_gpu, n = False, min(n, 2)
    return head_gpu, n, lc

def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)

def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def _is_directml(t):
    try: return t.device.type == 'privateuseone'
    except Exception: return False
def _is_directml_device(d):
    try: return d.type == 'privateuseone'
    except Exception: return False

def _is_accel_error(msg):
    m = msg.lower()
    return any(k in m for k in ("privateuse1", "unbox expects", "not implemented for", "not supported", "unsupported data type", "bfloat16", "internal assert", "dml", "scatter", "compilegraph", "out of memory", "allocation", "device_type", "not implemented for unique"))

def _sdpa_supports_gqa():
    """torch>=2.5 exposes enable_gqa on SDPA; older builds reject the kwarg."""
    try: return "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")
    except Exception: return False

_SDPA_GQA = _sdpa_supports_gqa()

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS; self.nkv = N_KV_HEADS; self.nrep = N_HEADS // N_KV_HEADS
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

        def _expand_kv(t):
            if t.size(1) == self.nh: return t
            return t.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)

        # SDPA can broadcast grouped KV heads, so keep K/V unexpanded on torch>=2.5.
        gqa = _SDPA_GQA and self.nkv != self.nh
        if not gqa: k, v = _expand_kv(k), _expand_kv(v)
        if q.dtype != v.dtype or k.dtype != v.dtype: q, k = q.to(v.dtype), k.to(v.dtype)
        gk = {"enable_gqa": True} if gqa else {}
        if _is_directml(q): o = F.scaled_dot_product_attention(q.contiguous().cpu(), k.contiguous().cpu(), v.contiguous().cpu(), is_causal=True, **gk).to(q.device)
        elif q.device.type == 'cuda' and getattr(torch.version, 'hip', None):
            if os.environ.get('CROWDGPT_ROCM_SDPA', '1') != '0':
                try: o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, **gk)
                except RuntimeError as exc:
                    if _is_oom(exc): raise
                    o = self._chunked_attn(q, _expand_kv(k), _expand_kv(v), T)
            else: o = self._chunked_attn(q, _expand_kv(k), _expand_kv(v), T)
        else:
            try: o = F.scaled_dot_product_attention(q, k, v, is_causal=True, **gk)
            except (TypeError, NotImplementedError, RuntimeError) as exc:
                if not gqa or _is_oom(exc): raise
                o = F.scaled_dot_product_attention(q, _expand_kv(k), _expand_kv(v), is_causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(B, T, C))

    def _chunked_attn(self, q, k, v, T, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)
            aw = (q[:, :, i:e, :] @ k[:, :, :e, :].transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            ri = torch.arange(i, e, device=q.device).unsqueeze(1); ci = torch.arange(e, device=q.device).unsqueeze(0)
            aw = aw.masked_fill(~(ci <= ri).unsqueeze(0).unsqueeze(0), float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v[:, :, :e, :])
        return torch.cat(chunks, dim=2)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False); self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False); self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(DIM); self.attn = GroupedQueryAttention()
        self.ln_2 = nn.LayerNorm(DIM); self.mlp = SwiGLU()
    def forward(self, x, cos, sin, use_chunked=True):
        x = x + self.attn(self.ln_1(x), cos, sin, use_chunked)
        return x + self.mlp(self.ln_2(x))

class EngramMemory(nn.Module):
    def __init__(self, bn, device):
        super().__init__()
        self.device = device
        rows = ENG_NUM_BUCKETS if ENGRAM_ACTIVE else 1
        self.table = nn.Embedding(rows, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        if self.table.num_embeddings <= 1:
            return torch.zeros(idx.shape[0], idx.shape[1], DIM, device=self.device, dtype=torch.float32)
        idx_cpu = idx.detach().cpu()
        px = torch.cat([torch.zeros_like(idx_cpu[:, :1]), idx_cpu[:, :-1]], dim=1)
        hi = (px * 1000003 + idx_cpu) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.reshape(-1), return_inverse=True)
        cr = self.table(uq)
        out = cr[inv].view(idx.shape[0], idx.shape[1], DIM)
        return out.to(self.device)

def _streamed_block(block, dev, dtype, x, cos, sin, use_chunked):
    if dtype is not None:
        params = {n: (p.to(dev, dtype=dtype) if p.dim() > 1 else p.to(dev)) for n, p in block.named_parameters()}
        with torch.autocast(device_type=dev.type, dtype=dtype):
            return torch.func.functional_call(block, params, (x, cos, sin, use_chunked))
    params = {n: p.to(dev) for n, p in block.named_parameters()}
    return torch.func.functional_call(block, params, (x, cos, sin, use_chunked))

class SotaGPT(nn.Module):
    def __init__(self, bn, device, gpu_layers=None, bs=1, sl=MAX_SEQ_LEN,
                 u8_override=None, act_bytes_override=None):
        super().__init__()
        self.main_device = device; self.accel_device = device; self.cpu_device  = torch.device('cpu'); self.backend_name = bn
        is_cpu = (device is None) or (device.type == 'cpu')
        if gpu_layers is None:
            u8 = u8_override if u8_override is not None else (
                has_bitsandbytes() and device is not None and device.type in ('cuda',))
            head_gpu, gpu_layers, self.loss_chunk_hint = _plan_placement(
                device, bn, memory_config["ram_gb"], bs=bs, sl=sl,
                u8=u8, act_bytes=act_bytes_override,
            )
        else:
            head_gpu = int(gpu_layers) >= N_LAYERS; self.loss_chunk_hint = LOSS_CHUNK
        self.gpu_layers = 0 if is_cpu else max(0, min(N_LAYERS, int(gpu_layers)))
        self.gpu_layer_set = set(range(self.gpu_layers))
        self.head_on_gpu = bool(head_gpu) and not is_cpu
        self.wte = nn.Embedding(VOCAB_SIZE, DIM); self.engram = EngramMemory(bn, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM); self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]: self.wte.weight = self.lm_head.weight
        cm_main, sm_main = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos_main", cm_main); self.register_buffer("freqs_sin_main", sm_main)
        cm_cpu, sm_cpu = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, self.cpu_device)
        self.register_buffer("freqs_cos_cpu", cm_cpu); self.register_buffer("freqs_sin_cpu", sm_cpu)
        for i, block in enumerate(self.blocks): block.to(device if i in self.gpu_layer_set else self.cpu_device)
        self._place_head(); self.engram.table.to('cpu')

    def _place_head(self):
        head_dev = self.main_device if self.head_on_gpu else self.cpu_device
        self.wte.to(head_dev); self.lm_head.to(head_dev); self.ln_f.to(self.main_device)

    def set_gpu_layers(self, n):
        n = max(0, min(N_LAYERS, int(n)))
        if self.main_device.type == 'cpu': n = 0
        self.gpu_layers = n; self.gpu_layer_set = set(range(n))
        if n < N_LAYERS: self.head_on_gpu = False
        for i, blk in enumerate(self.blocks): blk.to(self.main_device if i < n else self.cpu_device)
        self._place_head()

    def force_head_cpu(self): self.head_on_gpu = False; self._place_head()

    def to_cpu_only(self):
        self.gpu_layers = 0; self.gpu_layer_set = set(); self.head_on_gpu = False; self.main_device = self.cpu_device
        for blk in self.blocks: blk.to(self.cpu_device)
        self._place_head(); self.engram.table.to('cpu'); self.engram.device = self.cpu_device

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        if w.device != self.main_device: return w.to(self.main_device, dtype=dtype) if dtype is not None else w.to(self.main_device)
        if dtype is not None and w.dtype != dtype: return w.to(dtype)
        return w

    def forward_features(self, idx, use_chunked=True, use_ckpt=True, stream_dtype=None, head_w=None):
        dev = self.main_device; wdev = self.wte.weight.device
        if head_w is not None and not self.head_on_gpu and head_w.device == dev:
            x = F.embedding(idx.to(dev), head_w)
            if x.dtype != torch.float32: x = x.float()
        else:
            x = F.embedding(idx.to(wdev), self.wte.weight)
            if x.device != dev: x = x.to(dev)
        if ENGRAM_ACTIVE:
            eng = self.engram(idx)
            if eng.dtype != x.dtype: eng = eng.to(x.dtype)
            x = x + eng
        if dev.type == 'cpu': cos, sin = self.freqs_cos_cpu, self.freqs_sin_cpu
        else: cos, sin = self.freqs_cos_main, self.freqs_sin_main
        for i, block in enumerate(self.blocks):
            if dev.type == 'cpu' or i in self.gpu_layer_set:
                if use_ckpt: x = checkpoint(block, x, cos, sin, use_chunked, use_reentrant=False, preserve_rng_state=False)
                else: x = block(x, cos, sin, use_chunked)
            else:
                x = checkpoint(_streamed_block, block, dev, stream_dtype, x, cos, sin, use_chunked, use_reentrant=False, preserve_rng_state=False)
        return self.ln_f(x)

    def forward(self, idx, use_chunked=True, use_ckpt=True): return F.linear(self.forward_features(idx, use_chunked, use_ckpt), self.head_weight())

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw; o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = 1
                for d in p.shape: s *= int(d)
                src = ft[o:o + s].view(p.shape)
                if p.device.type == 'privateuseone':
                    src = src.to(device='cpu', dtype=p.dtype).contiguous(); src = src.to(device=p.device)
                else: src = src.to(device=p.device, dtype=p.dtype)
                p.data.copy_(src); o += s

def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True); r.raise_for_status(); return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries - 1: time.sleep(2 ** (a + 1))
            else: raise

class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500
    def __init__(self, repo_id, ci, ss=10 * 1024 * 1024, tps=65, slot=0, at=None, cohort_mode=False):
        self.repo_id = repo_id; self.ci = ci; self.sub_size = ss; self.tps = tps; self.auth_token = at
        self.cohort_mode = cohort_mode; self._fmt = "chunk_{:04d}.bin"; self.chunk_size = self._discover(ci)
        self.off = (slot * ss) % self.chunk_size; self.data = None; self.n = 0; self.steps_used = 0
        self._pft = None; self._pfr = None; self._lock = threading.Lock(); self._load(self.off); self._prefetch(self.off + ss)

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
                except Exception: time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")

    def _slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32); self.data = tk; self.n = len(tk) // self.tps; self.steps_used = 0

    def _load(self, off): self._set(self._slice(off)); self.off = off

    def _prefetch(self, off):
        if off >= self.chunk_size: return
        with self._lock: self._pfr = None
        def w():
            try:
                r = self._slice(off)
                with self._lock: self._pfr = (off, r)
            except Exception: pass
        self._pft = threading.Thread(target=w, daemon=True); self._pft.start()

    def needs_new_subchunk(self):
        if self.n == 0: return True
        if self.cohort_mode: return False
        return self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, srv=None, fmt="bf16"):
        no = self.off + self.sub_size
        if no < self.chunk_size:
            if self._pft: self._pft.join(timeout=120); self._pft = None
            with self._lock:
                r = self._pfr; self._pfr = None
            if r and r[0] == no: self.off = no; self._set(r[1])
            else: self._load(no)
            self._prefetch(no + self.sub_size); return True
        self._req(srv, fmt); self.chunk_size = self._discover(self.ci); self.off = 0; self._load(0); self._prefetch(self.sub_size); return True

    def _req(self, srv, fmt="bf16"):
        if not srv or self.cohort_mode: return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers={**h, **_hardware_headers()}, timeout=30)
            if r.status_code != 200: return False
            ct = (r.headers.get("content-type") or "").lower()
            if "json" in ct: md = r.json()
            else:
                raw = r.content
                if len(raw) < 4: return False
                ml = struct.unpack('<I', raw[:4])[0]
                if len(raw) < 4 + ml: return False
                md = json.loads(raw[4:4 + ml].decode())
            ni = md.get("datasetConfig", {}).get("chunkIdx", self.ci)
            if ni != self.ci: self.ci = ni; return True
        except Exception: pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0: self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed); s = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[x:x + self.tps - 1] for x in s]); tgt = np.stack([self.data[x + 1:x + self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        return None, "invalid_credentials"
    except Exception: return None, "network"

def _register_error_code(r):
    """Map coordinator responses to stable codes the GUI and CLI can share."""
    detail = ""
    try:
        body = r.json()
        if isinstance(body, dict):
            detail = " ".join(str(body.get(k, "")) for k in ("detail", "error", "code", "message")).lower()
    except Exception:
        pass
    if "username" in detail and any(w in detail for w in ("exist", "taken", "already")): return "username_taken"
    if "email" in detail and any(w in detail for w in ("exist", "taken", "already", "registered")): return "email_taken"
    if "password" in detail and any(w in detail for w in ("weak", "short", "6", "length")): return "weak_password"
    if r.status_code == 429: return "rate_limited"
    if r.status_code == 409: return "account_conflict"
    if r.status_code == 400: return "invalid_input"
    if r.status_code >= 500: return "server_error"
    return "register_failed"

def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200:
            try:
                tok = r.json().get("token")
            except Exception:
                return None, "server_error"
            return (tok, None) if tok else (None, "register_failed")
        return None, _register_error_code(r)
    except Exception: return None, "network"

class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv = srv; self.h = h; self.cr = cr
        self.stop_training = threading.Event(); self.shutdown = threading.Event()
        self.se = se; self.emit = emit; self.tid = None
        self.rejected = None   # reason text once the coordinator blocks this client; run_single_round_wrapper raises it
        self._lock = threading.Lock()
        self._closed = False        # set by close(): from then on a refusal can no longer change this round
        self._handed_over = False   # set by commit(): this round's delta file now belongs to the uploader
        self._t = threading.Thread(target=self._r, daemon=True)
    JOIN_TIMEOUT = 3.0
    def start(self): self._t.start()
    def stop(self): self.shutdown.set()
    def should_stop(self): return self.stop_training.is_set()
    def _reject(self, reason):
        """Heartbeat thread: record a refusal, unless the round was already closed (then the next round's
        wait_for_round reports it). A refusal after the hand-over is still recorded. Returns whether it was."""
        with self._lock:
            if self._closed: return False
            self.rejected = reason; self.stop_training.set(); return True
    def commit(self):
        """Main thread, before handing the delta to the uploader: True (and the file is the uploader's from now on)
        if no refusal was recorded yet, False if one was (the caller discards the delta). This does not stop
        refusals from being recorded: one arriving later still ends the client, it just doesn't touch the upload."""
        with self._lock:
            if self.rejected is None: self._handed_over = True
            return self.rejected is None
    def close(self):
        """Main thread, end of round: stop polling, let an in-flight check finish, then close the round.
        Returns the refusal recorded before closing. A check still running after JOIN_TIMEOUT can't record anything."""
        self.shutdown.set()
        if self._t.is_alive() and self._t is not threading.current_thread(): self._t.join(self.JOIN_TIMEOUT)
        with self._lock:
            self._closed = True; return self.rejected
    def _log_refusal(self, detail):
        tail = ("Stopping; the delta already handed to the uploader is left to finish." if self._handed_over
                else "Stopping without upload.")
        self.emit('log', f"⛔ coordinator refused this client during the round: {detail}. {tail}")
    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                tid = self.tid
                r = requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10, params={"taskId": tid} if tid else None)
                try: hj = r.json() if r.status_code == 200 else {}
                except ValueError: hj = {}
                if not isinstance(hj, dict): hj = {}
                blocked = hj.get("status") == "blocked"
                if tid and hj.get("task") == "gone":
                    self.emit('log', f"Task {tid} no longer exists on the server (restarted or pool closed). Stopping."); self.stop_training.set()
                if blocked:
                    # Record the block at once: the reason lookup below may fail or hang, and until the refusal is
                    # recorded commit() would still hand this round's delta to the uploader.
                    detail = "client blocked by the coordinator"
                    recorded = self._reject(f"coordinator rejected: {detail}")
                    try:   # heartbeat says only "blocked"; round_status carries the reason
                        r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                        if r.status_code == 403 and _http_detail(r) and self._reject(f"coordinator rejected: HTTP 403 — {_http_detail(r)}"):
                            detail = _http_detail(r)
                    except Exception: pass
                    if recorded: self._log_refusal(detail)
                    return
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 403:
                    detail = _http_detail(r) or "forbidden"
                    if self._reject(f"coordinator rejected: HTTP 403 — {detail}"): self._log_refusal(detail)
                    return
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr: self.emit('log', "Next round. Stopping."); self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0)) * 60
                    if rm <= 3: self.emit('log', f"ULTIMATUM: {rm:.0f} min left!"); self.stop_training.set()
            except Exception: pass
            self.shutdown.wait(timeout=15)

def wait_for_round(srv, h, se, emit):
    overlay_on = False
    last_error = None
    next_error_log = 0.0
    def show_wait():
        nonlocal overlay_on
        if not overlay_on: emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True}); overlay_on = True
    def hide():
        nonlocal overlay_on
        if overlay_on: emit('overlay_hide', {}); overlay_on = False
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                last_error = None
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting'); show_wait()
                    for _ in range(30):
                        if se.is_set(): hide(); return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False): hide(); return st
                emit('status', 'waiting'); show_wait()
                for _ in range(30):
                    if se.is_set(): hide(); return None
                    time.sleep(1)
            elif r.status_code == 403:
                detail = _http_detail(r); hide()
                emit('log', f"⛔ coordinator refused the round status request (403): {detail or 'forbidden'}")
                raise CoordinatorRejected(f"coordinator rejected: HTTP 403{' — ' + detail if detail else ''}", status=403)
            else:
                now = time.time()
                detail = _http_detail(r)
                key = (r.status_code, detail)
                if key != last_error or now >= next_error_log:
                    msg = f"[wait] coordinator status HTTP {r.status_code}"
                    if detail: msg += f": {detail}"
                    emit('log', msg + " — retrying in 10s")
                    next_error_log = now + 30
                last_error = key
                time.sleep(10)
        except CoordinatorRejected: raise
        except Exception as exc:
            now = time.time()
            detail = f"{type(exc).__name__}: {exc}"[:180]
            key = ("exception", detail)
            if key != last_error or now >= next_error_log:
                emit('log', f"[wait] coordinator status request failed: {detail} — retrying in 10s")
                next_error_log = now + 30
            last_error = key
            time.sleep(10)
    hide(); return None

def fetch_task_metadata(srv, h, se, emit, hb=None):
    bad_4xx = 0
    while not se.is_set() and (not hb or not hb.should_stop()):
        try:
            r = requests.get(f"{srv}/fl/task?format=bf16&skip_weights=true", headers={**(h or {}), **_hardware_headers()}, timeout=30)
            if r.status_code == 401:
                emit('log', "⚠️ auth rejected (401) — clearing the saved session")
                _clear_saved_token_if_matches(h)
                emit('login_error', {'code': 'session_expired'}); raise AuthRejected("auth rejected: HTTP 401")
            if r.status_code == 403:
                detail = _http_detail(r)
                emit('log', f"⛔ coordinator rejected the task request (403): {detail or 'forbidden'}")
                emit('login_error', {'code': 'forbidden'})
                raise CoordinatorRejected(f"coordinator rejected: HTTP 403{' — ' + detail if detail else ''}", status=403)
            if r.status_code == 426:
                detail = _http_detail(r)
                emit('log', f"⬆️ update required by the coordinator: {detail or 'install a newer client'}")
                raise CoordinatorRejected(f"update required: HTTP 426{' — ' + detail if detail else ''}", status=426)
            if r.headers.get("X-Status") == "wait":
                pool_fill = r.headers.get("X-Pool-Fill"); pool_size_hdr = r.headers.get("X-Pool-Size")
                if pool_fill:
                    msg = f"⏳ Pool matchmaking {pool_fill}"
                    if pool_size_hdr: msg += f"/{pool_size_hdr}"
                    emit('log', msg + " (waiting for peers)")
                else:
                    emit('log', f"⏳ {_http_detail(r) or 'waiting for available work'}")
                try: r.close()
                except Exception: pass
                for _ in range(10):
                    if se.is_set() or (hb and hb.should_stop()): return None
                    time.sleep(1)
                continue
            if r.status_code == 200:
                bad_4xx = 0
                ct = (r.headers.get("content-type") or "").lower()
                if "json" in ct:
                    md = r.json()
                    if md.get("status") == "wait":
                        fill = md.get("poolFill"); total = md.get("poolSize")
                        if fill and total: emit('log', f"⏳ Pool matchmaking {fill}/{total} (waiting for peers)")
                        elif md.get("message"): emit('log', f"⏳ {md.get('message')}")
                        for _ in range(10):
                            if se.is_set() or (hb and hb.should_stop()): return None
                            time.sleep(1)
                        continue
                    return md
                raw = r.content
                if len(raw) >= 4:
                    ml = struct.unpack('<I', raw[:4])[0]
                    if len(raw) >= 4 + ml: return json.loads(raw[4:4 + ml].decode())
                raise RuntimeError("Invalid task metadata response")
            if r.status_code in (429, 503, 504):
                detail = _http_detail(r)
                emit('log', f"[wait] /fl/task HTTP {r.status_code}{': ' + detail if detail else ''} — retrying in 10s")
                time.sleep(10); continue
            if 400 <= r.status_code < 500:
                bad_4xx += 1
                detail = _http_detail(r)
                emit('log', f"[wait] /fl/task HTTP {r.status_code}{': ' + detail if detail else ''} (attempt {bad_4xx}/3)")
                if bad_4xx >= 3:
                    return None
                time.sleep(5); continue
            detail = _http_detail(r)
            emit('log', f"[wait] /fl/task HTTP {r.status_code}{': ' + detail if detail else ''} — retrying in 10s")
            time.sleep(10)
        except (AuthRejected, CoordinatorRejected): raise
        except Exception as e:
            if se.is_set() or (hb and hb.should_stop()): return None
            emit('log', f"Task metadata fetch failed: {e}"); time.sleep(10)
    return None

def _weight_cache_path(revision, filename):
    rev = str(revision or "main").strip().replace("/", "_").replace("\\", "_")
    d = CHECKPOINT_DIR / "hf_weights" / rev; d.mkdir(parents=True, exist_ok=True); return d / filename

def _download_file_with_progress(url, dest, emit, stop_evt, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(expected_size) if expected_size is not None else None
    if dest.exists():
        sz = dest.stat().st_size
        if expected is not None and sz == expected: return dest
        if expected is None and sz > 10_000_000: return dest
        try: dest.unlink()
        except Exception: pass
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(3):
        if stop_evt.is_set(): raise KeyboardInterrupt("Stop requested")
        try: done = tmp.stat().st_size if tmp.exists() else 0
        except OSError: done = 0
        if expected is not None and done > expected:
            # Stale/oversized fragment; a Range request would be unsatisfiable.
            try: tmp.unlink()
            except Exception: pass
            done = 0
        if expected is not None and done == expected:
            os.replace(tmp, dest); return dest
        try:
            hdrs = {"Range": f"bytes={done}-"} if done > 0 else {}
            r = requests.get(url, headers=hdrs, stream=True, timeout=(15, 3600), allow_redirects=True)
            if r.status_code == 416:
                if expected is not None and done >= expected:
                    os.replace(tmp, dest); return dest
                try: tmp.unlink()
                except Exception: pass
                done = 0
                r.raise_for_status()
            if r.status_code not in (200, 206): r.raise_for_status()
            append = r.status_code == 206 and done > 0
            if not append: done = 0  # server ignored Range (or we had nothing): rewrite
            remaining = int(r.headers.get("content-length", 0) or 0)
            total = expected if expected is not None else done + remaining
            last_emit = 0.0
            with open(tmp, "ab" if append else "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_evt.is_set(): raise KeyboardInterrupt("Stop requested")
                    if not chunk: continue
                    f.write(chunk); done += len(chunk); now = time.time()
                    if now - last_emit > 0.25: emit("overlay_progress", {"done": done, "total": total}); last_emit = now
            if expected is not None and done != expected: raise RuntimeError(f"Size mismatch for {label}: got {done}, expected {expected}")
            os.replace(tmp, dest); return dest
        except KeyboardInterrupt:
            # Keep the partial file so a later run can resume from it.
            raise
        except Exception as e:
            last_err = e
            if attempt < 2: emit("log", f"Download retry {attempt + 1} for {label}: {e}"); time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {label}: {last_err}")

def _load_safetensors_flat(path, expected_numel, label="weights"):
    try: from safetensors.torch import load_file
    except ImportError as e: raise RuntimeError("Missing dependency: safetensors.") from e
    tensors = load_file(str(path))
    if "weights" in tensors: t = tensors["weights"]
    elif "flat" in tensors: t = tensors["flat"]
    elif len(tensors) == 1: t = next(iter(tensors.values()))
    else: raise RuntimeError(f"{label}: cannot identify tensor key.")
    t = t.contiguous().reshape(-1)
    if t.dtype not in (torch.bfloat16, torch.float32): t = t.to(torch.float32)
    del tensors; gc.collect()
    if t.numel() != int(expected_numel): raise RuntimeError(f"{label} size mismatch: got {t.numel()}, expected {expected_numel}.")
    return t

def fetch_task(srv, h, prec, se, emit, hb=None):
    while not se.is_set() and (not hb or not hb.should_stop()):
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})
            req_headers = {**(h or {}), **_hardware_headers()}
            r = requests.get(f"{srv}/fl/task?format={prec}", headers=req_headers, timeout=(15, 3600), stream=True)
            if r.status_code == 401:
                r.close()
                emit('log', "⚠️ auth rejected (401) — clearing the saved session")
                _clear_saved_token_if_matches(h)
                emit('login_error', {'code': 'session_expired'})
                raise AuthRejected("auth rejected: HTTP 401")
            if r.status_code == 403:
                detail = _http_detail(r); r.close(); emit('overlay_hide', {})
                emit('log', f"⛔ coordinator rejected the task request (403): {detail or 'forbidden'}")
                emit('login_error', {'code': 'forbidden'})
                raise CoordinatorRejected(f"coordinator rejected: HTTP 403{' — ' + detail if detail else ''}", status=403)
            if r.status_code == 426:
                detail = _http_detail(r); r.close(); emit('overlay_hide', {})
                emit('log', f"⬆️ update required by the coordinator: {detail or 'update the client'}")
                raise CoordinatorRejected(f"update required: HTTP 426{' — ' + detail if detail else ''}", status=426)
            if r.status_code != 200:
                detail = _http_detail(r); r.close(); emit('overlay_hide', {})
                emit('log', f"[wait] /fl/task HTTP {r.status_code}" + (f": {detail}" if detail else "") + " — retrying in 10s")
                time.sleep(10); continue
            if r.headers.get("X-Status") == "wait":
                pool_fill = r.headers.get("X-Pool-Fill")
                pool_size_hdr = r.headers.get("X-Pool-Size")
                r.close(); emit('overlay_hide', {})
                if pool_fill:
                    msg = f"⏳ Pool matchmaking {pool_fill}"
                    if pool_size_hdr: msg += f"/{pool_size_hdr}"
                    emit('log', msg + " (waiting for peers)")
                for _ in range(10):
                    if se.is_set() or (hb and hb.should_stop()): return None, None
                    time.sleep(1)
                continue
            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []; done = 0; last = 0.0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if not chunk: continue
                if se.is_set() or (hb and hb.should_stop()):
                    r.close()
                    raise KeyboardInterrupt("Stop requested")
                chunks.append(chunk); done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total}); last = now
            r.close()
            emit('overlay_progress', {'done': done, 'total': total}); emit('overlay_hide', {}); emit('status', 'preparing')
            raw = b"".join(chunks); del chunks
            if len(raw) < 4:
                emit('log', f"/fl/task returned only {len(raw)} bytes; retrying"); time.sleep(5); continue
            ml = struct.unpack('<I', raw[:4])[0]
            if 4 + ml > len(raw):
                emit('log', f"Truncated header; retrying"); time.sleep(5); continue
            md = json.loads(raw[4:4 + ml].decode()); wb = raw[4 + ml:]
            emit('log', f"/fl/task inline: metadata OK, weights={len(wb):,} bytes")
            if len(wb) < EXPECTED_WEIGHT_BYTES:
                emit('log', f"⚠️ inline truncated at {len(wb):,}; retrying"); time.sleep(10); continue
            return md, wb
        except (AuthRejected, CoordinatorRejected):
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set() or (hb and hb.should_stop()): return None, None
            emit('log', f"Task fetch failed: {e}")
            time.sleep(15)


# =================== LOSS / TRAINING ===================

def _hf_head_identity(url):
    """(content_id, size) of a Hugging Face file WITHOUT following the CDN redirect.
    HF answers the resolve URL with X-Linked-Etag (= sha256 of the LFS file) and X-Linked-Size, so this
    is a tiny request that tells us whether the bytes changed, whatever the commit sha says."""
    try:
        r = requests.head(url, allow_redirects=False, timeout=15)
        cid = r.headers.get("X-Linked-Etag") or r.headers.get("ETag") or ""
        cid = re.sub(r"[^0-9A-Za-z]", "", cid.replace("W/", ""))[:64] or None
        sz = r.headers.get("X-Linked-Size") or (r.headers.get("Content-Length") if r.status_code == 200 else None)
        return cid, (int(sz) if sz and str(sz).isdigit() else None)
    except Exception: return None, None

def _weight_blob_path(role, cid, size, revision):
    d = CHECKPOINT_DIR / "hf_weights" / "blobs"; d.mkdir(parents=True, exist_ok=True)
    key = cid or ("rev-" + re.sub(r"[^0-9A-Za-z]", "", str(revision or "main"))[:40])
    return d / f"{role}__{key}__{int(size or 0)}.safetensors"

def fetch_weights_from_hf(md, se, emit):
    wcfg = md.get("weights") or {}
    if wcfg.get("source") != "huggingface": raise RuntimeError("Task metadata does not advertise Hugging Face weights.")
    revision = wcfg.get("revision") or "main"; files = wcfg.get("files") or {}
    model_info = files.get("model") or {}; engram_info = files.get("engram") or {}
    model_url = model_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/model_bf16.safetensors"
    engram_url = engram_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/engram_bf16.safetensors"
    jobs = []     # (role, url, path, size, label, cached)
    for role, url, info, label, active in (("model", model_url, model_info, "model_bf16.safetensors", True),
                                           ("engram", engram_url, engram_info, "engram_bf16.safetensors", bool(ENGRAM_ACTIVE))):
        if not active: continue
        cid, hsize = _hf_head_identity(url)
        size = info.get("size") or hsize
        if not size:
            try: size = int(requests.head(url, allow_redirects=True, timeout=30).headers.get("content-length", 0) or 0) or None
            except Exception: pass
        path = _weight_blob_path(role, cid, size, revision)
        cached = path.exists() and ((size is not None and path.stat().st_size == int(size)) or (size is None and path.stat().st_size > 10_000_000))
        jobs.append((role, url, path, size, label, cached, cid))
    need = [j for j in jobs if not j[5]]
    if need:
        emit("status", "downloading"); emit("overlay_show", {"title_key": "dl_weights", "indeterminate": False})
        for role, url, path, size, label, cached, cid in need:
            _download_file_with_progress(url, path, emit, se, expected_size=size, label=label)
    for role, url, path, size, label, cached, cid in jobs:
        if cached:
            emit("log", f"♻️ {label} unchanged on Hugging Face ({(cid or 'rev ' + str(revision))[:10]}…): using cached copy, no download")
        try: os.utime(path, None)            # mark as the one in use for pruning
        except Exception: pass
    emit("overlay_hide", {}); emit("status", "preparing")
    paths = {j[0]: j[2] for j in jobs}
    iw = _load_safetensors_flat(paths["model"], EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(paths["engram"], EXPECTED_ENGRAM_SIZE, "engram weights") if ENGRAM_ACTIVE else None
    try:
        for p in CHECKPOINT_DIR.rglob("*.part"):
            try: p.unlink()
            except Exception: pass
    except Exception: pass
    return iw, ie

def _chunk_ce(xs, w, ys, pos_step=64):
    """Cross-entropy without materialising the whole logits chunk in fp32.

    The fp32 upcast is the peak-memory hotspot ([B, chunk, vocab] doubles in
    size), so project a slice of positions at a time and accumulate the fp32
    sum. The final divide reproduces cross_entropy(reduction='mean') numerics.
    """
    B, T, _ = xs.shape
    total = torch.zeros((), dtype=torch.float32, device=xs.device)
    for i in range(0, T, pos_step):
        logits = F.linear(xs[:, i:i + pos_step].to(w.dtype), w)
        total = total + F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE).float(),
            ys[:, i:i + pos_step].reshape(-1),
            reduction="sum",
        )
    return total / (B * T)

def _fwl(m, x, y, sl, ua, ad, ls=1.0, preview=False, loss_chunk=LOSS_CHUNK):
    sd = ad if ua else None; nc = max(1, math.ceil(sl / loss_chunk))
    def compute():
        w = m.head_weight(sd)
        xe = m.forward_features(x, use_chunked=True, use_ckpt=True, stream_dtype=sd, head_w=w)
        y_dev = y.to(xe.device)
        lo = sum(checkpoint(_chunk_ce, xe[:, i:i + loss_chunk, :], w, y_dev[:, i:i + loss_chunk], use_reentrant=False, preserve_rng_state=False) for i in range(0, sl, loss_chunk)) / nc
        return lo, xe, w
    if ua:
        with torch.autocast(device_type=m.main_device.type, dtype=ad): lo, xe, w = compute()
    else: lo, xe, w = compute()
    cards = []
    if preview:
        try:
            with torch.no_grad():
                B, T, C = xe.shape
                items = [(b, T - 1) for b in range(min(B, 3))]
                for offset in (24, 48, 72):
                    p = T - offset - 1
                    if p >= 9: items.append((0, p))
                items = items[:6]
                for b, p in items:
                    if p < 9 or p >= T: continue
                    start = max(0, p - 9); ctx_ids = x[b, start:p + 1].tolist(); target_id = int(y[b, p].item())
                    hidden = xe[b, p, :].unsqueeze(0).to(w.dtype); logits = F.linear(hidden, w).float()
                    probs = torch.softmax(logits[0], dim=-1); prob, pred = torch.max(probs, dim=0); pred_id = int(pred.item())
                    cards.append({'context': decode_ids(ctx_ids), 'target': decode_one(target_id), 'pred': decode_one(pred_id), 'match': bool(pred_id == target_id), 'prob': safe_float(prob.item()), 'target_id': int(target_id), 'pred_id': int(pred_id)})
        except Exception: cards = []
    return lo * ls, cards

def _is_oom(e):
    if isinstance(e, torch.cuda.OutOfMemoryError): return True
    if isinstance(e, RuntimeError): return "out of memory" in str(e).lower()
    return False

def _clear_accel_cache(device):
    try:
        if device is not None and device.type == 'cuda': torch.cuda.empty_cache()
        elif device is not None and device.type == 'xpu': torch.xpu.empty_cache()
    except Exception: pass
    gc.collect()

class DegradeLadder:
    def __init__(self, model, on_step_down=None):
        self.model = model; self.stage = 0
        base_chunk = getattr(model, 'loss_chunk_hint', LOSS_CHUNK)
        self.loss_chunk = min(base_chunk, 128) if model.backend_name == "ROCM" else base_chunk
        self.on_step_down = on_step_down
    def step_down(self, emit) -> bool:
        m = self.model
        while True:
            self.stage += 1; s = self.stage
            if s == 1:
                if not m.head_on_gpu: continue
                emit('log', "🔻 VRAM rung 1: head weights → system RAM"); m.force_head_cpu()
            elif s == 2:
                if m.gpu_layers <= 0: continue
                cur, nxt = m.gpu_layers, m.gpu_layers // 2
                emit('log', f"🔻 VRAM rung 2: resident blocks {cur} → {nxt}"); m.set_gpu_layers(nxt)
            elif s == 3:
                if m.gpu_layers <= 0: continue
                emit('log', "🔻 VRAM rung 3: all blocks streamed"); m.set_gpu_layers(0)
            elif s == 4:
                old, new = self.loss_chunk, max(32, self.loss_chunk // 2)
                if new == old: continue
                self.loss_chunk = new; emit('log', f"🔻 VRAM rung 4: LOSS_CHUNK {old} → {new}")
            elif s == 5:
                old, new = self.loss_chunk, max(16, self.loss_chunk // 2)
                if new == old: continue
                self.loss_chunk = new; emit('log', f"🔻 VRAM rung 5: LOSS_CHUNK {old} → {new}")
            elif s == 6:
                if m.main_device.type == 'cpu': return False
                emit('log', "🔻 VRAM rung 6: hard fallback — CPU-only"); m.to_cpu_only(); self.loss_chunk = 16
            else: return False
            break
        if self.on_step_down is not None:
            try: self.on_step_down(self)
            except CohortAbort: raise
            except Exception: pass
        _clear_accel_cache(m.accel_device); return True

def _train_step_safe(model, x, y, sl, ua, ad, ls, preview, ladder, emit):
    while True:
        try:
            lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=preview, loss_chunk=ladder.loss_chunk)
            lo.backward(); return lo, cards
        except RuntimeError as e:
            if not _is_oom(e): raise
            emit('log', f"💥 OOM detected: {str(e).splitlines()[0]}"); _clear_accel_cache(model.accel_device)
            # A backward that died mid-flight can leave partially written grads. Drop every
            # parameter's grad (including sparse engram rows) before retrying; optimizers keep
            # their Adam moments, so no rebuild is needed. This also discards grads accumulated
            # earlier in the window — correctness beats reusing a contaminated accumulator.
            for p in model.parameters(): p.grad = None
            if not ladder.step_down(emit): emit('log', "💀 OOM even at CPU-only"); raise
            emit('log', f"   retrying same batch at ladder stage {ladder.stage}")

def _sha256_hex(data): return hashlib.sha256(data).hexdigest()

def _sha256_file(path, piece=8 * 1024 * 1024):
    hs = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(piece), b''): hs.update(b)
    return hs.hexdigest()

def _post_json_with_retry(url, payload, h, emit, se, timeout=60, retries=4, max_total_wait_s=7200):
    last_err = None; real_attempts = 0; total_waited = 0.0
    while True:
        if se.is_set(): raise KeyboardInterrupt("Stop requested")
        try:
            r = requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code in (429, 503):
                retry_after = int(r.headers.get("Retry-After", "60"))
                if total_waited + retry_after > max_total_wait_s: raise RuntimeError("rate-limit wait budget exceeded")
                emit("log", f"⏳ {r.status_code} {url}; waiting {retry_after}s")
                time.sleep(min(retry_after, 3600)); total_waited += retry_after; continue
            r.raise_for_status(); return r
        except KeyboardInterrupt: raise
        except Exception as e:
            last_err = e; real_attempts += 1
            if real_attempts >= retries: break
            emit("log", f"POST retry {real_attempts}/{retries}: {e}"); time.sleep(min(20, 2 ** real_attempts))
    raise RuntimeError(f"POST {url} failed: {last_err}")

def _chunked_submit_once(srv, h, delta_path, meta, emit, se, chunk_size):
    total = os.path.getsize(delta_path); chunks = max(1, math.ceil(total / chunk_size)); checksum = _sha256_file(delta_path)
    init_payload = {**meta, "totalBytes": total, "chunkSize": chunk_size, "chunks": chunks, "checksum": checksum}
    emit("log", f"Initializing chunked upload: {chunks} chunks, {total / (1024 * 1024):.1f} MB total")
    r = _post_json_with_retry(f"{srv}/fl/submit/init", init_payload, h, emit, se, timeout=60, retries=4)
    j = r.json(); upload_id = j.get("uploadId")
    if not upload_id: raise RuntimeError("Server did not return uploadId")
    for idx in range(chunks):
        if se.is_set(): raise KeyboardInterrupt("Stop requested")
        start = idx * chunk_size; end = min(total, start + chunk_size)
        with open(delta_path, 'rb') as f: f.seek(start); part = f.read(end - start)
        part_sha = _sha256_hex(part); attempt = 0
        while True:
            try:
                rr = requests.post(f"{srv}/fl/submit/chunk?uploadId={upload_id}&index={idx}", data=part, headers={**h, "Content-Type": "application/octet-stream", "X-Chunk-Sha256": part_sha}, timeout=300)
                if rr.status_code == 413: raise RuntimeError("CHUNK_TOO_LARGE")
                rr.raise_for_status(); break
            except KeyboardInterrupt: raise
            except Exception as e:
                if "CHUNK_TOO_LARGE" in str(e): raise
                if getattr(e, "response", None) is not None and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", "60"))
                    emit("log", f"⏳ Server rate-limited chunk {idx+1}/{chunks}; waiting {retry_after}s"); time.sleep(min(retry_after, 3600)); continue
                attempt += 1
                if se.is_set() or attempt >= MAX_CHUNK_RETRY: raise
                emit("log", f"Chunk {idx + 1}/{chunks} retry {attempt}: {e}"); time.sleep(min(20, 2 ** attempt))
        emit("overlay_progress", {"done": end, "total": total}); emit("log", f"  ✓ chunk {idx + 1}/{chunks} ({end / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)")
    emit("log", "Finalizing chunked upload...")
    return _post_json_with_retry(f"{srv}/fl/submit/finalize", {"uploadId": upload_id}, h, emit, se, timeout=900, retries=2)

def chunked_submit_delta(srv, h, delta_path, meta, emit, se):
    chunk_size = UPLOAD_CHUNK_BYTES
    while True:
        try: return _chunked_submit_once(srv, h, delta_path, meta, emit, se, chunk_size)
        except Exception as e:
            msg = str(e)
            if ("CHUNK_TOO_LARGE" in msg or "413" in msg) and chunk_size > 8 * 1024 * 1024:
                chunk_size = max(8 * 1024 * 1024, chunk_size // 2)
                emit("log", f"Chunk too large. Reducing to {chunk_size / (1024 * 1024):.0f} MB and retrying."); continue
            raise

def _build_model(bn, device, iw, ie, emit, bs=1, sl=MAX_SEQ_LEN,
                 u8_override=None, act_bytes_override=None):
    model = SotaGPT(bn, device, gpu_layers=None, bs=bs, sl=sl,
                    u8_override=u8_override, act_bytes_override=act_bytes_override)
    model.engram.table.to('cpu'); model.load_base_weights(iw)
    if ENGRAM_ACTIVE and ie is not None: model.engram.table.weight.data.copy_(ie.view(model.engram.table.weight.shape))
    model.engram.table.to('cpu'); model.train(); return model

class _MultiOpt:
    def __init__(self, opts): self.opts = [o for o in opts if o is not None]
    def zero_grad(self, set_to_none=True):
        for o in self.opts: o.zero_grad(set_to_none=set_to_none)
    def step(self):
        for o in self.opts: o.step()

def _build_optimizers(model, lr=1e-4, force_mode=None):
    """force_mode: None = native, "adamw8bit" = force 8bit, "adamw32" = force fp32.
    Must be preserved across OOM-driven rebuilds (_on_step_down) or a client forced into an
    fp32 pool by the server will silently flip back to 8bit after a VRAM step-down."""
    bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
    ep = list(model.engram.parameters())
    groups = {}
    for p in bp: groups.setdefault(p.device.type, []).append(p)
    use_8bit = (_predict_optim(train_backend, force_mode=force_mode) == "adamw8bit"); opts = []
    for dev_type, ps in groups.items():
        o = None
        if dev_type == 'cuda' and use_8bit:
            try:
                import bitsandbytes as bnb
                o = bnb.optim.AdamW8bit(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
            except Exception: o = None
        if o is None: o = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, **({} if dev_type in ('cpu', 'cuda') else {"foreach": False}))
        opts.append(o)
    ob = _MultiOpt(opts)
    try: oe = torch.optim.SparseAdam(ep, lr=lr)
    except Exception: oe = torch.optim.AdamW(ep, lr=lr, weight_decay=0.01)
    return ob, oe

def _clip_base_grads(model, max_norm=1.0):
    """Clip only dense base-model grads; engram rows are sparse and coalesce in SparseAdam."""
    torch.nn.utils.clip_grad_norm_(
        [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('engram.')],
        max_norm,
    )

def _trim_heap():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

def _bf16_from_bytes(buf, offset, nbytes):
    a = np.frombuffer(buf, dtype=np.int16, count=nbytes // 2, offset=offset).copy()
    return torch.from_numpy(a).view(torch.bfloat16)

def _write_base_delta_bf16(gz, model, iw_bf16, piece=16 * 1024 * 1024):
    o = 0
    for n, p in model.named_parameters():
        if n.startswith('engram.'): continue
        s = p.numel(); flat = p.detach().reshape(-1)
        for i in range(0, s, piece):
            cur = flat[i:i + piece].to('cpu', torch.float32)
            d = (cur - iw_bf16[o + i:o + i + cur.numel()].float()).to(torch.bfloat16)
            gz.write(memoryview(d.view(torch.int16).numpy()).cast('B'))
        flat = cur = d = None; p.data = torch.empty(0, dtype=p.dtype, device=p.device); o += s
    if o != iw_bf16.numel(): raise RuntimeError(f"delta size mismatch: {o} vs {iw_bf16.numel()}")

def _engram_sparse_delta(table, iew_bf16, frac=0.10, chunk=32768):
    n = table.shape[0]; rn = torch.empty(n)
    for i in range(0, n, chunk): rn[i:i + chunk] = (table[i:i + chunk] - iew_bf16[i:i + chunk].float()).abs().sum(dim=1)
    k = max(1, int(n * frac)); tv, ti = torch.topk(rn, k); ai = ti[tv > 1e-8]
    if len(ai) == 0: return np.array([], dtype=np.uint32), np.array([], dtype=np.uint16)
    sv = (table[ai] - iew_bf16[ai].float()).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    return ai.numpy().astype(np.uint32), sv

def _write_delta_file(path, header_bytes, model, iw_bf16, engram_arrays, level=6, piece=64 * 1024 * 1024):
    with open(path, 'wb') as f, gzip.GzipFile(fileobj=f, mode='wb', compresslevel=level) as gz:
        gz.write(header_bytes); _write_base_delta_bf16(gz, model, iw_bf16)
        for a in engram_arrays:
            mv = memoryview(np.ascontiguousarray(a)).cast('B')
            for i in range(0, len(mv), piece): gz.write(mv[i:i + piece])
    return os.path.getsize(path)

def _pick_autocast(backend):
    if backend in ("DIRECTML", "CPU"): return False, None
    try:
        if backend in ("CUDA", "ROCM") and torch.cuda.is_available() and torch.cuda.is_bf16_supported(): return True, torch.bfloat16
        if backend == "XPU" and hasattr(torch, "xpu") and torch.xpu.is_bf16_supported(): return True, torch.bfloat16
    except Exception: pass
    return False, None

def _cpu_calibration_batch(x, y, full_sl):
    try: limit = int(os.environ.get("CROWDGPT_CPU_CAL_TOKENS", "32"))
    except (TypeError, ValueError): limit = 32
    limit = max(8, min(128, limit)); step_sl = max(1, min(int(full_sl), int(x.shape[1]), int(y.shape[1]), limit))
    return x[:, :step_sl].contiguous(), y[:, :step_sl].contiguous(), step_sl

def _parse_task_meta(md, emit):
    tid = md.get("taskId"); gs = md.get("globalStep"); sl = MAX_SEQ_LEN; cc_cfg = md.get("crossCheck") or {}
    cohort_mode = bool(cc_cfg.get("enabled", False))
    cc_seed = cc_cfg.get("targetSeed") if cohort_mode else None
    cc_steps = cc_cfg.get("targetLocalSteps") if cohort_mode else None
    cc_bs = cc_cfg.get("targetBatchSize") if cohort_mode else None
    cc_accum = cc_cfg.get("targetAccumSteps") if cohort_mode else None
    cc_autocast = cc_cfg.get("targetAutocast") if cohort_mode else None
    cc_optim = (cc_cfg.get("targetOptimizer") if cohort_mode else None) or "any"
    if cohort_mode and (cc_seed is None or cc_steps is None or cc_bs is None or cc_accum is None or cc_autocast is None):
        emit('log', "⚠️ Cross-check metadata incomplete; disabling cohort mode for this round")
        cohort_mode = False; cc_seed = cc_steps = cc_bs = cc_accum = cc_autocast = None
    if cohort_mode: emit('log', f"[COHORT] ENABLED: seed={cc_seed} steps={cc_steps} bs_target={cc_bs} accum={cc_accum} autocast={cc_autocast} optim={cc_optim}")
    else: emit('log', "[COHORT] disabled (server will use Krum for this shard)")
    return tid, gs, sl, cohort_mode, cc_seed, cc_steps, cc_bs, cc_accum, cc_autocast, cc_optim

# ---- v3.1: background uploader (max 1 upload in flight; training continues meanwhile) ----
_upload_lock = threading.Lock()
_upload_thread = None

def _upload_active():
    t = _upload_thread
    return bool(t is not None and t.is_alive())

def _wait_for_upload(se, emit=None, poll=0.5, hb=None):
    """Block until the in-flight upload (if any) finishes. Returns False if stopped (or refused, via hb) first."""
    announced = False
    while _upload_active():
        if se.is_set() or (hb is not None and hb.rejected is not None): return False
        if emit is not None and not announced:
            emit('log', "⏳ Previous upload still in progress; this round's delta will upload right after it."); emit('status', 'waiting'); announced = True
        time.sleep(poll)
    return True

def _bg_emit(emit):
    """The uploader must not touch the live training UI: no overlay, no status changes."""
    def _e(ev, *a, **k):
        if ev in ('overlay_show', 'overlay_progress', 'overlay_hide', 'status'): return
        return emit(ev, *a, **k)
    return _e

def _submit_delta_background(srv, h, delta_path, delta_size, meta, emit, se, hb=None):
    global _upload_thread
    # Only one upload at a time: if the previous one is still running, hold this (already written) delta until it is done.
    # The previous upload itself is never cancelled here. A refusal (hb) before the hand-over discards this delta.
    if not _wait_for_upload(se, emit, hb=hb) or (hb is not None and not hb.commit()):
        emit('log', "Coordinator refused this client; delta discarded, no upload." if hb is not None and hb.rejected is not None
             else "Stopped while waiting to upload; delta discarded.")
        try: os.unlink(delta_path)
        except OSError: pass
        return
    with _upload_lock:
        def _run():
            try: _upload_delta_file(srv, h, delta_path, delta_size, meta, _bg_emit(emit), se)
            except BaseException as e:
                try: emit('log', f"Upload failed: {e}")
                except Exception: pass
                try: os.unlink(delta_path)
                except OSError: pass
        _upload_thread = threading.Thread(target=_run, daemon=True, name="crowdgpt-upload")
        _upload_thread.start()

def _join_upload(se=None, timeout=None):
    """Wait for the background upload at shutdown / before a restart.

    ``se`` is accepted for call-site compatibility but deliberately does not
    abort the wait: on quit it is already set, and returning early can kill a
    half-written POST and leak the delta file. ``timeout=None`` waits as long as
    the upload takes (callers pass a cap only when the user is quitting).
    """
    t = _upload_thread
    if t is None: return
    end = None if timeout is None else time.time() + timeout
    while t.is_alive():
        if end is not None and time.time() > end: return
        t.join(0.5)

def _upload_delta_file(srv, h, delta_path, delta_size, meta, emit, se):
    """Upload one finished delta file and delete it. Runs in the background uploader thread."""
    upload_size_mb = delta_size / (1024 * 1024)
    emit('log', f"⬆️ Uploading {upload_size_mb:.1f} MB delta to server (in background)...")
    r = None
    try:
        if delta_size <= MAX_DIRECT_UPLOAD_BYTES:
            r = None
            for attempt in range(3):
                try:
                    with open(delta_path, 'rb') as f: body = f.read()
                    r = requests.post(f"{srv}/fl/submit", headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h}, data=body, timeout=600); body = None
                    if r.status_code in (413, 429, 500, 502, 503, 504):
                        r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
                    break
                except Exception as e:
                    body = None
                    if attempt == 2:
                        emit('log', f"Direct submit failed: {e}; trying chunked upload")
                        r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
                    else:
                        time.sleep(2 ** attempt)
        else: r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
        if r is not None and r.status_code == 200:
            try: j = r.json()
            except Exception: j = {}
            if j.get("duplicate"): emit("log", f"⏭️ Not counted by server ({j.get('reason','duplicate')})")
            else: emit("log", "✅ Submitted successfully!")
        elif r is not None:
            code = r.status_code; detail = _http_detail(r)
            if code == 401:
                _clear_saved_token_if_matches(h)
                emit("log", f"❌ Round lost — session expired during upload (HTTP 401): {detail}")
            elif code == 403:
                emit("log", f"❌ Round lost — coordinator rejected the upload (HTTP 403): {detail or 'forbidden'}")
            elif code == 426:
                emit("log", f"❌ Round lost — client update required (HTTP 426): {detail or 'update the client'}")
            else:
                emit("log", f"❌ Round lost — submit failed (HTTP {code}): {detail}")
    except Exception as e: emit("log", f"Upload failed: {e}")
    finally:
        try: os.unlink(delta_path)
        except OSError: pass
        gc.collect()

def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    if DEVICE_ID: h["X-Client-Device"] = DEVICE_ID
    emit('status', 'waiting'); rs = wait_for_round(srv, h, se, emit)
    if not rs: return
    cr = rs["current_round"]; hb = HeartbeatManager(srv, h, cr, se, emit); hb.start()
    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0; lval = 10.0; tt = 0; tid = ""; tried_cpu_fallback = False
    cohort_mode = False; cc_seed = cc_steps = cc_bs = cc_accum = cc_autocast = None; cc_optim = "any"
    optim_mode = None
    round_loss_sum = 0.0; round_loss_sq_sum = 0.0; round_loss_count = 0
    last_step_loss = None
    exc_type = None; exc_text = ""
    try:
        emit('log', "Fetching task metadata..."); md = fetch_task_metadata(srv, h, se, emit, hb=hb)
        if md is None: return
        tid, gs, sl, cohort_mode, cc_seed, cc_steps, cc_bs, cc_accum, cc_autocast, cc_optim = _parse_task_meta(md, emit)
        if not tid: return
        hb.tid = tid; iw = ie = None

        # Fail fast if selected GPU has no matching kernels in installed PyTorch
        if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
            _dev_idx = train_device.index if (train_device is not None and train_device.index is not None) else 0
            if not _check_arch_and_warn(emit, _dev_idx):
                raise ArchIncompatible(f"GPU {_dev_idx} has no matching kernels in installed PyTorch")

        if USE_HF_WEIGHTS:
            wcfg = md.get("weights") or {}
            if wcfg.get("source") == "huggingface":
                try:
                    emit('log', "Downloading weights from Hugging Face..."); iw, ie = fetch_weights_from_hf(md, se, emit)
                    prune_hf_cache(log_fn=lambda m: emit('log', m))
                except (KeyboardInterrupt, AuthRejected): raise
                except Exception as e:
                    emit('log', f"Hugging Face weight download failed: {e}"); emit('log', "Falling back to coordinator weight stream..."); iw = ie = None
        if iw is None or (ENGRAM_ACTIVE and ie is None):
            emit('log', "Downloading weights from coordinator...")
            md, wb = fetch_task(srv, h, "bf16", se, emit, hb=hb)
            if md is None: return
            tid, gs, sl, cohort_mode, cc_seed, cc_steps, cc_bs, cc_accum, cc_autocast, cc_optim = _parse_task_meta(md, emit)
            hb.tid = tid
            if not tid: emit('log', "Coordinator task metadata missing taskId."); return
            bbl = EXPECTED_MODEL_SIZE * 2; ebl = EXPECTED_ENGRAM_SIZE * 2
            needed = bbl + (ebl if ENGRAM_ACTIVE else 0)
            if len(wb) < needed: raise RuntimeError(f"Incomplete coordinator payload: got {len(wb)}, expected {needed}")
            iw = _bf16_from_bytes(wb, 0, bbl); ie = _bf16_from_bytes(wb, bbl, ebl) if ENGRAM_ACTIVE else None
            del wb; gc.collect()

        # v2: resolve optimizer mode from server's target (asymmetric forcing — fp32 pools accept any,
        # 8bit pools require native 8bit). If we can't honor the target, abort rather than submit a
        # delta that will silently fail cross-check or poison the aggregation.
        if cc_optim and cc_optim != "any":
            _native, _can_fp32, _can_8bit = _detect_optim_capability()
            if cc_optim in ("adamw_fp32", "adamw32"):
                optim_mode = "adamw32"
                if _native == "adamw8bit":
                    emit('log', f"[COHORT] pool forces fp32 (client native={_native}, downgrading)")
            elif cc_optim in ("adamw8bit", "adamw8"):
                if not _can_8bit:
                    emit('log', f"[COHORT] server requires adamw8bit but client cannot (backend={train_backend}, bnb={has_bitsandbytes()}) — aborting round")
                    return
                optim_mode = "adamw8bit"
            else:
                emit('log', f"[COHORT] unknown targetOptimizer={cc_optim!r} — using native")

        # Resolve autocast BEFORE model build so the VRAM planner sees the
        # ACTUAL activation dtype, not the client's native one.
        _ua, _ad = _pick_autocast(train_backend)
        if cohort_mode and cc_autocast and cc_autocast != "any":
            wants_bf16 = (cc_autocast == "bf16")
            if wants_bf16 and not _ua:
                emit('log', f"[COHORT] cannot reach bf16 on {train_backend} — aborting round"); return
            elif not wants_bf16 and _ua:
                emit('log', f"[COHORT] server requires fp32 — disabling autocast"); _ua, _ad = False, None
        ua, ad = _ua, _ad

        # Planner overrides from the ACTUAL (post-cohort) plan.
        _u8_override = True if optim_mode == "adamw8bit" else (False if optim_mode == "adamw32" else None)
        _act_bytes_override = 2 if ua else 4

        emit('status', 'preparing'); emit('log', "Preparing model...")
        dc = md.get("datasetConfig", {}); sc = md.get("shardConfig", {}); slot = sc.get("slot", 0)
        if cohort_mode: slot = 0; emit('log', "[COHORT] slot forced to 0 (identical shard for all members)")
        ds = StreamingShardDataset(dc.get("repoId", DATASET_REPO_ID), dc.get("chunkIdx", 0), dc.get("subChunkSize", 10 * 1024 * 1024), dc.get("tokensPerSample", sl + 1), slot, at, cohort_mode=cohort_mode)

        _bs_est = recommend_batch_size(sl, force_mode=optim_mode)
        bs = _bs_est
        if cohort_mode and cc_bs and int(cc_bs) > 0:
            if int(cc_bs) > _bs_est:
                emit('log', f"[COHORT] ⚠️ target bs={cc_bs} exceeds local estimate (bs={_bs_est}); will attempt with layout degradation")
            bs = int(cc_bs)

        model = None
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                gc.collect()
                model = _build_model(
                    train_backend, train_device, iw, ie, emit, bs=bs, sl=sl,
                    u8_override=_u8_override, act_bytes_override=_act_bytes_override,
                )
                break
            except RuntimeError as e:
                if not _is_oom(e): raise
                if model: del model; model = None
                _clear_accel_cache(train_device)
                if bs == 1 or (cohort_mode and cc_bs and bs <= int(cc_bs)):
                    emit('log', f"Model build OOM at bs={bs}; skipping this round (try a lower --vram-gb)"); break
                bs = max(1, bs // 2); emit('log', f"Model build OOM → bs={bs}")
        if model is None: return
        if cohort_mode and cc_bs and bs < int(cc_bs): emit('log', f"[COHORT] OOM at target bs={cc_bs}; aborting round"); return

        iew = ie.view(model.engram.table.weight.shape) if (ENGRAM_ACTIVE and ie is not None) else None; ie = None; _trim_heap()
        mbt = bs * sl; as_ = int(cc_accum) if (cohort_mode and cc_accum) else max(1, round(131072 / mbt))
        ls = 1.0 / as_; ob, oe = _build_optimizers(model, lr=1e-4, force_mode=optim_mode)

        def _on_step_down(_ladder):
            nonlocal ob, oe
            if cohort_mode and _ladder.stage >= 6:
                raise CohortAbort("OOM: would require CPU-only fallback in cohort mode")
            # Preserve optim_mode: flipping back to native mid-round would submit a delta trained
            # with the wrong optimizer for the pool, silently failing cross-check
            ob, oe = _build_optimizers(model, lr=1e-4, force_mode=optim_mode); emit('log', "   rebuilt optimizers for new placement")

        ladder = DegradeLadder(model, on_step_down=_on_step_down)
        emit('log', f"Backend: {train_backend} | autocast: {ua} | bs={bs} | accum={as_} | dtype={'bf16' if ua else 'fp32'} | engram={'on' if ENGRAM_ACTIVE else 'off'} | optim={_predict_optim(train_backend, force_mode=optim_mode)}")
        cs_ = time.time(); target_micro_steps = 100
        if cohort_mode: target_micro_steps = int(cc_steps); emit('status', 'training')
        else:
            CAL_MAX_MICRO = max(CAL_MIN_MICRO, min(as_, CAL_MAX_MICRO_HARD))
            if model.main_device.type == 'cpu': CAL_MAX_MICRO = 1
            emit('status', 'calibrating'); ob.zero_grad(set_to_none=True)
            while (micro_step < CAL_MAX_MICRO) and ((time.time() - cs_) < CAL_TIME_BUDGET):
                if hb.should_stop() or se.is_set(): return
                if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                x, y = ds.get_batch(bs, seed=(abs(hash("cal")) % 10000) + micro_step)
                step_x, step_y, step_sl = _cpu_calibration_batch(x, y, sl) if model.main_device.type == 'cpu' else (x, y, sl)
                want_preview = ((micro_step + 1) % PREVIEW_EVERY == 0) if train_backend == 'ROCM' else ((micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0))
                try: lo, cards = _train_step_safe(model, step_x, step_y, step_sl, ua, ad, ls, preview=want_preview, ladder=ladder, emit=emit)
                except CohortAbort: return
                lval = float(lo.item()); micro_step += 1
                _unscaled = lval * as_
                last_step_loss = _unscaled
                round_loss_sum += _unscaled; round_loss_sq_sum += _unscaled * _unscaled; round_loss_count += 1
                if cards: emit('model_preview', {'step': micro_step, 'cards': cards})
                elapsed_cal = max(0.001, time.time() - cs_); current_tps = (tt + step_x.numel()) / elapsed_cal
                if micro_step >= CAL_MIN_MICRO:
                    eta_elapsed = elapsed_cal * (sl / max(1, step_sl)) if model.main_device.type == 'cpu' and step_sl < sl else elapsed_cal
                    sps_micro = eta_elapsed / micro_step; rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                    remaining = max(60, (rh * 3600) - eta_elapsed - (UPLOAD_BUFFER_MIN * 60))
                    target_micro_steps = micro_step + int(remaining / sps_micro)
                emit('cal_stats', {'step': micro_step, 'total': target_micro_steps, 'loss': safe_float(lval * as_, 10.0), 'tps': safe_float(current_tps), 'time_left': safe_float((remaining if micro_step >= CAL_MIN_MICRO else 0) / 60.0)})
                tt += step_x.numel(); del lo, x, y, step_x, step_y
            if hb.should_stop() or se.is_set(): return
            _clip_base_grads(model)
            ob.step(); ob.zero_grad(set_to_none=True)
            oe.step(); oe.zero_grad(set_to_none=True); gc.collect()

        if se.is_set(): return
        if cohort_mode: dl = float('inf'); target_micro_steps = int(cc_steps)
        else:
            rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
            eta_elapsed = time.time() - cs_
            if model.main_device.type == 'cpu': eta_elapsed *= (sl / max(1, min(sl, int(os.environ.get("CROWDGPT_CPU_CAL_TOKENS", "32")))))
            remaining = max(60, (rh * 3600) - eta_elapsed - (UPLOAD_BUFFER_MIN * 60)); dl = time.time() + remaining
            target_micro_steps = micro_step + int(remaining / max(eta_elapsed / max(1, micro_step), 0.001)) if micro_step > 0 else 1000
        emit('status', 'training'); last_stats_emit = 0.0

        cc_step_cap = int(cc_steps) if cohort_mode else 0
        def _outer_continue(): return (not hb.should_stop()) and (not se.is_set()) and ((micro_step < cc_step_cap) if cohort_mode else (time.time() < dl))
        def _inner_continue(): return (not hb.should_stop()) and (not se.is_set()) and ((micro_step < cc_step_cap) if cohort_mode else (time.time() < dl))

        while _outer_continue():
            ob.zero_grad(set_to_none=True); als = 0.0; accum_count = 0
            for mi in range(as_):
                if not _inner_continue(): break
                if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                sd = (int(cc_seed) * 100_003 + micro_step * 1009) & 0xFFFFFFFF if cohort_mode else (abs(hash("t")) % 10000) + micro_step * 1000 + mi
                x, y = ds.get_batch(bs, seed=sd)
                want_preview = (model.main_device.type != 'cpu') and ((micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0))
                try: lo, cards = _train_step_safe(model, x, y, sl, ua, ad, ls, preview=want_preview, ladder=ladder, emit=emit)
                except CohortAbort: return
                lval = float(lo.item()) * as_
                als += lval; tt += x.numel(); accum_count += 1; micro_step += 1
                round_loss_sum += lval; round_loss_sq_sum += lval * lval; round_loss_count += 1
                last_step_loss = lval
                if cards: emit('model_preview', {'step': micro_step, 'cards': cards})
                elapsed_total = time.time() - cs_
                rem_time = (max(0, int(cc_steps) - micro_step) * max(1e-6, elapsed_total / max(1, micro_step))) if cohort_mode else max(1, dl - time.time())
                if not cohort_mode and micro_step > 5:
                    target_micro_steps = max(target_micro_steps, micro_step + int(rem_time / max(elapsed_total / micro_step, 0.001)))
                at_cohort_end = cohort_mode and micro_step >= int(cc_steps)
                if time.time() - last_stats_emit > 0.25 or at_cohort_end or (not cohort_mode and time.time() >= dl):
                    last_stats_emit = time.time()
                    emit('stats', {'step': micro_step, 'target': target_micro_steps, 'loss': safe_float(als / max(1, accum_count), 10.0), 'tps': safe_float(tt / max(elapsed_total, 1)), 'global_step': gs, 'round': cr, 'time_left': safe_float(max(0, rem_time / 60))})
                del lo, x, y
            if hb.should_stop() or se.is_set() or accum_count == 0: break
            _clip_base_grads(model)
            ob.step(); ob.zero_grad(set_to_none=True)
            oe.step(); oe.zero_grad(set_to_none=True)

        if se.is_set(): emit('log', "Stopped by user. Skipping upload."); return
        if hb.should_stop(): emit('log', f"Stopped mid-training (round {cr} ended or task gone); discarding partial delta, no upload."); return
        hb.tid = None
        if cohort_mode and int(micro_step) != int(cc_steps): emit('log', f"[COHORT] only completed {micro_step}/{cc_steps} steps; aborting upload"); return
        if round_loss_count > 0:
            fl = round_loss_sum / round_loss_count
            _var = max(0.0, round_loss_sq_sum / round_loss_count - fl * fl)
            fl_std = math.sqrt(_var)
        else:
            fl = float(last_step_loss) if last_step_loss is not None else 10.0
            fl_std = 0.0
        _last_str = f"{last_step_loss:.4f}" if last_step_loss is not None else "n/a"
        emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f} ± {fl_std:.4f} "
                    f"(mean of {round_loss_count}, last {_last_str})")
        emit('status', 'uploading')
        for _o in (ob, oe):
            try: _o.zero_grad(set_to_none=True)
            except Exception: pass
        ob = oe = ladder = None
        for _p in model.parameters(): _p.grad = None
        _clear_accel_cache(train_device); _trim_heap()

        if ENGRAM_ACTIVE and iew is not None:
            si, sv = _engram_sparse_delta(model.engram.table.weight.data, iew)
            if len(si) > 0: emit('log', f"[ENGRAM] uploading {len(si):,} sparse row updates")
        else: si, sv = np.array([], dtype=np.uint32), np.array([], dtype=np.uint16)
        iew = None; has_engram = bool(len(si) > 0); engram_sparse_count = int(len(si))

        submit_extra = {
            "crossCheckUsed": cohort_mode,
            "actualSeed": int(cc_seed) if (cohort_mode and cc_seed is not None) else None,
            "actualSteps": int(micro_step),
            "actualBatchSize": int(bs),
            "lastLoss": float(last_step_loss) if last_step_loss is not None else None,
            "lossStd": float(fl_std),
            "lossSamples": int(round_loss_count),
        }
        pl = json.dumps({"taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt, "loraRank": 0, "isDelta": True, "weightFormat": "bf16", "hasEngram": has_engram, "engramSparseCount": engram_sparse_count, **submit_extra}).encode()
        delta_fd, delta_path = tempfile.mkstemp(prefix="delta-", suffix=".gz", dir=str(CHECKPOINT_DIR)); os.close(delta_fd)
        try: delta_size = _write_delta_file(delta_path, struct.pack('<I', len(pl)) + pl, model, iw, (si, sv))
        except BaseException:
            try: os.unlink(delta_path)
            except OSError: pass
            raise
        finally: iw = None; model = None; del si, sv, pl; _clear_accel_cache(train_device); _trim_heap()
        
        upload_size_mb = delta_size / (1024 * 1024); meta = {"taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt, "loraRank": 0, "isDelta": True, "weightFormat": "bf16", "hasEngram": has_engram, "engramSparseCount": engram_sparse_count, **submit_extra}
        _submit_delta_background(srv, h, delta_path, delta_size, meta, emit, se, hb=hb)   # discards the delta if refused
    except BaseException as e:
        exc_type = type(e); exc_text = f"{type(e).__name__}: {e}"[:200] if isinstance(e, Exception) else ""; raise
    finally:
        rejected = hb.close(); model = ob = oe = x = y = lo = iw = ie = iew = ds = None; gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try: torch.cuda.empty_cache()
            except Exception: pass
        try: prune_hf_cache()
        except Exception: pass
        # The heartbeat thread can't raise into this thread: turn the refusal recorded before close() into the
        # exception the GUI/CLI loops handle (stop, keep the session, CLI exit 7). A user stop (se) and a login
        # error still win. A KeyboardInterrupt without se is fetch_task aborting its download for hb.should_stop().
        # Another round error is named in the message and stays chained as __context__.
        if rejected is not None and not se.is_set():
            if exc_type is None or issubclass(exc_type, KeyboardInterrupt) or (issubclass(exc_type, Exception) and not issubclass(exc_type, (AuthRejected, CoordinatorRejected))):
                raise CoordinatorRejected(rejected + (f" (the round had also failed: {exc_text})" if exc_text else ""), status=403)

def _do_backend_setup(selected_backend, vram_gb, emit):
    global train_device, train_backend
    train_device, train_backend = detect_training_backend(selected_backend)
    emit('meta', {'backend': train_backend}); emit('log', f"Backend: {train_backend} ({train_device})")
    auto_detect_vram_budget(); memory_config["auto_gb"] = memory_config["ram_gb"]; apply_vram_override(vram_gb)
    emit('log', f"Placement budget: {memory_config['ram_gb']:.2f} GB"); install_hard_vram_cap(train_device, SAFETY_MARGIN_GB)

def _check_updates_between_rounds(emit):
    try:
        if _upload_active(): return   # restarting would kill the in-flight upload; check again next round
        if check_for_update(): emit('log', "🔄 Updated to a new version — restarting..."); emit('status', 'idle'); time.sleep(2); _restart_self(resume_training=True)
    except Exception: pass

class Api:
    def __init__(self):
        self.window = None; self.stop_event = threading.Event(); self.thread = None
        self.auth_token = None; self.username = None; self.server_url = DEFAULT_SERVER
        self._lp = 0.0
        self._ui_queue = queue.Queue(maxsize=2048)
        self._ui_ready = threading.Event()
        threading.Thread(target=self._ui_pump, daemon=True, name="ui-emit").start()
        # v2.5: a saved session is used ONLY if "Remember me" was ticked, or if we just restarted because of an update
        # while training. A normal start without "Remember me" forgets the token and shows the login screen.
        self._saved = None
        self._resume = os.environ.pop("CROWDGPT_RESUME_TRAINING", "") == "1"
        try:
            if AUTH_TOKEN_FILE.exists():
                j = json.loads(AUTH_TOKEN_FILE.read_text())
                if j.get("token") and (j.get("remember") or self._resume):
                    self._saved = {"token": j["token"], "username": j.get("username"), "server": j.get("server"), "remember": bool(j.get("remember"))}
                else:
                    AUTH_TOKEN_FILE.unlink(missing_ok=True)
        except Exception: pass
        self._backend_file = CHECKPOINT_DIR / "selected_backend.json"; saved = None
        try:
            if self._backend_file.exists(): saved = json.loads(self._backend_file.read_text()).get("backend")
        except Exception: pass
        available = get_available_backends(); preferred_backend = get_best_default_backend()
        if saved and available.get(saved, False):
            should_upgrade = (os.environ.get("CROWDGPT_HONOR_SAVED_BACKEND") != "1" and ((saved == "cpu" and preferred_backend != "cpu") or (saved == "directml" and preferred_backend in ("cuda", "rocm", "mps", "xpu"))))
            self.selected_backend = preferred_backend if should_upgrade else saved
        else: self.selected_backend = preferred_backend
        self._vram_file = CHECKPOINT_DIR / "selected_vram.json"; self.vram_gb = 0.0
        try:
            if self._vram_file.exists(): self.vram_gb = max(0.0, float(json.loads(self._vram_file.read_text()).get("vram_gb", 0) or 0))
        except Exception: pass

    def emit(self, ev, data):
        if not self.window: return
        if ev in ('log', 'status'):
            try: print(f"[EMIT {ev}] {data}", flush=True)
            except Exception: pass
        if ev == 'preview_loss':
            n = time.time()
            if n - self._lp < 0.4: return
            self._lp = n
        if ev == 'stats':
            n = time.time()
            if n - getattr(self, "_last_stats_emit", 0.0) < 0.10: return
            self._last_stats_emit = n
        try: script = f"window.handleEvent({json.dumps(ev)}, {json.dumps(sanitize_for_json(data))});"
        except Exception: return
        try: self._ui_queue.put_nowait(script)
        except queue.Full: pass

    def _ui_pump(self):
        self._ui_ready.wait()
        time.sleep(0.1) 
        while True:
            script = self._ui_queue.get()
            try: self.window.evaluate_js(script)
            except Exception as e: log.warning(f"Emit failed: {e}")

    def _mark_ui_ready(self):
        if self._ui_ready.is_set(): return
        self._ui_ready.set()
        log.info("[ui] page loaded; starting emit pump and training thread")
        saved, resume = self._saved, self._resume
        try:
            if saved and saved.get("token"):                       # remember-me ticked, or update-resume
                self.auth_token = saved["token"]; self.username = saved.get("username")
                if saved.get("server"): self.server_url = saved["server"]
                self.emit('login_success', {'username': self.username or 'restored', 'autologin': True})
            elif resume:                                            # update-resume of an anonymous session
                self.auth_token = None; self.username = "anonymous"
                self.emit('login_success', {'username': 'anonymous', 'autologin': True})
            else:
                return                                              # normal start: login screen, nothing runs
            if resume and (self.thread is None or not self.thread.is_alive()):
                self.emit('log', "🔄 Updated — resuming training"); self.start()
        except Exception as e: log.warning(f"[auth] auto-login failed: {e}")

    def get_backends(self):
        self.emit('backends', {'available': get_available_backends(), 'current': self.selected_backend})
        self.emit('vram', {'current': self.vram_gb})
        gpus = get_gpu_list()
        saved = _load_saved_gpu_idx()
        if not gpus:
            current = 0
        elif saved is None or not any(g["idx"] == saved for g in gpus):
            current = gpus[0]["idx"]
        else:
            current = saved
        self.emit('gpus', {'list': gpus, 'current': current})

    def set_gpu(self, idx):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return
        _save_saved_gpu_idx(idx)
        busy = self.thread is not None and self.thread.is_alive()
        self.emit('log', f"GPU → {idx}" + (" (stopping current round; applies from the next)" if busy else ""))
        self.get_backends()
        if busy: self.stop_event.set()

    def set_vram(self, gb):
        try: gb = max(0.0, float(gb))
        except (TypeError, ValueError): gb = 0.0
        self.vram_gb = gb
        try: self._vram_file.write_text(json.dumps({"vram_gb": gb}))
        except Exception: pass
        apply_vram_override(gb); busy = self.thread is not None and self.thread.is_alive()
        self.emit('log', f"VRAM limit: {'Auto' if gb <= 0 else f'{gb:g} GB'}" + (" (applies from the next round)" if busy else ""))

    def set_backend(self, backend_name):
        self.selected_backend = backend_name
        try: self._backend_file.write_text(json.dumps({"backend": backend_name}))
        except Exception: pass
        self.get_backends()
        if self.thread and self.thread.is_alive(): self.stop_event.set()

    def login(self, s, u, p, remember=False):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p, bool(remember)), daemon=True).start()

    def _do_login(self, u, p, remember=False):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token = tok; self.username = u
            try: _save_auth_token(self.server_url, u, tok, remember)
            except Exception: pass
            self.emit('login_success', {'username': u})
        else: self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p, remember=False):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p, bool(remember)), daemon=True).start()

    def _do_register(self, u, email, p, remember=False):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token = tok; self.username = u
            try: _save_auth_token(self.server_url, u, tok, remember)
            except Exception: pass
            self.emit('login_success', {'username': u})
        else: self.emit('login_error', {'code': _GUI_REGISTER_CODES.get(err, err or 'register_failed')})

    def login_anon(self):
        self.auth_token = None; self.username = "anonymous"
        try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
        except Exception: pass
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive():
            if self.stop_event.is_set():
                self.emit('log', "Previous run is still shutting down — press Start again in a moment."); self.emit('start_refused', {})
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._rs, daemon=True)
        self.thread.start()
        self.emit('started', {})
      
    def stop(self):
        self.stop_event.set()
        self.emit('stopped', {})
    def logout(self):
        self.stop_event.set()                                      # stops training after the current step; a new login can't start until it has ended
        self.auth_token = None; self.username = None
        try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
        except Exception: pass
        self.emit('log', "Logged out.")
        self.emit('logged_out', {})
    def _check_updates_between_rounds(self): _check_updates_between_rounds(self.emit)

    def _rs(self):
        try: _do_backend_setup(self.selected_backend, self.vram_gb, self.emit)
        except Exception as e:
            self.emit('log', f"Error: {e}"); self.emit('status', 'idle'); return
        round_count = 0
        while not self.stop_event.is_set():
            round_count += 1; apply_vram_override(self.vram_gb)
            self.emit('log', f"{'='*40}\nStarting Round Cycle #{round_count}\n{'='*40}")
            try: run_single_round_wrapper(self.server_url, self.auth_token, self.stop_event, self.emit)
            except AuthRejected:
                self.auth_token = None; self.emit('status', 'idle'); self.emit('log', "🔒 Session expired — please log in again.")
                try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
                except Exception: pass
                self.emit('logged_out', {}); break
            except CoordinatorRejected as e:
                if e.status == 426:
                    self.emit('log', f"⛔ This client version is no longer accepted ({e}). Training stopped — checking for an update; if none installs, update the client and press Start again.")
                    self._check_updates_between_rounds()       # installs and restarts if a signed update exists
                else:                                          # 403: the reason may be temporary (e.g. a timed subnet block)
                    self.emit('log', f"⛔ The coordinator refused to hand out work ({e}). Training stopped — press Start again once this is resolved.")
                self.emit('status', 'idle'); return            # idle last: Start must not be re-enabled while the check runs
            except ArchIncompatible as e:
                self.emit('status', 'idle'); self.emit('log', f"⛔ Cannot train: {e}. Fix the PyTorch/GPU mismatch above, or pick a different GPU/backend, then press Start again."); break
            except KeyboardInterrupt: self.stop_event.set(); break
            except Exception as e:
                import traceback; tb = traceback.format_exc()
                for ln in tb.splitlines()[-14:]: self.emit('log', ln)
                self.emit('log', f"Round failed: {e}")
            if self.stop_event.is_set(): break
            self._check_updates_between_rounds(); self.emit('status', 'waiting'); self.emit('log', "✅ Round complete. Auto-relaunching next round in 10s...")
            for _ in range(10):
                if self.stop_event.is_set(): break
                time.sleep(1)
        self.emit('status', 'idle'); self.emit('log', "👋 Training stopped by user.")

def _apply_gtk_icon_list():
    """GTK backend only: give the WM every real size so it never has to scale one big image."""
    try:
        if "gtk" not in getattr(getattr(webview, "guilib", None), "__name__", ""): return
        import gi; gi.require_version("Gtk", "3.0"); gi.require_version("GdkPixbuf", "2.0")
        from gi.repository import Gtk, GdkPixbuf, GLib
        pix = []
        for _s, blob in sorted(_ico_pngs(base64.b64decode(_EMBEDDED_ICO_B64)).items()):
            try:
                ld = GdkPixbuf.PixbufLoader.new_with_type("png"); ld.write(blob); ld.close(); pix.append(ld.get_pixbuf())
            except Exception: pass
        if not pix: return
        def _set():
            try:
                Gtk.Window.set_default_icon_list(pix)
                for w in Gtk.Window.list_toplevels():
                    try: w.set_icon_list(pix)
                    except Exception: pass
            except Exception: pass
            return False
        GLib.idle_add(_set)
    except Exception: pass

def startup():
    try:
        scr = webview.screens[0]; aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80))); h = max(360, min(800, int(ah * 0.80)))
        w = max(320, min(w, aw - 16)); h = max(240, min(h, ah - 48))
        webview.windows[0].resize(w, h)
        webview.windows[0].move(max(0, (aw - w) // 2), max(0, (ah - h) // 2))
        if sys.platform == "win32" and WINDOWS_ICO_PATH:
            def _win_icon():
                if _wait_for_window(WINDOW_TITLE, timeout=10.0):
                    for delay in (0.0, 1.5):          # twice: some builds reset the icon right after the first paint
                        time.sleep(delay)
                        try: set_windows_icon(WINDOW_TITLE, WINDOWS_ICO_PATH)
                        except Exception: pass
            threading.Thread(target=_win_icon, daemon=True).start()
        elif sys.platform != "win32":
            threading.Thread(target=_apply_gtk_icon_list, daemon=True).start()
    except Exception: pass

def _create_window_safe(title, html, kwargs):
    try: return webview.create_window(title, html=html, **kwargs)
    except TypeError:
        kwargs.pop("min_size", None); return webview.create_window(title, html=html, **kwargs)

def _start_icon_kwargs():
    if sys.platform == "win32" or not ICON_PATH or not os.path.exists(ICON_PATH): return {}
    path = ICON_PATH
    if path.lower().endswith(".ico"):
        # Qt reads every size inside the ICO. GTK needs gdk-pixbuf to be able to open it, otherwise window creation fails.
        try:
            import gi; gi.require_version("GdkPixbuf", "2.0")
            from gi.repository import GdkPixbuf
            GdkPixbuf.Pixbuf.new_from_file(path)
        except (ImportError, ValueError): pass
        except Exception: path = _embedded_png(256) or path
    return {"icon": path}

_TUI_STATUS = {
    "idle": ("Idle", "dim"),
    "waiting": ("Waiting for coordinator", "yellow"),
    "downloading": ("Downloading weights", "cyan"),
    "preparing": ("Preparing model", "cyan"),
    "calibrating": ("Calibrating", "magenta"),
    "training": ("Training", "bold green"),
    "uploading": ("Uploading delta", "blue"),
    "paused": ("Paused — press s to start", "bold yellow"),
    "stopped": ("Stopped", "red"),
    "error": ("Error", "bold red"),
}
_TUI_OVERLAY = {
    "wait_coord": "Waiting for coordinator",
    "dl_weights": "Downloading weights",
    "uploading": "Uploading delta",
}

def _num_or_none(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def _fmt_int(value):
    return f"{int(value):,}" if value else "0"

def _fmt_eta(minutes):
    value = _num_or_none(minutes)
    if value is None or value < 0:
        return "—"
    total = int(round(value * 60))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"

def _bar_text(percent, width):
    pct = max(0.0, min(100.0, float(percent or 0.0)))
    usable = max(8, width)
    filled = int(round(usable * pct / 100.0))
    bar = Text()
    bar.append("█" * filled, style="green")
    bar.append("░" * (usable - filled), style="grey37")
    return bar

def _area_chart(values, width, rows=6):
    """A compact filled loss chart that uses vertical terminal space."""
    if not values:
        return [Text("—", style="dim")]
    points = list(values)[-max(8, width):]
    low, high = min(points), max(points)
    span = (high - low) or 1.0
    flat = high == low
    blocks = " ▁▂▃▄▅▆▇█"
    chart = []
    for row in range(rows - 1, -1, -1):
        line = Text()
        for value in points:
            level = rows * 4 if flat else int(round((value - low) / span * rows * 8))
            cell = max(0, min(8, level - row * 8))
            line.append(blocks[cell], style="green" if cell else "grey19")
        chart.append(line)
    return chart

class _DashboardLogHandler(logging.Handler):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record):
        try:
            self.dashboard.add_log(record.getMessage(), record.levelname)
        except Exception:
            pass  # a display failure must never break training

class HeadlessRunControl:
    """Coordinates the TUI keys with the headless training loop."""

    def __init__(self, dashboard=None):
        self.dashboard = dashboard
        self.quit_event = threading.Event()
        self.pause_event = threading.Event()
        self.halt_event = threading.Event()

    def request_quit(self, announce=True):
        if self.quit_event.is_set():
            return
        self.quit_event.set()
        self.halt_event.set()
        if announce:  # never log from a signal handler; logging can deadlock
            log.info("Stopping at the next safe step…")

    def toggle(self):
        if self.quit_event.is_set():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.halt_event.clear()
            if self.dashboard:
                self.dashboard.set_paused(False)
            log.info("Training started.")
        else:
            self.pause_event.set()
            self.halt_event.set()
            if self.dashboard:
                self.dashboard.set_paused(True)
            log.info("Training stopped at the next safe step. Press s to start again.")

    def wait_while_paused(self):
        while not self.quit_event.is_set() and self.pause_event.is_set():
            self.quit_event.wait(0.2)

class TerminalDashboard:
    """Full-screen terminal view over the same events as the GUI."""

    def __init__(self, console=None, refresh_per_second=8):
        if not _RICH_OK:
            raise RuntimeError("rich is not installed")
        self.console = console or Console()
        self.refresh_per_second = refresh_per_second
        self._live = None
        self._lock = threading.RLock()
        self._key_thread = None
        self._key_stop = threading.Event()
        self._on_toggle = None
        self._on_quit = None
        self.paused = False
        self._logs = deque(maxlen=300)
        self._losses = deque(maxlen=240)
        self._previews = []
        self._preview_step = 0
        self.username = "anonymous"
        self.server = DEFAULT_SERVER
        self.backend = "auto"
        self.device = ""
        self.status = "idle"
        self.round = 0
        self.step = 0
        self.target = 0
        self.loss = None
        self.best_loss = None
        self.tps = None
        self.time_left = None
        self.started_at = None
        self.overlay_title = ""
        self.overlay_done = 0
        self.overlay_total = 0
        self.overlay_indeterminate = False

    def update_context(self, **values):
        with self._lock:
            for key, value in values.items():
                if hasattr(self, key) and value is not None:
                    setattr(self, key, value)

    def set_paused(self, paused):
        with self._lock:
            self.paused = bool(paused)

    def add_log(self, message, level="INFO"):
        with self._lock:
            self._logs.append((time.time(), str(level).upper(), str(message)))

    def handle_event(self, event, data):
        payload = data if isinstance(data, dict) else {}
        with self._lock:
            if event == "log":
                self._logs.append((time.time(), "INFO", str(data)))
            elif event == "status":
                self.status = str(data).lower()
            elif event in ("stats", "cal_stats"):
                step = _num_or_none(payload.get("step"))
                target = _num_or_none(payload.get("target", payload.get("total")))
                loss = _num_or_none(payload.get("loss"))
                tps = _num_or_none(payload.get("tps"))
                time_left = _num_or_none(payload.get("time_left"))
                round_no = _num_or_none(payload.get("round"))
                if step is not None:
                    self.step = int(step)
                if target is not None:
                    self.target = int(target)
                if loss is not None and loss > 0:
                    self.loss = loss
                    self._losses.append(loss)
                    if self.best_loss is None or loss < self.best_loss:
                        self.best_loss = loss
                if tps is not None:
                    self.tps = tps
                if time_left is not None:
                    self.time_left = time_left
                if round_no is not None:
                    self.round = int(round_no)
            elif event == "model_preview":
                cards = payload.get("cards") or []
                self._previews = [dict(card) for card in cards if isinstance(card, dict)][:8]
                step = _num_or_none(payload.get("step"))
                if step is not None:
                    self._preview_step = int(step)
            elif event == "overlay_show":
                self.overlay_title = str(payload.get("title_key") or "")
                self.overlay_indeterminate = bool(payload.get("indeterminate"))
                self.overlay_done = 0
                self.overlay_total = 0
            elif event == "overlay_progress":
                self.overlay_done = _num_or_none(payload.get("done")) or 0
                self.overlay_total = _num_or_none(payload.get("total")) or 0
            elif event == "overlay_hide":
                self.overlay_title = ""
                self.overlay_done = 0
                self.overlay_total = 0
            elif event == "meta":
                self.backend = str(payload.get("backend") or self.backend)
            elif event == "login_success":
                self.username = str(payload.get("username") or self.username)
            elif event == "login_error":
                self.status = "error"
                self._logs.append((time.time(), "ERROR", f"Authentication failed: {payload.get('code', 'unknown')}"))
            elif event == "started":
                self.started_at = time.time()
            elif event == "stopped":
                self.status = "stopped"
            elif event == "backends":
                self.backend = str(payload.get("current") or self.backend)

    def emit(self, event, data):
        self.handle_event(event, data)

    def snapshot(self):
        with self._lock:
            return {
                "username": self.username,
                "server": self.server,
                "backend": self.backend,
                "status": self.status,
                "paused": self.paused,
                "round": self.round,
                "step": self.step,
                "target": self.target,
                "loss": self.loss,
                "best_loss": self.best_loss,
                "tps": self.tps,
                "time_left": self.time_left,
                "losses": list(self._losses),
                "previews": list(self._previews),
                "logs": list(self._logs),
                "overlay_title": self.overlay_title,
                "overlay_done": self.overlay_done,
                "overlay_total": self.overlay_total,
            }

    def _header_panel(self, snap, width):
        status_key = "paused" if snap["paused"] else snap["status"]
        status_text, status_style = _TUI_STATUS.get(status_key, (status_key.title(), "white"))
        if snap["overlay_title"] and not snap["paused"]:
            overlay = _TUI_OVERLAY.get(snap["overlay_title"], snap["overlay_title"])
            if snap["overlay_total"] > 0:
                overlay += f" {100.0 * snap['overlay_done'] / snap['overlay_total']:.0f}%"
            status_text = overlay
        title = Text()
        title.append("CrowdGPT", style="bold cyan")
        title.append("  terminal trainer", style="dim")
        title.append(f"   {snap['username']}", style="bold")
        if width >= 100:
            title.append(f" · {snap['server']}", style="dim")
        if snap["backend"]:
            title.append(f" · {snap['backend']}", style="dim")
        row = Table.grid(expand=True)
        row.add_column(ratio=1)
        row.add_column(justify="right")
        row.add_row(title, Text(f"● {status_text}", style=status_style))
        return Panel(row, box=rich_box.ROUNDED, border_style="cyan", padding=(0, 1))

    def _training_panel(self, snap, width):
        target = snap["target"]
        step = snap["step"]
        pct = (100.0 * step / target) if target > 0 else 0.0
        step_line = Text()
        step_line.append("Step ", style="dim")
        step_line.append(_fmt_int(step), style="bold")
        step_line.append(f" / {_fmt_int(target)}", style="dim")
        step_line.append(f"   {pct:5.1f}%", style="bold green")
        if snap["round"]:
            step_line.append(f"   round {snap['round']}", style="dim")
        bar = _bar_text(pct, max(12, width - 8))
        metrics = Text()
        metrics.append("loss ", style="dim")
        metrics.append(f"{snap['loss']:.4f}" if snap["loss"] is not None else "—", style="bold yellow")
        if snap["best_loss"] is not None:
            metrics.append("   best ", style="dim")
            metrics.append(f"{snap['best_loss']:.4f}", style="yellow")
        metrics.append("   tps ", style="dim")
        metrics.append(f"{snap['tps']:.0f}" if snap["tps"] is not None else "—", style="cyan")
        metrics.append("   eta ", style="dim")
        metrics.append(_fmt_eta(snap["time_left"]), style="cyan")
        if target <= 0 and not snap["losses"]:
            body = Group(Text("Waiting for a training assignment…", style="dim italic"), bar)
        else:
            body = Group(step_line, bar, metrics)
        return Panel(body, title="Training", box=rich_box.ROUNDED, border_style="green", padding=(0, 1))

    def _graph_panel(self, snap, width, rows=6):
        losses = snap["losses"]
        if losses:
            chart = _area_chart(losses, max(16, width - 4), rows=rows)
            axis = Text()
            axis.append(f"min {min(losses):.4f}", style="dim")
            axis.append(f"   max {max(losses):.4f}", style="dim")
            axis.append(f"   latest {losses[-1]:.4f}", style="bold yellow")
            body = Group(*chart, axis)
        else:
            body = Text("Waiting for the first loss value…", style="dim italic")
        point_label = "point" if len(losses) == 1 else "points"
        return Panel(body, title=f"Loss history · {len(losses)} {point_label}", box=rich_box.ROUNDED, border_style="yellow", padding=(0, 1))

    @staticmethod
    def _clip(text, limit, *, tail=False):
        text = " ".join(str(text).split())
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        if tail and limit > 1:
            return "…" + text[-(limit - 1):]
        return text[: max(0, limit - 1)] + "…"

    def _predictions_panel(self, snap, wide, width):
        content_width = max(20, width - 4)
        context_limit = max(10, content_width - 31)
        lines = []
        limit = 8 if wide else 2
        for card in snap["previews"][:limit]:
            context = self._clip(card.get("context", ""), context_limit, tail=True)
            prediction = self._clip(card.get("pred", ""), 10)
            target = self._clip(card.get("target", ""), 10)
            match = bool(card.get("match"))
            prob = _num_or_none(card.get("prob"))
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("✓ " if match else "✗ ", style="bold green" if match else "bold red")
            line.append(context, style="white")
            line.append(" → ", style="dim")
            line.append(prediction, style="bold cyan" if match else "bold red")
            if not match:
                line.append(" ≠ ", style="dim")
                line.append(target, style="yellow")
            if prob is not None:
                line.append(f" {prob * 100:.1f}%", style="cyan")
            lines.append(line)
        if not lines:
            lines = [Text("Waiting for the first model prediction…", style="dim italic")]
        title = "Model predictions"
        if snap["previews"]:
            title += f" · step {_fmt_int(self._preview_step)}"
        return Panel(Group(*lines), title=title, box=rich_box.ROUNDED, border_style="magenta", padding=(0, 1))

    def _activity_panel(self, snap, wide, width):
        logs = snap["logs"]
        limit = 10 if wide else 2
        message_limit = max(12, width - 12)
        lines = []
        for timestamp, level, message in logs[-limit:]:
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append(datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S "), style="dim")
            style = "red" if level in ("ERROR", "CRITICAL") else "yellow" if level == "WARNING" else "white"
            line.append(self._clip(message, message_limit, tail=True), style=style)
            lines.append(line)
        if not lines:
            lines = [Text("No activity yet.", style="dim italic")]
        return Panel(Group(*lines), title="Activity", box=rich_box.ROUNDED, border_style="blue", padding=(0, 1))

    def render(self):
        if not _RICH_OK:
            return Text("rich is not installed", style="red")
        try:
            snap = self.snapshot()
            width = self.console.width or 100
            height = self.console.height or 24
            wide = width >= 100 and height >= 28
            column_width = max(20, int(width * 0.5)) if wide else width
            panel_width = column_width - 4
            header = self._header_panel(snap, width)
            training = self._training_panel(snap, panel_width)
            graph_rows = (10 if height >= 36 else 8) if wide else 3
            graph = self._graph_panel(snap, panel_width, rows=graph_rows)
            predictions = self._predictions_panel(snap, wide, column_width)
            activity = self._activity_panel(snap, wide, column_width)
            idle = snap["status"] in ("idle", "stopped", "error")
            action = "start" if snap["paused"] or idle else "stop"
            footer = Text(f"[s] {action} training   [q] quit   ·   Ctrl+C quit   ·   --no-tui for plain logs", style="dim")

            layout = Layout()
            layout.split_column(
                Layout(header, name="header", size=3),
                Layout(name="body"),
                Layout(footer, name="footer", size=1),
            )
            if wide:
                layout["body"].split_row(
                    Layout(name="left", ratio=1),
                    Layout(name="right", ratio=1),
                )
                layout["left"].split_column(
                    Layout(training, name="training", ratio=1),
                    Layout(graph, name="graph", ratio=2),
                )
                layout["right"].split_column(
                    Layout(predictions, name="predictions", ratio=1),
                    Layout(activity, name="activity", ratio=1),
                )
            else:
                layout["body"].split_column(
                    Layout(training, name="training", ratio=3),
                    Layout(graph, name="graph", ratio=2),
                    Layout(predictions, name="predictions", ratio=3),
                    Layout(activity, name="activity", ratio=2),
                )
            return layout
        except Exception as exc:
            return Panel(Text(f"Dashboard render error: {exc}", style="red"), title="CrowdGPT")

    def bind_controls(self, on_toggle, on_quit):
        """Read single keys from the terminal without blocking Rich's refresh."""
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
        except Exception:
            return False
        if self._key_thread is not None and self._key_thread.is_alive():
            return True
        self._key_stop.clear()
        self._key_thread = threading.Thread(target=self._key_loop, name="crowdgpt-tui-keys", daemon=True)
        self._key_thread.start()
        return True

    def unbind_controls(self):
        self._key_stop.set()
        thread, self._key_thread = self._key_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    def _handle_key(self, key):
        key = (key or "").lower()
        if key in ("s", " "):
            if self._on_toggle:
                self._on_toggle()
        elif key in ("q", "\x03"):
            self._key_stop.set()
            if self._on_quit:
                self._on_quit()

    def _key_loop(self):
        try:
            if msvcrt is not None:
                self._key_loop_windows()
            elif _POSIX_KEYS:
                self._key_loop_posix()
            else:
                self.add_log("Keyboard controls are not supported on this platform.", "WARNING")
        except Exception as exc:
            self.add_log(f"Keyboard controls unavailable: {exc}", "WARNING")

    def _key_loop_windows(self):
        while not self._key_stop.is_set():
            if msvcrt.kbhit():
                self._handle_key(msvcrt.getwch())
            else:
                time.sleep(0.05)

    def _key_loop_posix(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._key_stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(fd, 1)
                if not data:
                    break
                self._handle_key(data.decode("utf-8", "ignore"))
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass  # terminal may already be closed during shutdown

    def start(self):
        if self._live is not None:
            return
        self._live = Live(
            get_renderable=self.render,
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            screen=True,
            transient=False,
        )
        try:
            self._live.start(refresh=True)
        except Exception:
            try:
                self._live.stop()  # restore the alternate screen if start failed midway
            except Exception:
                pass
            self._live = None
            raise

    def stop(self):
        self.unbind_controls()
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass  # terminal may already be torn down or redirected
        with self._lock:
            losses = list(self._losses)
            step, target, loss = self.step, self.target, self.loss
            round_no = self.round
        if step or target or losses:
            summary = Text()
            summary.append(f"Round {round_no} · ", style="dim")
            summary.append(f"{_fmt_int(step)} / {_fmt_int(target)} steps", style="bold")
            if loss is not None:
                summary.append(f" · last loss {loss:.4f}", style="yellow")
            self.console.print(Panel(summary, title="CrowdGPT session summary", box=rich_box.ROUNDED, border_style="cyan"))

def _tui_enabled(args):
    if getattr(args, "no_tui", False):
        return False
    if not _RICH_OK:
        return False
    if getattr(args, "tui", False):
        return True
    if getattr(args, "debug", False):
        return False
    if os.environ.get("TERM", "").lower() in ("dumb", "unknown"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False

_HEADLESS_EXIT_MESSAGES = {
    2: "Authentication failed or no saved session. Pass --user/--password or sign in with the desktop app.",
    3: "Backend setup failed.",
    5: "Could not fetch task metadata from the coordinator.",
    6: "The coordinator rejected this client's architecture.",
}

def parse_args():
    p = argparse.ArgumentParser(prog="client.py", description="CrowdGPT client. Runs GUI by default; use --headless for SSH/servers.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--headless", action="store_true", help="no window; train in the terminal")
    p.add_argument("--register", action="store_true", help="create an account, save the session, and exit")
    p.add_argument("--server", default=DEFAULT_SERVER, help="coordinator URL")
    p.add_argument("--user", default=None, help="username")
    p.add_argument("--email", default=None, help="email for --register (or env CROWDGPT_EMAIL)")
    p.add_argument("--password", default=None, help="password (or env CROWDGPT_PASSWORD)")
    p.add_argument("--token", default=None, help="auth token (or env CROWDGPT_TOKEN)")
    p.add_argument("--anon", action="store_true", help="train without an account")
    p.add_argument("--backend", default=None, choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu", "directml"])
    p.add_argument("--vram-gb", type=float, default=0.0)
    p.add_argument("--rounds", type=int, default=0)
    p.add_argument("--no-tokenizer", action="store_true")
    p.add_argument("--no-update", action="store_true")
    p.add_argument("--no-engram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--tui", action="store_true", help="force the full-screen terminal dashboard in headless mode")
    p.add_argument("--no-tui", action="store_true", help="use plain log lines instead of the dashboard")
    p.add_argument("--doctor", action="store_true", help="run read-only environment diagnostics and exit")
    p.add_argument("--json", action="store_true", help="with --doctor, print the report as JSON")
    p.add_argument("--version", action="version", version=f"crowdgpt-client {CLIENT_VERSION}")
    args = p.parse_args()
    if args.json and not args.doctor:
        p.error("--json requires --doctor")
    return args

def _headless_emit(verbose=False):
    _last = {"prog": 0.0, "last_stats": 0.0}
    def emit(ev, data):
        if ev == 'log': log.info(str(data))
        elif ev == 'status': log.info(f"[status] {data}")
        elif ev == 'stats':
            now = time.time()
            if now - _last["last_stats"] < 10.0: return
            _last["last_stats"] = now
            log.info(f"[train] step {data.get('step')}/{data.get('target')} loss={data.get('loss', 0):.4f} tps={data.get('tps', 0):.0f} left={data.get('time_left', 0):.0f}m")
        elif ev == 'cal_stats': log.info(f"[cal] step {data.get('step')}/{data.get('total')} loss={data.get('loss', 0):.4f}")
        elif ev == 'model_preview':
            for c in data.get('cards', []): log.info(f"[preview] {str(c.get('context', ''))[-40:]!r} → pred={c.get('pred')!r} target={c.get('target')!r} {'✓' if c.get('match') else '✗'}")
        elif ev == 'overlay_progress':
            now = time.time()
            if now - _last["prog"] < 5.0: return
            _last["prog"] = now
            done = data.get('done', 0) / (1024 * 1024); total = data.get('total', 0) / (1024 * 1024)
            pct = 100.0 * done / total if total > 0 else 0.0
            log.info(f"[progress] {pct:.1f}% ({done:.0f}/{total:.0f} MB)")
        elif verbose: log.debug(f"[{ev}] {data}")
    return emit

def _install_stop_signal_handlers(stop_event, on_stop=None):
    """First SIGINT/SIGTERM requests a graceful stop; a second one hard-exits.

    No logging from the handler: logging can block on locks held by the thread
    the signal interrupted, which is exactly how the old handler ended up
    ignoring repeated Ctrl-C presses.
    """
    seen = []
    def _on_signal(signum, frame):
        if seen:
            try: signal.signal(signum, signal.SIG_DFL)
            except (ValueError, OSError): pass
            os._exit(130)
        seen.append(signum)
        if on_stop is not None:
            try: on_stop()
            except Exception: pass
        stop_event.set()
    for sig_name in ("SIGINT", "SIGTERM"):
        s = getattr(signal, sig_name, None)
        if s is not None:
            try: signal.signal(s, _on_signal)
            except (ValueError, OSError): pass
    return _on_signal

def run_headless(args):
    dashboard = None
    if _tui_enabled(args):
        try:
            username = args.user
            if not username and AUTH_TOKEN_FILE.exists():
                try:
                    username = json.loads(AUTH_TOKEN_FILE.read_text()).get("username")
                except Exception:
                    username = None  # corrupt token file; normal auth handling will decide
            dashboard = TerminalDashboard()
            if getattr(getattr(dashboard, "console", None), "legacy_windows", False) and not getattr(args, "tui", False):
                log.warning("Legacy Windows console detected; using plain logs. Use Windows Terminal or enable VT processing for the dashboard.")
                dashboard = None
            if dashboard is not None:
                dashboard.update_context(username=username or "saved session", server=args.server)
                dashboard.start()
        except Exception as exc:
            log.warning(f"Terminal dashboard unavailable ({exc}); using plain logs.")
            dashboard = None
    if dashboard is None:
        return _run_headless(args, _headless_emit(verbose=args.debug))

    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    root.handlers = [_DashboardLogHandler(dashboard)]
    root.setLevel(logging.DEBUG if args.debug else logging.INFO)
    control = HeadlessRunControl(dashboard)
    rc = 1
    try:
        if not dashboard.bind_controls(control.toggle, control.request_quit):
            log.info("Keyboard controls unavailable; use Ctrl+C to quit.")
        rc = _run_headless(args, dashboard.emit, dashboard, control)
        return rc
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)
        dashboard.stop()
        if rc:
            log.error(_HEADLESS_EXIT_MESSAGES.get(rc, f"Headless run failed (exit {rc})."))

def _run_headless(args, emit, dashboard=None, control=None):
    control = control or HeadlessRunControl(dashboard)
    password = args.password or os.environ.get("CROWDGPT_PASSWORD"); token = args.token or os.environ.get("CROWDGPT_TOKEN")
    if args.anon: auth_token = None
    elif token: auth_token = token
    elif args.user and password:
        auth_token, err = do_login(args.server, args.user, password)
        if not auth_token: log.error(f"[auth] login failed: {err}"); return 2
    elif args.user:
        log.error("[auth] --user requires --password or CROWDGPT_PASSWORD.")
        return 2
    else:
        auth_token = None
        try:
            if AUTH_TOKEN_FILE.exists():
                saved = json.loads(AUTH_TOKEN_FILE.read_text())
                saved_server = (saved.get("server") or "").rstrip("/")
                if saved_server and saved_server == args.server.rstrip("/"):
                    auth_token = saved.get("token")
        except Exception: pass
        if not auth_token:
            log.error("[auth] no saved session for this server; run --register or pass --user/--password.")
            return 2

    selected = args.backend if args.backend and args.backend != "auto" else None
    if selected is None:
        bf = CHECKPOINT_DIR / "selected_backend.json"
        try:
            if bf.exists():
                saved = json.loads(bf.read_text()).get("backend")
                if saved and get_available_backends().get(saved, False): selected = saved
        except Exception: pass
    if selected is None: selected = get_best_default_backend()

    try: _do_backend_setup(selected, args.vram_gb, emit)
    except Exception: return 3
    if dashboard: dashboard.update_context(backend=selected)

    if not args.no_tokenizer: _load_tokenizer_worker()
    stop_event = control.halt_event
    _install_stop_signal_handlers(stop_event, lambda: control.request_quit(announce=False))

    if args.dry_run:
        try: md = fetch_task_metadata(args.server, {"Authorization": f"Bearer {auth_token}"} if auth_token else {}, stop_event, emit)
        except AuthRejected: return 2
        except CoordinatorRejected: return 7
        if md is None: return 5
        return 0

    max_rounds = args.rounds if args.rounds and args.rounds > 0 else 0; round_count = 0
    while not control.quit_event.is_set():
        if control.pause_event.is_set():
            control.wait_while_paused()
            if control.quit_event.is_set(): break
            continue
        round_count += 1; apply_vram_override(args.vram_gb)
        log.info(f"Round {round_count}" + (f"/{max_rounds}" if max_rounds else ""))
        try: run_single_round_wrapper(args.server, auth_token, stop_event, emit)
        except AuthRejected: return 2
        except CoordinatorRejected:
            _join_upload(stop_event, timeout=(15.0 if stop_event.is_set() else None)); return 7   # let an earlier round's upload finish; if the user quit, wait at most 15s (a second Ctrl+C force-quits)
        except ArchIncompatible: return 6
        except KeyboardInterrupt:
            if control.quit_event.is_set():
                break
            if control.pause_event.is_set():
                control.wait_while_paused()
                if control.quit_event.is_set():
                    break
                continue
            control.request_quit()
            break
        except Exception as e:
            import traceback; log.error(f"Round failed: {e}"); log.error(traceback.format_exc()[-2000:])

        if control.quit_event.is_set(): break
        if control.pause_event.is_set():
            control.wait_while_paused()
            if control.quit_event.is_set(): break
            continue
        if max_rounds and round_count >= max_rounds: break
        if not args.no_update: _check_updates_between_rounds(emit)
        if control.quit_event.is_set(): break
        stop_event.wait(timeout=10.0)
    # normal end of the run: the last round's upload must finish however long it takes; on quit wait at most 15s
    _join_upload(stop_event, timeout=(15.0 if stop_event.is_set() else None))
    return 0

def run_gui(args):
    if not _WEBVIEW_OK:
        log.error(f"GUI unavailable: {_WEBVIEW_ERR}"); return 4

    api = Api(); start_tokenizer_loader()
    kwargs = {"js_api": api, "width": 1000, "height": 720, "min_size": (320, 240)}

    html_path = Path(__file__).resolve().parent / "client.html"
    tmp_html = html_path.with_name(html_path.name + ".new")

    try:
        if not html_path.is_file():
            response = requests.get(HTML_URL, timeout=60); response.raise_for_status(); downloaded_html = response.text
            if ("<html" not in downloaded_html[:500].lower() or "__LOGO__" not in downloaded_html or "__LOGO_DARK__" not in downloaded_html or "__VERSION__" not in downloaded_html):
                raise ValueError("Downloaded client.html failed validation")
            try: tmp_html.write_text(downloaded_html, encoding="utf-8"); os.replace(str(tmp_html), str(html_path))
            finally:
                try: tmp_html.unlink(missing_ok=True)
                except OSError: pass
        html_template = html_path.read_text(encoding="utf-8")
        if "<html" not in html_template[:500].lower() or "__LOGO__" not in html_template or "__LOGO_DARK__" not in html_template or "__VERSION__" not in html_template:
            raise ValueError("Local client.html is invalid")
    except Exception as e:
        log.error(f"Could not load/download {html_path}: {e}"); return 4

    html_out = (html_template.replace("__LOGO__", LOGO_URI_LIGHT).replace("__LOGO_DARK__", LOGO_URI_DARK).replace("__VERSION__", CLIENT_VERSION))
    window = _create_window_safe("CrowdGPT", html_out, kwargs); api.window = window

    def _on_loaded():
        try: startup()
        except Exception: pass
        api._mark_ui_ready()

    subscribed = False
    for attr_chain in (("events", "loaded"), ("loaded",)):
        try:
            obj = window
            for a in attr_chain: obj = getattr(obj, a)
            obj += _on_loaded
            subscribed = True
            break
        except Exception: continue

    if not subscribed:
        log.warning("[ui] No 'loaded' event available; using fallback thread.")
        threading.Thread(target=lambda: (time.sleep(3.0), _on_loaded()), daemon=True).start()

    try: webview.start(debug=False, **_start_icon_kwargs())
    except TypeError: webview.start(debug=False)
    return 0

# The desktop UI predates these codes; keep its generic registration message.
_GUI_REGISTER_CODES = {
    "account_conflict": "register_failed",
    "rate_limited": "register_failed",
    "server_error": "register_failed",
}

_REGISTER_ERRORS = {
    "network": "cannot reach the coordinator",
    "username_taken": "that username is already taken",
    "email_taken": "that email is already registered",
    "weak_password": "the password is too weak (use at least 6 characters)",
    "invalid_input": "the username, email, or password was rejected",
    "account_conflict": "that username or email is already registered",
    "rate_limited": "too many attempts; try again later",
    "server_error": "the coordinator had a server error",
    "register_failed": "the coordinator rejected the registration",
}

def run_register(args):
    username = (args.user or "").strip()
    email = (getattr(args, "email", None) or os.environ.get("CROWDGPT_EMAIL") or "").strip()
    password = args.password or os.environ.get("CROWDGPT_PASSWORD")
    if not username:
        log.error("Registration needs --user.")
        return 2
    if not email:
        log.error("Registration needs --email (or CROWDGPT_EMAIL).")
        return 2
    if "@" not in email:
        log.error("Registration email does not look valid.")
        return 2
    if password is None:
        if not sys.stdin.isatty():
            log.error("Registration needs --password or CROWDGPT_PASSWORD when stdin is not a terminal.")
            return 2
        try:
            password = getpass.getpass("Password: ")
            confirm = getpass.getpass("Confirm password: ")
        except (EOFError, KeyboardInterrupt):
            log.error("Registration cancelled.")
            return 130
        if password != confirm:
            log.error("Passwords do not match.")
            return 2
    if len(password) < 6:
        log.error("Password must be at least 6 characters.")
        return 2
    try:
        token, err = do_register(args.server, username, email, password)
    except KeyboardInterrupt:
        log.error("Registration cancelled.")
        return 130
    if not token:
        log.error(f"Registration failed: {_REGISTER_ERRORS.get(err, err or 'unknown error')}.")
        return 1
    try:
        _save_auth_token(args.server, username, token, True)
    except Exception as exc:
        log.warning(f"Account created, but the session could not be saved: {exc}")
        print(f"Registered {username} on {args.server}. Log in with --user/--password next time.")
        return 0
    print(f"Registered {username} on {args.server}. Session saved for headless training.")
    return 0

if __name__ == "__main__":
    args = parse_args()
    if args.debug: logging.getLogger().setLevel(logging.DEBUG)

    if args.no_engram and ENGRAM_ACTIVE:
        os.environ["CROWDGPT_ENGRAM"] = "0"; ENGRAM_ACTIVE = False

    if args.doctor:
        sys.exit(run_doctor(args))
    if args.register:
        sys.exit(run_register(args))
    if sys.platform == "win32":
        try: ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Vxtzq.CrowdGPT.1")
        except Exception: pass

    if not args.no_update:
        try:
            if check_for_update():
                _restart_self()
        except Exception: pass

    if args.headless: sys.exit(run_headless(args))
    else: sys.exit(run_gui(args))
