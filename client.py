#!/usr/bin/env python3
"""
CrowdGPT GUI Client v2.8

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

import sys, io, json, time, struct, math, gc, threading, queue, base64, logging, tempfile, atexit, subprocess, hashlib, shutil, ctypes, signal, argparse, datetime, uuid
from pathlib import Path

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

if _WEBVIEW_OK:
    logging.getLogger("pywebview").setLevel(logging.CRITICAL)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

CLIENT_VERSION = "2.8"
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
    try:
        if DEVICE_ID_FILE.exists():
            v = json.loads(DEVICE_ID_FILE.read_text()).get("id")
            if isinstance(v, str) and 8 <= len(v) <= 32:
                return v
    except Exception:
        pass
    new_id = uuid.uuid4().hex[:16]
    try:
        DEVICE_ID_FILE.write_text(json.dumps({"id": new_id, "created": time.time()}))
    except Exception:
        pass
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
    def _log(msg):
        if log_fn is not None:
            try: log_fn(msg)
            except Exception: pass
        log.info(msg)
    try:
        hf_root = CHECKPOINT_DIR / "hf_weights"
        if hf_root.is_dir():
            revs = [p for p in hf_root.iterdir() if p.is_dir()]
            if keep_revision is not None:
                keep_norm = str(keep_revision).strip().replace("/", "_").replace("\\", "_")
                for r in revs:
                    if r.name != keep_norm:
                        sz = _dir_size_bytes(r); _safe_rmtree(r); _log(f"[cache] pruned stale revision {r.name} ({sz/1024**2:.0f} MB)")
                revs = [p for p in hf_root.iterdir() if p.is_dir()]
            if len(revs) > MAX_CACHED_REVISIONS:
                revs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
                for r in revs[:-MAX_CACHED_REVISIONS]:
                    sz = _dir_size_bytes(r); _safe_rmtree(r); _log(f"[cache] pruned old revision {r.name} ({sz/1024**2:.0f} MB)")
                revs = [p for p in hf_root.iterdir() if p.is_dir()]
            revs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
            total = sum(_dir_size_bytes(r) for r in revs); cap = int(MAX_HF_CACHE_GB * 1024**3)
            while total > cap and revs:
                victim = revs.pop(0); sz = _dir_size_bytes(victim)
                _safe_rmtree(victim); total -= sz; _log(f"[cache] pruned {victim.name} for size cap ({sz/1024**2:.0f} MB)")
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

def _cleanup_icon_files():
    for p in _cleanup_icon:
        try: os.unlink(p)
        except Exception: pass

atexit.register(_cleanup_icon_files)

def _find_best_icon():
    base = Path(__file__).parent
    for ico in [base / "icon.ico", base / "docs" / "icon.ico"]:
        if ico.is_file(): return str(ico), str(ico)
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
    hicon_small = user32.LoadImageW(None, ico_path, IMAGE_ICON, cx, cy, LR_LOADFROMFILE)
    hicon_big   = user32.LoadImageW(None, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)
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
    act = bs * sl * DIM * 4 * (2 + 1); loss = bs * LOSS_CHUNK * VOCAB_SIZE * 4
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
    }

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
        if n == N_LAYERS and remaining >= N_LAYERS * blk_res + head_res: head_gpu = True
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
        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        if q.dtype != v.dtype or k.dtype != v.dtype: q, k = q.to(v.dtype), k.to(v.dtype)
        if _is_directml(q): o = F.scaled_dot_product_attention(q.contiguous().cpu(), k.contiguous().cpu(), v.contiguous().cpu(), is_causal=True).to(q.device)
        elif q.device.type == 'cuda' and getattr(torch.version, 'hip', None):
            if os.environ.get('CROWDGPT_ROCM_SDPA', '1') != '0':
                try: o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                except RuntimeError as exc:
                    if _is_oom(exc): raise
                    o = self._chunked_attn(q, k, v, T)
            else: o = self._chunked_attn(q, k, v, T)
        else: o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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

def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        return None, "register_failed"
    except Exception: return None, "network"

class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv = srv; self.h = h; self.cr = cr
        self.stop_training = threading.Event(); self.shutdown = threading.Event()
        self.se = se; self.emit = emit; self.tid = None
        self._t = threading.Thread(target=self._r, daemon=True)
    def start(self): self._t.start()
    def stop(self): self.shutdown.set()
    def should_stop(self): return self.stop_training.is_set()
    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                tid = self.tid
                r = requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10, params={"taskId": tid} if tid else None)
                if tid and r.status_code == 200 and r.json().get("task") == "gone":
                    self.emit('log', f"Task {tid} no longer exists on the server (restarted or pool closed). Stopping."); self.stop_training.set()
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr: self.emit('log', "Next round. Stopping."); self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0)) * 60
                    if rm <= 3: self.emit('log', f"ULTIMATUM: {rm:.0f} min left!"); self.stop_training.set()
            except Exception: pass
            self.shutdown.wait(timeout=15)

def wait_for_round(srv, h, se, emit):
    overlay_on = False
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
            else: time.sleep(10)
        except Exception: time.sleep(10)
    hide(); return None

def fetch_task_metadata(srv, h, se, emit, hb=None):
    while not se.is_set() and (not hb or not hb.should_stop()):
        try:
            r = requests.get(f"{srv}/fl/task?format=bf16&skip_weights=true", headers={**(h or {}), **_hardware_headers()}, timeout=30)
            if r.status_code in (401, 403):
                emit('log', f"⚠️ auth rejected ({r.status_code}) — clearing session")
                try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
                except Exception: pass
                emit('login_error', {'code': 'session_expired'}); raise AuthRejected(f"auth rejected: HTTP {r.status_code}")
            if r.headers.get("X-Status") == "wait":
                pool_fill = r.headers.get("X-Pool-Fill"); pool_size_hdr = r.headers.get("X-Pool-Size")
                if pool_fill:
                    msg = f"⏳ Pool matchmaking {pool_fill}"
                    if pool_size_hdr: msg += f"/{pool_size_hdr}"
                    emit('log', msg + " (waiting for peers)")
                try: r.close()
                except Exception: pass
                for _ in range(10):
                    if se.is_set() or (hb and hb.should_stop()): return None
                    time.sleep(1)
                continue
            if r.status_code == 200:
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
            if r.status_code in (503, 504): time.sleep(10); continue
            time.sleep(5)
        except AuthRejected: raise
        except Exception as e:
            if se.is_set() or (hb and hb.should_stop()): return None
            emit('log', f"Task metadata fetch failed: {e}"); time.sleep(10)
    return None

def _weight_cache_path(revision, filename):
    rev = str(revision or "main").strip().replace("/", "_").replace("\\", "_")
    d = CHECKPOINT_DIR / "hf_weights" / rev; d.mkdir(parents=True, exist_ok=True); return d / filename

def _download_file_with_progress(url, dest, emit, stop_evt, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        leftover = dest.with_suffix(dest.suffix + ".part")
        if leftover.exists(): leftover.unlink()
    except Exception: pass
    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == int(expected_size): return dest
        if expected_size is None and sz > 10_000_000: return dest
        try: dest.unlink()
        except Exception: pass
    last_err = None
    for attempt in range(3):
        if stop_evt.is_set(): raise KeyboardInterrupt("Stop requested")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            if tmp.exists(): tmp.unlink()
        except Exception: pass
        try:
            r = requests.get(url, stream=True, timeout=(15, 3600), allow_redirects=True); r.raise_for_status()
            total = int(expected_size) if expected_size else int(r.headers.get("content-length", 0) or 0)
            done = 0; last_emit = 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_evt.is_set(): raise KeyboardInterrupt("Stop requested")
                    if not chunk: continue
                    f.write(chunk); done += len(chunk); now = time.time()
                    if now - last_emit > 0.25: emit("overlay_progress", {"done": done, "total": total}); last_emit = now
            if expected_size and done != int(expected_size): raise RuntimeError(f"Size mismatch for {label}: got {done}, expected {expected_size}")
            os.replace(tmp, dest); return dest
        except KeyboardInterrupt:
            try:
                if tmp.exists(): tmp.unlink()
            except Exception: pass
            raise
        except Exception as e:
            last_err = e
            try:
                if tmp.exists(): tmp.unlink()
            except Exception: pass
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
            if r.status_code in (401, 403):
                r.close()
                emit('log', f"⚠️ auth rejected ({r.status_code}) — clearing session")
                try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
                except Exception: pass
                emit('login_error', {'code': 'session_expired'})
                raise AuthRejected(f"auth rejected: HTTP {r.status_code}")
            if r.status_code != 200:
                r.close(); emit('overlay_hide', {}); emit('log', f"/fl/task HTTP {r.status_code}")
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
        except AuthRejected:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set() or (hb and hb.should_stop()): return None, None
            emit('log', f"Task fetch failed: {e}")
            time.sleep(15)


# =================== LOSS / TRAINING ===================

def fetch_weights_from_hf(md, se, emit):
    wcfg = md.get("weights") or {}
    if wcfg.get("source") != "huggingface": raise RuntimeError("Task metadata does not advertise Hugging Face weights.")
    revision = wcfg.get("revision") or "main"; files = wcfg.get("files") or {}
    model_info = files.get("model") or {}; engram_info = files.get("engram") or {}
    model_url = model_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/model_bf16.safetensors"
    engram_url = engram_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/engram_bf16.safetensors"
    model_size = model_info.get("size"); engram_size = engram_info.get("size")
    if not model_size:
        try: model_size = int(requests.head(model_url, allow_redirects=True, timeout=30).headers.get("content-length", 0) or 0) or None
        except Exception: pass
    if ENGRAM_ACTIVE and not engram_size:
        try: engram_size = int(requests.head(engram_url, allow_redirects=True, timeout=30).headers.get("content-length", 0) or 0) or None
        except Exception: pass
    model_path = _weight_cache_path(revision, "model_bf16.safetensors"); engram_path = _weight_cache_path(revision, "engram_bf16.safetensors")
    emit("status", "downloading"); emit("overlay_show", {"title_key": "dl_weights", "indeterminate": False})
    _download_file_with_progress(model_url, model_path, emit, se, expected_size=model_size, label="model_bf16.safetensors")
    if ENGRAM_ACTIVE: _download_file_with_progress(engram_url, engram_path, emit, se, expected_size=engram_size, label="engram_bf16.safetensors")
    emit("overlay_hide", {}); emit("status", "preparing")
    iw = _load_safetensors_flat(model_path, EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(engram_path, EXPECTED_ENGRAM_SIZE, "engram weights") if ENGRAM_ACTIVE else None
    try:
        for p in CHECKPOINT_DIR.rglob("*.part"):
            try: p.unlink()
            except Exception: pass
    except Exception: pass
    return iw, ie

def _chunk_ce(xs, w, ys):
    logits = F.linear(xs.to(w.dtype), w)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), ys.reshape(-1))

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
                    prune_hf_cache(keep_revision=wcfg.get("revision") or "main", log_fn=lambda m: emit('log', m))
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
                oe.step(); oe.zero_grad(set_to_none=True); tt += step_x.numel(); del lo, x, y, step_x, step_y
            if hb.should_stop() or se.is_set(): return
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            ob.step(); ob.zero_grad(set_to_none=True); gc.collect()

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
                lval = float(lo.item()) * as_; oe.step(); oe.zero_grad(set_to_none=True)
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); ob.step(); ob.zero_grad(set_to_none=True)

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
        emit('overlay_show', {'title_key': 'uploading', 'indeterminate': False}); emit('log', f"Uploading {upload_size_mb:.1f} MB delta to server...")
        r = None
        try:
            if delta_size <= MAX_DIRECT_UPLOAD_BYTES:
                try:
                    with open(delta_path, 'rb') as f: body = f.read()
                    r = requests.post(f"{srv}/fl/submit", headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h}, data=body, timeout=600); body = None
                    if r.status_code == 413: r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
                except Exception as e:
                    body = None
                    if delta_size > 8 * 1024 * 1024: r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
                    else: raise
            else: r = chunked_submit_delta(srv, h, delta_path, meta, emit, se)
            if r is not None and r.status_code == 200:
                try: j = r.json()
                except Exception: j = {}
                if j.get("duplicate"): emit("log", f"⏭️ Not counted by server ({j.get('reason','duplicate')})")
                else: emit("log", "✅ Submitted successfully!")
            elif r is not None: emit("log", f"Submit failed: {r.text[:200]}")
        except Exception as e: emit("log", f"Upload failed: {e}")
        finally:
            emit('overlay_hide', {})
            try: os.unlink(delta_path)
            except OSError: pass
            gc.collect()
    finally:
        hb.stop(); model = ob = oe = x = y = lo = iw = ie = iew = ds = None; gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try: torch.cuda.empty_cache()
            except Exception: pass
        try: prune_hf_cache()
        except Exception: pass

def _do_backend_setup(selected_backend, vram_gb, emit):
    global train_device, train_backend
    train_device, train_backend = detect_training_backend(selected_backend)
    emit('meta', {'backend': train_backend}); emit('log', f"Backend: {train_backend} ({train_device})")
    auto_detect_vram_budget(); memory_config["auto_gb"] = memory_config["ram_gb"]; apply_vram_override(vram_gb)
    emit('log', f"Placement budget: {memory_config['ram_gb']:.2f} GB"); install_hard_vram_cap(train_device, SAFETY_MARGIN_GB)

def _check_updates_between_rounds(emit):
    try:
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
            try: _write_private(AUTH_TOKEN_FILE, json.dumps({"token": tok, "username": u, "server": self.server_url, "remember": bool(remember)}))
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
            try: _write_private(AUTH_TOKEN_FILE, json.dumps({"token": tok, "username": u, "server": self.server_url, "remember": bool(remember)}))
            except Exception: pass
            self.emit('login_success', {'username': u})
        else: self.emit('login_error', {'code': err or 'register_failed'})

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

def startup():
    try:
        scr = webview.screens[0]; aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80))); h = max(360, min(800, int(ah * 0.80)))
        w = max(320, min(w, aw - 16)); h = max(240, min(h, ah - 48))
        webview.windows[0].resize(w, h)
        webview.windows[0].move(max(0, (aw - w) // 2), max(0, (ah - h) // 2))
        if sys.platform == "win32" and WINDOWS_ICO_PATH:
            set_windows_icon(WINDOW_TITLE, WINDOWS_ICO_PATH)
    except Exception: pass

def _create_window_safe(title, html, kwargs):
    try: return webview.create_window(title, html=html, **kwargs)
    except TypeError:
        kwargs.pop("min_size", None); return webview.create_window(title, html=html, **kwargs)

def _start_icon_kwargs():
    if sys.platform == "win32" or not ICON_PATH or not os.path.exists(ICON_PATH): return {}
    return {"icon": ICON_PATH}

def parse_args():
    p = argparse.ArgumentParser(prog="client.py", description="CrowdGPT client. Runs GUI by default; use --headless for SSH/servers.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--headless", action="store_true", help="no window; train in the terminal")
    p.add_argument("--server", default=DEFAULT_SERVER, help="coordinator URL")
    p.add_argument("--user", default=None, help="username")
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
    p.add_argument("--version", action="version", version=f"crowdgpt-client {CLIENT_VERSION}")
    return p.parse_args()

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

def run_headless(args):
    emit = _headless_emit(verbose=args.debug); password = args.password or os.environ.get("CROWDGPT_PASSWORD"); token = args.token or os.environ.get("CROWDGPT_TOKEN")
    if args.anon: auth_token = None
    elif token: auth_token = token
    elif args.user and password:
        auth_token, err = do_login(args.server, args.user, password)
        if not auth_token: log.error(f"[auth] login failed: {err}"); return 2
    else:
        auth_token = None
        try:
            if AUTH_TOKEN_FILE.exists(): auth_token = json.loads(AUTH_TOKEN_FILE.read_text()).get("token")
        except Exception: pass
        if not auth_token: return 2

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

    if not args.no_tokenizer: _load_tokenizer_worker()
    stop_event = threading.Event()
    def _on_signal(signum, frame): stop_event.set()
    for sig_name in ("SIGINT", "SIGTERM"):
        s = getattr(signal, sig_name, None)
        if s is not None:
            try: signal.signal(s, _on_signal)
            except (ValueError, OSError): pass

    if args.dry_run:
        try: md = fetch_task_metadata(args.server, {"Authorization": f"Bearer {auth_token}"} if auth_token else {}, stop_event, emit)
        except AuthRejected: return 2
        if md is None: return 5
        return 0

    max_rounds = args.rounds if args.rounds and args.rounds > 0 else 0; round_count = 0
    while not stop_event.is_set():
        round_count += 1; apply_vram_override(args.vram_gb)
        log.info(f"Round {round_count}" + (f"/{max_rounds}" if max_rounds else ""))
        try: run_single_round_wrapper(args.server, auth_token, stop_event, emit)
        except AuthRejected: return 2
        except ArchIncompatible: return 6
        except KeyboardInterrupt: stop_event.set(); break
        except Exception as e:
            import traceback; log.error(f"Round failed: {e}"); log.error(traceback.format_exc()[-2000:])

        if stop_event.is_set(): break
        if max_rounds and round_count >= max_rounds: break
        if not args.no_update: _check_updates_between_rounds(emit)
        if stop_event.is_set(): break
        stop_event.wait(timeout=10.0)
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

if __name__ == "__main__":
    args = parse_args()
    if args.debug: logging.getLogger().setLevel(logging.DEBUG)

    if args.no_engram and ENGRAM_ACTIVE:
        os.environ["CROWDGPT_ENGRAM"] = "0"; ENGRAM_ACTIVE = False

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
