#!/usr/bin/env python3
"""
CrowdGPT GUI Client v1.8

v1.8 changes:
  - Engram training ENABLED by default (CROWDGPT_ENGRAM=0 to disable)
  - Engram wired into forward pass (x = x + self.engram(idx))
  - Reports X-Client-Autocast and X-Client-Optim to the coordinator
  - Honors targetAutocast=fp32 from the pool (disables AMP)
  - CPU batch-size capped at 2 (prevents "bs=128 on CPU" advertisement)
  - Reports X-Client-Engram for forward-compat server matching
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import sys, io, json, time, struct, math, gc, threading, queue, base64, logging, tempfile, atexit, subprocess, hashlib, shutil, ctypes, signal, argparse, datetime
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

CLIENT_VERSION = "1.8"
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

def _restart_self():
    python = sys.executable; script = str(Path(__file__).resolve()); args = [python, script] + sys.argv[1:]
    if os.name == "nt":
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008); sys.exit(0)
    else: os.execv(python, args)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
AUTH_TOKEN_FILE = CHECKPOINT_DIR / "auth_token.json"

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

LOGO_URI_LIGHT = ""; LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_dark  = Path(__file__).parent / "docs/logo-white.svg"
if _lp_light.exists():
    try: LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except Exception: pass
if _lp_dark.exists():
    try: LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except Exception: pass
if not LOGO_URI_DARK: LOGO_URI_DARK = LOGO_URI_LIGHT

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
    """Generates a clean base64-less fallback icon if no SVG/ICO is found."""
    try:
        from PIL import Image, ImageDraw
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False); tmp.close()
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # Draw a simple orange rounded box with a white circle
        d.rounded_rectangle((20, 20, 236, 236), radius=40, fill=(245, 158, 11, 255))
        d.ellipse((78, 78, 178, 178), fill=(255, 255, 255, 255))
        img.save(tmp.name, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
        return tmp.name
    except Exception:
        return None

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
            
    # Fallback so we don't get the default Tkinter/WebView squares
    if sys.platform == "win32":
        fallback = _generate_fallback_ico()
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
    try:
        import intel_extension_for_pytorch
        backends['xpu'] = hasattr(torch, 'xpu') and torch.xpu.is_available()
    except ImportError: backends['xpu'] = False
    try:
        import torch_directml
        backends['directml'] = torch_directml.is_available()
    except ImportError: backends['directml'] = False
    backends['cpu'] = True
    return backends

def get_best_default_backend():
    avail = get_available_backends()
    for bk in ['cuda', 'rocm', 'mps', 'xpu', 'directml', 'cpu']:
        if avail.get(bk): return bk
    return 'cpu'

def _pick_rocm_device():
    candidates = []
    for idx in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(idx)
            candidates.append((idx, str(props.name), int(props.total_memory)))
        except Exception: pass
    if not candidates: raise RuntimeError("ROCm reports no usable GPU")
    preferred = [g for g in candidates if "radeon rx" in g[1].lower()]
    idx, name, total = max(preferred or candidates, key=lambda g: g[2])
    torch.cuda.set_device(idx)
    log.info(f"[ROCM-FIX] Selected cuda:{idx}: {name} ({total / 1024**3:.1f} GB VRAM)")
    return torch.device(f"cuda:{idx}")

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
    if force and force != "auto":
        n = force.lower()
        if n == "cpu": return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available() and not getattr(torch.version, "hip", None): return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return _pick_rocm_device(), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
            except ImportError: pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available(): return _pick_directml_device(torch_directml), "DIRECTML"
            except ImportError: pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip): return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return _pick_rocm_device(), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
    except ImportError: pass
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

def recommend_batch_size(sl=MAX_SEQ_LEN):
    u8 = has_bitsandbytes() and train_backend in ("CUDA", "ROCM")
    budget = int(memory_config["ram_gb"] * 1024**3); best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget: best = bs
        else: break
    if train_backend == "ROCM":
        rocm_bs_cap = max(1, min(8, int(os.environ.get("CROWDGPT_ROCM_BATCH", 1)))); best = min(best, rocm_bs_cap)
    if train_backend == "CPU": best = min(best, 2)
    return best

def _predict_optim(backend):
    if backend not in ("CUDA", "ROCM"): return "adamw32"
    try:
        import bitsandbytes as bnb
        if not hasattr(bnb, "optim") or not hasattr(bnb.optim, "AdamW8bit"): return "adamw32"
        return "adamw8bit"
    except Exception: return "adamw32"

def _hardware_headers():
    try: bs = recommend_batch_size(MAX_SEQ_LEN)
    except Exception: bs = 1
    gpu = "UNKNOWN"
    try:
        if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available(): gpu = torch.cuda.get_device_name(0)
        elif train_backend: gpu = str(train_backend)
    except Exception: pass
    try:
        _ua, _ad = _pick_autocast(train_backend); autocast_str = "bf16" if _ua else "fp32"
    except Exception: autocast_str = "fp32"
    return {
        "X-Client-Backend":   (train_backend or "UNKNOWN").upper(),
        "X-Client-Batch-Size": str(bs),
        "X-Client-Vram-Gb":   f"{memory_config.get('ram_gb', 0.0):.2f}",
        "X-Client-Gpu-Name":  gpu,
        "X-Client-Autocast":  autocast_str,
        "X-Client-Optim":     _predict_optim(train_backend),
        "X-Client-Engram":    "1" if ENGRAM_ACTIVE else "0",
    }

def _plan_placement(device, backend_name, budget_gb, bs=1, sl=MAX_SEQ_LEN, u8=False, loss_chunk=LOSS_CHUNK):
    if backend_name == "ROCM": loss_chunk = min(loss_chunk, 128)
    if device is None or device.type == "cpu": return False, 0, loss_chunk
    GB = 1024 ** 3; cd = 2 if device.type in ("cuda", "xpu") else 4
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
    def __init__(self, bn, device, gpu_layers=None, bs=1, sl=MAX_SEQ_LEN):
        super().__init__()
        self.main_device = device; self.accel_device = device; self.cpu_device  = torch.device('cpu'); self.backend_name = bn
        is_cpu = (device is None) or (device.type == 'cpu')
        if gpu_layers is None:
            head_gpu, gpu_layers, self.loss_chunk_hint = _plan_placement(device, bn, memory_config["ram_gb"], bs=bs, sl=sl, u8=(has_bitsandbytes() and device is not None and device.type in ('cuda',)))
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

def fetch_task_metadata(srv, h, se, emit):
    while not se.is_set():
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
                    if se.is_set(): return None
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
                            if se.is_set(): return None
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
            if se.is_set(): return None
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

def _build_model(bn, device, iw, ie, emit, bs=1, sl=MAX_SEQ_LEN):
    model = SotaGPT(bn, device, gpu_layers=None, bs=bs, sl=sl)
    model.engram.table.to('cpu'); model.load_base_weights(iw)
    if ENGRAM_ACTIVE and ie is not None: model.engram.table.weight.data.copy_(ie.view(model.engram.table.weight.shape))
    model.engram.table.to('cpu'); model.train(); return model

class _MultiOpt:
    def __init__(self, opts): self.opts = [o for o in opts if o is not None]
    def zero_grad(self, set_to_none=True):
        for o in self.opts: o.zero_grad(set_to_none=set_to_none)
    def step(self):
        for o in self.opts: o.step()

def _build_optimizers(model, lr=1e-4):
    bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
    ep = list(model.engram.parameters())
    groups = {}
    for p in bp: groups.setdefault(p.device.type, []).append(p)
    use_8bit = (_predict_optim(train_backend) == "adamw8bit"); opts = []
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
    if cohort_mode and (cc_seed is None or cc_steps is None or cc_bs is None or cc_accum is None or cc_autocast is None):
        emit('log', "⚠️ Cross-check metadata incomplete; disabling cohort mode for this round"); cohort_mode = False; cc_seed = cc_steps = cc_bs = cc_accum = cc_autocast = None
    if cohort_mode: emit('log', f"[COHORT] ENABLED: seed={cc_seed} steps={cc_steps} bs_target={cc_bs} accum={cc_accum} autocast={cc_autocast}")
    else: emit('log', "[COHORT] disabled (server will use Krum for this shard)")
    return tid, gs, sl, cohort_mode, cc_seed, cc_steps, cc_bs, cc_accum, cc_autocast

def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting'); rs = wait_for_round(srv, h, se, emit)
    if not rs: return
    cr = rs["current_round"]; hb = HeartbeatManager(srv, h, cr, se, emit); hb.start()
    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0; lval = 10.0; tt = 0; tid = ""; tried_cpu_fallback = False
    cohort_mode = False; cc_seed = cc_steps = cc_bs = cc_accum = cc_autocast = None
    try:
        emit('log', "Fetching task metadata..."); md = fetch_task_metadata(srv, h, se, emit)
        if md is None: return
        tid, gs, sl, cohort_mode, cc_seed, cc_steps, cc_bs, cc_accum, cc_autocast = _parse_task_meta(md, emit)
        if not tid: return
        hb.tid = tid; iw = ie = None
        
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
            r = requests.get(f"{srv}/fl/task?format=bf16", headers={**(h or {}), **_hardware_headers()}, timeout=(15, 3600), stream=True)
            if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}")
            raw = r.content; r.close()
            ml = struct.unpack('<I', raw[:4])[0]; md = json.loads(raw[4:4 + ml].decode()); wb = raw[4 + ml:]
            bbl = EXPECTED_MODEL_SIZE * 2; ebl = EXPECTED_ENGRAM_SIZE * 2
            iw = _bf16_from_bytes(wb, 0, bbl); ie = _bf16_from_bytes(wb, bbl, ebl) if ENGRAM_ACTIVE else None
            del wb, raw; gc.collect()

        emit('status', 'preparing'); emit('log', "Preparing model...")
        dc = md.get("datasetConfig", {}); sc = md.get("shardConfig", {}); slot = sc.get("slot", 0)
        if cohort_mode: slot = 0; emit('log', "[COHORT] slot forced to 0 (identical shard for all members)")
        ds = StreamingShardDataset(dc.get("repoId", DATASET_REPO_ID), dc.get("chunkIdx", 0), dc.get("subChunkSize", 10 * 1024 * 1024), dc.get("tokensPerSample", sl + 1), slot, at, cohort_mode=cohort_mode)

        bs = recommend_batch_size(sl)
        if cohort_mode and cc_bs and int(cc_bs) > 0: bs = int(cc_bs)
        model = None
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                gc.collect(); model = _build_model(train_backend, train_device, iw, ie, emit, bs=bs, sl=sl); break
            except RuntimeError as e:
                if _is_oom(e):
                    if model: del model; model = None
                    _clear_accel_cache(train_device); bs = max(1, bs // 2); emit('log', f"Model build OOM → bs={bs}")
                else: raise
        if model is None: return
        if cohort_mode and cc_bs and bs < int(cc_bs): emit('log', f"[COHORT] OOM at target bs={cc_bs}; aborting round"); return

        iew = ie.view(model.engram.table.weight.shape) if (ENGRAM_ACTIVE and ie is not None) else None; ie = None; _trim_heap()
        mbt = bs * sl; as_ = int(cc_accum) if (cohort_mode and cc_accum) else max(1, round(131072 / mbt))
        ls = 1.0 / as_; ob, oe = _build_optimizers(model, lr=1e-4)

        def _on_step_down(_ladder):
            nonlocal ob, oe
            if cohort_mode: raise CohortAbort("OOM during cohort training")
            ob, oe = _build_optimizers(model, lr=1e-4); emit('log', "   rebuilt optimizers for new placement")

        ladder = DegradeLadder(model, on_step_down=_on_step_down)
        ua, ad = _pick_autocast(train_backend)
        emit('log', f"Backend: {train_backend} | autocast: {ua} | bs={bs} | accum={as_} | dtype={'bf16' if ua else 'fp32'} | engram={'on' if ENGRAM_ACTIVE else 'off'} | optim={_predict_optim(train_backend)}")

        if cohort_mode and cc_autocast and cc_autocast != "any":
            wants_bf16 = (cc_autocast == "bf16")
            if wants_bf16 != bool(ua):
                if not wants_bf16: emit('log', f"[COHORT] server requires fp32 — disabling autocast"); ua, ad = False, None
                else: emit('log', f"[COHORT] cannot reach server bf16 on {train_backend} — aborting round"); return

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

        if se.is_set() or hb.should_stop(): return
        if cohort_mode and int(micro_step) != int(cc_steps): return
        fl = float(lval) if lval else 10.0; emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}"); emit('status', 'uploading')
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

        submit_extra = {"crossCheckUsed": cohort_mode, "actualSeed": int(cc_seed) if (cohort_mode and cc_seed is not None) else None, "actualSteps": int(micro_step), "actualBatchSize": int(bs)}
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
        if check_for_update(): emit('log', "🔄 Updated to a new version — restarting..."); emit('status', 'idle'); time.sleep(2); _restart_self()
    except Exception: pass

class Api:
    def __init__(self):
        self.window = None; self.stop_event = threading.Event(); self.thread = None
        self.auth_token = None; self.username = None; self.server_url = DEFAULT_SERVER
        self._lp = 0.0
        self._ui_queue = queue.Queue(maxsize=2048)
        self._ui_ready = threading.Event()
        threading.Thread(target=self._ui_pump, daemon=True, name="ui-emit").start()
        try:
            if AUTH_TOKEN_FILE.exists():
                j = json.loads(AUTH_TOKEN_FILE.read_text()); tok = j.get("token")
                if tok: self.auth_token = tok; self.username = j.get("username"); log.info(f"[auth] restored session for {self.username}")
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
        # BLOCK until the webview page fires 'loaded'. Prevents UI thread deadlock.
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
        if self.auth_token:
            try:
                self.emit('login_success', {'username': self.username or 'restored'})
                if self.thread is None or not self.thread.is_alive(): self.start()
            except Exception: pass

    def get_backends(self):
        self.emit('backends', {'available': get_available_backends(), 'current': self.selected_backend})
        self.emit('vram', {'current': self.vram_gb})

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

    def login(self, s, u, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p), daemon=True).start()

    def _do_login(self, u, p):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token = tok; self.username = u
            try: _write_private(AUTH_TOKEN_FILE, json.dumps({"token": tok, "username": u}))
            except Exception: pass
            self.emit('login_success', {'username': u})
        else: self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p), daemon=True).start()

    def _do_register(self, u, email, p):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token = tok; self.username = u
            try: _write_private(AUTH_TOKEN_FILE, json.dumps({"token": tok, "username": u}))
            except Exception: pass
            self.emit('login_success', {'username': u})
        else: self.emit('login_error', {'code': err or 'register_failed'})

    def login_anon(self):
        self.auth_token = None; self.username = "anonymous"
        try: AUTH_TOKEN_FILE.unlink(missing_ok=True)
        except Exception: pass
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear(); self.thread = threading.Thread(target=self._rs, daemon=True); self.thread.start()

    def stop(self): self.stop_event.set()
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
                self.auth_token = None; self.emit('status', 'idle'); self.emit('log', "🔒 Session expired — please log in again."); break
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
