#!/usr/bin/env python3
"""
CrowdGPT GUI Client v1, DirectML runtime fallback.

Memory-safety guarantees:
  * Allocator env vars are set before importing torch.
  * A hard per-process VRAM cap (total - SAFETY_MARGIN_GB) is installed on the
    accelerator, so any over-budget allocation becomes a catchable torch OOM
    instead of a driver-level failure.
  * The batch/placement estimator counts weights, grads, optimizer state,
    activations, loss chunks, and an allocator reserve.
  * A DegradeLadder reacts to any OOM by moving weight *storage* to system RAM
    (head, then blocks; compute stays on the accelerator and the weights are
    streamed in per use) and retrying the same micro-batch. The bottom rung is
    CPU-only, which has zero accelerator allocations.
  * Calibration is bounded by wall-clock time and step count so slow hardware
    does not hang in "Calibrating" for hours.

Disk-safety guarantees:
  * Only the newest Hugging Face weight revision is kept on disk.
  * Total hf_weights/ size is hard-capped.
  * Stale .part / .bak / .new files are removed at startup and after downloads.

Headless mode:
  * `python client.py --headless [--user U --password P | --token T | --anon]`
    runs the exact same training loop with no window (SSH / systemd / nohup).
  * Ctrl+C / SIGTERM finish the current micro-step and upload the partial delta.

Env vars (all use the CROWDGPT_ prefix):
  CROWDGPT_PASSWORD, CROWDGPT_TOKEN, CROWDGPT_VRAM_GB,
  CROWDGPT_ROCM_BATCH, CROWDGPT_ROCM_SDPA, CROWDGPT_ROCM_AUTOUPDATE
  (CROWDBPT_VRAM_GB still works but is deprecated.)
"""

import os

# ---------------------------------------------------------------------------
# Allocator configuration MUST happen before torch is imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import sys, io, json, time, struct, math, gc, threading, base64, logging, tempfile, atexit, subprocess, hashlib, shutil, ctypes, signal, argparse
from pathlib import Path

# `webview` may be unavailable on headless servers (no GTK/Qt/WebView2);
# import defensively so `--headless` never fails on that.
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

CLIENT_VERSION = "1"
VERSION_URL = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/version.txt"
CLIENT_URL  = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client.py"


def _parse_version(v):
    try:
        parts = str(v).strip().split("-")[0].split(".")
        return tuple(int(x) for x in parts[:4])
    except Exception:
        return (0,)


def _fetch_remote_version():
    try:
        r = requests.get(VERSION_URL, timeout=5)
        r.raise_for_status()
        line = r.text.strip().splitlines()[0].strip()
        return line or None
    except Exception:
        return None


def _download_new_client(expected_version):
    try:
        r = requests.get(CLIENT_URL, timeout=60)
        r.raise_for_status()
        code = r.text
    except Exception:
        return None
    try:
        compile(code, "client.py", "exec")
    except SyntaxError:
        return None
    markers = [f'CLIENT_VERSION = "{expected_version}"', f"CLIENT_VERSION = '{expected_version}'"]
    if not any(m in code for m in markers):
        return None
    return code


def _apply_update(new_code):
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
        os.replace(str(tmp), str(current))
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def check_for_update():
    # Keep this local ROCm compatibility fix from being silently overwritten.
    # Opt in to upstream updates with CROWDGPT_ROCM_AUTOUPDATE=1.
    if getattr(torch.version, "hip", None) and os.environ.get("CROWDGPT_ROCM_AUTOUPDATE") != "1":
        log.info("[ROCM-FIX] Auto-update paused; preserving local ROCm fix.")
        return False
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
    log.info(f"client.py updated to {remote}")
    return True


def _restart_self():
    python = sys.executable
    script = str(Path(__file__).resolve())
    args = [python, script] + sys.argv[1:]
    if os.name == "nt":
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008)
        sys.exit(0)
    else:
        os.execv(python, args)


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache pruning: keep disk usage bounded across rounds and revisions.
# ---------------------------------------------------------------------------
MAX_CACHED_REVISIONS = 1
MAX_HF_CACHE_GB      = 6.0
STALE_PART_AGE_SEC   = 60 * 60
STALE_BAK_AGE_SEC    = 24 * 60 * 60


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass
    return total


def _safe_rmtree(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def prune_hf_cache(keep_revision: str = None, log_fn=None):
    """
    Keep disk usage bounded:
      * Remove every hf_weights/<rev>/ except `keep_revision` (if given).
      * If not given, keep only the newest MAX_CACHED_REVISIONS folders.
      * Delete stale .part files anywhere under CHECKPOINT_DIR.
      * Delete stale .bak / .new files next to this script.
      * Enforce MAX_HF_CACHE_GB by dropping oldest revisions.
    Safe to call at any time. Never raises.
    """
    def _log(msg):
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:
                pass
        log.info(msg)

    try:
        hf_root = CHECKPOINT_DIR / "hf_weights"
        if hf_root.is_dir():
            revs = [p for p in hf_root.iterdir() if p.is_dir()]

            if keep_revision is not None:
                keep_norm = str(keep_revision).strip().replace("/", "_").replace("\\", "_")
                for r in revs:
                    if r.name != keep_norm:
                        sz = _dir_size_bytes(r)
                        _safe_rmtree(r)
                        _log(f"[cache] pruned stale revision {r.name} ({sz/1024**2:.0f} MB)")
                revs = [p for p in hf_root.iterdir() if p.is_dir()]

            if len(revs) > MAX_CACHED_REVISIONS:
                revs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
                for r in revs[:-MAX_CACHED_REVISIONS]:
                    sz = _dir_size_bytes(r)
                    _safe_rmtree(r)
                    _log(f"[cache] pruned old revision {r.name} ({sz/1024**2:.0f} MB)")
                revs = [p for p in hf_root.iterdir() if p.is_dir()]

            revs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
            total = sum(_dir_size_bytes(r) for r in revs)
            cap = int(MAX_HF_CACHE_GB * 1024**3)
            while total > cap and revs:
                victim = revs.pop(0)
                sz = _dir_size_bytes(victim)
                _safe_rmtree(victim)
                total -= sz
                _log(f"[cache] pruned {victim.name} for size cap ({sz/1024**2:.0f} MB)")

            try:
                if hf_root.is_dir() and not any(hf_root.iterdir()):
                    hf_root.rmdir()
            except Exception:
                pass

        now = time.time()
        try:
            for part in CHECKPOINT_DIR.rglob("*.part"):
                try:
                    if now - part.stat().st_mtime > STALE_PART_AGE_SEC:
                        part.unlink()
                        _log(f"[cache] removed stale {part.name}")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            here = Path(__file__).resolve().parent
            for pat in ("*.bak", "*.new"):
                for f in here.glob(pat):
                    try:
                        if now - f.stat().st_mtime > STALE_BAK_AGE_SEC:
                            f.unlink()
                            _log(f"[cache] removed stale {f.name}")
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as e:
        log.warning(f"[cache] prune failed: {e}")


# Run once at startup, before anything else touches disk.
try:
    prune_hf_cache()
except Exception:
    pass


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
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]
DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]
N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]
HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]
MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865
# The engram table is never read by SotaGPT.forward_features, so it gets no gradients and
# its delta is always empty. While that holds, don't allocate/load/diff its 1.4 GB.
ENGRAM_ACTIVE = False
LOSS_CHUNK = 256
UPLOAD_BUFFER_MIN = 25
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

# --- memory-safety knobs ----------------------------------------------------
SAFETY_MARGIN_GB = 1.0
ALLOCATOR_RESERVE_GB = 0.5
MIN_HEADROOM_GB = 0.25

# --- calibration budget -----------------------------------------------------
CAL_TIME_BUDGET = 90.0
CAL_MIN_MICRO   = 3
CAL_MAX_MICRO_HARD = 20


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
        if v is None or math.isnan(v) or math.isinf(v):
            return default
        return float(v)
    except Exception:
        return default


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


TOKENIZER = None
_tokenizer_started = False
PREVIEW_EVERY = 10


def start_tokenizer_loader():
    global _tokenizer_started
    if _tokenizer_started:
        return
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
                    TOKENIZER = tok
                    log.info(f"Tokenizer loaded: {repo} ({len(tok)} tokens)")
                    return
            except Exception:
                continue
        log.warning("No compatible tokenizer loaded.")
    except ImportError:
        log.warning("transformers not installed.")
    TOKENIZER = None


def decode_ids(ids, max_chars=220):
    try:
        ids = [int(i) for i in ids]
        if TOKENIZER is not None:
            s = TOKENIZER.decode(ids, skip_special_tokens=True)
            s = ' '.join(s.split())
            if len(s) > max_chars:
                s = s[:max_chars] + "…"
            return s or "…"
    except Exception:
        pass
    s = ' '.join(f"#{int(i)}" for i in ids)
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


def decode_one(tok_id):
    s = decode_ids([tok_id], max_chars=32)
    return s if s.strip() else "␠"


# =========================================================================
# Logos + window icon
# =========================================================================
LOGO_URI_LIGHT = ""
LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_app   = Path(__file__).parent / "docs/logo-app.svg"
_lp_dark  = Path(__file__).parent / "docs/logo-white.svg"

if _lp_light.exists():
    try:
        LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except Exception:
        pass
if _lp_dark.exists():
    try:
        LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except Exception:
        pass
if not LOGO_URI_DARK:
    LOGO_URI_DARK = LOGO_URI_LIGHT

ICON_PATH = None
WINDOWS_ICO_PATH = None
WINDOW_TITLE = "CrowdGPT"
_cleanup_icon = []


def _render_svg_to_png(svg_path, size: int = 256):
    svg_path = Path(svg_path)
    if not svg_path.exists():
        return None

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    out = tmp.name

    def _ok():
        try:
            return os.path.getsize(out) > 100
        except Exception:
            return False

    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=out,
                         output_width=size, output_height=size)
        if _ok():
            log.info("[icon] Rendered SVG via cairosvg")
            return out
    except Exception as e:
        log.debug(f"[icon] cairosvg failed: {e}")

    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            sw, sh = float(drawing.width or 0), float(drawing.height or 0)
            if sw > 0 and sh > 0:
                scale = size / max(sw, sh)
                drawing.scale(scale, scale)
                drawing.width = sw * scale
                drawing.height = sh * scale
            renderPM.drawToFile(drawing, out, fmt='PNG')
            if _ok():
                log.info("[icon] Rendered SVG via svglib+reportlab")
                return out
    except Exception as e:
        log.debug(f"[icon] svglib failed: {e}")

    try:
        import resvg_py
        resvg_py.svg_to_bytes(svg_path=str(svg_path), write_to=out,
                              width=size, height=size)
        if _ok():
            log.info("[icon] Rendered SVG via resvg-py")
            return out
    except Exception as e:
        log.debug(f"[icon] resvg-py failed: {e}")

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
            if not renderer.isValid():
                continue
            img = QImage(size, size, QImage.Format_ARGB32)
            img.fill(0)
            painter = QPainter(img)
            renderer.render(painter)
            painter.end()
            if img.save(out, "PNG") and _ok():
                log.info(f"[icon] Rendered SVG via {mod}")
                return out
        except Exception as e:
            log.debug(f"[icon] {mod} failed: {e}")

    try:
        r = subprocess.run(
            ["magick", "-background", "none", str(svg_path),
             "-resize", f"{size}x{size}", out],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and _ok():
            log.info("[icon] Rendered SVG via ImageMagick")
            return out
    except Exception as e:
        log.debug(f"[icon] magick failed: {e}")

    try:
        r = subprocess.run(
            ["inkscape", str(svg_path),
             "-w", str(size), "-h", str(size), "-o", out],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and _ok():
            log.info("[icon] Rendered SVG via Inkscape")
            return out
    except Exception as e:
        log.debug(f"[icon] inkscape failed: {e}")

    try:
        os.unlink(out)
    except Exception:
        pass
    return None


def _png_to_ico(png_path: str, sizes=(256, 128, 64, 48, 32, 16)):
    try:
        from PIL import Image
    except ImportError:
        log.warning("[icon] Pillow not installed — cannot build .ico for Windows. "
                    "Install with: pip install pillow")
        return None
    try:
        img = Image.open(png_path).convert("RGBA")
        tmp = tempfile.NamedTemporaryFile(suffix='.ico', delete=False)
        tmp.close()
        img.save(tmp.name, format="ICO", sizes=[(s, s) for s in sizes])
        return tmp.name
    except Exception as e:
        log.warning(f"[icon] PNG→ICO failed: {e}")
        return None


def _cleanup_icon_files():
    for p in _cleanup_icon:
        try:
            os.unlink(p)
        except Exception:
            pass


atexit.register(_cleanup_icon_files)


def _find_best_icon():
    base = Path(__file__).parent

    ico_candidates = [base / "icon.ico", base / "docs" / "icon.ico"]
    svg_candidates = [base / "docs" / "logo-app.svg",
                      base / "logo-app.svg"]
    png_candidates = [base / "icon.png",
                      base / "docs" / "logo-app.png"]

    for ico in ico_candidates:
        if ico.is_file():
            log.info(f"[icon] Using existing .ico: {ico}")
            return str(ico), str(ico)

    for svg in svg_candidates:
        if svg.is_file():
            rendered = _render_svg_to_png(svg, 256)
            if rendered:
                png_path = Path(rendered)
                _cleanup_icon.append(rendered)
                raster = str(png_path)
                ico_path = None
                if sys.platform == "win32":
                    ico_path = _png_to_ico(raster)
                    if ico_path:
                        _cleanup_icon.append(ico_path)
                        log.info(f"[icon] Built multi-size ICO: {ico_path}")
                    else:
                        log.warning("[icon] SVG rendered but PNG→ICO failed. "
                                    "Install Pillow: pip install pillow")
                return raster, ico_path
            if sys.platform != "win32":
                log.info(f"[icon] No SVG renderer found — using SVG directly: {svg}")
                return str(svg), None
            log.warning(
                "[icon] Could not render SVG. Install one of:\n"
                "    pip install svglib reportlab    (recommended on Windows)\n"
                "    pip install cairosvg            (Linux/macOS)\n"
                "    pip install resvg-py\n"
                "  Or drop a real icon.ico next to client.py."
            )
            break

    for png in png_candidates:
        if png.is_file():
            raster = str(png)
            ico_path = None
            if sys.platform == "win32":
                ico_path = _png_to_ico(raster)
                if ico_path:
                    _cleanup_icon.append(ico_path)
                    log.info(f"[icon] Built multi-size ICO: {ico_path}")
                else:
                    log.warning("[icon] Could not build .ico — window will "
                                "use the default. Install Pillow: pip install pillow")
            log.info(f"[icon] Using PNG: {png}")
            return raster, ico_path

    log.warning("[icon] No icon source found — window will use the default.")
    return None, None


ICON_PATH, WINDOWS_ICO_PATH = _find_best_icon()


# =========================================================================
# Windows native icon swap
# =========================================================================
def set_windows_icon(window_title: str, ico_path: str):
    if sys.platform != "win32" or not ico_path:
        return
    ico_path = os.path.abspath(ico_path)
    if not os.path.exists(ico_path):
        log.warning(f"[icon] not found: {ico_path}")
        return
    if not ico_path.lower().endswith(".ico"):
        log.warning(f"[icon] not an .ico file: {ico_path}")
        return

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes   = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype    = ctypes.c_void_p
    user32.LoadImageW.argtypes    = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.LoadImageW.restype     = ctypes.c_void_p
    user32.SendMessageW.argtypes  = [ctypes.c_void_p, ctypes.c_uint,
                                     ctypes.c_void_p, ctypes.c_void_p]
    user32.SendMessageW.restype   = ctypes.c_void_p
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype  = ctypes.c_int

    hwnd = user32.FindWindowW(None, window_title)
    if not hwnd:
        log.warning(f"[icon] window '{window_title}' not found")
        return

    IMAGE_ICON      = 1
    LR_LOADFROMFILE = 0x0010
    WM_SETICON      = 0x0080
    ICON_SMALL      = 0
    ICON_BIG        = 1
    SM_CXSMICON     = 49
    SM_CYSMICON     = 50

    cx = user32.GetSystemMetrics(SM_CXSMICON) or 16
    cy = user32.GetSystemMetrics(SM_CYSMICON) or 16

    hicon_small = user32.LoadImageW(None, ico_path, IMAGE_ICON, cx, cy, LR_LOADFROMFILE)
    hicon_big   = user32.LoadImageW(None, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)

    if hicon_small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
    if hicon_big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)

    if not hicon_small and not hicon_big:
        err = ctypes.get_last_error()
        log.warning(f"[icon] LoadImageW failed (err={err}) for {ico_path}")
    else:
        log.info(f"[icon] Applied window icon: {ico_path}")


def _wait_for_window(title: str, timeout: float = 5.0) -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype  = ctypes.c_void_p
    deadline = time.time() + timeout
    while time.time() < deadline:
        if user32.FindWindowW(None, title):
            return True
        time.sleep(0.1)
    return False


# =========================================================================
# Backend detection
# =========================================================================
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
        if avail.get(bk):
            return bk
    return 'cpu'


def _pick_rocm_device():
    """Prefer dedicated Radeon RX cards over integrated AMD graphics."""
    candidates = []
    for idx in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(idx)
            name = str(props.name)
            candidates.append((idx, name, int(props.total_memory)))
        except Exception as exc:
            log.warning(f"[ROCM-FIX] GPU {idx} unavailable: {exc}")
    if not candidates:
        raise RuntimeError("ROCm reports no usable GPU")
    preferred = [g for g in candidates if "radeon rx" in g[1].lower()]
    idx, name, total = max(preferred or candidates, key=lambda g: g[2])
    torch.cuda.set_device(idx)
    log.info(f"[ROCM-FIX] Selected cuda:{idx}: {name} ({total / 1024**3:.1f} GB VRAM)")
    return torch.device(f"cuda:{idx}")


def _pick_directml_device(dml):
    """Prefer a Radeon RX to an iGPU when DirectML exposes device names."""
    try:
        devices = [(i, str(dml.device_name(i))) for i in range(dml.device_count())]
        if devices:
            preferred = [gpu for gpu in devices if "radeon rx" in gpu[1].lower()]
            idx, name = (preferred or devices)[0]
            log.info(f"[AMD-AUTO] DirectML device {idx}: {name}")
            return dml.device(idx)
    except Exception as exc:
        log.warning(f"[AMD-AUTO] Could not enumerate DirectML GPUs: {exc}")
    return dml.device(0)


def detect_training_backend(force=None):
    if force and force != "auto":
        n = force.lower()
        if n == "cpu":
            return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available() and not getattr(torch.version, "hip", None):
            return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return _pick_rocm_device(), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    return torch.device('xpu'), "XPU"
            except ImportError:
                pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available():
                    return _pick_directml_device(torch_directml), "DIRECTML"
            except ImportError:
                pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return _pick_rocm_device(), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu'), "XPU"
    except ImportError:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            return _pick_directml_device(torch_directml), "DIRECTML"
    except ImportError:
        pass
    return torch.device('cpu'), "CPU"


def install_hard_vram_cap(device, safety_gb=SAFETY_MARGIN_GB):
    if device is None or device.type != "cuda":
        return None
    try:
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        total = int(props.total_memory)
        cap_bytes = max(int(MIN_HEADROOM_GB * 1024**3),
                        total - int(safety_gb * 1024**3))
        frac = max(0.05, min(1.0, cap_bytes / total))
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        log.info(f"[VRAM-CAP] hard cap {cap_bytes/1024**3:.2f} GB "
                 f"({frac*100:.1f}% of {total/1024**3:.2f} GB) on device {idx}")
        return cap_bytes / 1024**3
    except Exception as e:
        log.warning(f"[VRAM-CAP] could not install hard cap: {e}")
        return None


def auto_detect_vram_budget():
    global train_backend

    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            idx = train_device.index if train_device is not None and train_device.index is not None else 0
            fb, tb = torch.cuda.mem_get_info(idx)
            total_gb = tb / 1024**3
            free_gb  = fb / 1024**3
            usable = max(0.25, min(total_gb, free_gb)
                               - SAFETY_MARGIN_GB
                               - ALLOCATOR_RESERVE_GB)
            memory_config.update({"ram_gb": round(usable, 2),
                                  "is_auto_detected": True})
            log.info(f"[VRAM-BUDGET] free={free_gb:.2f}GB total={total_gb:.2f}GB "
                     f"margin={SAFETY_MARGIN_GB:.2f}GB reserve={ALLOCATOR_RESERVE_GB:.2f}GB "
                     f"→ placement={usable:.2f}GB")
            return
        except Exception:
            pass

    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024**3
            usable = max(0.5, total_gb * 0.75 - SAFETY_MARGIN_GB)
            memory_config.update({"ram_gb": round(usable, 2),
                                  "is_auto_detected": True})
            return
        except Exception:
            pass

    if train_backend == "XPU":
        try:
            total = int(torch.xpu.get_device_properties(0).total_memory)
            free = total
            try:
                free, _t = torch.xpu.mem_get_info(0)
            except Exception:
                pass
            total_gb, free_gb = total / 1024**3, int(free) / 1024**3
            usable = max(0.25, min(total_gb, free_gb) - SAFETY_MARGIN_GB - ALLOCATOR_RESERVE_GB)
            memory_config.update({"ram_gb": round(usable, 2), "is_auto_detected": True})
            log.info(f"[VRAM-BUDGET] XPU free={free_gb:.2f}GB total={total_gb:.2f}GB → placement={usable:.2f}GB")
            return
        except Exception:
            pass

    if train_backend == "DIRECTML":
        memory_config.update({"ram_gb": 1.5, "is_auto_detected": True})
        return

    try:
        import psutil
        usable = max(0.5, psutil.virtual_memory().available / 1024**3
                          - SAFETY_MARGIN_GB)
        memory_config.update({"ram_gb": round(usable, 2),
                              "is_auto_detected": True})
    except Exception:
        pass


def apply_vram_override(user_gb=0.0):
    """
    Cap how much VRAM the placement planner may fill. Priority: the UI dropdown, then
    the CROWDGPT_VRAM_GB env var (CROWDBPT_VRAM_GB is accepted as a deprecated alias);
    0 / unset = Auto. Lower = less VRAM used, more weights streamed from RAM (slower).
    Applied each round.
    """
    if "auto_gb" not in memory_config:          # budget not detected yet (before first Start)
        return
    auto = float(memory_config["auto_gb"])
    cap, src = 0.0, ""
    try:
        if user_gb and float(user_gb) > 0:
            cap, src = float(user_gb), "UI"
        else:
            v = os.environ.get("CROWDGPT_VRAM_GB", "").strip()
            if v:
                src = "CROWDGPT_VRAM_GB"
            else:
                v = os.environ.get("CROWDBPT_VRAM_GB", "").strip()
                if v:
                    src = "CROWDBPT_VRAM_GB (deprecated; use CROWDGPT_VRAM_GB)"
            if v and float(v) > 0:
                cap = float(v)
    except ValueError:
        log.warning("[VRAM-BUDGET] ignoring invalid VRAM limit")
    new = min(auto, cap) if cap > 0 else auto
    memory_config["ram_gb"] = round(new, 2)
    if cap > 0:
        memory_config["is_auto_detected"] = False
        log.info(f"[VRAM-BUDGET] limit {cap:.1f} GB ({src}) → placement budget {new:.2f} GB (auto {auto:.2f} GB)")


def has_bitsandbytes():
    try:
        import bitsandbytes
        return True
    except Exception:
        return False


def _bytes_per_block_params():
    return (
        DIM * 2
        + DIM * (N_HEADS * HEAD_DIM)
        + DIM * (N_KV_HEADS * HEAD_DIM)
        + DIM * (N_KV_HEADS * HEAD_DIM)
        + DIM * DIM
        + DIM * 2
        + DIM * MLP_HIDDEN
        + DIM * MLP_HIDDEN
        + MLP_HIDDEN * DIM
    )


def estimate_vram_bytes(bs, sl, u8):
    per_block = _bytes_per_block_params()

    head_params = VOCAB_SIZE * DIM
    head_bytes  = head_params * (4 + 4 + (2 if u8 else 8))

    blocks = N_LAYERS * per_block
    opt_state_per_param = 2 if u8 else 8
    block_bytes = blocks * (4 + 4 + opt_state_per_param)

    act = bs * sl * DIM * 4 * (2 + 1)

    loss = bs * LOSS_CHUNK * VOCAB_SIZE * 4

    workspace = int(ALLOCATOR_RESERVE_GB * 1024**3)

    return int(head_bytes + block_bytes + act + loss + workspace)


def recommend_batch_size(sl=MAX_SEQ_LEN):
    u8 = has_bitsandbytes() and train_backend in ("CUDA", "ROCM")
    budget = int(memory_config["ram_gb"] * 1024**3)
    best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget:
            best = bs
        else:
            break

    if train_backend == "ROCM":
        rocm_bs_cap = max(1, min(8, int(os.environ.get("CROWDGPT_ROCM_BATCH", 1))))
        best = min(best, rocm_bs_cap)

    log.info(f"[BATCH] recommended bs={best} for budget={memory_config['ram_gb']:.2f}GB "
             f"(8bit_opt={u8})")
    return best


def _plan_placement(device, backend_name, budget_gb, bs=1, sl=MAX_SEQ_LEN,
                    u8=False, loss_chunk=LOSS_CHUNK):
    """
    Decide where the *master copy* of each tensor lives. All compute (blocks,
    head, loss) always runs on the accelerator; only storage differs:

      * resident blocks / head : weights + grads + optimizer state on the accelerator
      * streamed blocks / head : master copy in system RAM, copied to the
                                 accelerator for each use (grads are copied back
                                 by autograd, the optimizer step runs on the CPU)

    Blocks are made resident first (their weights are streamed 3x per
    micro-step: forward, recompute, grads; the head only 2x). The head becomes
    resident only if every block fits as well. With a small budget the loss chunk
    (logits buffer) is shrunk first, which costs almost no speed.

    Returns (head_on_gpu, n_resident_blocks, loss_chunk).
    """
    if backend_name == "ROCM":
        loss_chunk = min(loss_chunk, 128)
    if device is None or device.type == "cpu":
        return False, 0, loss_chunk

    GB = 1024 ** 3
    cd = 2 if device.type in ("cuda", "xpu") else 4      # bytes/param of a streamed copy
    budget = int(max(0.0, budget_gb) * GB)
    per_block = _bytes_per_block_params()
    head_params = VOCAB_SIZE * DIM
    opt_b = 2 if u8 else 8

    def fixed_cost(lc):
        return (
            N_LAYERS * bs * sl * DIM * 4                  # checkpointed block inputs
            + bs * sl * (DIM * 12 + MLP_HIDDEN * 3) * 4   # one block's recompute workspace
            + per_block * cd * 2                          # one streamed block (weights + grads)
            + bs * lc * VOCAB_SIZE * 4 * 3                # loss-chunk logits (+ fp32 copy + grad)
            + int(ALLOCATOR_RESERVE_GB * GB)
        )

    blk_res = per_block * (4 + 4 + opt_b)
    head_res = head_params * (4 + 4 + opt_b)
    head_stream = head_params * cd * 2

    lc = loss_chunk
    while lc > 32 and budget - fixed_cost(lc) <= head_stream:
        lc //= 2
    fixed = fixed_cost(lc)
    remaining = budget - fixed

    head_gpu, n = False, 0
    if remaining > head_stream:
        n = int((remaining - head_stream) // blk_res)
        n = max(0, min(N_LAYERS, n))
        if n == N_LAYERS and remaining >= N_LAYERS * blk_res + head_res:
            head_gpu = True

    if backend_name == "DIRECTML":
        head_gpu, n = False, min(n, 2)

    log.info(f"[PLACEMENT] {backend_name}: budget={budget_gb:.2f}GB fixed={fixed/GB:.2f}GB "
             f"blk_res={blk_res/1024**2:.0f}MB head_res={head_res/GB:.2f}GB bs={bs} u8={u8} "
             f"loss_chunk={lc} → {n}/{N_LAYERS} blocks resident, head {'resident' if head_gpu else 'streamed'}")
    return head_gpu, n, lc


def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)


def _is_directml(t):
    try:
        return t.device.type == 'privateuseone'
    except Exception:
        return False


def _is_directml_device(d):
    try:
        return d.type == 'privateuseone'
    except Exception:
        return False


def _is_accel_error(msg):
    m = msg.lower()
    return any(k in m for k in (
        "privateuse1", "unbox expects", "not implemented for",
        "not supported", "unsupported data type", "bfloat16",
        "internal assert", "dml", "scatter", "compilegraph",
        "out of memory", "allocation", "device_type",
        "not implemented for unique",
    ))


# =========================================================================
# Model
# =========================================================================
class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS
        self.nkv = N_KV_HEADS
        self.nrep = N_HEADS // N_KV_HEADS
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
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q, k = q.to(v.dtype), k.to(v.dtype)
        if _is_directml(q):
            o = F.scaled_dot_product_attention(
                q.contiguous().cpu(), k.contiguous().cpu(), v.contiguous().cpu(),
                is_causal=True).to(q.device)
        elif q.device.type == 'cuda' and getattr(torch.version, 'hip', None):
            if os.environ.get('CROWDGPT_ROCM_SDPA', '1') != '0':
                try:
                    o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
                except RuntimeError as exc:
                    if _is_oom(exc):
                        raise
                    log.warning(f"[ROCM-FIX] SDPA unavailable, using chunked attention: {exc}")
                    o = self._chunked_attn(q, k, v, T)
            else:
                o = self._chunked_attn(q, k, v, T)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(B, T, C))

    def _chunked_attn(self, q, k, v, T, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)
            aw = (q[:, :, i:e, :] @ k[:, :, :e, :].transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            ri = torch.arange(i, e, device=q.device).unsqueeze(1)
            ci = torch.arange(e, device=q.device).unsqueeze(0)
            aw = aw.masked_fill(~(ci <= ri).unsqueeze(0).unsqueeze(0), float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v[:, :, :e, :])
        return torch.cat(chunks, dim=2)


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
    def __init__(self, bn, device):
        super().__init__()
        self.device = device
        rows = ENG_NUM_BUCKETS if ENGRAM_ACTIVE else 1
        self.table = nn.Embedding(rows, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        idx_cpu = idx.detach().cpu()
        px = torch.cat([torch.zeros_like(idx_cpu[:, :1]), idx_cpu[:, :-1]], dim=1)
        hi = (px * 1000003 + idx_cpu) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.reshape(-1), return_inverse=True)
        cr = self.table(uq)
        out = cr[inv].view(idx.shape[0], idx.shape[1], DIM)
        return out.to(self.device)


def _streamed_block(block, dev, dtype, x, cos, sin, use_chunked):
    if dtype is not None:
        params = {n: (p.to(dev, dtype=dtype) if p.dim() > 1 else p.to(dev))
                  for n, p in block.named_parameters()}
        with torch.autocast(device_type=dev.type, dtype=dtype):
            return torch.func.functional_call(block, params, (x, cos, sin, use_chunked))
    params = {n: p.to(dev) for n, p in block.named_parameters()}
    return torch.func.functional_call(block, params, (x, cos, sin, use_chunked))


class SotaGPT(nn.Module):
    def __init__(self, bn, device, gpu_layers=None, bs=1, sl=MAX_SEQ_LEN):
        super().__init__()
        self.main_device = device
        self.accel_device = device
        self.cpu_device  = torch.device('cpu')
        self.backend_name = bn
        is_cpu = (device is None) or (device.type == 'cpu')

        if gpu_layers is None:
            head_gpu, gpu_layers, self.loss_chunk_hint = _plan_placement(
                device, bn, memory_config["ram_gb"], bs=bs, sl=sl,
                u8=(has_bitsandbytes() and device is not None and device.type in ('cuda',)))
        else:
            head_gpu = int(gpu_layers) >= N_LAYERS
            self.loss_chunk_hint = LOSS_CHUNK

        self.gpu_layers = 0 if is_cpu else max(0, min(N_LAYERS, int(gpu_layers)))
        self.gpu_layer_set = set(range(self.gpu_layers))
        self.head_on_gpu = bool(head_gpu) and not is_cpu

        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory(bn, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]:
            self.wte.weight = self.lm_head.weight

        cm_main, sm_main = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos_main", cm_main)
        self.register_buffer("freqs_sin_main", sm_main)

        cm_cpu, sm_cpu = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, self.cpu_device)
        self.register_buffer("freqs_cos_cpu", cm_cpu)
        self.register_buffer("freqs_sin_cpu", sm_cpu)

        for i, block in enumerate(self.blocks):
            block.to(device if i in self.gpu_layer_set else self.cpu_device)

        self._place_head()
        self.engram.table.to('cpu')

        if is_cpu:
            log.info("🧠 SotaGPT: CPU-only, all weights in system RAM")
        else:
            log.info(
                f"🧠 SotaGPT: compute on {device.type}; {self.gpu_layers}/{N_LAYERS} blocks resident "
                f"(rest streamed from RAM), head {'resident' if self.head_on_gpu else 'streamed'}"
            )

    def _place_head(self):
        head_dev = self.main_device if self.head_on_gpu else self.cpu_device
        self.wte.to(head_dev)
        self.lm_head.to(head_dev)
        self.ln_f.to(self.main_device)

    def set_gpu_layers(self, n):
        n = max(0, min(N_LAYERS, int(n)))
        if self.main_device.type == 'cpu':
            n = 0
        self.gpu_layers = n
        self.gpu_layer_set = set(range(n))
        if n < N_LAYERS:
            self.head_on_gpu = False
        for i, blk in enumerate(self.blocks):
            blk.to(self.main_device if i < n else self.cpu_device)
        self._place_head()
        log.info(f"[PLACEMENT] {n}/{N_LAYERS} blocks resident, "
                 f"head {'resident' if self.head_on_gpu else 'streamed'}")

    def force_head_cpu(self):
        self.head_on_gpu = False
        self._place_head()
        log.info("[PLACEMENT] head master copy → system RAM (streamed per step)")

    def to_cpu_only(self):
        self.gpu_layers = 0
        self.gpu_layer_set = set()
        self.head_on_gpu = False
        self.main_device = self.cpu_device
        for blk in self.blocks:
            blk.to(self.cpu_device)
        self._place_head()
        self.engram.table.to('cpu')
        self.engram.device = self.cpu_device
        log.info("[PLACEMENT] model → CPU-only")

    def head_weight(self, dtype=None):
        w = self.lm_head.weight
        if w.device != self.main_device:
            return w.to(self.main_device, dtype=dtype) if dtype is not None else w.to(self.main_device)
        if dtype is not None and w.dtype != dtype:
            return w.to(dtype)
        return w

    def forward_features(self, idx, use_chunked=True, use_ckpt=True, stream_dtype=None, head_w=None):
        dev = self.main_device
        wdev = self.wte.weight.device
        if head_w is not None and not self.head_on_gpu and head_w.device == dev:
            x = F.embedding(idx.to(dev), head_w)
            if x.dtype != torch.float32:
                x = x.float()
        else:
            x = F.embedding(idx.to(wdev), self.wte.weight)
            if x.device != dev:
                x = x.to(dev)

        if dev.type == 'cpu':
            cos, sin = self.freqs_cos_cpu, self.freqs_sin_cpu
        else:
            cos, sin = self.freqs_cos_main, self.freqs_sin_main

        for i, block in enumerate(self.blocks):
            if dev.type == 'cpu' or i in self.gpu_layer_set:
                if use_ckpt:
                    x = checkpoint(block, x, cos, sin, use_chunked,
                                   use_reentrant=False, preserve_rng_state=False)
                else:
                    x = block(x, cos, sin, use_chunked)
            else:
                x = checkpoint(_streamed_block, block, dev, stream_dtype, x, cos, sin, use_chunked,
                               use_reentrant=False, preserve_rng_state=False)
        return self.ln_f(x)

    def forward(self, idx, use_chunked=True, use_ckpt=True):
        return F.linear(self.forward_features(idx, use_chunked, use_ckpt), self.head_weight())

    def get_base_weights(self):
        return np.concatenate([
            p.detach().float().flatten().cpu().numpy()
            for n, p in self.named_parameters() if not n.startswith('engram.')
        ])

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = 1
                for d in p.shape:
                    s *= int(d)
                src = ft[o:o + s].view(p.shape)
                # DirectML's Windows backend does not support BF16 tensors. The
                # checkpoint is stored as BF16, but model parameters are FP32.
                # Convert on CPU first, then transfer FP32 to DirectML; never
                # create a privateuseone/BF16 tensor during weight loading.
                if p.device.type == 'privateuseone':
                    src = src.to(device='cpu', dtype=p.dtype).contiguous()
                    src = src.to(device=p.device)
                else:
                    src = src.to(device=p.device, dtype=p.dtype)
                p.data.copy_(src)
                o += s


# =========================================================================
# Data + networking
# =========================================================================
def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries - 1:
                time.sleep(2 ** (a + 1))
            else:
                raise


class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500

    def __init__(self, repo_id, ci, ss=10 * 1024 * 1024, tps=65, slot=0, at=None):
        self.repo_id = repo_id
        self.ci = ci
        self.sub_size = ss
        self.tps = tps
        self.auth_token = at
        self._fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover(ci)
        self.off = (slot * ss) % self.chunk_size
        self.data = None
        self.n = 0
        self.steps_used = 0
        self._pft = None
        self._pfr = None
        self._lock = threading.Lock()
        self._load(self.off)
        self._prefetch(self.off + ss)

    def _url(self):
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._fmt.format(self.ci)

    def _discover(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        sz = int(r.headers.get('content-length', 0))
                        if sz > 0:
                            self._fmt = fmt
                            return sz
                except Exception:
                    time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")

    def _slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data = tk
        self.n = len(tk) // self.tps
        self.steps_used = 0

    def _load(self, off):
        self._set(self._slice(off))
        self.off = off

    def _prefetch(self, off):
        if off >= self.chunk_size:
            return
        with self._lock:
            self._pfr = None

        def w():
            try:
                r = self._slice(off)
                with self._lock:
                    self._pfr = (off, r)
            except Exception:
                pass
        self._pft = threading.Thread(target=w, daemon=True)
        self._pft.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, srv=None, fmt="bf16"):
        no = self.off + self.sub_size
        if no < self.chunk_size:
            if self._pft:
                self._pft.join(timeout=120)
                self._pft = None
            with self._lock:
                r = self._pfr
                self._pfr = None
            if r and r[0] == no:
                self.off = no
                self._set(r[1])
            else:
                self._load(no)
            self._prefetch(no + self.sub_size)
            return True
        self._req(srv, fmt)
        self.chunk_size = self._discover(self.ci)
        self.off = 0
        self._load(0)
        self._prefetch(self.sub_size)
        return True

    def _req(self, srv, fmt="bf16"):
        if not srv:
            return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers=h, timeout=30)
            if r.status_code != 200:
                return False
            ct = (r.headers.get("content-type") or "").lower()
            if "json" in ct:
                md = r.json()
            else:
                raw = r.content
                if len(raw) < 4:
                    return False
                ml = struct.unpack('<I', raw[:4])[0]
                if len(raw) < 4 + ml:
                    return False
                md = json.loads(raw[4:4 + ml].decode())
            ni = md.get("datasetConfig", {}).get("chunkIdx", self.ci)
            if ni != self.ci:
                self.ci = ni
                return True
        except Exception:
            pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0:
            self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        s = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[x:x + self.tps - 1] for x in s])
        tgt = np.stack([self.data[x + 1:x + self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()


def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        if "invalid" in msg.lower() or "credentials" in msg.lower():
            return None, "invalid_credentials"
        return None, "login_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"


def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        msg_l = msg.lower()
        if "username" in msg_l and "taken" in msg_l:
            return None, "username_taken"
        if "email" in msg_l and "registered" in msg_l:
            return None, "email_taken"
        if "password" in msg_l:
            return None, "weak_password"
        if "invalid" in msg_l:
            return None, "invalid_input"
        return None, "register_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"


class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv = srv
        self.h = h
        self.cr = cr
        self.stop_training = threading.Event()
        self.shutdown = threading.Event()
        self.se = se
        self.emit = emit
        self._t = threading.Thread(target=self._r, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self.shutdown.set()

    def should_stop(self):
        return self.stop_training.is_set()

    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10)
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr:
                        self.emit('log', "Next round. Stopping.")
                        self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0)) * 60
                    if rm <= 3:
                        self.emit('log', f"ULTIMATUM: {rm:.0f} min left!")
                        self.stop_training.set()
            except Exception:
                pass
            self.shutdown.wait(timeout=15)


def wait_for_round(srv, h, se, emit):
    overlay_on = False

    def show_wait():
        nonlocal overlay_on
        if not overlay_on:
            emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True})
            overlay_on = True

    def hide():
        nonlocal overlay_on
        if overlay_on:
            emit('overlay_hide', {})
            overlay_on = False

    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting')
                    show_wait()
                    for _ in range(30):
                        if se.is_set():
                            hide()
                            return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False):
                    hide()
                    return st
                emit('status', 'waiting')
                show_wait()
                for _ in range(30):
                    if se.is_set():
                        hide()
                        return None
                    time.sleep(1)
            else:
                time.sleep(10)
        except Exception:
            time.sleep(10)
    hide()
    return None


def fetch_task_metadata(srv, h, se, emit):
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/task?format=bf16&skip_weights=true", headers=h, timeout=30)
            if r.headers.get("X-Status") == "wait":
                for _ in range(10):
                    if se.is_set():
                        return None
                    time.sleep(1)
                continue
            if r.status_code == 200:
                ct = (r.headers.get("content-type") or "").lower()
                if "json" in ct:
                    md = r.json()
                    if md.get("status") == "wait":
                        for _ in range(10):
                            if se.is_set():
                                return None
                            time.sleep(1)
                        continue
                    return md
                raw = r.content
                if len(raw) >= 4:
                    ml = struct.unpack('<I', raw[:4])[0]
                    if len(raw) >= 4 + ml:
                        return json.loads(raw[4:4 + ml].decode())
                raise RuntimeError("Invalid task metadata response")
            if r.status_code in (503, 504):
                time.sleep(10)
                continue
            time.sleep(5)
        except Exception as e:
            if se.is_set():
                return None
            emit('log', f"Task metadata fetch failed: {e}")
            time.sleep(10)
    return None


def _weight_cache_path(revision, filename):
    rev = str(revision or "main").strip().replace("/", "_").replace("\\", "_")
    d = CHECKPOINT_DIR / "hf_weights" / rev
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _download_file_with_progress(url, dest, emit, stop_evt, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        leftover = dest.with_suffix(dest.suffix + ".part")
        if leftover.exists():
            leftover.unlink()
    except Exception:
        pass

    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == int(expected_size):
            log.info(f"Using cached weight file: {dest} ({sz} bytes)")
            return dest
        if expected_size is None and sz > 10_000_000:
            log.info(f"Using cached weight file: {dest} ({sz} bytes, unverified)")
            return dest
        try:
            dest.unlink()
        except Exception:
            pass
    last_err = None
    for attempt in range(3):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        try:
            log.info(f"Downloading {label}: {url}")
            r = requests.get(url, stream=True, timeout=(15, 3600), allow_redirects=True)
            r.raise_for_status()
            content_length = int(r.headers.get("content-length", 0) or 0)
            total = int(expected_size) if expected_size else content_length
            done = 0
            last_emit = 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_evt.is_set():
                        raise KeyboardInterrupt("Stop requested")
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_emit > 0.25:
                        emit("overlay_progress", {"done": done, "total": total})
                        last_emit = now
            if expected_size and done != int(expected_size):
                raise RuntimeError(f"Size mismatch for {label}: got {done}, expected {expected_size}")
            os.replace(tmp, dest)
            log.info(f"Finished downloading {label}: {dest}")
            return dest
        except KeyboardInterrupt:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt < 2:
                emit("log", f"Download retry {attempt + 1} for {label}: {e}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {label}: {last_err}")


def _load_safetensors_flat(path, expected_numel, label="weights"):
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError("Missing dependency: safetensors.") from e
    tensors = load_file(str(path))
    if "weights" in tensors:
        t = tensors["weights"]
    elif "flat" in tensors:
        t = tensors["flat"]
    elif len(tensors) == 1:
        t = next(iter(tensors.values()))
    else:
        raise RuntimeError(f"{label}: cannot identify tensor key. keys={list(tensors.keys())[:20]}")
    t = t.contiguous().reshape(-1)
    if t.dtype not in (torch.bfloat16, torch.float32):
        t = t.to(torch.float32)
    del tensors
    gc.collect()
    if t.numel() != int(expected_numel):
        raise RuntimeError(f"{label} size mismatch: got {t.numel()}, expected {expected_numel}.")
    _ = float(t[0]) + float(t[-1])
    return t


def fetch_weights_from_hf(md, se, emit):
    wcfg = md.get("weights") or {}
    if wcfg.get("source") != "huggingface":
        raise RuntimeError("Task metadata does not advertise Hugging Face weights.")
    revision = wcfg.get("revision") or "main"
    files = wcfg.get("files") or {}
    model_info = files.get("model") or {}
    engram_info = files.get("engram") or {}
    model_url = model_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/model_bf16.safetensors"
    engram_url = engram_info.get("url") or f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/engram_bf16.safetensors"
    model_size = model_info.get("size")
    engram_size = engram_info.get("size")
    if not model_size:
        try:
            hr = requests.head(model_url, allow_redirects=True, timeout=30)
            model_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            model_size = None
    if ENGRAM_ACTIVE and not engram_size:
        try:
            hr = requests.head(engram_url, allow_redirects=True, timeout=30)
            engram_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            engram_size = None
    model_path = _weight_cache_path(revision, "model_bf16.safetensors")
    engram_path = _weight_cache_path(revision, "engram_bf16.safetensors")
    emit("status", "downloading")
    emit("overlay_show", {"title_key": "dl_weights", "indeterminate": False})
    _download_file_with_progress(model_url, model_path, emit, se, expected_size=model_size, label="model_bf16.safetensors")
    if ENGRAM_ACTIVE:
        _download_file_with_progress(engram_url, engram_path, emit, se, expected_size=engram_size, label="engram_bf16.safetensors")
    emit("overlay_hide", {})
    emit("status", "preparing")
    emit("log", "Loading weights from safetensors...")
    iw = _load_safetensors_flat(model_path, EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(engram_path, EXPECTED_ENGRAM_SIZE, "engram weights") if ENGRAM_ACTIVE else None

    try:
        for p in CHECKPOINT_DIR.rglob("*.part"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass
    return iw, ie


def fetch_task(srv, h, prec, se, emit):
    while not se.is_set():
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})
            r = requests.get(f"{srv}/fl/task?format={prec}", headers=h, timeout=(15, 3600), stream=True)
            if r.status_code != 200:
                r.close()
                emit('overlay_hide', {})
                emit('log', f"/fl/task HTTP {r.status_code}")
                time.sleep(10)
                continue
            if r.headers.get("X-Status") == "wait":
                r.close()
                emit('overlay_hide', {})
                for _ in range(10):
                    if se.is_set():
                        return None, None
                    time.sleep(1)
                continue
            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []
            done = 0
            last = 0.0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                chunks.append(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total})
                    last = now
            r.close()
            emit('overlay_progress', {'done': done, 'total': total})
            emit('overlay_hide', {})
            emit('status', 'preparing')
            raw = b"".join(chunks)
            del chunks
            if len(raw) < 4:
                emit('log', f"/fl/task returned only {len(raw)} bytes; retrying")
                time.sleep(5)
                continue
            ml = struct.unpack('<I', raw[:4])[0]
            if 4 + ml > len(raw):
                emit('log', f"Truncated header; retrying")
                time.sleep(5)
                continue
            md = json.loads(raw[4:4 + ml].decode())
            wb = raw[4 + ml:]
            emit('log', f"/fl/task inline: metadata OK, weights={len(wb):,} bytes")
            if len(wb) < EXPECTED_WEIGHT_BYTES:
                emit('log', f"⚠️ inline truncated at {len(wb):,}; retrying")
                time.sleep(10)
                continue
            return md, wb
        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set():
                return None, None
            emit('log', f"Task fetch failed: {e}")
            time.sleep(15)


# =========================================================================
# Loss + training step + degrade ladder
# =========================================================================
def _chunk_ce(xs, w, ys):
    logits = F.linear(xs.to(w.dtype), w)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(), ys.reshape(-1))


def _fwl(m, x, y, sl, ua, ad, ls=1.0, preview=False, loss_chunk=LOSS_CHUNK):
    sd = ad if ua else None
    nc = max(1, math.ceil(sl / loss_chunk))

    def compute():
        w = m.head_weight(sd)
        xe = m.forward_features(x, use_chunked=True, use_ckpt=True, stream_dtype=sd, head_w=w)
        y_dev = y.to(xe.device)
        lo = sum(
            checkpoint(_chunk_ce,
                       xe[:, i:i + loss_chunk, :], w, y_dev[:, i:i + loss_chunk],
                       use_reentrant=False, preserve_rng_state=False)
            for i in range(0, sl, loss_chunk)
        ) / nc
        return lo, xe, w

    if ua:
        with torch.autocast(device_type=m.main_device.type, dtype=ad):
            lo, xe, w = compute()
    else:
        lo, xe, w = compute()

    cards = []
    if preview:
        try:
            with torch.no_grad():
                B, T, C = xe.shape
                items = [(b, T - 1) for b in range(min(B, 3))]
                for offset in (24, 48, 72):
                    p = T - offset - 1
                    if p >= 9:
                        items.append((0, p))
                items = items[:6]
                for b, p in items:
                    if p < 9 or p >= T:
                        continue
                    start = max(0, p - 9)
                    ctx_ids = x[b, start:p + 1].tolist()
                    target_id = int(y[b, p].item())
                    hidden = xe[b, p, :].unsqueeze(0).to(w.dtype)
                    logits = F.linear(hidden, w).float()
                    probs = torch.softmax(logits[0], dim=-1)
                    prob, pred = torch.max(probs, dim=0)
                    pred_id = int(pred.item())
                    cards.append({
                        'context': decode_ids(ctx_ids),
                        'target': decode_one(target_id),
                        'pred': decode_one(pred_id),
                        'match': bool(pred_id == target_id),
                        'prob': safe_float(prob.item()),
                        'target_id': int(target_id),
                        'pred_id': int(pred_id),
                    })
        except Exception as e:
            log.warning(f"Preview extraction failed: {e}")
            cards = []
    return lo * ls, cards


def _is_oom(e):
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError):
        return "out of memory" in str(e).lower()
    return False


def _clear_accel_cache(device):
    try:
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device is not None and device.type == 'xpu':
            torch.xpu.empty_cache()
    except Exception:
        pass
    gc.collect()


class DegradeLadder:
    def __init__(self, model, on_step_down=None):
        self.model = model
        self.stage = 0
        base_chunk = getattr(model, 'loss_chunk_hint', LOSS_CHUNK)
        self.loss_chunk = min(base_chunk, 128) if model.backend_name == "ROCM" else base_chunk
        self.on_step_down = on_step_down

    def step_down(self, emit) -> bool:
        m = self.model
        while True:
            self.stage += 1
            s = self.stage
            if s == 1:
                if not m.head_on_gpu:
                    continue
                emit('log', "🔻 VRAM rung 1: head weights → system RAM (streamed per step)")
                m.force_head_cpu()
            elif s == 2:
                if m.gpu_layers <= 0:
                    continue
                cur, nxt = m.gpu_layers, m.gpu_layers // 2
                emit('log', f"🔻 VRAM rung 2: resident blocks {cur} → {nxt} (rest streamed)")
                m.set_gpu_layers(nxt)
            elif s == 3:
                if m.gpu_layers <= 0:
                    continue
                emit('log', "🔻 VRAM rung 3: all blocks streamed from system RAM")
                m.set_gpu_layers(0)
            elif s == 4:
                old, new = self.loss_chunk, max(32, self.loss_chunk // 2)
                if new == old:
                    continue
                self.loss_chunk = new
                emit('log', f"🔻 VRAM rung 4: LOSS_CHUNK {old} → {new}")
            elif s == 5:
                old, new = self.loss_chunk, max(16, self.loss_chunk // 2)
                if new == old:
                    continue
                self.loss_chunk = new
                emit('log', f"🔻 VRAM rung 5: LOSS_CHUNK {old} → {new}")
            elif s == 6:
                if m.main_device.type == 'cpu':
                    return False
                emit('log', "🔻 VRAM rung 6: hard fallback — CPU-only model")
                m.to_cpu_only()
                self.loss_chunk = 16
            else:
                return False
            break

        if self.on_step_down is not None:
            try:
                self.on_step_down(self)
            except Exception as e:
                log.warning(f"on_step_down failed: {e}")
        _clear_accel_cache(m.accel_device)
        return True


def _train_step_safe(model, x, y, sl, ua, ad, ls, preview, ladder, emit):
    while True:
        try:
            lo, cards = _fwl(model, x, y, sl, ua, ad, ls,
                             preview=preview, loss_chunk=ladder.loss_chunk)
            lo.backward()
            return lo, cards
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            emit('log', f"💥 OOM detected: {str(e).splitlines()[0]}")
            _clear_accel_cache(model.accel_device)
            if not ladder.step_down(emit):
                emit('log', "💀 OOM even at CPU-only — this is a bug, not VRAM.")
                raise
            emit('log', f"   retrying same batch at ladder stage {ladder.stage}")


# =========================================================================
# Upload
# =========================================================================
def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def _post_json_with_retry(url, payload, h, emit, se, timeout=60, retries=4):
    last_err = None
    for attempt in range(retries):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested")
        try:
            r = requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 503:
                emit("log", "Server busy, waiting...")
                last_err = RuntimeError("server busy 503")
                time.sleep(10)
                continue
            r.raise_for_status()
            return r
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                emit("log", f"POST retry {attempt + 1}: {e}")
                time.sleep(min(20, 2 ** (attempt + 1)))
    raise RuntimeError(f"POST {url} failed: {last_err}")


def _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size):
    total = len(compressed_data)
    chunks = max(1, math.ceil(total / chunk_size))
    checksum = _sha256_hex(compressed_data)
    init_payload = {**meta, "totalBytes": total, "chunkSize": chunk_size, "chunks": chunks, "checksum": checksum}
    emit("log", f"Initializing chunked upload: {chunks} chunks, {total / (1024 * 1024):.1f} MB total")
    r = _post_json_with_retry(f"{srv}/fl/submit/init", init_payload, h, emit, se, timeout=60, retries=4)
    j = r.json()
    upload_id = j.get("uploadId")
    if not upload_id:
        raise RuntimeError("Server did not return uploadId")
    for idx in range(chunks):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested")
        start = idx * chunk_size
        end = min(total, start + chunk_size)
        part = compressed_data[start:end]
        part_sha = _sha256_hex(part)
        attempt = 0
        while True:
            try:
                rr = requests.post(
                    f"{srv}/fl/submit/chunk?uploadId={upload_id}&index={idx}",
                    data=part,
                    headers={**h, "Content-Type": "application/octet-stream", "X-Chunk-Sha256": part_sha},
                    timeout=300,
                )
                if rr.status_code == 413:
                    raise RuntimeError("CHUNK_TOO_LARGE")
                rr.raise_for_status()
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if "CHUNK_TOO_LARGE" in str(e):
                    raise
                attempt += 1
                if se.is_set() or attempt >= MAX_CHUNK_RETRY:
                    raise
                emit("log", f"Chunk {idx + 1}/{chunks} retry {attempt}: {e}")
                time.sleep(min(20, 2 ** attempt))
        emit("overlay_progress", {"done": end, "total": total})
        emit("log", f"  ✓ chunk {idx + 1}/{chunks} ({end / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)")
    emit("log", "Finalizing chunked upload...")
    return _post_json_with_retry(f"{srv}/fl/submit/finalize", {"uploadId": upload_id}, h, emit, se, timeout=900, retries=2)


def chunked_submit_delta(srv, h, compressed_data, meta, emit, se):
    chunk_size = UPLOAD_CHUNK_BYTES
    while True:
        try:
            return _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size)
        except Exception as e:
            msg = str(e)
            if ("CHUNK_TOO_LARGE" in msg or "413" in msg) and chunk_size > 8 * 1024 * 1024:
                chunk_size = max(8 * 1024 * 1024, chunk_size // 2)
                emit("log", f"Chunk too large. Reducing to {chunk_size / (1024 * 1024):.0f} MB and retrying.")
                continue
            raise


def stream_with_progress(data, emit_cb, total_size, stop_evt):
    chunk_size = 128 * 1024
    uploaded = 0
    last_emit = 0.0
    for i in range(0, len(data), chunk_size):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested")
        chunk = data[i:i + chunk_size]
        uploaded += len(chunk)
        now = time.time()
        if now - last_emit > 0.25:
            emit_cb("overlay_progress", {"done": uploaded, "total": total_size})
            last_emit = now
        yield chunk
    emit_cb("overlay_progress", {"done": total_size, "total": total_size})


# =========================================================================
# Model build + optimizer construction
# =========================================================================
def _build_model(bn, device, iw, ie, emit, bs=1, sl=MAX_SEQ_LEN):
    model = SotaGPT(bn, device, gpu_layers=None, bs=bs, sl=sl)
    model.engram.table.to('cpu')
    model.load_base_weights(iw)
    if ENGRAM_ACTIVE and ie is not None:
        model.engram.table.weight.data.copy_(ie.view(model.engram.table.weight.shape))
    model.engram.table.to('cpu')
    model.train()
    return model


class _MultiOpt:
    def __init__(self, opts):
        self.opts = [o for o in opts if o is not None]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def _build_optimizers(model, lr=1e-4):
    bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
    ep = list(model.engram.parameters())

    groups = {}
    for p in bp:
        groups.setdefault(p.device.type, []).append(p)

    opts = []
    for dev_type, ps in groups.items():
        o = None
        if dev_type == 'cuda':
            try:
                import bitsandbytes as bnb
                o = bnb.optim.AdamW8bit(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
            except Exception:
                o = None
        if o is None:
            kw = {} if dev_type in ('cpu', 'cuda') else {"foreach": False}
            o = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.01, **kw)
        opts.append(o)
    ob = _MultiOpt(opts)

    try:
        oe = torch.optim.SparseAdam(ep, lr=lr)
    except Exception:
        oe = torch.optim.AdamW(ep, lr=lr, weight_decay=0.01)
    return ob, oe


def _trim_heap():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _bf16_from_bytes(buf, offset, nbytes):
    a = np.frombuffer(buf, dtype=np.int16, count=nbytes // 2, offset=offset).copy()
    return torch.from_numpy(a).view(torch.bfloat16)


def _to_bf16_exact(arr):
    return torch.from_numpy(arr).to(torch.bfloat16)


def _base_delta_bf16(model, iw_bf16):
    out = torch.empty(iw_bf16.numel(), dtype=torch.bfloat16)
    o = 0
    for n, p in model.named_parameters():
        if n.startswith('engram.'):
            continue
        s = p.numel()
        cur = p.detach().to('cpu', torch.float32).reshape(-1)
        out[o:o + s] = cur - iw_bf16[o:o + s].float()
        o += s
    if o != iw_bf16.numel():
        raise RuntimeError(f"delta size mismatch: {o} vs {iw_bf16.numel()}")
    return out.view(torch.int16).numpy().view(np.uint16)


def _engram_sparse_delta(table, iew_bf16, frac=0.10, chunk=32768):
    n = table.shape[0]
    rn = torch.empty(n)
    for i in range(0, n, chunk):
        rn[i:i + chunk] = (table[i:i + chunk] - iew_bf16[i:i + chunk].float()).abs().sum(dim=1)
    k = max(1, int(n * frac))
    tv, ti = torch.topk(rn, k)
    ai = ti[tv > 1e-8]
    if len(ai) == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint16)
    sv = (table[ai] - iew_bf16[ai].float()).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    return ai.numpy().astype(np.uint32), sv


def _gzip_stream(header_bytes, arrays, level=6, piece=64 * 1024 * 1024):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=level) as gz:
        gz.write(header_bytes)
        for a in arrays:
            mv = memoryview(np.ascontiguousarray(a)).cast('B')
            for i in range(0, len(mv), piece):
                gz.write(mv[i:i + piece])
    return buf.getvalue()


def _pick_autocast(backend):
    # DirectML and CPU use FP32. In particular, do not pass BF16 tensors to
    # torch-directml: its Windows backend aborts on unsupported BF16 dtypes.
    if backend in ("DIRECTML", "CPU"):
        return False, None
    try:
        if backend in ("CUDA", "ROCM") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        if backend == "XPU" and hasattr(torch, "xpu") and torch.xpu.is_bf16_supported():
            return True, torch.bfloat16
    except Exception:
        pass
    return False, None


def _cpu_calibration_batch(x, y, full_sl):
    """Use a small, bounded sequence for CPU calibration, not the full 2048 tokens."""
    try:
        limit = int(os.environ.get("CROWDGPT_CPU_CAL_TOKENS", "32"))
    except (TypeError, ValueError):
        limit = 32
    limit = max(8, min(128, limit))
    step_sl = max(1, min(int(full_sl), int(x.shape[1]), int(y.shape[1]), limit))
    return x[:, :step_sl].contiguous(), y[:, :step_sl].contiguous(), step_sl


# =========================================================================
# Round
# =========================================================================
def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting')
    rs = wait_for_round(srv, h, se, emit)
    if not rs:
        return
    cr = rs["current_round"]
    hb = HeartbeatManager(srv, h, cr, se, emit)
    hb.start()
    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0
    lval = 10.0
    tt = 0
    tid = ""
    tried_cpu_fallback = False
    try:
        emit('log', "Fetching task metadata...")
        md = fetch_task_metadata(srv, h, se, emit)
        if md is None:
            return
        tid = md.get("taskId")
        gs = md.get("globalStep")
        sl = MAX_SEQ_LEN
        if not tid:
            emit("log", "Task metadata missing taskId.")
            return
        iw = ie = None
        if USE_HF_WEIGHTS:
            wcfg = md.get("weights") or {}
            if wcfg.get("source") == "huggingface":
                try:
                    emit('log', "Downloading weights from Hugging Face...")
                    iw, ie = fetch_weights_from_hf(md, se, emit)
                    prune_hf_cache(keep_revision=wcfg.get("revision") or "main",
                                   log_fn=lambda m: emit('log', m))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    import traceback as _tb
                    emit('log', f"Hugging Face weight download failed: {e}")
                    print(f"[HF-FAIL-TRACEBACK]\n{_tb.format_exc()}\n", flush=True)
                    emit('log', "Falling back to coordinator weight stream...")
                    iw = ie = None
        if iw is None or (ENGRAM_ACTIVE and ie is None):
            emit('log', "Downloading weights from coordinator...")
            md2, wb = fetch_task(srv, h, "bf16", se, emit)
            if md2 is None:
                return
            md = md2
            tid = md.get("taskId")
            gs = md.get("globalStep")
            if not tid:
                emit("log", "Coordinator task metadata missing taskId.")
                return
            bbl = EXPECTED_MODEL_SIZE * 2
            ebl = EXPECTED_ENGRAM_SIZE * 2
            needed = bbl + (ebl if ENGRAM_ACTIVE else 0)
            if len(wb) < needed:
                raise RuntimeError(f"Incomplete coordinator payload: got {len(wb)}, expected {needed}")
            iw = _bf16_from_bytes(wb, 0, bbl)
            ie = _bf16_from_bytes(wb, bbl, ebl) if ENGRAM_ACTIVE else None
            del wb
            gc.collect()

        emit('status', 'preparing')
        emit('log', "Preparing model...")
        dc = md.get("datasetConfig", {})
        sc = md.get("shardConfig", {})
        ds = StreamingShardDataset(
            dc.get("repoId", DATASET_REPO_ID),
            dc.get("chunkIdx", 0),
            dc.get("subChunkSize", 10 * 1024 * 1024),
            dc.get("tokensPerSample", sl + 1),
            sc.get("slot", 0),
            at,
        )
        bs = recommend_batch_size(sl)

        model = None
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                gc.collect()
                model = _build_model(train_backend, train_device, iw, ie, emit,
                                     bs=bs, sl=sl)
                break
            except RuntimeError as e:
                if _is_oom(e):
                    if model:
                        del model
                        model = None
                    _clear_accel_cache(train_device)
                    bs = max(1, bs // 2)
                    emit('log', f"Model build OOM → bs={bs}")
                else:
                    raise
        if model is None:
            emit('log', "Failed to initialize model.")
            return

        iew = ie.view(model.engram.table.weight.shape) if (ENGRAM_ACTIVE and ie is not None) else None
        ie = None
        _trim_heap()

        mbt = bs * sl
        as_ = max(1, round(131072 / mbt))
        ls = 1.0 / as_

        ob, oe = _build_optimizers(model, lr=1e-4)

        def _on_step_down(_ladder):
            nonlocal ob, oe
            ob, oe = _build_optimizers(model, lr=1e-4)
            emit('log', "   rebuilt optimizers for new placement")

        ladder = DegradeLadder(model, on_step_down=_on_step_down)

        ua, ad = _pick_autocast(train_backend)

        emit('log', f"Backend: {train_backend} | autocast: {ua} | "
                   f"bs={bs} | accum={as_} | dtype={'bf16' if ua else 'fp32'}")
        if train_backend == 'ROCM':
            emit('log', '[ROCM-FIX] Attention: SDPA=' +
                         ('enabled' if os.environ.get('CROWDGPT_ROCM_SDPA', '1') != '0' else 'disabled') +
                         '; first micro-step may exceed 90s (timer checks between steps).')

        emit('log', f"Calibrating speed (max {CAL_TIME_BUDGET:.0f}s / {CAL_MAX_MICRO_HARD} micro-steps)...")
        cs_ = time.time()
        CAL_MAX_MICRO = max(CAL_MIN_MICRO, min(as_, CAL_MAX_MICRO_HARD))
        if model.main_device.type == 'cpu':
            # The wall-clock guard is checked between micro-steps, so a single
            # full-context CPU forward/backward could leave the UI in Calibrating
            # for a very long time. Use one short calibration step; full training
            # keeps the original sequence length.
            CAL_MAX_MICRO = 1
            emit('log', "[CPU-CAL] CPU-only calibration uses one short sequence (default 32 tokens) to avoid an unbounded wait.")
        target_micro_steps = 100
        emit('status', 'calibrating')

        ob.zero_grad(set_to_none=True)
        while (micro_step < CAL_MAX_MICRO) and ((time.time() - cs_) < CAL_TIME_BUDGET):
            if hb.should_stop() or se.is_set():
                return
            if ds.needs_new_subchunk():
                ds.advance(srv, "bf16")
            sd = (abs(hash("cal")) % 10000) + micro_step
            x, y = ds.get_batch(bs, seed=sd)
            step_x, step_y, step_sl = x, y, sl
            if model.main_device.type == 'cpu':
                step_x, step_y, step_sl = _cpu_calibration_batch(x, y, sl)
                want_preview = False
            else:
                want_preview = ((micro_step + 1) % PREVIEW_EVERY == 0) if train_backend == 'ROCM' else ((micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0))
            if train_backend == 'ROCM':
                emit('log', f'[ROCM-FIX] Calibration micro-step {micro_step + 1}/{CAL_MAX_MICRO} starting...')

            try:
                lo, cards = _train_step_safe(model, step_x, step_y, step_sl, ua, ad, ls,
                                             preview=want_preview,
                                             ladder=ladder, emit=emit)
            except RuntimeError as re:
                if (not tried_cpu_fallback) and _is_accel_error(str(re)):
                    emit('log', f"⚠️ {train_backend} op failed at fwd: {str(re)[:120]}")
                    emit('log', "🔻 Falling back to CPU-only model...")
                    model.to_cpu_only()
                    ob, oe = _build_optimizers(model, lr=1e-4)
                    ua = False; ad = None
                    tried_cpu_fallback = True
                    step_x, step_y, step_sl = _cpu_calibration_batch(x, y, sl)
                    want_preview = False
                    CAL_MAX_MICRO = min(CAL_MAX_MICRO, micro_step + 1)
                    emit('log', f"[CPU-CAL] Retrying with {step_sl} tokens for calibration.")
                    lo, cards = _train_step_safe(model, step_x, step_y, step_sl, ua, ad, ls,
                                                 preview=False,
                                                 ladder=ladder, emit=emit)
                else:
                    raise

            lval = float(lo.item())
            micro_step += 1
            if cards:
                emit('model_preview', {'step': micro_step, 'cards': cards})

            elapsed_cal = max(0.001, time.time() - cs_)
            current_tps = (tt + step_x.numel()) / elapsed_cal
            time_left_mins = 0.0
            if micro_step >= CAL_MIN_MICRO:
                eta_elapsed = elapsed_cal
                if model.main_device.type == 'cpu' and step_sl < sl:
                    eta_elapsed *= (sl / max(1, step_sl))
                sps_micro = eta_elapsed / micro_step
                rh = max(0.1, rs.get("max_round_hours", 2.0)
                              - rs.get("round_elapsed_hours", 0))
                remaining = max(60, (rh * 3600) - eta_elapsed
                                     - (UPLOAD_BUFFER_MIN * 60))
                target_micro_steps = micro_step + int(remaining / sps_micro)
                time_left_mins = remaining / 60.0
            emit('cal_stats', {
                'step': micro_step, 'total': target_micro_steps,
                'loss': safe_float(lval * as_, 10.0),
                'microbatch': micro_step,
                'tps': safe_float(current_tps),
                'time_left': safe_float(time_left_mins),
            })
            oe.step()
            oe.zero_grad(set_to_none=True)
            tt += step_x.numel()
            del lo, x, y, step_x, step_y

        if hb.should_stop() or se.is_set():
            return
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        ob.step()
        ob.zero_grad(set_to_none=True)
        gc.collect()
        emit('log', f"Calibration done: {micro_step} micro-steps in "
                    f"{time.time() - cs_:.1f}s (est. target {target_micro_steps})")

        if se.is_set():
            return
        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
        eta_elapsed = time.time() - cs_
        if model.main_device.type == 'cpu':
            try:
                cpu_cal_tokens = max(8, min(128, int(os.environ.get("CROWDGPT_CPU_CAL_TOKENS", "32"))))
            except (TypeError, ValueError):
                cpu_cal_tokens = 32
            eta_elapsed *= (sl / max(1, min(sl, cpu_cal_tokens)))
        remaining = max(60, (rh * 3600) - eta_elapsed - (UPLOAD_BUFFER_MIN * 60))
        dl = time.time() + remaining
        if micro_step > 0:
            sps_micro = eta_elapsed / micro_step
            target_micro_steps = micro_step + int(remaining / max(sps_micro, 0.001))
        else:
            target_micro_steps = 1000
        emit('status', 'training')
        emit('log', f"Target: ~{target_micro_steps} micro-steps")
        last_stats_emit = 0.0
        while time.time() < dl:
            if hb.should_stop() or se.is_set():
                break
            ob.zero_grad(set_to_none=True)
            als = 0.0
            cc = 0
            for mi in range(as_):
                if hb.should_stop() or se.is_set() or time.time() >= dl:
                    break
                if ds.needs_new_subchunk():
                    ds.advance(srv, "bf16")
                sd = (abs(hash("t")) % 10000) + micro_step * 1000 + mi
                x, y = ds.get_batch(bs, seed=sd)
                want_preview = (model.main_device.type != 'cpu') and ((micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0))

                try:
                    lo, cards = _train_step_safe(model, x, y, sl, ua, ad, ls,
                                                 preview=want_preview,
                                                 ladder=ladder, emit=emit)
                except RuntimeError as re:
                    if (not tried_cpu_fallback) and _is_accel_error(str(re)):
                        emit('log', f"⚠️ {train_backend} op failed at fwd: {str(re)[:120]}")
                        emit('log', "🔻 Falling back to CPU-only model...")
                        model.to_cpu_only()
                        ob, oe = _build_optimizers(model, lr=1e-4)
                        ua = False; ad = None
                        tried_cpu_fallback = True
                        lo, cards = _train_step_safe(model, x, y, sl, ua, ad, ls,
                                                     preview=want_preview,
                                                     ladder=ladder, emit=emit)
                    else:
                        raise

                lval = float(lo.item()) * as_
                oe.step()
                oe.zero_grad(set_to_none=True)
                als += lval
                tt += x.numel()
                cc += 1
                micro_step += 1
                if cards:
                    emit('model_preview', {'step': micro_step, 'cards': cards})
                rem_time = max(1, dl - time.time())
                elapsed_total = time.time() - cs_
                if micro_step > 5:
                    sps_micro = elapsed_total / micro_step
                    future_steps = int(rem_time / sps_micro)
                    current_target = micro_step + future_steps
                    if current_target > target_micro_steps:
                        target_micro_steps = current_target
                if time.time() - last_stats_emit > 0.25 or time.time() >= dl:
                    last_stats_emit = time.time()
                    lv = als / cc if cc > 0 else 10.0
                    ctps = tt / max(time.time() - cs_, 1)
                    emit('stats', {
                        'step': micro_step, 'target': target_micro_steps,
                        'loss': safe_float(lv, 10.0), 'tps': safe_float(ctps),
                        'global_step': gs, 'round': cr,
                        'time_left': safe_float(max(0, (dl - time.time()) / 60)),
                    })
                del lo, x, y
            if hb.should_stop() or se.is_set():
                break
            if cc == 0:
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            ob.step()
            ob.zero_grad(set_to_none=True)

        if se.is_set():
            emit('log', "Stopped by user. Skipping upload.")
            return
        fl = float(lval) if lval else 10.0
        emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}")
        emit('status', 'uploading')

        for _o in (ob, oe):
            try:
                _o.zero_grad(set_to_none=True)
            except Exception:
                pass
        ob = oe = ladder = None
        for _p in model.parameters():
            _p.grad = None
        _clear_accel_cache(train_device)
        _trim_heap()

        dbf = _base_delta_bf16(model, iw)
        iw = None
        if ENGRAM_ACTIVE and iew is not None:
            si, sv = _engram_sparse_delta(model.engram.table.weight.data, iew)
        else:
            si, sv = np.array([], dtype=np.uint32), np.array([], dtype=np.uint16)
        iew = None
        model = None
        _clear_accel_cache(train_device)
        _trim_heap()
        has_engram = bool(len(si) > 0)
        engram_sparse_count = int(len(si))
        pl = json.dumps({
            "taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt,
            "loraRank": 0, "isDelta": True, "weightFormat": "bf16",
            "hasEngram": has_engram, "engramSparseCount": engram_sparse_count,
        }).encode()
        compressed_data = _gzip_stream(struct.pack('<I', len(pl)) + pl, (dbf, si, sv))
        upload_size_mb = len(compressed_data) / (1024 * 1024)
        del dbf, si, sv, pl
        _trim_heap()
        meta = {
            "taskId": tid, "loss": fl, "localSteps": micro_step, "tokensProcessed": tt,
            "loraRank": 0, "isDelta": True, "weightFormat": "bf16",
            "hasEngram": has_engram, "engramSparseCount": engram_sparse_count,
        }
        emit('overlay_show', {'title_key': 'uploading', 'indeterminate': False})
        emit('log', f"Uploading {upload_size_mb:.1f} MB delta to server...")
        r = None
        try:
            if len(compressed_data) <= MAX_DIRECT_UPLOAD_BYTES:
                try:
                    r = requests.post(
                        f"{srv}/fl/submit",
                        headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h},
                        data=compressed_data,
                        timeout=600,
                    )
                    if r.status_code == 413:
                        emit("log", "Direct upload rejected. Switching to chunked...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                except Exception as e:
                    if len(compressed_data) > 8 * 1024 * 1024:
                        emit("log", f"Direct upload failed ({e}). Trying chunked...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                    else:
                        raise
            else:
                r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
            if r is not None and r.status_code == 200:
                emit("log", "Submitted successfully!")
            elif r is not None:
                emit("log", f"Submit failed: {r.text[:200]}")
        except KeyboardInterrupt:
            emit("log", "Upload aborted by user.")
        except Exception as e:
            emit("log", f"Upload failed: {e}")
        finally:
            emit('overlay_hide', {})
            try:
                del compressed_data
            except Exception:
                pass
            gc.collect()
    finally:
        hb.stop()
        model = ob = oe = x = y = lo = iw = ie = iew = ds = None
        gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            prune_hf_cache()
        except Exception:
            pass


# =========================================================================
# UI (HTML/JS) — unchanged from the previous version
# =========================================================================
HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#fafaf9;--bg-soft:#fff;--bg-softer:#f0f0ee;--text:#111113;--text-dim:#3a3a40;--text-muted:#6b6b73;--border:#e4e4e1;--border-strong:#d0d0cb;--accent:#16a34a;--accent-dim:#15803d;--warn:#b45309;--err:#dc2626;--err-soft:rgba(220,38,38,.06);--err-border:rgba(220,38,38,.28);--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;--r:6px}
body.dark{--bg:#1a1a1e;--bg-soft:#242428;--bg-softer:#2e2e33;--text:#e8e8ec;--text-dim:#b8b8be;--text-muted:#888890;--border:#3a3a40;--border-strong:#505058;--accent:#22c55e;--accent-dim:#4ade80;--warn:#fbbf24;--err:#f87171;--err-soft:rgba(248,113,113,.08);--err-border:rgba(248,113,113,.35)}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;width:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);display:flex;flex-direction:column;overflow:hidden;font-size:14px;line-height:1.5}
header{flex:0 0 auto;height:54px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 clamp(12px,2vw,20px);gap:10px}
.logo{display:flex;align-items:center;gap:9px;min-width:0}
.logo-img{height:22px;width:auto}
.logo-fallback{width:24px;height:24px;border:1px solid var(--border-strong);display:none;place-items:center;font-size:13px;font-weight:600}
.logo-text{font-size:15px;font-weight:600;white-space:nowrap}
.logo-ver{font-family:var(--mono);font-size:10px;color:var(--text-muted);border:1px solid var(--border);padding:1px 5px;border-radius:3px}
.header-right{display:flex;align-items:center;gap:8px}
#user-info{font-family:var(--mono);font-size:11px;color:var(--text-muted);max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.theme-toggle{width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);border-radius:4px;cursor:pointer;font-size:14px;padding:0}
.theme-toggle:hover{border-color:var(--border-strong)}
.theme-toggle .icon-sun{display:none}
body.dark .theme-toggle .icon-sun{display:inline}
body.dark .theme-toggle .icon-moon{display:none}
body:not(.dashboard-active) #backend-select,body:not(.dashboard-active) #vram-select{display:none}
.backend-select,.lang-select{font-family:var(--sans);font-size:12px;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);padding:5px 8px;border-radius:4px;cursor:pointer;outline:none}
.backend-select:hover,.lang-select:hover{border-color:var(--border-strong)}
.backend-select:disabled{opacity:.5;cursor:not-allowed}
main{flex:1 1 auto;min-height:0;overflow-y:auto;padding:clamp(10px,2vw,18px) clamp(12px,2vw,20px)}
.view{display:none;height:100%;min-height:0}
.view.active{display:flex;flex-direction:column}
#view-login.active{flex:1 1 auto;align-items:center;justify-content:center;padding:10px 0;min-height:0}
.login-box{width:min(380px,100%);max-height:100%;overflow-y:auto;background:var(--bg-soft);border:1px solid var(--border);padding:clamp(14px,2.5vw,24px);border-radius:8px;display:flex;flex-direction:column}
.auth-screen{display:none;flex-direction:column}
.auth-screen.active{display:flex}
.login-box h2{font-size:clamp(17px,3vw,21px);font-weight:600;text-align:center;margin-bottom:3px}
.login-box h2 .em{color:var(--accent);font-weight:700}
.login-sub{font-size:12px;color:var(--text-dim);text-align:center;margin-bottom:14px}
.fg{margin-bottom:9px}
.fg label{display:block;font-size:9px;color:var(--text-muted);margin-bottom:3px;font-family:var(--mono);text-transform:uppercase}
input{width:100%;font-family:inherit;font-size:12.5px;background:var(--bg-softer);color:var(--text);border:1px solid var(--border);padding:8px 10px;border-radius:4px;outline:none}
input:focus{border-color:var(--accent)}
@font-face{font-family:"Twemoji Country Flags";unicode-range:U+1F1E6-1F1FF,U+1F3F4,U+E0062-E0063,U+E0065,U+E0067,U+E006C,U+E006E,U+E0073-E0074,U+E0077,U+E007F;src:url("https://cdn.jsdelivr.net/npm/country-flag-emoji-polyfill@0.1/dist/TwemojiCountryFlags.woff2") format("woff2");font-display:swap}
:root{--sans:"Twemoji Country Flags",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
button{font-family:inherit;font-size:12.5px;font-weight:500;cursor:pointer;border-radius:4px}
.btn-primary{background:var(--text);color:var(--bg);border:1px solid var(--text);padding:9px 14px}
.btn-primary:hover:not(:disabled){opacity:.85}
.btn-danger{background:transparent;color:var(--err);border:1px solid var(--err);padding:9px 14px}
.btn-danger:hover:not(:disabled){background:var(--err);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
.auth-btn{width:100%;margin-top:2px}
.error-box{display:none;margin-top:8px;padding:8px 10px;background:var(--err-soft);border:1px solid var(--err-border);border-left:3px solid var(--err);border-radius:4px;font-size:12px;color:var(--err)}
.error-box.show{display:block}
.auth-switch{margin-top:12px;font-size:12px;color:var(--text-muted);text-align:center}
.auth-switch a{color:var(--accent-dim);cursor:pointer;text-decoration:none}
.anon-hint{margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:11px;color:var(--text-muted);text-align:center}
.anon-hint a{color:var(--text-dim);cursor:pointer;text-decoration:none}
.dash{display:flex;flex-direction:column;gap:12px;flex:1 1 auto;min-height:0}
.dash-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.dash-title{font-size:17px;font-weight:600}
.status-badge{font-family:var(--mono);font-size:10px;padding:3px 10px;border-radius:100px;background:rgba(22,163,74,.08);color:var(--accent-dim);border:1px solid rgba(22,163,74,.35)}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat-card{background:var(--bg-soft);border:1px solid var(--border);padding:12px 14px;border-radius:var(--r);min-width:0}
.stat-val{font-family:var(--mono);font-size:clamp(17px,2.4vw,22px);font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stat-val.accent{color:var(--accent-dim)}
.stat-lbl{font-size:11px;color:var(--text-muted);margin-top:2px}
.panel{background:var(--bg-soft);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px}
.panel.flex{flex:1 1 auto;display:flex;flex-direction:column;min-height:160px}
.panel-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:8px;font-family:var(--mono);font-size:11px;color:var(--text-muted);flex-wrap:wrap}
.panel-head .pv{color:var(--text)}
.progress-track{height:6px;background:var(--bg-softer);border:1px solid var(--border);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;width:0%;background:var(--accent);transition:width .4s ease}
.dash-body{display:grid;grid-template-columns:1fr;gap:12px;flex:1 1 auto;min-height:0}
@media(min-width:800px){.dash-body{grid-template-columns:1fr 1fr}}
.col{display:flex;flex-direction:column;gap:12px;min-height:0}
.panel-content{flex:1 1 auto;min-height:0;overflow-y:auto}
.graph-container{position:relative;height:230px}
#loss-graph{position:absolute;inset:0;width:100%;height:100%;display:block;transition:opacity .2s}
#loss-graph.hidden-graph{opacity:0;pointer-events:none}
.graph-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-family:var(--mono);font-size:11px;transition:opacity .2s}
.graph-empty.hidden-empty{opacity:0;pointer-events:none}
.pred-lines{display:flex;flex-direction:column;gap:6px;font-family:var(--mono);font-size:12px}
.preview-empty{color:var(--text-muted);text-align:center;padding:24px 0;font-size:11px}
.pred-line{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:6px 8px;border-radius:4px;border:1px solid var(--border);background:var(--bg-softer);animation:lineIn .24s ease}
@keyframes lineIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.pred-line.good{border-left:3px solid var(--accent)}
.pred-line.bad{border-left:3px solid var(--err)}
.pred-main{display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0;flex:1}
.pred-text{color:var(--text-dim);font-size:12px;word-break:break-word}
.pred-token{font-weight:600;padding:1px 5px;border-radius:3px;font-size:12px}
.good .pred-token{background:rgba(22,163,74,.10);color:var(--accent-dim)}
.bad .pred-token{background:rgba(220,38,38,.08);color:var(--err)}
.pred-side{display:flex;align-items:center;gap:8px;color:var(--text-muted);font-size:10px;white-space:nowrap}
.pred-badge{font-weight:600}
.good .pred-badge{color:var(--accent-dim)}
.bad .pred-badge{color:var(--err)}
.status-strip{font-family:var(--mono);font-size:11px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 0}
.ss-event{display:block;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.controls{display:flex;gap:8px;justify-content:flex-end;align-items:center;flex-wrap:wrap}
.overlay{position:fixed;inset:0;background:rgba(250,250,249,.86);backdrop-filter:blur(3px);display:flex;align-items:center;justify-content:center;z-index:50;opacity:0;pointer-events:none;transition:opacity .25s}
body.dark .overlay{background:rgba(26,26,30,.86)}
.overlay.show{opacity:1;pointer-events:auto}
.overlay-card{width:min(420px,88vw);background:var(--bg-soft);border:1px solid var(--border);border-radius:8px;padding:22px 24px}
.overlay-title{font-size:15px;font-weight:600;margin-bottom:4px}
.overlay-subtitle{font-size:12.5px;color:var(--text-muted);margin-bottom:14px;display:none}
.overlay-subtitle.show{display:block}
.overlay-bar{height:10px;background:var(--bg-softer);border:1px solid var(--border);border-radius:5px;overflow:hidden;position:relative}
.overlay-fill{height:100%;width:0%;background:linear-gradient(90deg,#15803d,#16a34a 55%,#4ade80);transition:width .25s ease}
.overlay-bar.indet .overlay-fill{width:35%;transition:none;animation:indet 1.15s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.overlay-sub{margin-top:10px;font-family:var(--mono);font-size:11px;color:var(--text-muted);display:flex;justify-content:space-between;gap:8px}
.motd-overlay{position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-20px);width:min(520px,92vw);background:var(--bg-soft);border:1px solid var(--border);border-left:4px solid var(--warn);border-radius:6px;padding:16px 20px;box-shadow:0 12px 32px rgba(0,0,0,.08);z-index:200;opacity:0;pointer-events:none;transition:opacity .4s,transform .4s;display:flex;flex-direction:column;gap:12px}
.motd-overlay.show{opacity:1;pointer-events:auto;transform:translateX(-50%) translateY(0)}
.motd-header{display:flex;align-items:center;justify-content:space-between;gap:10px}
.motd-title{font-size:11px;font-weight:600;color:var(--warn);display:flex;align-items:center;gap:8px;font-family:var(--mono);text-transform:uppercase}
.motd-close{background:transparent;border:1px solid var(--border);color:var(--text-muted);cursor:pointer;font-size:12px;padding:4px 8px;border-radius:4px;font-family:var(--mono)}
.motd-close:hover{color:var(--text);border-color:var(--border-strong)}
.motd-text{font-size:13px;color:var(--text-dim);line-height:1.65;white-space:pre-wrap;word-break:break-word;max-height:240px;overflow-y:auto}
@media(max-width:520px){header{height:auto;flex-wrap:wrap;padding:8px 12px}#user-info{display:none}.controls button{flex:1}}
</style></head>
<body>
<header>
  <div class="logo">
    <img src="__LOGO__" id="logo-img" data-light="__LOGO__" data-dark="__LOGO_DARK__" class="logo-img" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='grid';">
    <div class="logo-fallback">C</div>
    <span class="logo-text">CrowdGPT</span>
    <span class="logo-ver">v__VERSION__</span>
  </div>
  <div class="header-right">
    <span id="user-info"></span>
    <select class="backend-select" id="backend-select" disabled></select>
    <select class="backend-select" id="vram-select" title="Max VRAM to use. Lower = less VRAM, more streaming from RAM (slower). Applies from the next round.">
      <option value="0">VRAM: Auto</option><option value="3">VRAM ≤ 3 GB</option><option value="4">VRAM ≤ 4 GB</option><option value="5">VRAM ≤ 5 GB</option><option value="6">VRAM ≤ 6 GB</option><option value="8">VRAM ≤ 8 GB</option><option value="10">VRAM ≤ 10 GB</option><option value="12">VRAM ≤ 12 GB</option>
    </select>
    <select class="lang-select" id="lang-select">
      <option value="en">🇬🇧 English</option>
      <option value="fr">🇫🇷 Français</option>
      <option value="es">🇪🇸 Español</option>
      <option value="de">🇩🇪 Deutsch</option>
      <option value="tr">🇹🇷 Türkçe</option>
    </select>
    <button class="theme-toggle" id="theme-toggle"><span class="icon-moon">🌙</span><span class="icon-sun">☀️</span></button>
  </div>
</header>
<main>
  <div id="view-login" class="view active">
    <div class="login-box">
      <div class="auth-screen active" id="screen-login">
        <h2><span data-i18n="login_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="login_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="login-server" value="https://server.crowdgpt.net"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="login-user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="login-pass" autocomplete="current-password"></div>
        <div class="error-box" id="login-error"></div>
        <button id="btn-login" class="btn-primary auth-btn" data-i18n="login_btn"></button>
        <div class="auth-switch"><span data-i18n="no_account"></span> <a id="go-to-register" data-i18n="register_link"></a></div>
        <div class="anon-hint"><span data-i18n="anon_preface"></span> <a id="btn-anon" data-i18n="anon_link"></a></div>
      </div>
      <div class="auth-screen" id="screen-register">
        <h2><span data-i18n="register_title_pre"></span><span class="em">CrowdGPT</span></h2>
        <p class="login-sub" data-i18n="register_sub"></p>
        <div class="fg"><label data-i18n="server_url"></label><input type="text" id="reg-server" value="https://server.crowdgpt.net"></div>
        <div class="fg"><label data-i18n="username"></label><input type="text" id="reg-user" autocomplete="username"></div>
        <div class="fg"><label data-i18n="email"></label><input type="email" id="reg-email" autocomplete="email"></div>
        <div class="fg"><label data-i18n="password"></label><input type="password" id="reg-pass" autocomplete="new-password"></div>
        <div class="error-box" id="reg-error"></div>
        <button id="btn-register" class="btn-primary auth-btn" data-i18n="register_btn"></button>
        <div class="auth-switch"><span data-i18n="have_account"></span> <a id="go-to-login" data-i18n="login_link"></a></div>
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
        <div class="stat-card"><div class="stat-val" id="stat-time">—</div><div class="stat-lbl" data-i18n="time_left"></div></div>
      </div>
      <div class="panel">
        <div class="panel-head"><span><span data-i18n="round"></span> <span class="pv" id="stat-round">—</span> · <span data-i18n="step"></span> <span class="pv" id="stat-step-cur">0</span>/<span class="pv" id="stat-step-tot">0</span></span><span class="pv" id="progress-text">0%</span></div>
        <div class="progress-track"><div class="progress-fill" id="progress-bar"></div></div>
      </div>
      <div class="dash-body">
        <div class="col"><div class="panel flex">
          <div class="panel-head"><span data-i18n="loss_history"></span><span class="pv" id="graph-last"></span></div>
          <div class="graph-container"><canvas id="loss-graph"></canvas><div class="graph-empty" id="graph-empty" data-i18n="waiting_data"></div></div>
        </div></div>
        <div class="col"><div class="panel flex">
          <div class="panel-head"><span data-i18n="model_thoughts"></span><span class="pv" id="preview-step">—</span></div>
          <div class="panel-content"><div id="model-preview" class="pred-lines"><div class="preview-empty" data-i18n="waiting_preview"></div></div></div>
        </div></div>
      </div>
      <div class="status-strip"><span class="ss-event" id="status-event"></span></div>
      <div class="controls">
        <button id="btn-start" class="btn-primary" data-i18n="start"></button>
        <button id="btn-stop" class="btn-danger" data-i18n="stop" disabled></button>
      </div>
    </div>
  </div>
</main>
<div class="overlay" id="overlay"><div class="overlay-card">
  <div class="overlay-title" id="overlay-title"></div>
  <div class="overlay-subtitle" id="overlay-subtitle"></div>
  <div class="overlay-bar" id="overlay-bar"><div class="overlay-fill" id="overlay-fill"></div></div>
  <div class="overlay-sub"><span id="overlay-pct"></span><span id="overlay-mb"></span></div>
</div></div>
<div class="motd-overlay" id="motd-overlay">
  <div class="motd-header">
    <div class="motd-title">⚠️ <span data-i18n="motd_title">System Notice</span></div>
    <button class="motd-close" id="motd-close" data-i18n="close">Close</button>
  </div>
  <div class="motd-text" id="motd-text"></div>
</div>
<script>
const I18N={
en:{login_pre:"Sign in to ",login_sub:"Welcome back to the swarm.",register_title_pre:"Join ",register_sub:"Create your account.",server_url:"Server address",username:"Username",password:"Password",email:"Email",login_btn:"Log in",register_btn:"Create account",no_account:"No account?",register_link:"Create one",have_account:"Already registered?",login_link:"Log in",anon_preface:"Stay anonymous?",anon_link:"Skip and contribute anonymously",dash_title:"Training",loss:"Loss",tokens_sec:"Tokens/sec",time_left:"Time left",round:"Round",step:"Step",loss_history:"Loss history",waiting_data:"Waiting for data…",model_thoughts:"Model Predictions",waiting_preview:"Waiting for first prediction…",idle:"Idle",training:"Training…",waiting:"Waiting…",uploading:"Uploading…",downloading:"Downloading…",preparing:"Preparing…",calibrating:"Calibrating…",dl_weights:"Downloading weights…",wait_coord:"Waiting for coordinator…",wait_coord_sub:"This can take a few minutes.",start:"Start training",stop:"Stop",err_invalid_credentials:"Invalid username or password.",err_username_taken:"Username taken.",err_email_taken:"Email already registered.",err_weak_password:"Password must be ≥6 chars.",err_invalid_input:"Check your input.",err_login_failed:"Login failed.",err_register_failed:"Registration failed.",err_network:"Cannot reach server.",motd_title:"System Notice",close:"Close"},
fr:{login_pre:"Connexion à ",login_sub:"Bon retour parmi nous.",register_title_pre:"Rejoignez ",register_sub:"Créez votre compte.",server_url:"Adresse",username:"Nom d'utilisateur",password:"Mot de passe",email:"E-mail",login_btn:"Se connecter",register_btn:"Créer le compte",no_account:"Pas de compte ?",register_link:"Créer un compte",have_account:"Déjà inscrit ?",login_link:"Se connecter",anon_preface:"Rester anonyme ?",anon_link:"Continuer anonymement",dash_title:"Entraînement",loss:"Perte",tokens_sec:"Tokens/sec",time_left:"Reste",round:"Manche",step:"Étape",loss_history:"Historique",waiting_data:"En attente…",model_thoughts:"Prédictions",waiting_preview:"En attente…",idle:"Inactif",training:"Entraînement…",waiting:"Attente…",uploading:"Envoi…",downloading:"Téléchargement…",preparing:"Préparation…",calibrating:"Calibration…",dl_weights:"Téléchargement des poids…",wait_coord:"Attente du coordinateur…",wait_coord_sub:"Cela peut prendre quelques minutes.",start:"Démarrer",stop:"Arrêter",err_invalid_credentials:"Identifiants invalides.",err_username_taken:"Nom pris.",err_email_taken:"E-mail déjà enregistré.",err_weak_password:"Mot de passe ≥6 caractères.",err_invalid_input:"Vérifiez la saisie.",err_login_failed:"Échec de connexion.",err_register_failed:"Échec d'inscription.",err_network:"Serveur injoignable.",motd_title:"Avis système",close:"Fermer"},
es:{login_pre:"Inicia sesión en ",login_sub:"Bienvenido de vuelta.",register_title_pre:"Únete a ",register_sub:"Crea tu cuenta.",server_url:"Servidor",username:"Usuario",password:"Contraseña",email:"Correo",login_btn:"Iniciar sesión",register_btn:"Crear cuenta",no_account:"¿Sin cuenta?",register_link:"Crear una",have_account:"¿Registrado?",login_link:"Iniciar sesión",anon_preface:"¿Anónimo?",anon_link:"Continuar anónimamente",dash_title:"Entrenamiento",loss:"Pérdida",tokens_sec:"Tokens/seg",time_left:"Queda",round:"Ronda",step:"Paso",loss_history:"Historial",waiting_data:"Esperando…",model_thoughts:"Predicciones",waiting_preview:"Esperando…",idle:"Inactivo",training:"Entrenando…",waiting:"Esperando…",uploading:"Subiendo…",downloading:"Descargando…",preparing:"Preparando…",calibrating:"Calibrando…",dl_weights:"Descargando pesos…",wait_coord:"Esperando coordinador…",wait_coord_sub:"Puede tardar unos minutos.",start:"Iniciar",stop:"Detener",err_invalid_credentials:"Credenciales inválidas.",err_username_taken:"Usuario en uso.",err_email_taken:"Correo ya registrado.",err_weak_password:"Contraseña ≥6 caracteres.",err_invalid_input:"Revisa los datos.",err_login_failed:"Falló el inicio de sesión.",err_register_failed:"Falló el registro.",err_network:"Servidor inaccesible.",motd_title:"Aviso",close:"Cerrar"},
de:{login_pre:"Anmelden bei ",login_sub:"Willkommen zurück.",register_title_pre:"Tritt ",register_sub:"Erstelle dein Konto.",server_url:"Serveradresse",username:"Benutzername",password:"Passwort",email:"E-Mail",login_btn:"Anmelden",register_btn:"Konto erstellen",no_account:"Kein Konto?",register_link:"Erstellen",have_account:"Bereits registriert?",login_link:"Anmelden",anon_preface:"Anonym bleiben?",anon_link:"Anonym fortfahren",dash_title:"Training",loss:"Verlust",tokens_sec:"Tokens/Sek",time_left:"Rest",round:"Runde",step:"Schritt",loss_history:"Verlauf",waiting_data:"Warte…",model_thoughts:"Vorhersagen",waiting_preview:"Warte…",idle:"Bereit",training:"Training…",waiting:"Warten…",uploading:"Upload…",downloading:"Herunterladen…",preparing:"Vorbereitung…",calibrating:"Kalibrierung…",dl_weights:"Gewichte werden geladen…",wait_coord:"Warte auf Koordinator…",wait_coord_sub:"Kann ein paar Minuten dauern.",start:"Training starten",stop:"Stopp",err_invalid_credentials:"Ungültig.",err_username_taken:"Benutzername vergeben.",err_email_taken:"E-Mail bereits registriert.",err_weak_password:"Passwort ≥6 Zeichen.",err_invalid_input:"Eingabe prüfen.",err_login_failed:"Anmeldung fehlgeschlagen.",err_register_failed:"Registrierung fehlgeschlagen.",err_network:"Server nicht erreichbar.",motd_title:"Systemmeldung",close:"Schließen"},
tr:{login_pre:"Giriş yap: ",login_sub:"Sürüye hoş geldin.",register_title_pre:"Katıl: ",register_sub:"Hesabını oluştur.",server_url:"Sunucu",username:"Kullanıcı adı",password:"Şifre",email:"E-posta",login_btn:"Giriş yap",register_btn:"Hesap oluştur",no_account:"Hesap yok mu?",register_link:"Oluştur",have_account:"Kayıtlı mısın?",login_link:"Giriş yap",anon_preface:"Anonim kal?",anon_link:"Anonim devam et",dash_title:"Eğitim",loss:"Kayıp",tokens_sec:"Token/sn",time_left:"Kalan",round:"Tur",step:"Adım",loss_history:"Geçmiş",waiting_data:"Bekleniyor…",model_thoughts:"Tahminler",waiting_preview:"Bekleniyor…",idle:"Boşta",training:"Eğitiliyor…",waiting:"Bekleniyor…",uploading:"Yükleniyor…",downloading:"İndiriliyor…",preparing:"Hazırlanıyor…",calibrating:"Kalibre ediliyor…",dl_weights:"Ağırlıklar yükleniyor…",wait_coord:"Koordinatör bekleniyor…",wait_coord_sub:"Birkaç dakika sürebilir.",start:"Eğitimi başlat",stop:"Durdur",err_invalid_credentials:"Geçersiz.",err_username_taken:"Kullanıcı adı alınmış.",err_email_taken:"E-posta kayıtlı.",err_weak_password:"Şifre ≥6 karakter.",err_invalid_input:"Girdiyi kontrol et.",err_login_failed:"Giriş başarısız.",err_register_failed:"Kayıt başarısız.",err_network:"Sunucuya ulaşılamıyor.",motd_title:"Sistem Uyarısı",close:"Kapat"}
};
let lang='en',statusKey='idle';
const lossHistory=[];const MAXP=80;
function t(k){return (I18N[lang]&&I18N[lang][k])||I18N.en[k]||k}
function applyT(){document.querySelectorAll('[data-i18n]').forEach(e=>e.textContent=t(e.getAttribute('data-i18n')));document.getElementById('status-indicator').textContent=t(statusKey)}
document.getElementById('lang-select').addEventListener('change',e=>{lang=e.target.value;applyT()});
function showAuthScreen(name){document.querySelectorAll('.auth-screen').forEach(s=>s.classList.remove('active'));document.getElementById('screen-'+name).classList.add('active');document.getElementById('login-error').classList.remove('show');document.getElementById('reg-error').classList.remove('show')}
document.getElementById('go-to-register').addEventListener('click',e=>{e.preventDefault();showAuthScreen('register')});
document.getElementById('go-to-login').addEventListener('click',e=>{e.preventDefault();showAuthScreen('login')});
function callApi(n,...a){if(window.pywebview&&window.pywebview.api&&window.pywebview.api[n]){window.pywebview.api[n](...a);return true}return false}
window.addEventListener('pywebviewready',()=>callApi('get_backends'));
document.getElementById('vram-select').addEventListener('change',e=>{callApi('set_vram',parseFloat(e.target.value)||0)});
document.getElementById('backend-select').addEventListener('change',e=>{const nb=e.target.value;if(nb===window._currentBackend)return;callApi('set_backend',nb);});
document.getElementById('btn-login').addEventListener('click',()=>{const srv=document.getElementById('login-server').value.trim();const u=document.getElementById('login-user').value.trim();const p=document.getElementById('login-pass').value;if(!u||!p){showLoginError('err_invalid_credentials');return}document.getElementById('login-error').classList.remove('show');document.getElementById('btn-login').disabled=true;callApi('login',srv,u,p)});
document.getElementById('btn-register').addEventListener('click',()=>{const srv=document.getElementById('reg-server').value.trim();const u=document.getElementById('reg-user').value.trim();const email=document.getElementById('reg-email').value.trim();const p=document.getElementById('reg-pass').value;if(!u||!email||!p){showRegError('err_invalid_input');return}document.getElementById('reg-error').classList.remove('show');document.getElementById('btn-register').disabled=true;callApi('register',srv,u,email,p)});
document.getElementById('btn-anon').addEventListener('click',e=>{e.preventDefault();callApi('login_anon')});
document.getElementById('btn-start').addEventListener('click',()=>{if(callApi('start')){document.getElementById('btn-start').disabled=true;document.getElementById('btn-stop').disabled=false}});
document.getElementById('btn-stop').addEventListener('click',()=>{callApi('stop');document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true});
function showLoginError(code){const el=document.getElementById('login-error');el.textContent=t(code);el.classList.add('show');document.getElementById('btn-login').disabled=false}
function showRegError(code){const el=document.getElementById('reg-error');el.textContent=t(code);el.classList.add('show');document.getElementById('btn-register').disabled=false}
function mb(b){return (b/1048576).toFixed(1)}
function overlayShow(d){document.getElementById('overlay').classList.add('show');document.getElementById('overlay-title').textContent=t(d.title_key);const s=document.getElementById('overlay-subtitle');if(d.subtitle_key){s.textContent=t(d.subtitle_key);s.classList.add('show')}else{s.classList.remove('show')}const b=document.getElementById('overlay-bar');b.classList.toggle('indet',!!d.indeterminate);document.getElementById('overlay-fill').style.width=d.indeterminate?'':'0%'}
function overlayHide(){document.getElementById('overlay').classList.remove('show')}
function drawGraph(){const c=document.getElementById('loss-graph');if(!c)return;const empty=document.getElementById('graph-empty');if(lossHistory.length<2){c.classList.add('hidden-graph');if(empty)empty.classList.remove('hidden-empty');return}c.classList.remove('hidden-graph');if(empty)empty.classList.add('hidden-empty');const rect=c.getBoundingClientRect();const W=rect.width,H=rect.height;if(W<10||H<10){requestAnimationFrame(drawGraph);return}const ctx=c.getContext('2d');const dpr=window.devicePixelRatio||1;c.width=Math.round(W*dpr);c.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);const data=lossHistory.slice(-MAXP);const mn=Math.min(...data),mx=Math.max(...data),range=(mx-mn)||1;const pad=6,pH=H-pad*2,pW=W-pad*2,st=pW/(data.length-1);const isDark=document.body.classList.contains('dark');ctx.strokeStyle=isDark?'rgba(255,255,255,.06)':'rgba(17,17,19,.07)';for(let g=1;g<4;g++){const y=pad+pH*g/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(pad+pW,y);ctx.stroke()}const pt=i=>[pad+i*st,pad+pH-(data[i]-mn)/range*pH];ctx.beginPath();data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.lineTo(pad+pW,pad+pH);ctx.lineTo(pad,pad+pH);ctx.closePath();const gr=ctx.createLinearGradient(0,0,0,H);gr.addColorStop(0,isDark?'rgba(34,197,94,.20)':'rgba(22,163,74,.16)');gr.addColorStop(1,'rgba(22,163,74,0)');ctx.fillStyle=gr;ctx.fill();ctx.beginPath();data.forEach((v,i)=>{const[x,y]=pt(i);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=isDark?'#22c55e':'#16a34a';ctx.lineWidth=1.5;ctx.stroke()}
window.handleEvent=function(ev,data){
if(ev==='login_success'){document.getElementById('view-login').classList.remove('active');document.getElementById('view-dashboard').classList.add('active');document.body.classList.add('dashboard-active');document.getElementById('user-info').textContent=data.username;document.getElementById('btn-login').disabled=false;document.getElementById('btn-register').disabled=false;requestAnimationFrame(drawGraph)}
else if(ev==='login_error'){const code=data.code||'err_invalid_credentials';const map={invalid_credentials:'err_invalid_credentials',login_failed:'err_login_failed',username_taken:'err_username_taken',email_taken:'err_email_taken',weak_password:'err_weak_password',invalid_input:'err_invalid_input',register_failed:'err_register_failed',network:'err_network'};const tk=map[code]||'err_login_failed';const loginActive=document.getElementById('screen-login').classList.contains('active');if(loginActive)showLoginError(tk);else showRegError(tk)}
else if(ev==='log'){document.getElementById('status-event').textContent=data}
else if(ev==='backends'){const sel=document.getElementById('backend-select');sel.innerHTML='';window._currentBackend=data.current;const order=[{key:'cuda',label:'CUDA (NVIDIA)'},{key:'rocm',label:'ROCm (AMD)'},{key:'mps',label:'MPS (Apple)'},{key:'xpu',label:'XPU (Intel)'},{key:'directml',label:'DirectML (Windows)'},{key:'cpu',label:'CPU'}];order.forEach(bk=>{const opt=document.createElement('option');opt.value=bk.key;opt.textContent=bk.label;opt.disabled=!data.available[bk.key];if(bk.key===data.current)opt.selected=true;sel.appendChild(opt)});sel.disabled=false}
else if(ev==='vram'){const vs=document.getElementById('vram-select');if(vs){const vv=String(Math.round(data.current||0));vs.value=[...vs.options].some(o=>o.value===vv)?vv:'0'}}
else if(ev==='overlay_show'){overlayShow(data)}
else if(ev==='overlay_progress'){const bar=document.getElementById('overlay-bar');if(data.total>0){bar.classList.remove('indet');const pct=Math.min(100,data.done/data.total*100);document.getElementById('overlay-fill').style.width=pct+'%';document.getElementById('overlay-pct').textContent=pct.toFixed(1)+'%';document.getElementById('overlay-mb').textContent=mb(data.done)+' / '+mb(data.total)+' MB'}}
else if(ev==='overlay_hide'){overlayHide()}
else if(ev==='cal_stats'||ev==='stats'){const loss=Number(data.loss);const tps=Number(data.tps);const step=Number(data.step);const target=ev==='cal_stats'?Number(data.total):Number(data.target);const tl=Number(data.time_left);if(isFinite(loss)){document.getElementById('stat-loss').textContent=loss.toFixed(4);document.getElementById('graph-last').textContent=loss.toFixed(4);if(loss>0){lossHistory.push(loss);if(lossHistory.length>MAXP*2)lossHistory.splice(0,lossHistory.length-MAXP*2)}}if(isFinite(tps))document.getElementById('stat-tps').textContent=Math.round(tps);if(isFinite(tl))document.getElementById('stat-time').textContent=Math.round(tl)+'m';document.getElementById('stat-round').textContent=ev==='cal_stats'?'—':(data.round!=null?data.round:'—');document.getElementById('stat-step-cur').textContent=isFinite(step)?step:0;document.getElementById('stat-step-tot').textContent=isFinite(target)?target:0;const pct=(isFinite(step)&&isFinite(target)&&target>0)?Math.min(100,(step/target)*100):0;document.getElementById('progress-bar').style.width=pct+'%';document.getElementById('progress-text').textContent=pct.toFixed(1)+'%';requestAnimationFrame(drawGraph)}
else if(ev==='model_preview'){const container=document.getElementById('model-preview');const stepEl=document.getElementById('preview-step');if(!container||!stepEl)return;stepEl.textContent='Step '+data.step;let html='';if(data.cards&&data.cards.length>0){data.cards.forEach(card=>{const cls=card.match?'good':'bad';const badge=card.match?'✓':'✗';const prob=(typeof card.prob==='number'&&isFinite(card.prob))?(card.prob*100).toFixed(1)+'%':'';html+='<div class="pred-line '+cls+'"><div class="pred-main"><span class="pred-text">'+escapeHtml(card.context)+'</span><span class="pred-token">'+escapeHtml(card.pred)+'</span></div><div class="pred-side"><span class="pred-badge">'+badge+'</span><span>target: '+escapeHtml(card.target)+'</span><span>'+prob+'</span></div></div>'})}else{html='<div class="preview-empty">'+escapeHtml(t('waiting_preview'))+'</div>'}container.innerHTML=html}
else if(ev==='status'){statusKey=data;document.getElementById('status-indicator').textContent=t(statusKey);if(data==='training'||data==='idle'||data==='calibrating'||data==='preparing')overlayHide();if(data==='idle'){document.getElementById('btn-start').disabled=false;document.getElementById('btn-stop').disabled=true}}
};
function escapeHtml(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
window.addEventListener('resize',()=>requestAnimationFrame(drawGraph));
function updateLogo(){try{const img=document.getElementById('logo-img');if(!img)return;const isDark=document.body.classList.contains('dark');const src=isDark?img.getAttribute('data-dark'):img.getAttribute('data-light');if(src&&src.length>30)img.src=src}catch(e){}}
(function(){try{const saved=localStorage.getItem('crowdgpt-theme');const prefersDark=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;if(saved==='dark'||(!saved&&prefersDark))document.body.classList.add('dark');updateLogo()}catch(e){}const tbtn=document.getElementById('theme-toggle');if(tbtn){tbtn.addEventListener('click',function(){document.body.classList.toggle('dark');try{localStorage.setItem('crowdgpt-theme',document.body.classList.contains('dark')?'dark':'light')}catch(e){}updateLogo();requestAnimationFrame(drawGraph)})}})();
applyT();
(function(){const key='crowdgpt_motd_last_day';const today=new Date().toISOString().slice(0,10);let last=null;try{last=localStorage.getItem(key)}catch(e){}if(last===today)return;fetch('https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/motd.txt').then(r=>r.ok?r.text():'').then(text=>{text=(text||'').trim();if(!text)return;document.getElementById('motd-text').textContent=text;document.getElementById('motd-overlay').classList.add('show');try{localStorage.setItem(key,today)}catch(e){}}).catch(()=>{})})();
document.getElementById('motd-close').addEventListener('click',()=>document.getElementById('motd-overlay').classList.remove('show'));
</script></body></html>"""


# =========================================================================
# Shared backend setup + update check (used by both GUI and headless)
# =========================================================================
def _do_backend_setup(selected_backend, vram_gb, emit):
    """Detect backend + VRAM budget + install hard cap. Shared by GUI and headless."""
    global train_device, train_backend
    train_device, train_backend = detect_training_backend(selected_backend)
    emit('meta', {'backend': train_backend})
    emit('log', f"Backend: {train_backend} ({train_device})")
    auto_detect_vram_budget()
    memory_config["auto_gb"] = memory_config["ram_gb"]
    apply_vram_override(vram_gb)
    emit('log', f"Placement budget: {memory_config['ram_gb']:.2f} GB "
                f"(margin {SAFETY_MARGIN_GB:.2f} GB)")
    install_hard_vram_cap(train_device, SAFETY_MARGIN_GB)


def _check_updates_between_rounds(emit):
    """Shared between GUI and headless. Exits by re-exec if an update lands."""
    try:
        if check_for_update():
            emit('log', "🔄 Updated to a new version — restarting...")
            emit('status', 'idle')
            time.sleep(2)
            _restart_self()
    except Exception as e:
        log.warning(f"Update check failed: {e}")


# =========================================================================
# Api (GUI)
# =========================================================================
class Api:
    def __init__(self):
        self.window = None
        self.stop_event = threading.Event()
        self.thread = None
        self.auth_token = None
        self.username = None
        self.server_url = DEFAULT_SERVER
        self._lp = 0.0
        self._backend_file = CHECKPOINT_DIR / "selected_backend.json"
        saved = None
        try:
            if self._backend_file.exists():
                saved = json.loads(self._backend_file.read_text()).get("backend")
        except Exception:
            saved = None
        available = get_available_backends()
        preferred_backend = get_best_default_backend()
        honor_saved = os.environ.get("CROWDGPT_HONOR_SAVED_BACKEND") == "1"
        if saved and available.get(saved, False):
            # A previous CPU choice commonly persists through reinstallations.
            # Prefer an installed accelerator by default; set
            # CROWDGPT_HONOR_SAVED_BACKEND=1 to keep a saved CPU/DirectML choice.
            should_upgrade = (
                not honor_saved and (
                    (saved == "cpu" and preferred_backend != "cpu") or
                    (saved == "directml" and preferred_backend in ("cuda", "rocm", "mps", "xpu"))
                )
            )
            if should_upgrade:
                self.selected_backend = preferred_backend
                log.info(f"[AMD-AUTO] Replacing saved backend {saved!r} with available {preferred_backend!r}.")
            else:
                self.selected_backend = saved
        else:
            self.selected_backend = preferred_backend

        if sys.platform == "win32" and not available.get("rocm", False) and available.get("directml", False):
            log.warning(
                "[ROCM] ROCm is disabled in the selector because this environment's PyTorch "
                "is not a HIP/ROCm build. Reinstall with install.bat --backend rocm; "
                "if the installer falls back to DirectML, review its ROCm install/runtime-test warning."
            )
        log.info(f"[CLIENT] Selected backend: {self.selected_backend}")
        self._vram_file = CHECKPOINT_DIR / "selected_vram.json"
        self.vram_gb = 0.0
        try:
            if self._vram_file.exists():
                self.vram_gb = max(0.0, float(json.loads(self._vram_file.read_text()).get("vram_gb", 0) or 0))
        except Exception:
            self.vram_gb = 0.0

    def emit(self, ev, data):
        if not self.window:
            return
        if ev in ('log', 'status'):
            try:
                print(f"[EMIT {ev}] {data}", flush=True)
            except Exception:
                pass
        if ev == 'preview_loss':
            n = time.time()
            if n - self._lp < 0.4:
                return
            self._lp = n
        try:
            clean_data = sanitize_for_json(data)
            self.window.evaluate_js(f"window.handleEvent({json.dumps(ev)}, {json.dumps(clean_data)});")
        except Exception as e:
            log.warning(f"Emit failed for {ev}: {e}")

    def get_backends(self):
        available = get_available_backends()
        self.emit('backends', {'available': available, 'current': self.selected_backend})
        self.emit('vram', {'current': self.vram_gb})

    def set_vram(self, gb):
        try:
            gb = max(0.0, float(gb))
        except (TypeError, ValueError):
            gb = 0.0
        self.vram_gb = gb
        try:
            self._vram_file.write_text(json.dumps({"vram_gb": gb}))
        except Exception as e:
            log.warning(f"Failed to save VRAM limit: {e}")
        apply_vram_override(gb)
        busy = self.thread is not None and self.thread.is_alive()
        label = "Auto" if gb <= 0 else f"{gb:g} GB"
        self.emit('log', f"VRAM limit: {label}" + (" (applies from the next round)" if busy else ""))

    def set_backend(self, backend_name):
        log.info(f"[CLIENT] set_backend called with: {backend_name} (was {self.selected_backend})")
        self.selected_backend = backend_name
        try:
            self._backend_file.write_text(json.dumps({"backend": backend_name}))
        except Exception as e:
            log.warning(f"Failed to save backend choice: {e}")
        self.get_backends()
        if self.thread and self.thread.is_alive():
            self.stop_event.set()

    def login(self, s, u, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p), daemon=True).start()

    def _do_login(self, u, p):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p), daemon=True).start()

    def _do_register(self, u, email, p):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'register_failed'})

    def login_anon(self):
        self.auth_token = None
        self.username = "anonymous"
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._rs, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _check_updates_between_rounds(self):
        _check_updates_between_rounds(self.emit)

    def _rs(self):
        try:
            _do_backend_setup(self.selected_backend, self.vram_gb, self.emit)
        except Exception as e:
            self.emit('log', f"Error: {e}")
            self.emit('status', 'idle')
            return
        round_count = 0
        while not self.stop_event.is_set():
            round_count += 1
            apply_vram_override(self.vram_gb)
            self.emit('log', f"{'='*40}")
            self.emit('log', f"Starting Round Cycle #{round_count}")
            self.emit('log', f"{'='*40}")
            try:
                run_single_round_wrapper(self.server_url, self.auth_token, self.stop_event, self.emit)
            except KeyboardInterrupt:
                self.stop_event.set()
                break
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"\n[ROUND-FAIL-TRACEBACK]\n{tb}\n", flush=True)
                for ln in tb.splitlines()[-14:]:
                    self.emit('log', ln)
                self.emit('log', f"Round failed: {e}")
            if self.stop_event.is_set():
                break
            self._check_updates_between_rounds()
            self.emit('status', 'waiting')
            self.emit('log', "✅ Round complete. Auto-relaunching next round in 10s...")
            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
        self.emit('status', 'idle')
        self.emit('log', "👋 Training stopped by user.")


# =========================================================================
# GUI windowing
# =========================================================================
def startup():
    if sys.platform == "win32" and WINDOWS_ICO_PATH:
        _wait_for_window(WINDOW_TITLE, timeout=5.0)
        time.sleep(0.15)
        set_windows_icon(WINDOW_TITLE, WINDOWS_ICO_PATH)

    try:
        scr = webview.screens[0]
        aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80)))
        h = max(360, min(800, int(ah * 0.80)))
        w = min(w, aw - 16)
        h = min(h, ah - 48)
        w = max(320, w)
        h = max(240, h)
        window.resize(w, h)
        window.move(max(0, (aw - w) // 2), max(0, (ah - h) // 2))
    except Exception:
        pass


def _create_window_safe(title, html, kwargs):
    try:
        return webview.create_window(title, html=html, **kwargs)
    except TypeError:
        kwargs.pop("min_size", None)
        return webview.create_window(title, html=html, **kwargs)


def _start_icon_kwargs():
    if sys.platform == "win32" or not ICON_PATH or not os.path.exists(ICON_PATH):
        return {}
    log.info(f"[icon] passing icon to webview.start: {ICON_PATH}")
    return {"icon": ICON_PATH}


# =========================================================================
# CLI / headless
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        prog="client.py",
        description="CrowdGPT client. Runs the GUI by default; use --headless for SSH/servers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--headless", action="store_true",
                   help="no window; train in the terminal (for SSH / systemd / nohup)")
    p.add_argument("--server", default=DEFAULT_SERVER,
                   help="coordinator URL")
    p.add_argument("--user", default=None, help="username")
    p.add_argument("--password", default=None,
                   help="password (or env CROWDGPT_PASSWORD)")
    p.add_argument("--token", default=None,
                   help="auth token (or env CROWDGPT_TOKEN)")
    p.add_argument("--anon", action="store_true",
                   help="train without an account")
    p.add_argument("--backend", default=None,
                   choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu", "directml"],
                   help="force a backend (default: best available)")
    p.add_argument("--vram-gb", type=float, default=0.0,
                   help="cap placement budget in GB (0 = auto, also reads $CROWDGPT_VRAM_GB)")
    p.add_argument("--rounds", type=int, default=0,
                   help="stop after N rounds (0 = run forever)")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="skip loading the tokenizer (~500 MB RAM); previews show #tok ids")
    p.add_argument("--no-update", action="store_true",
                   help="don't check for client.py updates")
    p.add_argument("--dry-run", action="store_true",
                   help="check auth + backend + task metadata, then exit without training")
    p.add_argument("--debug", action="store_true",
                   help="verbose logging")
    p.add_argument("--version", action="version",
                   version=f"crowdgpt-client {CLIENT_VERSION}")
    return p.parse_args()


def _headless_emit(verbose=False):
    """Console emit for --headless. Throttles noisy events, prefixes the rest."""
    _last = {"prog": 0.0, "last_stats": 0.0}
    def emit(ev, data):
        if ev == 'log':
            log.info(str(data))
        elif ev == 'status':
            log.info(f"[status] {data}")
        elif ev == 'meta':
            pass
        elif ev == 'stats':
            now = time.time()
            if now - _last["last_stats"] < 10.0:
                return
            _last["last_stats"] = now
            log.info(
                f"[train] step {data.get('step')}/{data.get('target')} "
                f"loss={data.get('loss', 0):.4f} "
                f"tps={data.get('tps', 0):.0f} "
                f"left={data.get('time_left', 0):.0f}m"
            )
        elif ev == 'cal_stats':
            log.info(
                f"[cal]   step {data.get('step')}/{data.get('total')} "
                f"loss={data.get('loss', 0):.4f}"
            )
        elif ev == 'model_preview':
            for c in data.get('cards', []):
                log.info(
                    f"[preview] {str(c.get('context', ''))[-40:]!r} → "
                    f"pred={c.get('pred')!r} target={c.get('target')!r} "
                    f"{'✓' if c.get('match') else '✗'}"
                )
        elif ev == 'overlay_progress':
            now = time.time()
            if now - _last["prog"] < 5.0:
                return
            _last["prog"] = now
            done  = data.get('done', 0) / (1024 * 1024)
            total = data.get('total', 0) / (1024 * 1024)
            pct = 100.0 * done / total if total > 0 else 0.0
            log.info(f"[progress] {pct:.1f}% ({done:.0f}/{total:.0f} MB)")
        elif ev in ('overlay_show', 'overlay_hide'):
            pass
        else:
            if verbose:
                log.debug(f"[{ev}] {data}")
    return emit


def run_headless(args):
    """No window. Runs the same training loop as the GUI, prints to stdout."""
    emit = _headless_emit(verbose=args.debug)

    # ---- credentials ----
    password = args.password or os.environ.get("CROWDGPT_PASSWORD")
    token    = args.token    or os.environ.get("CROWDGPT_TOKEN")
    if args.anon:
        auth_token = None
        log.info("[auth] anonymous")
    elif token:
        auth_token = token
        log.info("[auth] using provided token")
    elif args.user and password:
        log.info(f"[auth] logging in as {args.user}")
        auth_token, err = do_login(args.server, args.user, password)
        if not auth_token:
            log.error(f"[auth] login failed: {err}")
            return 2
    else:
        log.error("[auth] need --token, --user + --password, or --anon "
                  "(password may also come from $CROWDGPT_PASSWORD)")
        return 2

    log.info(f"[cfg] server={args.server} rounds={args.rounds or '∞'} "
             f"vram={args.vram_gb or 'auto'}GB backend={args.backend or 'auto'}")

    # ---- backend / VRAM ----
    selected = args.backend if args.backend and args.backend != "auto" else None
    if selected is None:
        bf = CHECKPOINT_DIR / "selected_backend.json"
        try:
            if bf.exists():
                saved = json.loads(bf.read_text()).get("backend")
                if saved and get_available_backends().get(saved, False):
                    selected = saved
        except Exception:
            pass
    if selected is None:
        selected = get_best_default_backend()

    try:
        _do_backend_setup(selected, args.vram_gb, emit)
    except Exception as e:
        log.error(f"[backend] setup failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return 3

    # ---- tokenizer (for readable previews) ----
    if not args.no_tokenizer:
        log.info("[tokenizer] loading (skippable with --no-tokenizer)...")
        _load_tokenizer_worker()
    else:
        log.info("[tokenizer] skipped (--no-tokenizer)")

    # ---- signal handling ----
    stop_event = threading.Event()
    def _on_signal(signum, frame):
        log.info("[signal] stop requested; finishing current micro-step...")
        stop_event.set()
    for sig_name in ("SIGINT", "SIGTERM"):
        s = getattr(signal, sig_name, None)
        if s is not None:
            try:
                signal.signal(s, _on_signal)
            except (ValueError, OSError):
                pass

    # ---- dry-run ----
    if args.dry_run:
        log.info("[dry-run] fetching task metadata...")
        h = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        md = fetch_task_metadata(args.server, h, stop_event, emit)
        if md is None:
            log.error("[dry-run] no task available")
            return 5
        log.info(f"[dry-run] task OK: taskId={md.get('taskId')} "
                 f"round={md.get('current_round') or md.get('round')} "
                 f"step={md.get('globalStep')}")
        return 0

    # ---- rounds ----
    max_rounds = args.rounds if args.rounds and args.rounds > 0 else 0
    round_count = 0
    while not stop_event.is_set():
        round_count += 1
        apply_vram_override(args.vram_gb)
        log.info("=" * 60)
        log.info(f"Round {round_count}" + (f"/{max_rounds}" if max_rounds else ""))
        log.info("=" * 60)
        try:
            run_single_round_wrapper(args.server, auth_token, stop_event, emit)
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            import traceback
            log.error(f"Round failed: {e}")
            log.error(traceback.format_exc()[-2000:])

        if stop_event.is_set():
            break
        if max_rounds and round_count >= max_rounds:
            log.info(f"Completed {round_count} round(s); exiting.")
            break

        if not args.no_update:
            _check_updates_between_rounds(emit)
        if stop_event.is_set():
            break

        log.info("Round complete; next round in 10s (Ctrl+C to stop)...")
        stop_event.wait(timeout=10.0)

    log.info("Bye.")
    return 0


def run_gui(args):
    if not _WEBVIEW_OK:
        log.error(f"GUI unavailable: {_WEBVIEW_ERR}")
        log.error("Run with --headless to train without a window.")
        return 4

    api = Api()
    start_tokenizer_loader()

    kwargs = {"js_api": api, "width": 1000, "height": 720, "min_size": (320, 240)}
    html_out = (HTML.replace("__LOGO__", LOGO_URI_LIGHT)
                    .replace("__LOGO_DARK__", LOGO_URI_DARK)
                    .replace("__VERSION__", CLIENT_VERSION))

    window = _create_window_safe("CrowdGPT", html_out, kwargs)
    api.window = window
    try:
        webview.start(startup, debug=False, **_start_icon_kwargs())
    except TypeError:
        webview.start(startup, debug=False)
    return 0


# =========================================================================
# Entry point
# =========================================================================
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"CrowdGPT client v{CLIENT_VERSION}"
          + (" [headless]" if args.headless else ""), flush=True)

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Vxtzq.CrowdGPT.1")
        except Exception:
            pass

    if not args.no_update:
        try:
            if check_for_update():
                print("Update applied. Restarting...", flush=True)
                _restart_self()
        except Exception as e:
            log.warning(f"Auto-update check error: {e}")

    if args.headless:
        sys.exit(run_headless(args))
    else:
        sys.exit(run_gui(args))
