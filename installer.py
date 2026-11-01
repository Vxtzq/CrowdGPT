#!/usr/bin/env python3
"""
CrowdGPT GUI installer (pywebview).

Platform backends for the INSTALLER window:
  - Windows:  EdgeChromium (native WebView2 via pythonnet; no Qt)
  - macOS:    cocoa (bundled with pywebview; no Qt)
  - Linux:    Qt (PyQt6-WebEngine) or GTK fallback

Platform backends for the INSTALLED CLIENT:
  - Windows:  EdgeChromium via pywebview (pythonnet pulled transitively)
  - macOS:    cocoa (bundled with pywebview)
  - Linux:    GTK if PyGObject is present, else PyQt6-WebEngine

All install logic mirrors the v12 bash installer / v13 batch installer.
Every step streams to the GUI in real time (batched every 150 ms).
"""

import argparse
import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    import webview
except ImportError:
    print("ERROR: pywebview not installed. Run via install.sh / install.bat (bootstrap).")
    sys.exit(1)


# ============================================================
# HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#fafaf9;--bg-soft:#fff;--bg-softer:#f0f0ee;--text:#111113;--text-dim:#3a3a40;--text-muted:#6b6b73;--border:#e4e4e1;--border-strong:#d0d0cb;--accent:#16a34a;--accent-dim:#15803d;--warn:#b45309;--err:#dc2626;--err-soft:rgba(220,38,38,.06);--err-border:rgba(220,38,38,.28);--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;--r:6px}
body.dark{color-scheme:dark;--bg:#1a1a1e;--bg-soft:#242428;--bg-softer:#2e2e33;--text:#e8e8ec;--text-dim:#b8b8be;--text-muted:#888890;--border:#3a3a40;--border-strong:#505058;--accent:#22c55e;--accent-dim:#4ade80;--warn:#fbbf24;--err:#f87171;--err-soft:rgba(248,113,113,.08);--err-border:rgba(248,113,113,.35)}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;width:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);display:flex;flex-direction:column;overflow:hidden;font-size:14px;line-height:1.5}
header{flex:0 0 auto;height:54px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 20px;gap:10px}
.logo{display:flex;align-items:center;gap:9px;min-width:0}
.logo-img{height:22px;width:auto;display:block}
.logo-fallback{width:24px;height:24px;border:1px solid var(--border-strong);display:none;place-items:center;font-size:13px;font-weight:600}
.logo-text{font-size:15px;font-weight:600;white-space:nowrap}
.logo-ver{font-family:var(--mono);font-size:10px;color:var(--text-muted);border:1px solid var(--border);padding:1px 5px;border-radius:3px}
.theme-toggle{width:30px;height:30px;display:flex;align-items:center;justify-content:center;background:var(--bg-soft);color:var(--text);border:1px solid var(--border);border-radius:4px;cursor:pointer;font-size:14px;padding:0}
.theme-toggle:hover{border-color:var(--border-strong)}
.theme-toggle .icon-sun{display:none}
body.dark .theme-toggle .icon-sun{display:inline}
body.dark .theme-toggle .icon-moon{display:none}
main{flex:1 1 auto;min-height:0;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;align-items:center;gap:12px}
.card{width:100%;max-width:640px;background:var(--bg-soft);border:1px solid var(--border);border-radius:8px;padding:22px 24px;display:flex;flex-direction:column;gap:14px}
h1{font-size:20px;font-weight:600}
h1 .em{color:var(--accent);font-weight:700}
.subtitle{font-size:13px;color:var(--text-dim);margin-top:-8px;min-height:18px}
.bar-track{height:6px;background:var(--bg-softer);border:1px solid var(--border);border-radius:3px;overflow:hidden;position:relative}
.bar-fill{height:100%;width:0%;background:var(--accent);transition:width .3s ease}
.bar-track.indet .bar-fill{width:35%;transition:none;animation:indet 1.15s ease-in-out infinite}
@keyframes indet{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
.steps{display:flex;flex-direction:column;gap:2px;font-family:var(--mono);font-size:12px}
.step{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:4px;color:var(--text-muted);border:1px solid transparent}
.step .icon{width:16px;text-align:center;flex-shrink:0}
.step .label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.step .detail{font-size:10.5px;color:var(--text-muted);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono)}
.step.pending .icon::before{content:"○"}
.step.running{color:var(--text);background:var(--bg-softer);border-color:var(--border)}
.step.running .icon::before{content:"◐";color:var(--accent);animation:spin 1s linear infinite;display:inline-block}
.step.done{color:var(--text-dim)}
.step.done .icon::before{content:"✓";color:var(--accent);font-weight:700}
.step.failed{color:var(--err);background:var(--err-soft);border-color:var(--err-border)}
.step.failed .icon::before{content:"✕";color:var(--err);font-weight:700}
@keyframes spin{to{transform:rotate(360deg)}}
.log-toggle{background:transparent;border:1px solid var(--border);color:var(--text-muted);padding:5px 10px;border-radius:4px;font-family:var(--mono);font-size:11px;cursor:pointer;align-self:flex-start}
.log-toggle:hover{border-color:var(--border-strong);color:var(--text)}
.log-panel{display:none;max-height:280px;overflow-y:auto;background:#0f0f12;color:#e8e8ec;border:1px solid var(--border);border-radius:4px;padding:10px 12px;font-family:var(--mono);font-size:11px;line-height:1.55}
body.dark .log-panel{background:#0a0a0d}
.log-panel.show{display:block}
.log-line{white-space:pre-wrap;word-break:break-all}
.log-line.err{color:#f87171}
.log-line.warn{color:#fbbf24}
.log-line.ok{color:#4ade80}
.actions{display:flex;gap:8px;justify-content:flex-end;margin-top:4px;flex-wrap:wrap}
button{font-family:inherit;font-size:12.5px;font-weight:500;cursor:pointer;border-radius:4px;padding:9px 14px}
button:disabled{opacity:.4;cursor:not-allowed}
.btn-primary{background:var(--text);color:var(--bg);border:1px solid var(--text)}
.btn-primary:hover:not(:disabled){opacity:.85}
body.dark .btn-primary{background:var(--bg-softer)!important;color:var(--text)!important;border-color:var(--border-strong)!important}
.btn-danger{background:transparent;color:var(--err);border:1px solid var(--err)}
.btn-danger:hover:not(:disabled){background:var(--err);color:#fff}
.btn-secondary{background:transparent;color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover:not(:disabled){border-color:var(--border-strong)}
.result-card{display:none;flex-direction:column;gap:10px;padding:14px;border-radius:6px;border:1px solid}
.result-card.show{display:flex}
.result-card.success{background:rgba(22,163,74,.05);border-color:rgba(22,163,74,.28)}
.result-card.failure{background:var(--err-soft);border-color:var(--err-border)}
.result-title{font-size:15px;font-weight:600}
.result-card.success .result-title{color:var(--accent-dim)}
.result-card.failure .result-title{color:var(--err)}
.result-body{font-family:var(--mono);font-size:11.5px;color:var(--text-dim);white-space:pre-wrap;word-break:break-word;max-height:180px;overflow-y:auto}
</style></head>
<body>
<header>
  <div class="logo">
    __LOGO_WRAP__
    <span class="logo-text">CrowdGPT</span>
    <span class="logo-ver" id="ver">installer</span>
  </div>
  <button class="theme-toggle" id="theme-toggle">
    <span class="icon-moon">🌙</span><span class="icon-sun">☀️</span>
  </button>
</header>
<main>
  <div class="card">
    <div>
      <h1>Installing <span class="em">CrowdGPT</span></h1>
      <p class="subtitle" id="subtitle">Preparing installer…</p>
    </div>

    <div class="bar-track indet" id="bar-track"><div class="bar-fill" id="bar-fill"></div></div>

    <div class="steps" id="steps"></div>

    <button class="log-toggle" id="log-toggle">Show log ▾</button>
    <div class="log-panel" id="log-panel"></div>

    <div class="result-card" id="result">
      <div class="result-title" id="result-title"></div>
      <div class="result-body" id="result-body"></div>
      <div class="actions" id="result-actions"></div>
    </div>

    <div class="actions" id="main-actions">
      <button class="btn-danger" id="btn-cancel" disabled>Cancel</button>
    </div>
  </div>
</main>
<script>
const STEPS = [
  ["check",     "Checking system"],
  ["git",       "Syncing repository"],
  ["detect",    "Detecting hardware"],
  ["venv",      "Creating virtual environment"],
  ["torch",     "Installing PyTorch"],
  ["reqs",      "Installing dependencies"],
  ["extras",    "Installing extras (bitsandbytes, transformers)"],
  ["webview",   "Installing GUI backend"],
  ["integrate", "Installing desktop integration"],
  ["finalize",  "Finalizing"],
];
const state = { steps: {}, done: false, failed: false, canRelaunch: false };

function renderSteps() {
  const box = document.getElementById('steps');
  box.innerHTML = '';
  for (const [id, label] of STEPS) {
    const s = state.steps[id] || { status: 'pending', detail: '' };
    const div = document.createElement('div');
    div.className = 'step ' + s.status;
    div.innerHTML = '<span class="icon"></span>'
                  + '<span class="label">' + esc(label) + '</span>'
                  + (s.detail ? '<span class="detail">' + esc(s.detail) + '</span>' : '');
    box.appendChild(div);
  }
}

function setSubtitle(t) { document.getElementById('subtitle').textContent = t; }
function setIndeterminate(on) { document.getElementById('bar-track').classList.toggle('indet', !!on); }
function setProgress(pct) {
  setIndeterminate(false);
  document.getElementById('bar-fill').style.width = Math.min(100, Math.max(0, pct)) + '%';
}

window.onStep = function(id, status, detail) {
  state.steps[id] = { status, detail: detail || '' };
  renderSteps();
};

window.onLog = function(line, kind) {
  window.onLogBatch([[line, kind || ""]]);
};

window.onLogBatch = function(batch) {
  const panel = document.getElementById('log-panel');
  for (let i = 0; i < batch.length; i++) {
    const [line, kind] = batch[i];
    const div = document.createElement('div');
    div.className = 'log-line' + (kind ? ' ' + kind : '');
    div.textContent = line;
    panel.appendChild(div);
  }
  while (panel.children.length > 2000) panel.removeChild(panel.firstChild);
  panel.scrollTop = panel.scrollHeight;
};

window.onSubtitle = function(t) { setSubtitle(t); };
window.onProgress = function(pct) { setProgress(pct); };

window.onFinish = function(ok, title, body, canRelaunch) {
  state.done = true;
  state.failed = !ok;
  state.canRelaunch = !!canRelaunch;
  document.getElementById('bar-track').classList.remove('indet');
  document.getElementById('bar-fill').style.width = ok ? '100%' : '0%';
  document.getElementById('btn-cancel').disabled = true;
  const rc = document.getElementById('result');
  rc.classList.remove('success', 'failure');
  rc.classList.add('show', ok ? 'success' : 'failure');
  document.getElementById('result-title').textContent = title;
  document.getElementById('result-body').textContent = body || '';
  const acts = document.getElementById('result-actions');
  acts.innerHTML = '';
  if (ok && canRelaunch) {
    const b = document.createElement('button');
    b.className = 'btn-primary';
    b.textContent = 'Launch CrowdGPT';
    b.onclick = () => window.pywebview.api.launch_client();
    acts.appendChild(b);
  }
  if (!ok) {
    const retry = document.createElement('button');
    retry.className = 'btn-secondary';
    retry.textContent = 'Retry';
    retry.onclick = () => {
      rc.classList.remove('show');
      document.getElementById('btn-cancel').disabled = false;
      for (const id of Object.keys(state.steps)) state.steps[id] = { status: 'pending', detail: '' };
      state.done = false; state.failed = false;
      renderSteps();
      window.pywebview.api.start_install();
    };
    acts.appendChild(retry);
  }
  const closeBtn = document.createElement('button');
  closeBtn.className = ok ? 'btn-secondary' : 'btn-primary';
  closeBtn.textContent = 'Close';
  closeBtn.onclick = () => window.pywebview.api.close_window();
  acts.appendChild(closeBtn);
};

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function updateLogo() {
  const img = document.getElementById('logo-img');
  if (!img) return;
  const isDark = document.body.classList.contains('dark');
  const next = isDark ? img.dataset.dark : img.dataset.light;
  if (next && img.src !== next) img.src = next;
}

(function(){
  try {
    const saved = localStorage.getItem('crowdgpt-installer-theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) document.body.classList.add('dark');
  } catch(e) {}
  updateLogo();
  document.getElementById('theme-toggle').addEventListener('click', () => {
    document.body.classList.toggle('dark');
    try { localStorage.setItem('crowdgpt-installer-theme', document.body.classList.contains('dark') ? 'dark' : 'light'); } catch(e) {}
    updateLogo();
  });
})();

document.getElementById('log-toggle').addEventListener('click', () => {
  const panel = document.getElementById('log-panel');
  const shown = panel.classList.toggle('show');
  document.getElementById('log-toggle').textContent = (shown ? 'Hide log ▴' : 'Show log ▾');
});

document.getElementById('btn-cancel').addEventListener('click', () => {
  if (state.done) return;
  if (confirm('Cancel the installation?')) window.pywebview.api.cancel_install();
});

window.addEventListener('pywebviewready', () => {
  renderSteps();
  window.pywebview.api.start_install();
});

window.addEventListener('keydown', e => {
  if (e.key === 'Escape') window.pywebview.api.close_window();
});
</script>
</body></html>
"""


# ============================================================
# Logo + window icon
# ============================================================

def _logo_data_uri(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/svg+xml"
    try:
        data = p.read_bytes()
    except OSError as e:
        print(f"[installer] Could not read logo {p}: {e}", flush=True)
        return ""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _render_installer_html(logo_light_uri: str, logo_dark_uri: str) -> str:
    if not logo_light_uri:
        return HTML.replace(
            "__LOGO_WRAP__",
            '<div class="logo-fallback" style="display:grid">C</div>',
        )
    dark = logo_dark_uri or logo_light_uri
    wrap = (
        f'<img id="logo-img" src="{logo_light_uri}" '
        f'data-light="{logo_light_uri}" '
        f'data-dark="{dark}" '
        'class="logo-img" alt="" '
        "onerror=\"this.style.display='none';this.nextElementSibling.style.display='grid';\">"
        '<div class="logo-fallback">C</div>'
    )
    return HTML.replace("__LOGO_WRAP__", wrap)


def _prepare_qt_icon_early(source_path):
    """Pre-create QApplication with a crisp multi-size QIcon BEFORE webview.start().
    Only used on Linux where pywebview picks Qt. Harmless if the backend ends up
    being GTK — QApplication just sits unused."""
    if not source_path:
        return False
    p = Path(source_path)
    if not p.exists():
        return False
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QIcon, QPixmap, QPainter
        from PyQt6.QtCore import Qt
    except ImportError:
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtGui import QIcon, QPixmap, QPainter
            from PySide6.QtCore import Qt
        except ImportError:
            return False
    try:
        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setApplicationName("CrowdGPT")
        app.setApplicationDisplayName("CrowdGPT")
        try:
            app.setDesktopFileName("crowdgpt")
        except AttributeError:
            pass
        icon = QIcon()
        if p.suffix.lower() == ".svg":
            try:
                from PyQt6.QtSvg import QSvgRenderer
            except ImportError:
                try:
                    from PySide6.QtSvg import QSvgRenderer
                except ImportError:
                    QSvgRenderer = None
            if QSvgRenderer is not None:
                renderer = QSvgRenderer(str(p))
                if renderer.isValid():
                    for size in (16, 24, 32, 48, 64, 128, 256):
                        pm = QPixmap(size, size)
                        pm.fill(Qt.GlobalColor.transparent)
                        painter = QPainter(pm)
                        renderer.render(painter)
                        painter.end()
                        icon.addPixmap(pm)
        if icon.isNull() or not icon.availableSizes():
            icon = QIcon(str(p))
        if icon.isNull():
            return False
        app.setWindowIcon(icon)
        print(f"[installer] Pre-set QApplication window icon from {p.name}", flush=True)
        return True
    except Exception as e:
        print(f"[installer] Early Qt setup failed: {e}", flush=True)
        return False


# ============================================================
# General helpers
# ============================================================

def run_capture(cmd, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(str(c) for c in cmd)}: timeout"


def which(binary):
    return shutil.which(binary)


def _webview_backend_ok() -> bool:
    """Return True if any pywebview GUI backend can be imported on this OS."""
    _os = platform.system()
    if _os == "Windows":
        try:
            from webview.platforms import edgechromium  # noqa: F401
            return True
        except Exception:
            pass
    if _os == "Darwin":
        try:
            import webview.platforms.cocoa  # noqa: F401
            return True
        except Exception:
            pass
    try:
        from qtpy import QtCore  # noqa: F401
        from PyQt6 import QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import gi  # noqa: F401
        return True
    except Exception:
        return False


# ============================================================
# InstallerAPI
# ============================================================

class InstallerAPI:
    def __init__(self, repo_dir, backend_override, launch, integrate, headless):
        self.repo_dir = Path(repo_dir).resolve()
        self.backend_override = (backend_override or "").strip().lower()
        self.launch = launch
        self.integrate = integrate
        self.headless = headless
        self._window = None
        self.cancelled = False
        self.finished = False
        self._proc = None
        self._proc_lock = threading.Lock()

        self._log_buffer: list[tuple[str, str]] = []
        self._log_lock = threading.Lock()
        self._log_flusher_started = False

        self.os_name = platform.system()
        self.arch = platform.machine().lower()
        self.backend = None
        self.torch_index = None
        self.wheel_note = ""
        self.torch_ver = ""
        self.installed_venv_py = None
        self.installed_app_dir = None

    # ---------- GUI callbacks ----------
    def _js(self, code):
        if self.headless or not self._window:
            return
        try:
            self._window.evaluate_js(code)
        except Exception:
            pass

    def log(self, line, kind=""):
        line = str(line).rstrip()
        if not line:
            return
        if self.headless:
            prefix = {"err": "[FAIL] ", "warn": "[WARN] ", "ok": "[ OK ] "}.get(kind, "")
            print(prefix + line, flush=True)
            return
        with self._log_lock:
            self._log_buffer.append((line, kind))
            if len(self._log_buffer) >= 25:
                self._flush_logs_locked()

    def _flush_logs_locked(self):
        if not self._log_buffer:
            return
        batch = list(self._log_buffer)
        self._log_buffer.clear()
        try:
            self._window.evaluate_js("window.onLogBatch(" + json.dumps(batch) + ");")
        except Exception:
            pass

    def _flush_logs(self):
        if self.headless or not self._window:
            return
        with self._log_lock:
            self._flush_logs_locked()

    def _log_flusher_loop(self):
        while not self.finished and not self.cancelled:
            time.sleep(0.15)
            self._flush_logs()

    def step(self, step_id, status, detail=""):
        self._js(f"window.onStep({json.dumps(step_id)}, {json.dumps(status)}, {json.dumps(detail)})")

    def subtitle(self, text):
        if self.headless:
            print(f"\n== {text} ==", flush=True)
            return
        self._js(f"window.onSubtitle({json.dumps(text)})")

    def progress(self, pct):
        self._js(f"window.onProgress({pct})")

    def finish(self, ok, title, body="", can_relaunch=False):
        self._flush_logs()
        self.finished = True
        if self.headless:
            print(f"\n{'[DONE]' if ok else '[FAILED]'} {title}", flush=True)
            if body:
                print(body, flush=True)
            return
        self._js(f"window.onFinish({str(ok).lower()}, "
                 f"{json.dumps(title)}, {json.dumps(body)}, {str(can_relaunch).lower()})")

    # ---------- Process runner ----------
    def run(self, cmd, cwd=None, env=None, check=True, kind=""):
        self.log(f"$ {' '.join(str(c) for c in cmd)}", kind)
        if self.cancelled:
            raise RuntimeError("Cancelled by user")
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
            )
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {cmd[0]}")
        with self._proc_lock:
            self._proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if self.cancelled:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise RuntimeError("Cancelled by user")
                line = line.rstrip()
                if line:
                    self.log(line)
            rc = proc.wait()
        finally:
            with self._proc_lock:
                self._proc = None
        if check and rc != 0:
            raise RuntimeError(f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")
        return rc

    # ---------- Public API ----------
    def start_install(self):
        self.cancelled = False
        self.finished = False
        if not self._log_flusher_started:
            self._log_flusher_started = True
            threading.Thread(target=self._log_flusher_loop, daemon=True).start()
        threading.Thread(target=self._run_install, daemon=True).start()

    def cancel_install(self):
        self.cancelled = True
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def close_window(self):
        if self._window:
            self._window.destroy()
        else:
            sys.exit(0)

    def launch_client(self):
        if not (self.installed_venv_py and self.installed_app_dir):
            self.close_window()
            return
        if self.os_name == "Windows":
            launcher = self.installed_app_dir / "CrowdGPT.pyw"
            if launcher.exists():
                pyw = self.installed_venv_py.parent / "pythonw.exe"
                exe = pyw if pyw.exists() else self.installed_venv_py
                subprocess.Popen([str(exe), str(launcher)], cwd=str(self.installed_app_dir))
            else:
                client = self.installed_app_dir / "client.py"
                if client.exists():
                    subprocess.Popen([str(self.installed_venv_py), str(client)],
                                     cwd=str(self.installed_app_dir))
        else:
            client = self.installed_app_dir / "client.py"
            if client.exists():
                subprocess.Popen([str(self.installed_venv_py), str(client)])
        self.close_window()

    # ---------- Install flow ----------
    def _run_install(self):
        try:
            self._do_install()
        except RuntimeError as e:
            msg = str(e)
            self.log(msg, "err")
            self._mark_current_failed()
            self.finish(False, "Installation failed" if msg != "Cancelled by user" else "Cancelled", msg)
        except Exception as e:
            self.log(f"Unexpected error: {e}", "err")
            self.log(traceback.format_exc(), "err")
            self._mark_current_failed()
            self.finish(False, "Installation failed", str(e))

    def _mark_current_failed(self):
        if self.headless:
            return
        try:
            self._js("""
              (function(){
                const box = document.getElementById('steps');
                const last = [...box.querySelectorAll('.step.running')].pop();
                if (last) { last.className = 'step failed'; }
              })();
            """)
        except Exception:
            pass

    def _do_install(self):
        total_steps = 10
        done = 0

        def _begin(sid):
            self.step(sid, "running")
            self.progress(done / total_steps * 100)

        def _end(sid, detail=""):
            nonlocal done
            self.step(sid, "done", detail)
            done += 1
            self.progress(done / total_steps * 100)

        _begin("check");   self.subtitle("Checking system…");                  self.check_system();             _end("check", f"{self.os_name}/{self.arch}")
        _begin("git");     self.subtitle("Syncing repository…");               self.sync_repo();                _end("git")
        _begin("detect");  self.subtitle("Detecting hardware…");               self.detect_backend();           _end("detect", self.backend or "cpu")
        self.log(f"Backend: {self.backend}", "ok")
        if self.torch_index:
            self.log(f"Torch index: {self.torch_index}", "ok")
        else:
            self.log("Torch index: (torch-directml from PyPI)", "ok")
        self.log(f"Note: {self.wheel_note}", "ok")

        _begin("venv");    self.subtitle("Creating virtual environment…");     self.create_venv();              _end("venv")
        _begin("torch");   self.subtitle("Installing PyTorch (few minutes)…"); self.install_torch();            _end("torch", self.torch_ver)
        _begin("reqs");    self.subtitle("Installing project dependencies…");  self.install_requirements();     _end("reqs")
        _begin("extras");  self.subtitle("Installing extras…");                self.install_extras();           _end("extras")
        _begin("webview"); self.subtitle("Installing GUI backend…");           self.install_webview_for_client(); _end("webview")

        if self.integrate:
            _begin("integrate")
            self.subtitle("Installing desktop integration…")
            self.install_integration()
            _end("integrate")
        else:
            self.step("integrate", "done", "skipped")
            done += 1
            self.progress(done / total_steps * 100)

        _begin("finalize"); self.subtitle("Finalizing…"); self.finalize(); _end("finalize")

        self.subtitle("Installation complete")
        self.log("", "ok")
        self.log(f"✅ CrowdGPT installed to {self.installed_app_dir}", "ok")
        self.log(f"   Python: {self.installed_venv_py}", "ok")
        self.log(f"   Backend: {self.backend}", "ok")
        self.log(f"   Torch: {self.torch_ver}", "ok")

        if self.os_name == "Windows":
            launch_hint = "Start Menu > CrowdGPT"
        elif self.os_name == "Darwin":
            launch_hint = "Spotlight > CrowdGPT"
        elif self.integrate:
            launch_hint = "crowdgpt"
        else:
            launch_hint = f"{self.installed_venv_py} {self.installed_app_dir/'client.py'}"

        self.finish(
            True, "CrowdGPT is ready",
            f"Installed to {self.installed_app_dir}\n"
            f"Backend: {self.backend}\n"
            f"Torch: {self.torch_ver}\n"
            f"Launch: {launch_hint}",
            can_relaunch=self.launch,
        )

    # ---------- System ----------
    def check_system(self):
        self.log(f"OS: {self.os_name} {platform.release()}")
        self.log(f"Arch: {self.arch}")
        self.log(f"Python: {sys.version.split()[0]} ({sys.executable})")
        if which("nvidia-smi"):
            rc, out = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            if rc == 0:
                first = out.strip().splitlines()[0] if out.strip() else "detected"
                self.log(f"nvidia-smi: {first}", "ok")
        if which("rocminfo") or Path("/opt/rocm").exists():
            self.log("ROCm: detected", "ok")

    def sync_repo(self):
        if not (self.repo_dir / "client.py").exists():
            raise RuntimeError(f"client.py not found in {self.repo_dir}")
        rc, out = run_capture(["git", "-C", str(self.repo_dir), "log", "-1", "--format=%h %s"])
        if rc == 0:
            self.log(f"HEAD: {out.strip()}", "ok")

    # ============================================================
    # Windows GPU discovery (WMI-based, no PATH dependency)
    # ============================================================

    def _wmi_gpu_names(self) -> list:
        """GPU names via PowerShell CIM. Works even if nvidia-smi is
        missing or on a non-PATH directory."""
        if self.os_name != "Windows":
            return []
        rc, out = run_capture([
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
            "Get-CimInstance -ClassName Win32_VideoController | ForEach-Object { $_.Name }"
        ])
        if rc != 0:
            self.log(f"WMI GPU query failed (rc={rc}): {out[:200]}", "warn")
            return []
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    def _find_nvidia_smi(self):
        """Locate nvidia-smi regardless of PATH."""
        p = which("nvidia-smi")
        if p:
            return p
        if self.os_name == "Windows":
            candidates = [
                Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                    / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            ]
            for cand in candidates:
                if cand.is_file():
                    return str(cand)
        return None

    def _detect_nvidia_windows(self) -> bool:
        """NVIDIA on Windows: nvidia-smi (even off-PATH) then WMI fallback."""
        smi = self._find_nvidia_smi()
        if smi:
            self.log(f"nvidia-smi: {smi}")
            rc, out = run_capture([smi, "--query-gpu=driver_version",
                                   "--format=csv,noheader"])
            driver = out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else ""
            try:
                major = int(driver.split(".")[0]) if driver else 0
            except ValueError:
                major = 0

            rc, out = run_capture([smi, "--query-gpu=name", "--format=csv,noheader"])
            names = out.strip() if rc == 0 else ""
            if names:
                self.log(f"nvidia-smi GPUs: {' | '.join(names.splitlines()[:3])}")

            is_bw = bool(re.search(
                r"RTX\s*50\d\d|B100|B200|GB200|GB10(\D|$)|DGX.?Spark|Blackwell",
                names, re.I,
            ))

            if is_bw:
                self.backend = "nvidia"
                if major >= 580:
                    self.torch_index = "https://download.pytorch.org/whl/cu130"
                    self.wheel_note = f"NVIDIA Blackwell -> cu130 (driver {driver})"
                else:
                    self.torch_index = "https://download.pytorch.org/whl/cu128"
                    self.wheel_note = f"NVIDIA Blackwell -> cu128 (driver {driver or 'unknown'})"
                return True

            self.backend = "nvidia"
            if not major:
                self.torch_index = "https://download.pytorch.org/whl/cu124"
                self.wheel_note = "NVIDIA (driver unreadable) -> cu124"
            elif major >= 525:
                self.torch_index = "https://download.pytorch.org/whl/cu124"
                self.wheel_note = f"NVIDIA driver {driver} -> cu124"
            elif major >= 470:
                self.torch_index = "https://download.pytorch.org/whl/cu118"
                self.wheel_note = f"NVIDIA driver {driver} -> cu118"
            else:
                raise RuntimeError(f"NVIDIA driver {driver} is too old (need >= 470)")
            return True

        # nvidia-smi missing everywhere → fall back to WMI
        names = self._wmi_gpu_names()
        nv = [n for n in names if any(
            k in n.lower() for k in ("nvidia", "geforce", "quadro", "tesla", "rtx")
        )]
        if not nv:
            return False

        self.log(f"nvidia-smi missing; WMI reports: {nv[0]}")
        joined = " ".join(nv).lower()
        is_bw = bool(re.search(r"rtx\s*50\d\d|b100|b200|gb200|gb10", joined))
        self.backend = "nvidia"
        if is_bw:
            self.torch_index = "https://download.pytorch.org/whl/cu128"
            self.wheel_note = f"NVIDIA Blackwell via WMI ({nv[0]}) -> cu128"
        else:
            self.torch_index = "https://download.pytorch.org/whl/cu124"
            self.wheel_note = f"NVIDIA via WMI ({nv[0]}) -> cu124"
        return True

    def _detect_amd_windows(self) -> bool:
        """AMD on Windows → DirectML (Windows ROCm is experimental/absent)."""
        names = self._wmi_gpu_names()
        amd = [n for n in names if any(
            k in n.lower() for k in ("amd", "radeon", "ati ")
        )]
        if not amd:
            return False
        self.backend = "amd"
        self.torch_index = ""  # DirectML installs from PyPI, not a torch index
        self.wheel_note = f"AMD {amd[0]} -> DirectML (torch-directml)"
        return True

    def _detect_intel_windows(self) -> bool:
        names = self._wmi_gpu_names()
        arc = [n for n in names if "intel" in n.lower() and "arc" in n.lower()]
        if not arc:
            return False
        self.backend = "intel"
        self.torch_index = "https://download.pytorch.org/whl/xpu"
        self.wheel_note = f"Intel {arc[0]} -> XPU"
        return True

    # ============================================================
    # Linux GPU detection (unchanged from v12)
    # ============================================================

    def _is_jetson(self) -> bool:
        if Path("/etc/nv_tegra_release").exists():
            return True
        if which("dpkg"):
            rc, out = run_capture(["dpkg", "-l"])
            if rc == 0 and re.search(r"^ii\s+nvidia-l4t-core", out, re.M):
                return True
        if which("nvidia-smi"):
            rc, out = run_capture(["nvidia-smi", "-q"])
            if rc == 0 and re.search(r"Jetson|Orin|Xavier|Tegra", out, re.I):
                return True
        return False

    def _is_blackwell(self) -> bool:
        if not which("nvidia-smi"):
            return False
        rc, out = run_capture(["nvidia-smi", "--query-gpu=compute_cap",
                               "--format=csv,noheader"])
        if rc == 0 and out.strip():
            for cap in out.splitlines():
                cap = cap.strip()
                if not cap:
                    continue
                try:
                    if int(cap.split(".")[0]) >= 12:
                        return True
                except ValueError:
                    continue
            return False
        rc, out = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if rc == 0 and out:
            if re.search(r"RTX\s*50\d\d|B100|B200|GB200|GB10(\D|$)|DGX.?Spark|Blackwell",
                         out, re.I):
                return True
        return False

    def _is_gfx906(self) -> bool:
        if not which("rocminfo"):
            return False
        rc, out = run_capture(["rocminfo"])
        return rc == 0 and "gfx906" in out

    def _detect_nvidia(self, forced):
        """Linux NVIDIA detection."""
        rc, out = run_capture(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"])
        driver = out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else ""
        try:
            major = int(driver.split(".")[0]) if driver else 0
        except ValueError:
            major = 0

        is_bw = self._is_blackwell()
        is_aarch = self.arch in ("aarch64", "arm64")

        if is_bw and is_aarch:
            if major and major < 525:
                raise RuntimeError(f"Blackwell aarch64 needs driver >= 525 (got {driver})")
            self.backend = "nvidia"
            self.torch_index = "https://download.pytorch.org/whl/nightly/cu128"
            self.wheel_note = "Blackwell aarch64 -> nightly cu128"
            return
        if is_bw:
            if not major:
                self.backend = "nvidia"
                self.torch_index = "https://download.pytorch.org/whl/cu128"
                self.wheel_note = "Blackwell (driver unknown) -> cu128"
            elif major >= 580:
                self.backend = "nvidia"
                self.torch_index = "https://download.pytorch.org/whl/cu130"
                self.wheel_note = f"Blackwell -> cu130 (driver {driver})"
            elif major >= 525:
                self.backend = "nvidia"
                self.torch_index = "https://download.pytorch.org/whl/cu128"
                self.wheel_note = f"Blackwell -> cu128 (driver {driver})"
            else:
                raise RuntimeError(f"Blackwell needs driver >= 525 (got {driver})")
            return

        if is_aarch:
            if self._is_jetson():
                self.backend = "cpu"
                self.torch_index = "https://download.pytorch.org/whl/cpu"
                self.wheel_note = "Jetson (Tegra) — generic CUDA wheels incompatible, using CPU"
                self.log("Jetson detected — PyTorch installs CPU wheel.", "warn")
                return
            self.backend = "nvidia"
            self.torch_index = "https://download.pytorch.org/whl/cu124"
            self.wheel_note = "aarch64 NVIDIA server -> cu124"
            return

        if not driver:
            self.backend = "nvidia"
            self.torch_index = "https://download.pytorch.org/whl/cu124"
            self.wheel_note = "NVIDIA (driver unknown) -> cu124"
        elif major >= 525:
            self.backend = "nvidia"
            self.torch_index = "https://download.pytorch.org/whl/cu124"
            self.wheel_note = f"NVIDIA driver {driver} -> cu124"
        elif major >= 470:
            self.backend = "nvidia"
            self.torch_index = "https://download.pytorch.org/whl/cu118"
            self.wheel_note = f"NVIDIA driver {driver} -> cu118"
        else:
            raise RuntimeError(f"NVIDIA driver {driver} is too old (need >= 470)")

    def _detect_amd(self) -> bool:
        """Linux AMD (ROCm) detection."""
        rocm_ver = ""
        info = Path("/opt/rocm/.info/version")
        if info.exists():
            try:
                rocm_ver = ".".join(info.read_text().strip().split(".")[:2])
            except Exception:
                pass
        if not rocm_ver and which("rocminfo"):
            rc, out = run_capture(["rocminfo"])
            if rc == 0:
                m = re.search(r"ROCm\s+(\d+\.\d+)", out)
                if m:
                    rocm_ver = m.group(1)
        if not rocm_ver and which("hipconfig"):
            rc, out = run_capture(["hipconfig", "--version"])
            if rc == 0:
                m = re.search(r"^(\d+\.\d+)", out)
                if m:
                    rocm_ver = m.group(1)
        if not rocm_ver and not Path("/opt/rocm").exists() and not which("rocm-smi"):
            return False

        if self._is_gfx906():
            self.backend = "amd"
            self.torch_index = "https://download.pytorch.org/whl/rocm6.0"
            self.wheel_note = "gfx906 -> rocm6.0"
            return True

        self.backend = "amd"
        vmaj = rocm_ver.split(".")[0] if rocm_ver else ""
        if vmaj == "7":
            self.torch_index = "https://download.pytorch.org/whl/rocm7.0"
        elif vmaj == "6" and rocm_ver.startswith(("6.3", "6.4")):
            self.torch_index = "https://download.pytorch.org/whl/rocm6.3"
        elif vmaj == "6" and rocm_ver.startswith("6.2"):
            self.torch_index = "https://download.pytorch.org/whl/rocm6.2"
        elif vmaj == "5":
            self.torch_index = "https://download.pytorch.org/whl/rocm5.7"
        else:
            self.torch_index = "https://download.pytorch.org/whl/rocm6.1"
        self.wheel_note = f"AMD ROCm {rocm_ver or 'unknown'} -> {self.torch_index.rsplit('/', 1)[-1]}"
        return True

    # ============================================================
    # Backend dispatch
    # ============================================================

    def detect_backend(self):
        forced = self.backend_override
        if forced:
            self.backend = forced
            if forced == "apple":
                self.torch_index = "https://pypi.org/simple"
                self.wheel_note = "forced: apple/mps"
            elif forced == "intel":
                self.torch_index = "https://download.pytorch.org/whl/xpu"
                self.wheel_note = "forced: intel/xpu"
            elif forced == "cpu":
                self.torch_index = "https://download.pytorch.org/whl/cpu"
                self.wheel_note = "forced: cpu"
            elif forced == "amd":
                if self.os_name == "Windows":
                    self.torch_index = ""
                    self.wheel_note = "forced: amd/directml"
                elif self._is_gfx906():
                    self.torch_index = "https://download.pytorch.org/whl/rocm6.0"
                    self.wheel_note = "forced: amd/gfx906 -> rocm6.0"
                else:
                    self.torch_index = "https://download.pytorch.org/whl/rocm6.1"
                    self.wheel_note = "forced: amd/rocm6.1"
            elif forced == "nvidia":
                if self.os_name == "Windows":
                    if not self._detect_nvidia_windows():
                        raise RuntimeError(
                            "--backend nvidia: no NVIDIA GPU found "
                            "(nvidia-smi missing and WMI shows no NVIDIA device)"
                        )
                else:
                    self._detect_nvidia(forced=True)
            else:
                raise RuntimeError(f"Unknown --backend: {forced}")
            return

        if self.os_name == "Darwin":
            if self.arch == "arm64":
                self.backend = "apple"
                self.torch_index = "https://pypi.org/simple"
                self.wheel_note = "Apple Silicon (MPS)"
            else:
                self.backend = "cpu"
                self.torch_index = "https://download.pytorch.org/whl/cpu"
                self.wheel_note = "Intel Mac — CPU only"
            return

        # Windows: WMI-based detection (nvidia-smi is not always on PATH)
        if self.os_name == "Windows":
            try:
                if self._detect_nvidia_windows():
                    return
                if self._detect_amd_windows():
                    return
                if self._detect_intel_windows():
                    return
            except RuntimeError as e:
                self.log(f"GPU detection error: {e}", "warn")

            names = self._wmi_gpu_names()
            if names:
                self.log(f"WMI GPUs: {' | '.join(names[:5])}")
            self.backend = "cpu"
            self.torch_index = "https://download.pytorch.org/whl/cpu"
            self.wheel_note = "No usable GPU detected via WMI — CPU wheel"
            return

        # Linux
        if which("nvidia-smi"):
            self._detect_nvidia(forced=False)
            return
        if self._detect_amd():
            return
        if which("sycl-ls") or Path("/opt/intel/oneapi").exists():
            self.backend = "intel"
            self.torch_index = "https://download.pytorch.org/whl/xpu"
            self.wheel_note = "Intel Arc / Data Center GPU -> XPU"
            return
        self.backend = "cpu"
        self.torch_index = "https://download.pytorch.org/whl/cpu"
        self.wheel_note = "No GPU detected — CPU wheel"

    # ---------- venv ----------
    def create_venv(self):
        if self.os_name == "Darwin":
            app_dir = Path.home() / "Library" / "Application Support" / "CrowdGPT"
        elif self.os_name == "Windows":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            app_dir = Path(base) / "CrowdGPT"
        else:
            app_dir = Path.home() / ".local" / "share" / "crowdgpt"

        venv_dir = app_dir / ".venv" if self.integrate else self.repo_dir / ".venv"
        app_dir.mkdir(parents=True, exist_ok=True)

        if venv_dir.exists():
            self.log(f"Removing existing venv at {venv_dir}")
            shutil.rmtree(venv_dir, ignore_errors=True)

        self.log(f"Creating venv at {venv_dir}")
        rc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir), "--system-site-packages"],
            capture_output=True, text=True,
        ).returncode
        if rc != 0:
            self.log("venv failed, falling back to uv...", "warn")
            self.run(["uv", "venv", "--clear", "--system-site-packages",
                      "--python", "3.11", str(venv_dir)])
        sub = "Scripts/python.exe" if self.os_name == "Windows" else "bin/python"
        venv_py = venv_dir / sub
        if not venv_py.exists():
            raise RuntimeError(f"venv Python not created: {venv_py}")
        self.run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
        self.installed_venv_py = venv_py
        self.installed_app_dir = app_dir
        self.log(f"venv Python: {venv_py}", "ok")

    # ---------- Torch ----------
    def _index_has_wheel_for(self) -> bool:
        if not self.torch_index or not self.torch_index.startswith(
                "https://download.pytorch.org/whl/"):
            return True
        py = str(self.installed_venv_py)
        rc, out = run_capture([py, "-c", "import sys; print('%d%d' % sys.version_info[:2])"])
        if rc != 0:
            return True
        tag = f"cp{out.strip()}"
        if self.os_name == "Windows":
            arch_tag = "win_amd64"
        elif self.arch in ("aarch64", "arm64"):
            arch_tag = "linux_aarch64"
        else:
            arch_tag = "linux_x86_64"
        try:
            import urllib.request
            url = self.torch_index.rstrip("/") + "/torch/"
            with urllib.request.urlopen(url, timeout=15) as r:
                page = r.read().decode("utf-8", "replace")
            return tag in page and arch_tag in page
        except Exception:
            return True

    def install_torch(self):
        py = str(self.installed_venv_py)
        subprocess.run(
            [py, "-m", "pip", "uninstall", "-y",
             "torch", "torchvision", "torchaudio", "torch-directml"],
            capture_output=True, text=True,
        )

        # Windows AMD → torch-directml (installs its own torch 2.4.1)
        if self.os_name == "Windows" and self.backend == "amd":
            self.log("Installing torch-directml (Windows AMD DirectML backend)...")
            rc = self.run([py, "-m", "pip", "install", "torch-directml"], check=False)
            if rc != 0:
                self.log("torch-directml failed; falling back to CPU", "warn")
                self.backend = "cpu"
                self.torch_index = "https://download.pytorch.org/whl/cpu"
                self.run([py, "-m", "pip", "install",
                          "torch", "torchvision", "torchaudio",
                          "--index-url", "https://download.pytorch.org/whl/cpu"])
            else:
                rc2, out2 = run_capture([py, "-c",
                    "import torch, torch_directml as d; "
                    "print(f'torch={torch.__version__} directml_devices={d.device_count()}')"
                ])
                if rc2 == 0:
                    self.log(out2.strip(), "ok")
                else:
                    self.log("torch-directml installed but import failed", "warn")
            rc, out = run_capture([py, "-c", "import torch; print(torch.__version__)"])
            self.torch_ver = out.strip() if rc == 0 else "unknown"
            return

        if not self._index_has_wheel_for():
            self.log(
                f"{self.torch_index} has no torch wheels for python "
                f"{sys.version.split()[0]} on {self.arch} — switching to Python 3.11",
                "warn",
            )
            self.run(["uv", "python", "install", "3.11"])
            venv_dir = self.installed_app_dir / ".venv"
            shutil.rmtree(venv_dir, ignore_errors=True)
            self.run(["uv", "venv", "--clear", "--system-site-packages",
                      "--python", "3.11", str(venv_dir)])
            sub = "Scripts/python.exe" if self.os_name == "Windows" else "bin/python"
            self.installed_venv_py = venv_dir / sub
            py = str(self.installed_venv_py)
            self.run([py, "-m", "pip", "install", "--upgrade", "pip"])

        args = [py, "-m", "pip", "install"]
        is_nightly = "/nightly/" in self.torch_index
        if is_nightly:
            args += ["torch"]
        else:
            args += ["torch", "torchvision", "torchaudio"]
        args += ["--index-url", self.torch_index]
        if is_nightly:
            args.append("--pre")
        self.run(args)

        rc, out = run_capture([py, "-c", "import torch; print(torch.__version__)"])
        self.torch_ver = out.strip() if rc == 0 else "unknown"
        if self.torch_ver == "unknown" or not self.torch_ver[:1].isdigit():
            raise RuntimeError("torch failed to import after install")
        self.log(f"torch {self.torch_ver} installed", "ok")

    def install_requirements(self):
        req = self.repo_dir / "requirements.txt"
        if not req.exists():
            self.log("requirements.txt not found — skipping", "warn")
            return
        lines = req.read_text().splitlines()
        filtered = [
            l for l in lines
            if not re.match(r"^\s*(torch|torchvision|torchaudio|torch-directml)(\s*[<>=!~].*)?$", l)
        ]
        tmp = self.repo_dir / ".requirements.stripped"
        tmp.write_text("\n".join(filtered) + "\n")
        try:
            self.run([str(self.installed_venv_py), "-m", "pip", "install", "-r", str(tmp)])
        finally:
            tmp.unlink(missing_ok=True)

    def install_extras(self):
        py = str(self.installed_venv_py)
        m = re.match(r"(\d+)\.(\d+)", self.torch_ver)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major == 2 and minor < 5:
                self.log(f"torch {self.torch_ver} < 2.5 → pinning transformers 4.x")
                self.run([py, "-m", "pip", "install", "transformers>=4.45,<5.0"])
        if self.backend == "nvidia":
            self.log("Installing bitsandbytes (8-bit optimizer)...")
            rc = self.run([py, "-m", "pip", "install", "bitsandbytes"], check=False)
            if rc != 0:
                self.log(
                    "bitsandbytes failed (8-bit Adam unavailable). Client will fall back "
                    "to adamw_fp32 — expect cross-check flags in mixed-optimizer pools.",
                    "warn",
                )

    def install_webview_for_client(self):
        py = str(self.installed_venv_py)

        # Windows: native EdgeChromium via pythonnet (pulled by pywebview).
        if self.os_name == "Windows":
            self.log("Installing pywebview + pythonnet (native WebView2, no Qt)...")
            rc = self.run([py, "-m", "pip", "install", "pywebview", "pythonnet"], check=False)
            if rc == 0:
                rc2, out = run_capture([py, "-c",
                    "from webview.platforms import edgechromium; print('EDGE_OK')"])
                if rc2 == 0 and "EDGE_OK" in out:
                    self.log("EdgeChromium backend OK (no Qt needed)", "ok")
                    return
                self.log(f"EdgeChromium check failed: {out[:200]}", "warn")
            self.log("Falling back to PyQt6 for Windows...", "warn")
            rc = self.run([py, "-m", "pip", "install",
                           "QtPy", "PyQt6", "PyQt6-WebEngine"], check=False)
            if rc != 0:
                raise RuntimeError("Could not install any webview backend on Windows")
            return

        # macOS: cocoa backend ships with pywebview; no extra packages.
        if self.os_name == "Darwin":
            rc = self.run([py, "-m", "pip", "install", "pywebview"], check=False)
            if rc == 0:
                self.log("pywebview cocoa backend is built-in (no Qt needed)", "ok")
            else:
                self.log("pywebview install failed (should already be in requirements)", "warn")
            return

        # Linux: try GTK (system PyGObject) first, then Qt fallback.
        rc, _ = run_capture([py, "-c", "import gi"])
        if rc == 0:
            self.log("GTK bindings already available (system PyGObject)", "ok")
            return

        self.log("Installing Qt backend (PyQt6 + WebEngine)...")
        rc = self.run([py, "-m", "pip", "install",
                       "QtPy", "PyQt6", "PyQt6-WebEngine"], check=False)
        if rc != 0:
            self.log("PyQt6 failed, trying PyQt5 fallback...", "warn")
            rc = self.run([py, "-m", "pip", "install",
                           "QtPy", "PyQt5", "PyQtWebEngine"], check=False)
            if rc != 0:
                raise RuntimeError("Could not install any Qt backend")

        rc, out = run_capture([py, "-c", "import webview; print('ok')"])
        if rc != 0:
            self.log(f"pywebview import failed: {out}", "warn")
        else:
            self.log("webview backend OK", "ok")

    # ---------- Integration ----------
    def install_integration(self):
        src = self.repo_dir
        dst = self.installed_app_dir
        self.log(f"Copying app files to {dst}")
        self._copy_tree(src, dst)

        src_client = src / "client.py"
        dst_client = dst / "client.py"
        if src_client.exists() and dst_client.exists():
            try:
                if src_client.read_bytes() != dst_client.read_bytes():
                    raise RuntimeError(
                        f"Installed client.py does not match source — copy_tree failed.\n"
                        f"  Source:    {src_client}\n"
                        f"  Installed: {dst_client}"
                    )
            except OSError:
                pass

        if self.os_name == "Darwin":
            self._install_macos_integration()
        elif self.os_name == "Windows":
            self._install_windows_integration()
        else:
            self._install_linux_integration()

    def _copy_tree(self, src, dst):
        excludes = {".git", ".venv", ".installer-venv", "checkpoints",
                    "pending_uploads", "__pycache__", ".installed-version"}
        for item in src.iterdir():
            if item.name in excludes:
                continue
            target = dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, target)

        rc, out = run_capture(["git", "-C", str(src), "log", "-1", "--format=%H|%ci|%s"])
        if rc == 0:
            parts = out.strip().split("|", 2)
            stamp = (f"commit: {parts[0]}\n"
                     f"date:   {parts[1] if len(parts) > 1 else '?'}\n"
                     f"subject:{parts[2] if len(parts) > 2 else '?'}\n")
            (dst / ".installed-version").write_text(stamp)

    def _pick_icon_src(self):
        for cand in ("docs/logo-app.svg", "docs/logo-app.png",
                     "docs/logo-black.svg", "docs/logo-white.svg"):
            p = self.installed_app_dir / cand
            if p.exists():
                return p, (cand if "logo-app" in cand else f"monochrome fallback: {cand}")
        return None, ""

    def _install_linux_integration(self):
        bin_dir = Path.home() / ".local" / "bin"
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        icon_dir = Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        icon_scalable = Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps"
        for d in (bin_dir, desktop_dir, icon_dir, icon_scalable):
            d.mkdir(parents=True, exist_ok=True)

        launcher = bin_dir / "crowdgpt"
        launcher.write_text(
            f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" '
            f'"{self.installed_app_dir}/client.py" "$@"\n'
        )
        launcher.chmod(0o755)

        icon_src, note = self._pick_icon_src()
        if icon_src:
            if icon_src.suffix == ".png":
                shutil.copy2(icon_src, icon_dir / "crowdgpt.png")
                self.log(f"Icon installed: {icon_dir/'crowdgpt.png'} ({note})", "ok")
            else:
                png = icon_dir / "crowdgpt.png"
                rc, _ = run_capture(["rsvg-convert", "-a", "-w", "256", "-h", "256",
                                     str(icon_src), "-o", str(png)])
                if rc == 0 and png.exists():
                    self.log(f"Icon installed (rasterised): {png} ({note})", "ok")
                else:
                    shutil.copy2(icon_src, icon_scalable / "crowdgpt.svg")
                    self.log(f"Icon installed (SVG): {icon_scalable/'crowdgpt.svg'} ({note})", "ok")
        else:
            self.log("No icon found under docs/ — generic icon will be used.", "warn")

        desktop = desktop_dir / "crowdgpt.desktop"
        desktop.write_text(f"""[Desktop Entry]
Version=1.0
Type=Application
Name=CrowdGPT
GenericName=Decentralized AI Training Client
Comment=Contribute GPU cycles to the CrowdGPT network
Exec="{launcher}"
Icon=crowdgpt
Terminal=false
Categories=Science;Network;Utility;
Keywords=AI;ML;Training;Distributed;LLM;
StartupNotify=true
""")
        desktop.chmod(0o755)

        uninstall = bin_dir / "crowdgpt-uninstall"
        uninstall.write_text(f"""#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -rf "{self.installed_app_dir}"
rm -f "{launcher}" "{uninstall}"
rm -f "{desktop}"
rm -f "{icon_dir}/crowdgpt.png" "{icon_scalable}/crowdgpt.svg"
echo "Done."
""")
        uninstall.chmod(0o755)

        for cmd in (["update-desktop-database", str(desktop_dir)],
                    ["gtk-update-icon-cache", "-f", "-t",
                     str(Path.home() / "icons" / "hicolor")]):
            run_capture(cmd)

        self.log(f"Desktop entry: {desktop}", "ok")
        self.log(f"Command: {launcher}", "ok")
        if str(bin_dir) not in os.environ.get("PATH", "").split(":"):
            self.log(f"⚠ {bin_dir} not on PATH — add: export PATH=\"$HOME/.local/bin:$PATH\"", "warn")

    def _install_windows_integration(self):
        """Windows: .pyw launcher + Start Menu + Desktop shortcut + Add/Remove Programs."""
        import winreg

        app_dir = self.installed_app_dir
        venv_py = self.installed_venv_py
        venv_pyw = venv_py.parent / "pythonw.exe"
        if not venv_pyw.exists():
            venv_pyw = venv_py

        # .pyw launcher: no console window, no cwd dependency
        launcher = app_dir / "CrowdGPT.pyw"
        client_py = app_dir / "client.py"
        launcher.write_text(
            "import runpy, sys, os\n"
            f"os.chdir(r'{app_dir}')\n"
            f"sys.argv = [r'{client_py}'] + sys.argv[1:]\n"
            f"runpy.run_path(r'{client_py}', run_name='__main__')\n",
            encoding="utf-8",
        )

        # Icon for the shortcut (best-effort: .ico, then .png; SVG can't be used directly)
        ico_src = None
        for cand in ("docs/icon.ico", "docs/logo-app.ico",
                     "icon.ico", "docs/logo-app.png"):
            p = app_dir / cand
            if p.exists():
                ico_src = p
                break
        if ico_src is None:
            self.log(
                "No .ico/.png icon in docs/ — shortcut will use default icon. "
                "Add docs/logo-app.ico for a proper icon.",
                "warn",
            )

        def _ps_shortcut(path):
            ps = (
                f'$s = (New-Object -COM WScript.Shell).CreateShortcut("{path}")\n'
                f'$s.TargetPath = "{venv_pyw}"\n'
                f"$s.Arguments = '\"{launcher}\"'\n"
                f'$s.WorkingDirectory = "{app_dir}"\n'
                f'$s.Description = "CrowdGPT - Decentralized AI Training Client"\n'
                f'$s.WindowStyle = 1\n'
            )
            if ico_src:
                ps += f'$s.IconLocation = "{ico_src}"\n'
            ps += '$s.Save()\n'
            return run_capture(["powershell", "-NoProfile", "-ExecutionPolicy",
                                "Bypass", "-Command", ps])

        start_menu = Path(os.environ.get("APPDATA", str(Path.home()))) \
                     / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        start_menu.mkdir(parents=True, exist_ok=True)
        lnk = start_menu / "CrowdGPT.lnk"
        rc, out = _ps_shortcut(lnk)
        if rc == 0 and lnk.exists():
            self.log(f"Start Menu shortcut: {lnk}", "ok")
        else:
            self.log(f"Start Menu shortcut failed: {out[:200]}", "warn")

        try:
            desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
            if desktop.exists():
                desk = desktop / "CrowdGPT.lnk"
                rc, _ = _ps_shortcut(desk)
                if desk.exists():
                    self.log(f"Desktop shortcut: {desk}", "ok")
        except Exception:
            pass

        try:
            uninstall_key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT"
            uninstaller = app_dir / "uninstall.bat"
            uninstaller.write_text(
                "@echo off\r\n"
                "echo Uninstalling CrowdGPT...\r\n"
                f'rmdir /s /q "{app_dir}"\r\n'
                f'del /q "{lnk}"\r\n'
                f'reg delete "HKCU\\{uninstall_key}" /f >nul 2>&1\r\n'
                "echo Done.\r\n"
                "pause\r\n",
                encoding="ascii",
            )
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, uninstall_key) as k:
                winreg.SetValueEx(k, "DisplayName",     0, winreg.REG_SZ,    "CrowdGPT")
                winreg.SetValueEx(k, "DisplayVersion",  0, winreg.REG_SZ,    "1.0")
                winreg.SetValueEx(k, "Publisher",       0, winreg.REG_SZ,    "CrowdGPT Project")
                winreg.SetValueEx(k, "InstallLocation", 0, winreg.REG_SZ,    str(app_dir))
                winreg.SetValueEx(k, "UninstallString", 0, winreg.REG_SZ,    str(uninstaller))
                winreg.SetValueEx(k, "NoModify",        0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(k, "NoRepair",        0, winreg.REG_DWORD, 1)
            self.log("Registered in Add/Remove Programs", "ok")
        except Exception as e:
            self.log(f"Registry entry failed (continuing): {e}", "warn")

        self.log(f"Launch: Start Menu > CrowdGPT, or double-click {launcher}", "ok")

    def _make_icns(self, src: Path, icns_out: Path) -> bool:
        if not which("iconutil"):
            self.log("iconutil not found (not macOS?) — skipping icns", "warn")
            return False
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "crowdgpt.iconset"
            iconset.mkdir()
            sizes = [(16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
                     (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
                     (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
                     (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
                     (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png")]
            try:
                from PyQt6.QtGui import QPixmap, QPainter
                from PyQt6.QtCore import Qt
                if src.suffix.lower() == ".svg":
                    from PyQt6.QtSvg import QSvgRenderer
                    r = QSvgRenderer(str(src))
                    if not r.isValid():
                        return False
                    for size, name in sizes:
                        pm = QPixmap(size, size)
                        pm.fill(Qt.GlobalColor.transparent)
                        p = QPainter(pm)
                        r.render(p)
                        p.end()
                        pm.save(str(iconset / name), "PNG")
                else:
                    base = QPixmap(str(src))
                    for size, name in sizes:
                        scaled = base.scaled(
                            size, size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        scaled.save(str(iconset / name), "PNG")
            except Exception as e:
                self.log(f"icns render failed: {e}", "warn")
                return False
            rc, _ = run_capture(["iconutil", "-c", "icns", str(iconset),
                                 "-o", str(icns_out)])
            return rc == 0 and icns_out.exists()

    def _install_macos_integration(self):
        bin_dir = Path("/usr/local/bin")
        apps_dir = Path.home() / "Applications"
        apps_dir.mkdir(parents=True, exist_ok=True)
        app_bundle = apps_dir / "CrowdGPT.app"

        launcher = bin_dir / "crowdgpt"
        script = (f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" '
                  f'"{self.installed_app_dir}/client.py" "$@"\n')
        try:
            launcher.write_text(script)
            launcher.chmod(0o755)
        except PermissionError:
            self.log(f"Cannot write {launcher} (needs sudo). Skipping command launcher.", "warn")

        if app_bundle.exists():
            shutil.rmtree(app_bundle)
        (app_bundle / "Contents" / "MacOS").mkdir(parents=True)
        (app_bundle / "Contents" / "Resources").mkdir(parents=True)

        (app_bundle / "Contents" / "Info.plist").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>CrowdGPT</string>
  <key>CFBundleDisplayName</key><string>CrowdGPT</string>
  <key>CFBundleIdentifier</key><string>net.crowdgpt.client</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>icon.icns</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
""")
        exe = app_bundle / "Contents" / "MacOS" / "launcher"
        exe.write_text(f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" '
                       f'"{self.installed_app_dir}/client.py"\n')
        exe.chmod(0o755)

        icon_src, note = self._pick_icon_src()
        if icon_src:
            icns = app_bundle / "Contents" / "Resources" / "icon.icns"
            if self._make_icns(icon_src, icns):
                self.log(f"App icon: icon.icns ({note})", "ok")
            else:
                self.log(f"icns generation failed for {icon_src.name}", "warn")
        else:
            self.log("No icon found under docs/ — .app will use default icon.", "warn")

        # Register with LaunchServices so Spotlight/Launchpad see it immediately
        run_capture([
            "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
            "LaunchServices.framework/Support/lsregister",
            "-f", str(app_bundle),
        ])

        self.log(f"App bundle: {app_bundle}", "ok")

    def finalize(self):
        py = str(self.installed_venv_py)
        rc, out = run_capture([py, "-c",
            "import torch; import transformers; "
            "assert transformers.utils.is_torch_available(); "
            "print(f'OK torch={torch.__version__} transformers={transformers.__version__}')"
        ])
        if rc != 0:
            raise RuntimeError(
                "torch + transformers don't link. Model loading will fail.\n"
                f"{out}\n"
                f"Fix: \"{py}\" -m pip install 'transformers>=4.45,<5.0'"
            )
        self.log(out.strip(), "ok")


# ============================================================
# Headless
# ============================================================

def run_headless(api):
    print("\n=== CrowdGPT installer (text mode) ===\n")
    api.start_install()
    while not api.finished and not api.cancelled:
        time.sleep(0.2)


# ============================================================
# Windows console visibility
# ============================================================

def _set_windows_console_visible(visible: bool) -> bool:
    """Hide the launching CMD while the GUI is open; return whether it worked.

    The console is restored when the GUI closes, so a terminal from which the
    batch file was launched remains usable. This affects only the console
    window, not the installer process or its GUI.
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return False
        # SW_HIDE=0; SW_RESTORE=9 (also restores a previously minimized window).
        ctypes.windll.user32.ShowWindow(hwnd, 9 if visible else 0)
        return True
    except Exception as e:
        print(f"[installer] Could not change console visibility: {e}", flush=True)
        return False


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="CrowdGPT GUI installer")
    p.add_argument("--repo-dir", default=".")
    p.add_argument("--backend", default="", choices=["", "nvidia", "amd", "intel", "apple", "cpu"])
    p.add_argument("--no-launch", action="store_true")
    p.add_argument("--no-integrate", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--logo", default="", help="Path to full-colour logo (SVG or PNG)")
    p.add_argument("--logo-dark", default="", help="Path to dark-theme logo (optional)")
    args = p.parse_args()

    headless = args.headless
    if not headless:
        _os = platform.system()
        # Windows + macOS always have a GUI available — they do NOT set
        # DISPLAY / WAYLAND_DISPLAY (those are X11/Wayland only).
        has_display = (
            _os in ("Windows", "Darwin")
            or os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )
        if not has_display:
            print("[installer] No DISPLAY / WAYLAND_DISPLAY — using headless mode.", flush=True)
            headless = True
        elif not _webview_backend_ok():
            print("[installer] No pywebview GUI backend available — using headless mode.", flush=True)
            headless = True

    api = InstallerAPI(
        repo_dir=args.repo_dir,
        backend_override=args.backend,
        launch=not args.no_launch,
        integrate=not args.no_integrate,
        headless=headless,
    )

    if headless:
        run_headless(api)
        return 0

    logo_light_uri = _logo_data_uri(args.logo)
    logo_dark_uri  = _logo_data_uri(args.logo_dark)
    if not logo_light_uri:
        print("[installer] No --logo supplied — using 'C' fallback.", flush=True)

    # Linux/Qt: pre-create QApplication with a crisp icon before pywebview.start().
    # Windows/macOS don't use Qt for pywebview; the window icon comes from the
    # shortcut .ico or from Info.plist respectively.
    if platform.system() == "Linux":
        _icon_src = args.logo or args.logo_dark
        if _icon_src and os.path.exists(_icon_src):
            _prepare_qt_icon_early(_icon_src)

    window = webview.create_window(
        title="CrowdGPT Installer",
        html=_render_installer_html(logo_light_uri, logo_dark_uri),
        width=720,
        height=780,
        resizable=True,
        js_api=api,
    )
    api._window = window
    window.events.closing += lambda: (setattr(api, "cancelled", True), True)[1]

    # Hide the console only after the GUI window has been created. Keep the
    # process alive and restore the same console when the user closes the GUI.
    console_hidden = _set_windows_console_visible(False)
    try:
        webview.start(debug=False)
    except Exception as e:
        if console_hidden:
            _set_windows_console_visible(True)
            console_hidden = False
        print(f"\n[installer] GUI failed to start: {e}", file=sys.stderr)
        print("[installer] Falling back to headless mode...\n", file=sys.stderr)
        api.headless = True
        api._window = None
        if not api.finished:
            run_headless(api)
    finally:
        if console_hidden:
            _set_windows_console_visible(True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
