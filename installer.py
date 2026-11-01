#!/usr/bin/env python3
"""
CrowdGPT GUI installer (pywebview + PyQt6).

Same install logic as the v12 bash installer:
  1. sync repo
  2. detect backend (nvidia / amd / intel / apple / cpu)
  3. pick torch index
  4. create final venv at APP_DIR/.venv
  5. install torch
  6. install requirements
  7. pin transformers to 4.x if torch < 2.5
  8. install bitsandbytes on nvidia
  9. install GUI backend for pywebview (GTK preferred, Qt fallback)
 10. desktop / command integration
 11. optionally launch client

Progress streamed to a window styled exactly like the client.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

try:
    import webview
except ImportError:
    print("ERROR: pywebview not installed. Run via install.sh (bootstrap).")
    sys.exit(1)


# ============================================================
# HTML — same design system as the client
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
.logo{display:flex;align-items:center;gap:9px}
.logo-fallback{width:24px;height:24px;border:1px solid var(--border-strong);display:grid;place-items:center;font-size:13px;font-weight:600}
.logo-text{font-size:15px;font-weight:600}
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
    <div class="logo-fallback">C</div>
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
  const panel = document.getElementById('log-panel');
  const div = document.createElement('div');
  div.className = 'log-line' + (kind ? ' ' + kind : '');
  div.textContent = line;
  panel.appendChild(div);
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

(function(){
  try {
    const saved = localStorage.getItem('crowdgpt-installer-theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) document.body.classList.add('dark');
  } catch(e) {}
  document.getElementById('theme-toggle').addEventListener('click', () => {
    document.body.classList.toggle('dark');
    try { localStorage.setItem('crowdgpt-installer-theme', document.body.classList.contains('dark') ? 'dark' : 'light'); } catch(e) {}
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
# Helpers
# ============================================================

def run_capture(cmd, **kw):
    """Run a command, return (rc, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timeout"


def which(binary):
    return shutil.which(binary)


class InstallerAPI:
    def __init__(self, repo_dir, backend_override, launch, integrate, headless):
        self.repo_dir = Path(repo_dir).resolve()
        self.backend_override = (backend_override or "").strip().lower()
        self.launch = launch
        self.integrate = integrate
        self.headless = headless
        self.window = None
        self.cancelled = False
        self.proc = None
        self.proc_lock = threading.Lock()

        # Install state
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
        if self.headless or not self.window:
            return
        try:
            self.window.evaluate_js(code)
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
        self._js(f"window.onLog({json.dumps(line)}, {json.dumps(kind)})")

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
        if self.headless:
            print(f"\n{'[DONE]' if ok else '[FAILED]'} {title}", flush=True)
            if body:
                print(body, flush=True)
            return
        self._js(f"window.onFinish({str(ok).lower()}, "
                 f"{json.dumps(title)}, {json.dumps(body)}, {str(can_relaunch).lower()})")

    # ---------- Process runner ----------
    def run(self, cmd, cwd=None, env=None, check=True, kind=""):
        """Run a command, streaming output to the log."""
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
        with self.proc_lock:
            self.proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if self.cancelled:
                    try: proc.terminate()
                    except Exception: pass
                    raise RuntimeError("Cancelled by user")
                line = line.rstrip()
                if line:
                    self.log(line)
            rc = proc.wait()
        finally:
            with self.proc_lock:
                self.proc = None
        if check and rc != 0:
            raise RuntimeError(f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")
        return rc

    # ---------- Public API (called from JS) ----------
    def start_install(self):
        self.cancelled = False
        threading.Thread(target=self._run_install, daemon=True).start()

    def cancel_install(self):
        self.cancelled = True
        with self.proc_lock:
            if self.proc and self.proc.poll() is None:
                try: self.proc.terminate()
                except Exception: pass

    def close_window(self):
        if self.window:
            self.window.destroy()
        else:
            sys.exit(0)

    def launch_client(self):
        if self.installed_venv_py and self.installed_app_dir:
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
            # Mark the currently-running step as failed
            self._mark_current_failed()
            self.finish(False, "Installation failed" if msg != "Cancelled by user" else "Cancelled", msg)
        except Exception as e:
            self.log(f"Unexpected error: {e}", "err")
            self.log(traceback.format_exc(), "err")
            self._mark_current_failed()
            self.finish(False, "Installation failed", str(e))

    def _mark_current_failed(self):
        # Walk all steps; the last "running" one is the failure
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

        def _step_begin(sid):
            self.step(sid, "running")
            self.progress(done / total_steps * 100)

        def _step_end(sid, detail=""):
            nonlocal done
            self.step(sid, "done", detail)
            done += 1
            self.progress(done / total_steps * 100)

        # ── 1. Check system ──
        _step_begin("check")
        self.subtitle("Checking system…")
        self.check_system()
        _step_end("check", f"{self.os_name}/{self.arch}")

        # ── 2. Sync repo ──
        _step_begin("git")
        self.subtitle("Syncing repository…")
        self.sync_repo()
        _step_end("git")

        # ── 3. Detect hardware ──
        _step_begin("detect")
        self.subtitle("Detecting hardware…")
        self.detect_backend()
        _step_end("detect", self.backend or "cpu")
        self.log(f"Backend: {self.backend}", "ok")
        self.log(f"Torch index: {self.torch_index}", "ok")
        self.log(f"Note: {self.wheel_note}", "ok")

        # ── 4. Create venv ──
        _step_begin("venv")
        self.subtitle("Creating virtual environment…")
        self.create_venv()
        _step_end("venv")

        # ── 5. Install torch ──
        _step_begin("torch")
        self.subtitle("Installing PyTorch (this can take a few minutes)…")
        self.install_torch()
        _step_end("torch", self.torch_ver)

        # ── 6. Install requirements ──
        _step_begin("reqs")
        self.subtitle("Installing project dependencies…")
        self.install_requirements()
        _step_end("reqs")

        # ── 7. Extras (transformers pin, bitsandbytes) ──
        _step_begin("extras")
        self.subtitle("Installing extras…")
        self.install_extras()
        _step_end("extras")

        # ── 8. Webview for the client ──
        _step_begin("webview")
        self.subtitle("Installing GUI backend…")
        self.install_webview_for_client()
        _step_end("webview")

        # ── 9. Integration ──
        if self.integrate:
            _step_begin("integrate")
            self.subtitle("Installing desktop integration…")
            self.install_integration()
            _step_end("integrate")
        else:
            self.step("integrate", "done", "skipped")
            done += 1
            self.progress(done / total_steps * 100)

        # ── 10. Finalize ──
        _step_begin("finalize")
        self.subtitle("Finalizing…")
        self.finalize()
        _step_end("finalize")

        self.subtitle("Installation complete")
        self.log("", "ok")
        self.log(f"✅ CrowdGPT installed to {self.installed_app_dir}", "ok")
        self.log(f"   Python: {self.installed_venv_py}", "ok")
        self.log(f"   Backend: {self.backend}", "ok")
        self.log(f"   Torch: {self.torch_ver}", "ok")

        self.finish(
            True,
            "CrowdGPT is ready",
            f"Installed to {self.installed_app_dir}\n"
            f"Backend: {self.backend}\n"
            f"Torch: {self.torch_ver}\n"
            f"Launch: crowdgpt" if self.integrate else f"Launch: {self.installed_venv_py} {self.installed_app_dir/'client.py'}",
            can_relaunch=self.launch,
        )

    # ---------- Steps ----------
    def check_system(self):
        self.log(f"OS: {self.os_name} {platform.release()}")
        self.log(f"Arch: {self.arch}")
        self.log(f"Python: {sys.version.split()[0]} ({sys.executable})")
        if which("nvidia-smi"):
            rc, out = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
            if rc == 0:
                self.log(f"nvidia-smi: {out.strip().splitlines()[0] if out.strip() else 'detected'}", "ok")
        if which("rocminfo") or Path("/opt/rocm").exists():
            self.log("ROCm: detected", "ok")

    def sync_repo(self):
        # Bootstrap already synced; just verify.
        if not (self.repo_dir / "client.py").exists():
            raise RuntimeError(f"client.py not found in {self.repo_dir}")
        rc, out = run_capture(["git", "-C", str(self.repo_dir), "log", "-1", "--format=%h %s"])
        if rc == 0:
            self.log(f"HEAD: {out.strip()}", "ok")

    def detect_backend(self):
        """Port of the v12 detection logic (condensed)."""
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
                if self._is_gfx906():
                    self.torch_index = "https://download.pytorch.org/whl/rocm6.0"
                    self.wheel_note = "forced: amd/gfx906 -> rocm6.0"
                else:
                    self.torch_index = "https://download.pytorch.org/whl/rocm6.1"
                    self.wheel_note = "forced: amd/rocm6.1"
            elif forced == "nvidia":
                self._detect_nvidia(forced=True)
            else:
                raise RuntimeError(f"Unknown --backend: {forced}")
            return

        # Auto-detect
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

    def _detect_nvidia(self, forced):
        rc, out = run_capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        driver = out.strip().splitlines()[0].strip() if rc == 0 and out.strip() else ""
        major_str = driver.split(".")[0] if driver else ""
        try:
            major = int(major_str)
        except ValueError:
            major = 0

        is_bw = self._is_blackwell()
        is_aarch = self.arch in ("aarch64", "arm64")

        if is_bw and is_aarch:
            # DGX Spark GB10 / GB200 — nightly cu128 for aarch64 kernels
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

    def _is_blackwell(self):
        if not which("nvidia-smi"):
            return False
        rc, out = run_capture(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
        if rc == 0 and out.strip():
            for cap in out.splitlines():
                cap = cap.strip()
                if not cap: continue
                try:
                    maj = int(cap.split(".")[0])
                except ValueError:
                    continue
                if maj >= 12:
                    return True
            return False
        rc, out = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if rc == 0 and out:
            if re.search(r"RTX\s*50\d\d|B100|B200|GB200|GB10(\D|$)|DGX.?Spark|Blackwell", out, re.I):
                return True
        return False

    def _detect_amd(self):
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
                if m: rocm_ver = m.group(1)
        if not rocm_ver and which("hipconfig"):
            rc, out = run_capture(["hipconfig", "--version"])
            if rc == 0:
                m = re.search(r"^(\d+\.\d+)", out)
                if m: rocm_ver = m.group(1)

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
        self.wheel_note = f"AMD ROCm {rocm_ver or 'unknown'} -> {self.torch_index.rsplit('/',1)[-1]}"
        return True

    def _is_gfx906(self):
        if not which("rocminfo"):
            return False
        rc, out = run_capture(["rocminfo"])
        return rc == 0 and "gfx906" in out

    def create_venv(self):
        if self.os_name == "Darwin":
            app_dir = Path.home() / "Library" / "Application Support" / "CrowdGPT"
        else:
            app_dir = Path.home() / ".local" / "share" / "crowdgpt"

        if self.integrate:
            venv_dir = app_dir / ".venv"
        else:
            venv_dir = self.repo_dir / ".venv"
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
            self.run(["uv", "venv", "--system-site-packages", "--python", "3.11", str(venv_dir)])
        venv_py = venv_dir / ("Scripts/python.exe" if self.os_name == "Windows" else "bin/python")
        if not venv_py.exists():
            raise RuntimeError(f"venv Python not created: {venv_py}")
        self.run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
        self.installed_venv_py = venv_py
        self.installed_app_dir = app_dir
        self.log(f"venv Python: {venv_py}", "ok")

    def install_torch(self):
        py = str(self.installed_venv_py)
        # Remove any prior torch
        subprocess.run([py, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                       capture_output=True, text=True)

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
        # Strip torch lines
        lines = req.read_text().splitlines()
        filtered = [l for l in lines if not re.match(r"^\s*(torch|torchvision|torchaudio)(\s*[<>=!~].*)?$", l)]
        tmp = self.repo_dir / ".requirements.stripped"
        tmp.write_text("\n".join(filtered) + "\n")
        try:
            self.run([str(self.installed_venv_py), "-m", "pip", "install", "-r", str(tmp)])
        finally:
            tmp.unlink(missing_ok=True)

    def install_extras(self):
        py = str(self.installed_venv_py)

        # Pin transformers to 4.x if torch < 2.5
        m = re.match(r"(\d+)\.(\d+)", self.torch_ver)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major == 2 and minor < 5:
                self.log(f"torch {self.torch_ver} < 2.5 → pinning transformers 4.x")
                self.run([py, "-m", "pip", "install", "transformers>=4.45,<5.0"])

        # bitsandbytes on nvidia
        if self.backend == "nvidia":
            self.log("Installing bitsandbytes (8-bit optimizer)...")
            rc = self.run([py, "-m", "pip", "install", "bitsandbytes"], check=False)
            if rc != 0:
                self.log("bitsandbytes failed (8-bit Adam unavailable — GPUs <12GB may not fit)", "warn")

    def install_webview_for_client(self):
        py = str(self.installed_venv_py)

        # Try GTK first (system), then PyQt6 (self-contained)
        rc, _ = run_capture([py, "-c", "import gi"])
        if rc == 0:
            self.log("GTK bindings already available (system PyGObject)", "ok")
            return

        self.log("Installing Qt backend (PyQt6 + WebEngine)...")
        rc = self.run([py, "-m", "pip", "install", "QtPy", "PyQt6", "PyQt6-WebEngine"], check=False)
        if rc != 0:
            self.log("PyQt6 failed, trying PyQt5 fallback...", "warn")
            rc = self.run([py, "-m", "pip", "install", "QtPy", "PyQt5", "PyQtWebEngine"], check=False)
            if rc != 0:
                raise RuntimeError("Could not install any Qt backend")

        # Verify pywebview can find a backend
        self.log("Verifying webview import...")
        rc, out = run_capture([py, "-c", "import webview; print('ok')"])
        if rc != 0:
            self.log(f"pywebview import failed: {out}", "warn")
        else:
            self.log("webview backend OK", "ok")

    def install_integration(self):
        src = self.repo_dir
        dst = self.installed_app_dir

        # Copy tree (excluding venv, git, checkpoints)
        self.log(f"Copying app files to {dst}")
        self._copy_tree(src, dst)

        if self.os_name == "Darwin":
            self._install_macos_integration()
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
                if target.exists(): shutil.rmtree(target)
                shutil.copytree(item, target,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, target)

        # Stamp provenance
        rc, out = run_capture(["git", "-C", str(src), "log", "-1", "--format=%H|%ci|%s"])
        if rc == 0:
            parts = out.strip().split("|", 2)
            stamp = (f"commit: {parts[0]}\n"
                     f"date:   {parts[1] if len(parts) > 1 else '?'}\n"
                     f"subject:{parts[2] if len(parts) > 2 else '?'}\n")
            (dst / ".installed-version").write_text(stamp)

    def _install_linux_integration(self):
        bin_dir = Path.home() / ".local" / "bin"
        desktop_dir = Path.home() / ".local" / "share" / "applications"
        icon_dir = Path.home() / ".local" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        icon_scalable = Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps"
        for d in (bin_dir, desktop_dir, icon_dir, icon_scalable):
            d.mkdir(parents=True, exist_ok=True)

        # Command launcher
        launcher = bin_dir / "crowdgpt"
        launcher.write_text(f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" "{self.installed_app_dir}/client.py" "$@"\n')
        launcher.chmod(0o755)

        # Icon
        icon_src = None
        for cand in ("docs/logo-app.svg", "docs/logo-app.png"):
            p = self.installed_app_dir / cand
            if p.exists(): icon_src = p; break
        if icon_src:
            if icon_src.suffix == ".png":
                shutil.copy2(icon_src, icon_dir / "crowdgpt.png")
            else:
                # Try conversion, else copy SVG
                png = icon_dir / "crowdgpt.png"
                rc, _ = run_capture(["rsvg-convert", "-a", "-w", "256", "-h", "256",
                                     str(icon_src), "-o", str(png)])
                if rc != 0:
                    shutil.copy2(icon_src, icon_scalable / "crowdgpt.svg")

        # Desktop entry
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

        # Uninstall
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

        # Refresh caches
        for cmd in (["update-desktop-database", str(desktop_dir)],
                    ["gtk-update-icon-cache", "-f", "-t", str(Path.home()/"icons"/"hicolor")]):
            run_capture(cmd)

        self.log(f"Desktop entry: {desktop}", "ok")
        self.log(f"Command: {launcher}", "ok")

        if str(bin_dir) not in os.environ.get("PATH", "").split(":"):
            self.log(f"⚠ {bin_dir} not on PATH — add: export PATH=\"$HOME/.local/bin:$PATH\"", "warn")

    def _install_macos_integration(self):
        bin_dir = Path("/usr/local/bin")
        apps_dir = Path.home() / "Applications"
        apps_dir.mkdir(parents=True, exist_ok=True)
        app_bundle = apps_dir / "CrowdGPT.app"

        launcher = bin_dir / "crowdgpt"
        script = f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" "{self.installed_app_dir}/client.py" "$@"\n'
        try:
            launcher.write_text(script)
            launcher.chmod(0o755)
        except PermissionError:
            self.log(f"Cannot write {launcher} (needs sudo). Skipping command launcher.", "warn")

        # App bundle
        if app_bundle.exists(): shutil.rmtree(app_bundle)
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
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
""")
        exe = app_bundle / "Contents" / "MacOS" / "launcher"
        exe.write_text(f'#!/usr/bin/env bash\nexec "{self.installed_venv_py}" "{self.installed_app_dir}/client.py"\n')
        exe.chmod(0o755)
        self.log(f"App bundle: {app_bundle}", "ok")

    def finalize(self):
        # Quick sanity: verify transformers can import torch in the final venv
        py = str(self.installed_venv_py)
        rc, out = run_capture([py, "-c",
            "import torch; import transformers; "
            "assert transformers.utils.is_torch_available(); "
            f"print(f'OK torch={{torch.__version__}} transformers={{transformers.__version__}}')"
        ])
        if rc != 0:
            raise RuntimeError(
                "torch + transformers don't link. Model loading will fail.\n"
                f"{out}\n"
                f"Fix: \"{py}\" -m pip install 'transformers>=4.45,<5.0'"
            )
        self.log(out.strip(), "ok")


# ============================================================
# Headless mode (no GUI) — used when no display is available
# ============================================================

def run_headless(api):
    print("\n=== CrowdGPT installer (text mode) ===\n")
    api._run_install()


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
    args = p.parse_args()

    api = InstallerAPI(
        repo_dir=args.repo_dir,
        backend_override=args.backend,
        launch=not args.no_launch,
        integrate=not args.no_integrate,
        headless=args.headless,
    )

    if args.headless:
        run_headless(api)
        return 0

    window = webview.create_window(
        title="CrowdGPT Installer",
        html=HTML,
        width=720,
        height=780,
        resizable=True,
        js_api=api,
    )
    api.window = window

    def on_closing():
        api.cancelled = True
        return True

    window.events.closing += on_closing

    # Force the Qt backend (we installed PyQt6 in the bootstrap venv)
    try:
        webview.start(gui="qt", debug=False)
    except Exception:
        # Fall back to auto-detect if PyQt6 isn't there (GTK might be)
        webview.start(debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
