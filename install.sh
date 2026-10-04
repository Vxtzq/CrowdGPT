#!/usr/bin/env bash
# ============================================================
# CrowdGPT universal installer — Linux + macOS
#
# Detects hardware, installs matching PyTorch wheel, sets up a
# `crowdgpt` command, registers the app with the OS launcher,
# and ensures a working pywebview backend (GTK → PyQt fallback).
#
# Usage:  bash install.sh [--backend nvidia|amd|intel|apple|cpu]
#                         [--dir PATH] [--no-launch] [--no-integrate]
#                         [--help]
# ============================================================
set -Eeuo pipefail

REPO_URL="https://github.com/Vxtzq/CrowdGPT.git"
REPO_DIR="CrowdGPT"
BACKEND_OVERRIDE=""
LAUNCH=1
INTEGRATE=1

# ---------- colors ----------
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    BLUE=$'\033[34m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; RESET=""
fi
log()  { printf '\n%s[INFO]%s %s\n' "$BLUE" "$RESET" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'   "$GREEN" "$RESET" "$*"; }
warn() { printf '%s[WARN]%s %s\n'   "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%s[FAIL]%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }
hdr()  { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

trap 'printf "\n%s[FAIL]%s Installer failed near line %s.\n" "$RED" "$RESET" "$LINENO" >&2' ERR

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      BACKEND_OVERRIDE="${2:-}"; shift 2 ;;
        --dir)          REPO_DIR="${2:-}"; shift 2 ;;
        --no-launch)    LAUNCH=0; shift ;;
        --no-integrate) INTEGRATE=0; shift ;;
        --help|-h)
            cat <<EOF
CrowdGPT installer

Options:
  --backend <name>   Force backend: nvidia | amd | intel | apple | cpu
  --dir <path>       Directory to clone into (default: ./CrowdGPT)
  --no-launch        Install only, don't launch client.py
  --no-integrate     Skip desktop/command integration
  --help             Show this message
EOF
            exit 0
            ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

echo
echo "============================================================"
echo "                 CrowdGPT Installer"
echo "============================================================"
echo

OS="$(uname -s)"
ARCH="$(uname -m)"

# ---------- helpers ----------
run_as_root() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Root privileges required but sudo is unavailable."
    fi
}

install_uv() {
    command -v uv >/dev/null 2>&1 && return
    log "Bootstrapping uv (Python manager)..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Need curl or wget to bootstrap uv."
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH."
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"; return
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python)"; return
    fi
    install_uv
    log "Downloading Python via uv..."
    uv python install --default
    export PATH="$HOME/.local/bin:$PATH"
    PYTHON_EXE="$(uv python find)"
}

copy_tree() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete \
            --exclude='.git' \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='checkpoints' \
            --exclude='pending_uploads' \
            "$src"/ "$dst"/
    else
        find "$dst" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} + 2>/dev/null || true
        ( cd "$src" && tar cf - \
            --exclude='.git' \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='checkpoints' \
            --exclude='pending_uploads' \
            . ) | ( cd "$dst" && tar xf - )
    fi
}

# ============================================================
# 1. Python
# ============================================================
PYTHON_EXE=""
find_python
ok "Python: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ============================================================
# 2. Git
# ============================================================
if ! command -v git >/dev/null 2>&1; then
    log "Installing Git..."
    if [[ "$OS" == "Linux" ]]; then
        if   command -v apt-get >/dev/null 2>&1; then run_as_root apt-get update && run_as_root apt-get install -y git
        elif command -v dnf     >/dev/null 2>&1; then run_as_root dnf install -y git
        elif command -v pacman  >/dev/null 2>&1; then run_as_root pacman -Sy --noconfirm git
        elif command -v zypper  >/dev/null 2>&1; then run_as_root zypper --non-interactive install git
        else die "Unsupported Linux package manager. Install git manually."
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install git
        else
            xcode-select --install >/dev/null 2>&1 || true
            die "Install Xcode Command Line Tools or Homebrew, then rerun."
        fi
    fi
fi
ok "Git: $(git --version)"

# ============================================================
# 3. Clone / update repo (build dir)
# ============================================================
if [[ -f "client.py" && -d ".git" ]]; then
    REPO_DIR="."
elif [[ -d "$REPO_DIR/.git" ]]; then
    log "Updating existing checkout..."
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed; continuing."
else
    [[ -e "$REPO_DIR" ]] && die "$REPO_DIR exists but is not a git repo."
    log "Cloning CrowdGPT..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
[[ -f client.py ]] || die "client.py not found after checkout."

# ============================================================
# 4. Hardware detection
# ============================================================
detect_backend() {
    BACKEND=""
    TORCH_INDEX=""
    TORCH_EXTRA=""
    WHEEL_NOTE=""

    # ---------- Apple ----------
    if [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            BACKEND="apple"
            TORCH_INDEX="https://pypi.org/simple"
            WHEEL_NOTE="Apple Silicon (MPS) — bundled with default PyTorch wheel"
        else
            BACKEND="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            WHEEL_NOTE="Intel Mac — CPU only"
        fi
        return
    fi

    # ---------- Linux ----------

    # NVIDIA
    if command -v nvidia-smi >/dev/null 2>&1; then
        local DRIVER MAJOR
        DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"

        if [[ "$ARCH" == "aarch64" ]]; then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="Jetson (aarch64) — you may need NVIDIA's Jetson-specific wheels"
            warn "$WHEEL_NOTE"
            return
        fi

        MAJOR="${DRIVER%%.*}"
        if [[ -z "$DRIVER" ]]; then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="NVIDIA GPU — could not read driver version, defaulting to cu124"
        elif [[ "$MAJOR" =~ ^[0-9]+$ ]] && (( MAJOR >= 525 )); then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="NVIDIA driver $DRIVER → CUDA 12.4 wheel"
        elif [[ "$MAJOR" =~ ^[0-9]+$ ]] && (( MAJOR >= 470 )); then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            WHEEL_NOTE="NVIDIA driver $DRIVER → CUDA 11.8 wheel (older driver)"
        else
            die "NVIDIA driver '$DRIVER' is too old. Upgrade to >= 470."
        fi
        return
    fi

    # AMD ROCm
    local ROCM_VER=""
    if [[ -f /opt/rocm/.info/version ]]; then
        ROCM_VER="$(cut -d. -f1,2 < /opt/rocm/.info/version 2>/dev/null || true)"
    fi
    if [[ -z "$ROCM_VER" ]] && command -v rocminfo >/dev/null 2>&1; then
        ROCM_VER="$(rocminfo 2>/dev/null | grep -oE 'ROCm[[:space:]]+[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}' || true)"
    fi
    if [[ -n "$ROCM_VER" ]] || [[ -d /opt/rocm ]] || command -v rocm-smi >/dev/null 2>&1; then
        BACKEND="amd"
        case "$ROCM_VER" in
            6.*)   TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1" ;;
            5.*)   TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7" ;;
            *)     TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1" ;;
        esac
        WHEEL_NOTE="AMD ROCm ${ROCM_VER:-unknown} → ${TORCH_INDEX##*/}"
        warn "Consumer Radeon (RX 6000/7000) often needs:"
        warn "    export HSA_OVERRIDE_GFX_VERSION=10.3.0"
        warn "  Add that to your shell profile if torch doesn't see the GPU."
        return
    fi

    # Intel XPU
    if command -v sycl-ls >/dev/null 2>&1 || [[ -d /opt/intel/oneapi ]]; then
        BACKEND="intel"
        TORCH_INDEX="https://download.pytorch.org/whl/xpu"
        TORCH_EXTRA="https://pypi.org/simple"
        WHEEL_NOTE="Intel Arc / Data Center GPU → XPU wheel"
        return
    fi

    # CPU fallback
    BACKEND="cpu"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    WHEEL_NOTE="No GPU detected — CPU wheel (training will be very slow)"
}

if [[ -n "$BACKEND_OVERRIDE" ]]; then
    case "$BACKEND_OVERRIDE" in
        nvidia) BACKEND="nvidia"; TORCH_INDEX="https://download.pytorch.org/whl/cu124"; WHEEL_NOTE="forced: nvidia/cu124" ;;
        amd)    BACKEND="amd";    TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1"; WHEEL_NOTE="forced: amd/rocm6.1" ;;
        intel)  BACKEND="intel";  TORCH_INDEX="https://download.pytorch.org/whl/xpu";    WHEEL_NOTE="forced: intel/xpu" ;;
        apple)  BACKEND="apple";  TORCH_INDEX="https://pypi.org/simple";                 WHEEL_NOTE="forced: apple/mps" ;;
        cpu)    BACKEND="cpu";    TORCH_INDEX="https://download.pytorch.org/whl/cpu";    WHEEL_NOTE="forced: cpu" ;;
        *) die "--backend must be one of: nvidia | amd | intel | apple | cpu" ;;
    esac
else
    detect_backend
fi

hdr "Backend"
echo "  Detected:    $BACKEND"
echo "  Torch index: $TORCH_INDEX"
echo "  Note:        $WHEEL_NOTE"

# ============================================================
# 5. Install paths
# ============================================================
if [[ "$OS" == "Darwin" ]]; then
    APP_DIR="$HOME/Library/Application Support/CrowdGPT"
    BIN_DIR="/usr/local/bin"
    APPS_DIR="$HOME/Applications"
    APP_BUNDLE="$APPS_DIR/CrowdGPT.app"
else
    APP_DIR="$HOME/.local/share/crowdgpt"
    BIN_DIR="$HOME/.local/bin"
    DESKTOP_DIR="$HOME/.local/share/applications"
    ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
fi

# ============================================================
# 6. Build venv in the source dir
# ============================================================
# We use --system-site-packages so that PyGObject (gi) from the
# system Python is visible inside the venv. If the user has GTK
# installed, we use it; otherwise we fall back to PyQt6 below.
BUILD_VENV="$PWD/.venv"

log "Creating build venv (with system site-packages for GTK access)..."
rm -rf "$BUILD_VENV"
if ! "$PYTHON_EXE" -m venv "$BUILD_VENV" --system-site-packages 2>/dev/null; then
    install_uv
    uv venv --system-site-packages "$BUILD_VENV"
fi

BUILD_PY="$BUILD_VENV/bin/python"
[[ -x "$BUILD_PY" ]] || die "Build venv Python missing."

log "Upgrading pip..."
"$BUILD_PY" -m pip install --upgrade pip >/dev/null

# ============================================================
# 7. Install PyTorch from the correct index
# ============================================================
hdr "Installing PyTorch"

"$BUILD_PY" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

PIP_TORCH_ARGS=(
    install
    torch torchvision torchaudio
    --index-url "$TORCH_INDEX"
)
if [[ -n "$TORCH_EXTRA" ]]; then
    PIP_TORCH_ARGS+=( --extra-index-url "$TORCH_EXTRA" )
fi

log "pip ${PIP_TORCH_ARGS[*]}"
if ! "$BUILD_PY" -m pip "${PIP_TORCH_ARGS[@]}"; then
    die "PyTorch install failed from $TORCH_INDEX"
fi

# ============================================================
# 8. Install project requirements (torch lines stripped)
# ============================================================
if [[ -f requirements.txt ]]; then
    hdr "Installing project dependencies"
    TMP_REQ="$(mktemp -t crowdgpt-req.XXXXXX.txt)"
    grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]*[<>=!~].*)?$' \
        requirements.txt > "$TMP_REQ" || true
    if [[ -s "$TMP_REQ" ]]; then
        "$BUILD_PY" -m pip install -r "$TMP_REQ"
    else
        warn "requirements.txt contained only torch packages; nothing to install."
    fi
    rm -f "$TMP_REQ"
else
    warn "requirements.txt not found; skipping project deps."
fi

# ============================================================
# 9. Ensure pywebview has a working GUI backend
# ============================================================
hdr "GUI backend for pywebview"

# --- 9a. Try to expose system GTK inside the venv ---
GTK_WORKS=0

if python3 -c "import gi" 2>/dev/null; then
    SYS_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    VENV_SITE="$("$BUILD_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"

    if [[ -n "$SYS_SITE" && -n "$VENV_SITE" ]]; then
        for pkg in gi gi_cairo pygobject; do
            if [[ -e "$SYS_SITE/$pkg" ]]; then
                ln -sfn "$SYS_SITE/$pkg" "$VENV_SITE/$pkg" 2>/dev/null || true
            fi
        done
        for so in "$SYS_SITE"/_gi*.so; do
            [[ -e "$so" ]] && ln -sfn "$so" "$VENV_SITE/$(basename "$so")" 2>/dev/null || true
        done
    fi

    if "$BUILD_PY" -c "import gi" 2>/dev/null; then
        GTK_WORKS=1
        ok "GTK bindings available (system PyGObject)"
    fi
fi

# --- 9b. If not, install PyQt6 from pip (works everywhere, no sudo) ---
if [[ "$GTK_WORKS" -eq 0 ]]; then
    log "GTK not available inside venv — installing PyQt6 fallback..."
    if ! "$BUILD_PY" -m pip install PyQt6 PyQt6-WebEngine; then
        warn "PyQt6 install failed. Trying PyQt5 as last resort..."
        if ! "$BUILD_PY" -m pip install PyQt5 PyQtWebEngine; then
            die "Could not install any Qt backend. pywebview needs GTK or Qt."
        fi
    fi
    ok "Qt backend installed"
fi

# --- 9c. Quick sanity check: pywebview can find its dependencies ---
"$BUILD_PY" - <<'PYEOF' || warn "pywebview preflight failed — the client may not open a window"
try:
    import webview  # noqa
    print("  pywebview: OK")
except Exception as e:
    print(f"  pywebview import error: {e}")
    raise SystemExit(1)
PYEOF

# ============================================================
# 10. Verify the venv
# ============================================================
hdr "Verifying install"
"$BUILD_PY" - <<'PYEOF'
import sys, torch
print("  Python:", sys.version.split()[0])
print("  Torch:  ", torch.__version__)
print("  CUDA:   ", torch.cuda.is_available(), end="")
if torch.cuda.is_available():
    print(f"  ({torch.cuda.get_device_name(0)})")
else:
    print()
print("  MPS:    ", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
try:
    import torch.version
    if torch.version.hip:
        print("  ROCm:   ", torch.version.hip)
except Exception:
    pass
PYEOF

# ============================================================
# 11. Desktop + command integration
# ============================================================
if [[ "$INTEGRATE" -eq 1 ]]; then
    hdr "Installing desktop integration"
    log "Install location: $APP_DIR"

    mkdir -p "$APP_DIR"
    if [[ "$OS" == "Darwin" ]]; then
        mkdir -p "$APPS_DIR"
    else
        mkdir -p "$BIN_DIR" "$DESKTOP_DIR" "$ICON_DIR"
    fi

    log "Copying app files..."
    copy_tree "$PWD" "$APP_DIR"

    # Move the build venv into the app dir
    if [[ -d "$APP_DIR/.venv" ]]; then
        rm -rf "$APP_DIR/.venv"
    fi
    mv "$BUILD_VENV" "$APP_DIR/.venv"

    APP_PY="$APP_DIR/.venv/bin/python"
    [[ -x "$APP_PY" ]] || die "Installed venv Python missing at $APP_PY"

    # Rewrite shebangs that still point at the old path
    log "Rewriting venv paths..."
    python3 - <<EOF
import pathlib
old = "$PWD/.venv"
new = "$APP_DIR/.venv"
root = pathlib.Path(new)
if not root.exists():
    raise SystemExit(0)
for p in root.rglob("*"):
    if not p.is_file():
        continue
    try:
        if p.stat().st_size > 200_000:
            continue
    except Exception:
        continue
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    if old in text:
        try:
            p.write_text(text.replace(old, new), encoding="utf-8")
        except Exception:
            pass
EOF

    # ---------- Linux: command + .desktop ----------
    if [[ "$OS" != "Darwin" ]]; then
        cat > "$BIN_DIR/crowdgpt" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
        chmod +x "$BIN_DIR/crowdgpt"

        # Icon (best-effort PNG; some systems handle SVG fine)
        ICON_SRC=""
        for c in "$APP_DIR/docs/logo-app.svg" "$APP_DIR/docs/logo-black.svg" "$APP_DIR/docs/logo-white.svg"; do
            [[ -f "$c" ]] && ICON_SRC="$c" && break
        done

        ICON_NAME="crowdgpt"
        if [[ -n "$ICON_SRC" ]]; then
            if command -v rsvg-convert >/dev/null 2>&1; then
                rsvg-convert -w 256 -h 256 "$ICON_SRC" -o "$ICON_DIR/crowdgpt.png" 2>/dev/null || true
            elif "$APP_PY" -c "import cairosvg" 2>/dev/null; then
                "$APP_PY" -c "import cairosvg; cairosvg.svg2png(url='$ICON_SRC', write_to='$ICON_DIR/crowdgpt.png', output_width=256, output_height=256)" 2>/dev/null || true
            fi
            if [[ ! -f "$ICON_DIR/crowdgpt.png" ]]; then
                cp "$ICON_SRC" "$ICON_DIR/crowdgpt.svg" 2>/dev/null || true
            fi
        fi

        cat > "$DESKTOP_DIR/crowdgpt.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CrowdGPT
GenericName=Decentralized AI Training Client
Comment=Contribute GPU cycles to the CrowdGPT network
Exec=$BIN_DIR/crowdgpt
Icon=$ICON_NAME
Terminal=false
Categories=Science;Network;Utility;
Keywords=AI;ML;Training;Distributed;LLM;
StartupNotify=true
StartupWMClass=CrowdGPT
EOF
        chmod +x "$DESKTOP_DIR/crowdgpt.desktop"

        command -v update-desktop-database >/dev/null 2>&1 && \
            update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
        command -v gtk-update-icon-cache >/dev/null 2>&1 && \
            gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true

        cat > "$BIN_DIR/crowdgpt-uninstall" <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -rf "$APP_DIR"
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -f "$DESKTOP_DIR/crowdgpt.desktop"
rm -f "$ICON_DIR/crowdgpt.png" "$ICON_DIR/crowdgpt.svg"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
echo "Done."
EOF
        chmod +x "$BIN_DIR/crowdgpt-uninstall"

        ok "Command:   $BIN_DIR/crowdgpt"
        ok "App entry: $DESKTOP_DIR/crowdgpt.desktop"
        ok "Uninstall: crowdgpt-uninstall"

        if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
            warn "$BIN_DIR is not on your PATH."
            warn "Add this to ~/.bashrc or ~/.zshrc:"
            warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    fi

    # ---------- macOS: command + .app bundle ----------
    if [[ "$OS" == "Darwin" ]]; then
        if [[ -w "$BIN_DIR" ]]; then
            cat > "$BIN_DIR/crowdgpt" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
            chmod +x "$BIN_DIR/crowdgpt"
        else
            log "Creating $BIN_DIR/crowdgpt (requires sudo)..."
            run_as_root tee "$BIN_DIR/crowdgpt" >/dev/null <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
            run_as_root chmod +x "$BIN_DIR/crowdgpt"
        fi

        rm -rf "$APP_BUNDLE"
        mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

        cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>CrowdGPT</string>
    <key>CFBundleDisplayName</key><string>CrowdGPT</string>
    <key>CFBundleIdentifier</key><string>net.crowdgpt.client</string>
    <key>CFBundleVersion</key><string>0.5.0</string>
    <key>CFBundleShortVersionString</key><string>0.5</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIconFile</key><string>icon.icns</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

        cat > "$APP_BUNDLE/Contents/MacOS/launcher" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py"
EOF
        chmod +x "$APP_BUNDLE/Contents/MacOS/launcher"

        ICON_SRC=""
        for c in "$APP_DIR/docs/logo-app.svg" "$APP_DIR/docs/logo-black.svg"; do
            [[ -f "$c" ]] && ICON_SRC="$c" && break
        done
        if [[ -n "$ICON_SRC" ]] && command -v rsvg-convert >/dev/null 2>&1; then
            TMP_ICONSET="$(mktemp -d)/crowdgpt.iconset"
            mkdir -p "$TMP_ICONSET"
            for size in 16 32 64 128 256 512; do
                rsvg-convert -w $size -h $size "$ICON_SRC" -o "$TMP_ICONSET/icon_${size}x${size}.png" 2>/dev/null || true
                rsvg-convert -w $((size*2)) -h $((size*2)) "$ICON_SRC" -o "$TMP_ICONSET/icon_${size}x${size}@2x.png" 2>/dev/null || true
            done
            iconutil -c icns "$TMP_ICONSET" -o "$APP_BUNDLE/Contents/Resources/icon.icns" 2>/dev/null || true
            rm -rf "$TMP_ICONSET"
        fi

        if [[ -w "$BIN_DIR" ]]; then
            cat > "$BIN_DIR/crowdgpt-uninstall" <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -rf "$APP_DIR" "$APP_BUNDLE"
echo "Done."
EOF
            chmod +x "$BIN_DIR/crowdgpt-uninstall"
        else
            run_as_root tee "$BIN_DIR/crowdgpt-uninstall" >/dev/null <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -rf "$APP_DIR" "$APP_BUNDLE"
echo "Done."
EOF
            run_as_root chmod +x "$BIN_DIR/crowdgpt-uninstall"
        fi

        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_BUNDLE" 2>/dev/null || true

        ok "Command:    crowdgpt"
        ok "App bundle: $APP_BUNDLE"
        ok "Uninstall:  crowdgpt-uninstall"

        if [[ ! -w "$BIN_DIR" ]]; then
            warn "/usr/local/bin was not writable; uninstall requires sudo."
        fi
    fi

    echo
    echo "============================================================"
    echo "             Installation complete  ✓"
    echo "============================================================"
    echo "  Backend: $BACKEND"
    echo
    echo "  Launch from terminal:  crowdgpt"
    if [[ "$OS" == "Darwin" ]]; then
        echo "  Launch from GUI:       Spotlight → CrowdGPT"
    else
        echo "  Launch from GUI:       Activities / app menu → CrowdGPT"
    fi
    echo "  Uninstall:             crowdgpt-uninstall"
    echo
else
    hdr "Skipping desktop integration (--no-integrate)"
    echo "  Build venv: $BUILD_VENV"
fi

# ============================================================
# 12. Launch
# ============================================================
if [[ "$LAUNCH" -eq 1 ]]; then
    echo "Launching client.py..."
    echo
    if [[ "$INTEGRATE" -eq 1 ]]; then
        exec "$APP_PY" "$APP_DIR/client.py"
    else
        exec "$BUILD_PY" "$PWD/client.py"
    fi
else
    if [[ "$INTEGRATE" -eq 1 ]]; then
        echo "To launch later:  crowdgpt"
    else
        echo "To launch later:  $BUILD_PY $PWD/client.py"
    fi
fi
