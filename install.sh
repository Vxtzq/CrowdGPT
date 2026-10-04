#!/usr/bin/env bash
# ============================================================
# CrowdGPT universal installer — Linux + macOS
# Detects hardware, installs matching PyTorch wheel, launches client.
# Usage:  bash install.sh [--backend nvidia|amd|intel|apple|cpu]
#                         [--dir PATH] [--no-launch] [--help]
# ============================================================
set -Eeuo pipefail

REPO_URL="https://github.com/Vxtzq/CrowdGPT.git"
REPO_DIR="CrowdGPT"
BACKEND_OVERRIDE=""
LAUNCH=1

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
        --backend)   BACKEND_OVERRIDE="${2:-}"; shift 2 ;;
        --dir)       REPO_DIR="${2:-}"; shift 2 ;;
        --no-launch) LAUNCH=0; shift ;;
        --help|-h)
            cat <<EOF
CrowdGPT installer

Options:
  --backend <name>   Force backend: nvidia | amd | intel | apple | cpu
  --dir <path>       Directory to clone into (default: ./CrowdGPT)
  --no-launch        Install only, don't launch client.py
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
    if [[ "$(uname -s)" == "Linux" ]]; then
        if   command -v apt-get >/dev/null 2>&1; then run_as_root apt-get update && run_as_root apt-get install -y git
        elif command -v dnf     >/dev/null 2>&1; then run_as_root dnf install -y git
        elif command -v pacman  >/dev/null 2>&1; then run_as_root pacman -Sy --noconfirm git
        elif command -v zypper  >/dev/null 2>&1; then run_as_root zypper --non-interactive install git
        else die "Unsupported Linux package manager. Install git manually."
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
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
# 3. Clone / update repo
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
# Sets: BACKEND, TORCH_INDEX, TORCH_EXTRA (optional), WHEEL_NOTE
detect_backend() {
    BACKEND=""
    TORCH_INDEX=""
    TORCH_EXTRA=""
    WHEEL_NOTE=""

    local OS ARCH
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    # ---------- Apple ----------
    if [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            BACKEND="apple"
            TORCH_INDEX="https://pypi.org/simple"    # MPS is bundled
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

        # Jetson: special wheels — warn explicitly
        if [[ "$ARCH" == "aarch64" ]]; then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="Jetson (aarch64) — you may need NVIDIA's Jetson-specific wheels instead: https://developer.nvidia.com/embedded/pytorch"
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
# 5. Virtual environment
# ============================================================
if [[ ! -x ".venv/bin/python" ]]; then
    log "Creating .venv..."
    if ! "$PYTHON_EXE" -m venv .venv 2>/dev/null; then
        install_uv
        uv venv .venv
    fi
fi
PYTHON_EXE="$PWD/.venv/bin/python"
[[ -x "$PYTHON_EXE" ]] || die "venv Python missing."
log "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip >/dev/null

# ============================================================
# 6. Install PyTorch from the correct index
# ============================================================
hdr "Installing PyTorch"

# Clean any pre-existing torch so we never end up with a mixed install.
"$PYTHON_EXE" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

PIP_TORCH_ARGS=(
    install
    torch torchvision torchaudio
    --index-url "$TORCH_INDEX"
)
if [[ -n "$TORCH_EXTRA" ]]; then
    PIP_TORCH_ARGS+=( --extra-index-url "$TORCH_EXTRA" )
fi

log "pip ${PIP_TORCH_ARGS[*]}"
if ! "$PYTHON_EXE" -m pip "${PIP_TORCH_ARGS[@]}"; then
    die "PyTorch install failed from $TORCH_INDEX"
fi

# ============================================================
# 7. Install the rest of requirements.txt (torch lines stripped)
# ============================================================
if [[ -f requirements.txt ]]; then
    hdr "Installing project dependencies"
    TMP_REQ="$(mktemp -t crowdgpt-req.XXXXXX.txt)"
    # Strip torch/torchvision/torchaudio so pip never overrides our wheel.
    grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]*[<>=!~].*)?$' \
        requirements.txt > "$TMP_REQ" || true
    if [[ -s "$TMP_REQ" ]]; then
        "$PYTHON_EXE" -m pip install -r "$TMP_REQ"
    else
        warn "requirements.txt contained only torch packages; nothing to install."
    fi
    rm -f "$TMP_REQ"
else
    warn "requirements.txt not found; skipping project deps."
fi

# ============================================================
# 8. Verify
# ============================================================
hdr "Verifying install"
"$PYTHON_EXE" - <<'PYEOF'
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
# 9. Launch
# ============================================================
echo
echo "============================================================"
echo "             Installation complete  ✓"
echo "============================================================"
echo "  Backend: $BACKEND"
echo "  Venv:    .venv"
echo

if [[ "$LAUNCH" -eq 1 ]]; then
    echo "Launching client.py..."
    echo
    exec "$PYTHON_EXE" client.py
else
    echo "To launch later:  $PYTHON_EXE client.py"
fi
