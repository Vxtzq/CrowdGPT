#!/usr/bin/env bash
set -Eeuo pipefail

# CrowdGPT universal installer / launcher for Linux + macOS.
# Detects Python + Git + GPU backend, installs the matching requirements,
# and launches client.py.

REPO_URL="https://github.com/Vxtzq/CrowdGPT.git"
REPO_DIR="CrowdGPT"
PYTHON_EXE=""
REQ_FILE=""
BACKEND=""

log()  { printf '\n[INFO] %s\n' "$*"; }
ok()   { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die()  { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

trap 'printf "\n[ERROR] Installer failed near line %s.\n" "$LINENO" >&2' ERR

echo
echo "============================================================"
echo "                CrowdGPT Installer"
echo "============================================================"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
run_as_root() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "This operation needs root privileges, but sudo is unavailable."
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    log "Python is unavailable. Installing uv so it can download the latest stable Python..."

    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Neither curl nor wget is installed; cannot install uv."
    fi

    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    command -v uv >/dev/null 2>&1 || die "uv was installed but is not in PATH."
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python)"
        return
    fi

    install_uv
    log "Downloading the latest stable Python..."
    uv python install --default

    export PATH="$HOME/.local/bin:$PATH"
    PYTHON_EXE="$(uv python find)"
}

# ------------------------------------------------------------
# 1. Locate / install Python
# ------------------------------------------------------------
find_python
ok "Python: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ------------------------------------------------------------
# 2. Locate / install Git
# ------------------------------------------------------------
if ! command -v git >/dev/null 2>&1; then
    log "Git was not found. Installing Git..."

    if [[ "$(uname -s)" == "Linux" ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            run_as_root apt-get update
            run_as_root apt-get install -y git
        elif command -v dnf >/dev/null 2>&1; then
            run_as_root dnf install -y git
        elif command -v pacman >/dev/null 2>&1; then
            run_as_root pacman -Sy --noconfirm git
        elif command -v zypper >/dev/null 2>&1; then
            run_as_root zypper --non-interactive install git
        else
            die "No supported Linux package manager found. Install Git manually."
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install git
        else
            # This invokes Apple's command-line-tools installer if available.
            xcode-select --install >/dev/null 2>&1 || true
            die "Git is unavailable. Install Xcode Command Line Tools or Homebrew, then rerun."
        fi
    else
        die "Unsupported operating system."
    fi
fi

command -v git >/dev/null 2>&1 || die "Git is still unavailable."
ok "Git: $(git --version)"

# ------------------------------------------------------------
# 3. Clone / reuse CrowdGPT
# ------------------------------------------------------------
if [[ -f "client.py" && -d ".git" ]]; then
    REPO_DIR="."
elif [[ -d "$REPO_DIR/.git" ]]; then
    log "CrowdGPT already exists. Updating it..."
    if ! git -C "$REPO_DIR" pull --ff-only; then
        warn "git pull failed; continuing with the existing checkout."
    fi
else
    if [[ -e "$REPO_DIR" ]]; then
        die "$REPO_DIR exists but is not a Git repository. Move/delete it and rerun."
    fi

    log "Cloning CrowdGPT..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
[[ -f client.py ]] || die "client.py was not found in the CrowdGPT repository."

# ------------------------------------------------------------
# 4. Detect GPU / backend
# ------------------------------------------------------------
log "Detecting hardware backend..."

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # Apple Silicon / MPS. The repository currently does not ship
    # requirements_mps.txt, so requirements.txt is the safe fallback.
    if [[ "$(uname -m)" == "arm64" ]]; then
        BACKEND="mps"
        REQ_FILE="requirements_mps.txt"
    else
        BACKEND="cpu"
        REQ_FILE="requirements.txt"
    fi

elif [[ "$OS" == "Linux" ]]; then
    # NVIDIA
    if command -v nvidia-smi >/dev/null 2>&1; then
        BACKEND="cuda"
        REQ_FILE="requirements_cuda.txt"

    # AMD ROCm
    elif command -v rocminfo >/dev/null 2>&1 || \
         command -v rocm-smi >/dev/null 2>&1 || \
         [[ -d /opt/rocm ]]; then
        BACKEND="rocm"
        REQ_FILE="requirements_rocm.txt"

    # Intel oneAPI/XPU
    elif command -v sycl-ls >/dev/null 2>&1 || \
         [[ -d /opt/intel/oneapi ]] || \
         [[ -d /opt/intel/oneapi/compiler/latest/linux/lib/oclfpga ]] ; then
        BACKEND="xpu"
        REQ_FILE="requirements_xpu.txt"

    else
        BACKEND="cpu"
        REQ_FILE="requirements.txt"
    fi
else
    BACKEND="cpu"
    REQ_FILE="requirements.txt"
fi

# The current public CrowdGPT repository contains CUDA, DirectML,
# ROCm and CPU requirements. XPU/MPS are selected only if the
# corresponding files are actually present in a future checkout.
if [[ ! -f "$REQ_FILE" ]]; then
    warn "$REQ_FILE is not present in this repository."
    if [[ "$BACKEND" == "xpu" || "$BACKEND" == "mps" ]]; then
        warn "Falling back to requirements.txt. PyTorch can still expose MPS/XPU if the installed build supports it."
    else
        warn "Falling back to requirements.txt."
    fi
    BACKEND="cpu"
    REQ_FILE="requirements.txt"
fi

ok "Backend: $BACKEND"
ok "Requirements: $REQ_FILE"

# ------------------------------------------------------------
# 5. Create isolated virtual environment
# ------------------------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
    log "Creating virtual environment..."
    if ! "$PYTHON_EXE" -m venv .venv; then
        if ! command -v uv >/dev/null 2>&1; then
            install_uv
        fi
        uv venv .venv
    fi
fi

PYTHON_EXE="$PWD/.venv/bin/python"
[[ -x "$PYTHON_EXE" ]] || die "Virtual-environment Python was not created."

log "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

# ------------------------------------------------------------
# 6. Install requirements
# ------------------------------------------------------------
log "Installing CrowdGPT dependencies..."

if [[ "$BACKEND" == "rocm" ]]; then
    # CrowdGPT's ROCm requirements explicitly document the ROCm
    # PyTorch index because ROCm wheels are not normal PyPI wheels.
    if ! "$PYTHON_EXE" -m pip install -r "$REQ_FILE" \
        --index-url https://download.pytorch.org/whl/rocm6.0 \
        --extra-index-url https://pypi.org/simple; then
        warn "ROCm 6.0 wheel installation failed."
        warn "Your ROCm/PyTorch combination may need a different PyTorch ROCm index."
        exit 1
    fi
else
    "$PYTHON_EXE" -m pip install -r "$REQ_FILE"
fi

# ------------------------------------------------------------
# 7. Launch client.py
# ------------------------------------------------------------
echo
echo "============================================================"
echo "                Installation complete"
echo "============================================================"
echo "Backend: $BACKEND"
echo "Starting client.py..."
echo

exec "$PYTHON_EXE" client.py
