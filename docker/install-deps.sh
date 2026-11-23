#!/usr/bin/env bash
set -euo pipefail

BACKEND="cuda"
TORCH_INDEX=""
TORCH_SPEC="torch"
BITSANDBYTES="auto"
BNB_ROCM_ARCH=""
BNB_REF="0.50.2"

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)      BACKEND="$2"; shift 2 ;;
        --torch-index)  TORCH_INDEX="$2"; shift 2 ;;
        --torch-spec)   TORCH_SPEC="$2"; shift 2 ;;
        --bitsandbytes) BITSANDBYTES="$2"; shift 2 ;;
        --bnb-rocm-arch) BNB_ROCM_ARCH="$2"; shift 2 ;;
        --bnb-ref)      BNB_REF="$2"; shift 2 ;;
        *) echo "[deps] unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$BACKEND" in
    cuda|rocm|xpu|cpu) ;;
    *) echo "[deps] ERROR: --backend must be one of cuda|rocm|xpu|cpu (got '$BACKEND')" >&2; exit 2 ;;
esac

log() { printf '[deps] %s\n' "$*"; }

# Distro-managed pip (e.g. Ubuntu system Python in rocm/pytorch 10.x) cannot be uninstalled;
# keep going with the existing pip rather than failing the build.
python -m pip install --upgrade pip \
    || log "WARNING: could not upgrade pip; continuing with $(python -m pip --version)"

# 1. Reuse the base image's PyTorch if it is >= 2.6 and matches the requested backend.
torch_ok() {
    python - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
try:
    import torch
    v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if v < (2, 6):
        sys.exit(1)
    hip = bool(getattr(torch.version, "hip", None))
    cuda_ver = getattr(torch.version, "cuda", None)
    if backend == "rocm" and not hip:
        sys.exit(1)
    if backend == "cuda" and (hip or not cuda_ver):
        sys.exit(1)
    if backend == "xpu" and not hasattr(torch, "xpu"):
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

if torch_ok; then
    log "reusing base image torch: $(python -c 'import torch; print(torch.__version__)')"
else
    log "installing $TORCH_SPEC (index: ${TORCH_INDEX:-PyPI})"
    # shellcheck disable=SC2086
    if [ -n "$TORCH_INDEX" ]; then
        python -m pip install --upgrade $TORCH_SPEC --index-url "$TORCH_INDEX"
    else
        python -m pip install --upgrade $TORCH_SPEC
    fi
fi

# 2. Install runtime requirements, stripping torch* and pywebview (unused in --headless mode).
grep -Ev '^\s*(torch|torchvision|torchaudio|torch-directml|pywebview)(\s*[<>=!~].*)?$' \
    /tmp/requirements.txt > /tmp/requirements.docker.txt
python -m pip install -r /tmp/requirements.docker.txt

# 3. Install bitsandbytes only on CUDA/ROCm (client.py gates 8-bit AdamW to CUDA and ROCm).
#    "source" builds it for the given ROCm GPU arch(s), for ROCm releases or GPUs the
#    PyPI wheel does not ship kernels for (e.g. ROCm 10.x, gfx1201).
build_bnb_from_source() {
    if [ "$BACKEND" != "rocm" ]; then
        log "ERROR: --bitsandbytes source is only supported for --backend rocm" >&2; exit 2
    fi
    local missing=()
    command -v git   >/dev/null 2>&1 || missing+=(git)
    command -v cmake >/dev/null 2>&1 || missing+=(cmake)
    command -v make  >/dev/null 2>&1 || missing+=(make)
    command -v g++   >/dev/null 2>&1 || missing+=(g++)
    if [ "${#missing[@]}" -gt 0 ]; then
        log "installing build tools: ${missing[*]}"
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
        rm -rf /var/lib/apt/lists/*
    fi
    # ROCm may come from pip wheels (rocm-sdk, as in rocm/pytorch 10.x) instead of /opt/rocm.
    local hip_args=()
    # An empty arch must not be passed: CMake would see BNB_ROCM_ARCH defined but blank.
    # Without it, bitsandbytes builds its own default list of supported GPUs.
    if [ -n "$BNB_ROCM_ARCH" ]; then
        hip_args+=(-DBNB_ROCM_ARCH="$BNB_ROCM_ARCH")
    fi
    if [ -z "${ROCM_PATH:-}" ] && [ ! -d /opt/rocm ] && command -v rocm-sdk >/dev/null 2>&1; then
        ROCM_PATH="$(rocm-sdk path --root)"
        export ROCM_PATH
        export PATH="$ROCM_PATH/bin:$PATH"
        hip_args+=(-DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)")
    fi
    if [ -n "${ROCM_PATH:-}" ] && [ -x "$ROCM_PATH/lib/llvm/bin/clang++" ]; then
        hip_args+=(-DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++")
    fi
    log "ROCm for bitsandbytes build: ${ROCM_PATH:-/opt/rocm} ($(hipconfig --version 2>/dev/null || echo 'hipconfig not found'))"
    local src
    src="$(mktemp -d)"
    log "building bitsandbytes ${BNB_REF} from source for ${BNB_ROCM_ARCH:-bitsandbytes default GPU list}"
    git clone --depth 1 --branch "$BNB_REF" \
        https://github.com/bitsandbytes-foundation/bitsandbytes.git "$src"
    cmake -S "$src" -B "$src/build" -DCOMPUTE_BACKEND=hip \
        "${hip_args[@]}"
    cmake --build "$src/build" -j"$(nproc)"
    if ! ls "$src"/bitsandbytes/libbitsandbytes_rocm*.so >/dev/null 2>&1; then
        log "ERROR: CMake finished but no libbitsandbytes_rocm*.so was produced" >&2; exit 1
    fi
    log "built $(basename "$src"/bitsandbytes/libbitsandbytes_rocm*.so)"
    # The native library is already built above; package it without re-running CMake.
    BNB_SKIP_CMAKE=1 python -m pip install "$src"
    rm -rf "$src"
}

if [ "$BITSANDBYTES" = "source" ]; then
    build_bnb_from_source
elif [ "$BITSANDBYTES" = "1" ] \
   || { [ "$BITSANDBYTES" = "auto" ] && { [ "$BACKEND" = "cuda" ] || [ "$BACKEND" = "rocm" ]; }; }; then
    log "installing bitsandbytes"
    python -m pip install bitsandbytes \
        || log "WARNING: bitsandbytes install failed; client will fall back to fp32 AdamW"
fi

log "finished: torch $(python -c 'import torch; print(torch.__version__)')"
