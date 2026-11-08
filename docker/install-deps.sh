#!/usr/bin/env bash
set -euo pipefail

BACKEND="cuda"
TORCH_INDEX=""
TORCH_SPEC="torch"
BITSANDBYTES="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)      BACKEND="$2"; shift 2 ;;
        --torch-index)  TORCH_INDEX="$2"; shift 2 ;;
        --torch-spec)   TORCH_SPEC="$2"; shift 2 ;;
        --bitsandbytes) BITSANDBYTES="$2"; shift 2 ;;
        *) echo "[deps] unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$BACKEND" in
    cuda|rocm|xpu|cpu) ;;
    *) echo "[deps] ERROR: --backend must be one of cuda|rocm|xpu|cpu (got '$BACKEND')" >&2; exit 2 ;;
esac

log() { printf '[deps] %s\n' "$*"; }

python -m pip install --upgrade pip

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
if [ "$BITSANDBYTES" = "1" ] \
   || { [ "$BITSANDBYTES" = "auto" ] && { [ "$BACKEND" = "cuda" ] || [ "$BACKEND" = "rocm" ]; }; }; then
    log "installing bitsandbytes"
    python -m pip install bitsandbytes \
        || log "WARNING: bitsandbytes install failed; client will fall back to fp32 AdamW"
fi

log "finished: torch $(python -c 'import torch; print(torch.__version__)')"
