#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

has_intel_gpu() {
    command -v sycl-ls >/dev/null 2>&1 && return 0
    for v in /sys/class/drm/card*/device/vendor; do
        [ -f "$v" ] && grep -qi '0x8086' "$v" && return 0
    done
    return 1
}

has_nvidia_runtime() {
    command -v nvidia-container-runtime >/dev/null 2>&1 \
        || command -v nvidia-ctk >/dev/null 2>&1 \
        || return 1
    grep -qs 'nvidia' /etc/docker/daemon.json "${HOME:-}/.config/docker/daemon.json" 2>/dev/null && return 0
    docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q 'nvidia' && return 0
    return 1
}

usage() {
    echo "Usage: $0 [auto|cuda|cuda-noruntime|rocm|xpu|cpu] <docker compose args...>" >&2
    echo "Example: $0 auto up -d" >&2
    exit 2
}

[ "$#" -gt 0 ] || usage

backend="$1"
case "$backend" in
    auto|cuda|cuda-noruntime|rocm|xpu|cpu)
        shift
        [ "$#" -gt 0 ] || usage
        ;;
    build|up|down|logs|run|config|ps|exec|pull|push|images|restart|stop|start|rm|top|events|kill|pause|unpause|port|ls|version)
        # User passed a Compose subcommand without a backend; default to auto.
        backend=auto
        ;;
    *)
        usage
        ;;
esac

if [ "$backend" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 || [ -e /proc/driver/nvidia/version ]; then
        backend=cuda
    elif [ -e /dev/kfd ]; then
        backend=rocm
    elif has_intel_gpu; then
        backend=xpu
    else
        backend=cpu
    fi
fi

if [ "$backend" = "cuda" ]; then
    case "${CROWDGPT_CUDA_NORUNTIME:-auto}" in
        1|true|yes) backend=cuda-noruntime ;;
        0|false|no) backend=cuda ;;
        *)
            if ! has_nvidia_runtime; then
                backend=cuda-noruntime
            fi
            ;;
    esac
fi

exec docker compose -f docker-compose.yml -f "docker-compose.${backend}.yml" "$@"
