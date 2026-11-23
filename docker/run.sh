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

# ---------------------------------------------------------------------------
# ROCm: choose the base image and bitsandbytes GPU target from the host GPU.
# ---------------------------------------------------------------------------
ROCM_NEW_IMAGE="rocm/pytorch:rocm10.2.0_ubuntu24.04_py3.12_pytorch_release_2.15.0"
ROCM_NEW_TORCH_INDEX="https://download.pytorch.org/whl/rocm10.1"
# GPU targets the ROCm 10.2 image ships device libraries for (rocm-sdk-device-* packages).
ROCM_NEW_ARCHS=" gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx1250 "
ROCM_OLD_IMAGE="rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.6.0"
ROCM_OLD_TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"

# Value of KEY from the environment, else from ./.env (Compose gives the shell priority over
# .env, so exporting our own value must not shadow one the user put in .env).
env_or_dotenv() {
    local key="$1" val="${!1:-}"
    if [ -z "$val" ] && [ -f .env ]; then
        val="$(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=[[:space:]]*//p" .env | tail -n 1)"
        val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    fi
    printf '%s' "$val"
}

# KFD reports gfx_target_version as MMmmss (e.g. 120001); turn it into gfx1201 / gfx90a.
gfx_from_target_version() {
    local v="$1"
    printf 'gfx%d%d%x' $((v / 10000)) $(((v / 100) % 100)) $((v % 100))
}

# HSA_OVERRIDE_GFX_VERSION is "major.minor.stepping" (e.g. 10.3.0 -> gfx1030).
gfx_from_hsa_override() {
    local major minor step
    IFS=. read -r major minor step <<<"$1"
    printf 'gfx%d%d%x' "$major" "${minor:-0}" "${step:-0}"
}

# Space-separated, de-duplicated gfx targets of the GPUs the kernel exposes (CPU nodes report 0).
detect_rocm_gfx() {
    local f v out=" "
    for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$f" ] || continue
        v="$(awk '$1 == "gfx_target_version" { print $2 }' "$f")"
        case "$v" in ''|0|*[!0-9]*) continue ;; esac
        v="$(gfx_from_target_version "$v")"
        case "$out" in *" $v "*) ;; *) out="$out$v " ;; esac
    done
    out="${out# }"; printf '%s' "${out% }"
}

configure_rocm() {
    local override gfx g all_new=1
    override="$(env_or_dotenv HSA_OVERRIDE_GFX_VERSION)"
    if [ -n "$override" ]; then
        # The runtime will treat the GPU as this target, so build for it too.
        gfx="$(gfx_from_hsa_override "$override")"
    else
        gfx="$(detect_rocm_gfx)"
    fi

    for g in $gfx; do
        case "$ROCM_NEW_ARCHS" in *" $g "*) ;; *) all_new=0 ;; esac
    done

    local base torch_index arch reason
    base="$(env_or_dotenv BASE_IMAGE)"
    torch_index="$(env_or_dotenv TORCH_INDEX)"
    arch="$(env_or_dotenv BNB_ROCM_ARCH)"

    if [ -n "$base" ]; then
        reason="BASE_IMAGE set by you"
    elif [ -z "$gfx" ]; then
        base="$ROCM_NEW_IMAGE"; torch_index="${torch_index:-$ROCM_NEW_TORCH_INDEX}"
        reason="GPU not detected, defaulting to ROCm 10.2"
    elif [ "$all_new" = 1 ]; then
        base="$ROCM_NEW_IMAGE"; torch_index="${torch_index:-$ROCM_NEW_TORCH_INDEX}"
        reason="supported by ROCm 10.2"
    else
        base="$ROCM_OLD_IMAGE"; torch_index="${torch_index:-$ROCM_OLD_TORCH_INDEX}"
        reason="not in the ROCm 10.2 image, using ROCm 6.4"
    fi
    if [ -z "$arch" ] && [ -n "$gfx" ]; then
        arch="${gfx// /;}"
    fi

    export BASE_IMAGE="$base"
    [ -n "$torch_index" ] && export TORCH_INDEX="$torch_index"
    [ -n "$arch" ] && export BNB_ROCM_ARCH="$arch"

    echo "[run.sh] ROCm GPU: ${gfx:-unknown}${override:+ (HSA_OVERRIDE_GFX_VERSION=$override)} -> ${reason}" >&2
    echo "[run.sh]   base image: ${BASE_IMAGE}" >&2
    echo "[run.sh]   bitsandbytes arch: ${BNB_ROCM_ARCH:-bitsandbytes default list}" >&2
}

if [ "$backend" = "rocm" ]; then
    configure_rocm
fi

compose_files=(-f docker-compose.yml -f "docker-compose.${backend}.yml")

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    exec docker compose "${compose_files[@]}" "$@"
elif command -v podman-compose >/dev/null 2>&1; then
    # podman-compose ignores the service-level `ipc:` key, and ROCm crashes on the first
    # GPU allocation without host IPC, so pass it to `podman run` explicitly.
    podman_args=()
    if [ "$backend" = "rocm" ]; then
        podman_args+=(--podman-run-args=--ipc=host)
    fi
    exec podman-compose "${podman_args[@]}" "${compose_files[@]}" "$@"
else
    echo "Error: neither 'docker compose' nor 'podman-compose' was found." >&2
    exit 1
fi
