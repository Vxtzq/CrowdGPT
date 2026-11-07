#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/app
DATA_DIR="${CROWDGPT_DATA_DIR:-/data}"
USER_NAME=crowd
SELF=/usr/local/bin/docker-entrypoint.sh

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# Unset empty hardware-selector variables so CUDA/HIP/SYCL/OpenMP do not treat "" as 0 devices/threads.
for _var in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
            HSA_OVERRIDE_GFX_VERSION ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK OMP_NUM_THREADS; do
    if [ -z "${!_var:-}" ]; then
        unset "$_var"
    fi
done
unset _var

if [ "$(id -u)" = "0" ]; then
    # 1. Optional NVIDIA device node creation for hosts without nvidia-container-toolkit.
    if [ "${CROWDGPT_MKNOD_NVIDIA:-0}" = "1" ] && [ -e /proc/driver/nvidia/version ]; then
        nv_major=$(awk '$2=="nvidia" {print $1; exit}' /proc/devices)
        uvm_major=$(awk '$2=="nvidia-uvm" {print $1; exit}' /proc/devices)
        if [ -n "$nv_major" ]; then
            [ -e /dev/nvidiactl ] || mknod -m 666 /dev/nvidiactl c "$nv_major" 255 || true
            [ -e /dev/nvidia0 ]   || mknod -m 666 /dev/nvidia0 c "$nv_major" 0 || true
        fi
        if [ -n "$uvm_major" ]; then
            [ -e /dev/nvidia-uvm ]       || mknod -m 666 /dev/nvidia-uvm c "$uvm_major" 0 || true
            [ -e /dev/nvidia-uvm-tools ] || mknod -m 666 /dev/nvidia-uvm-tools c "$uvm_major" 1 || true
        fi
    fi

    # 2. Match host GPU device GIDs (/dev/kfd, /dev/dri/*) so `gosu` preserves device access.
    for dev in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
        [ -e "$dev" ] || continue
        gid=$(stat -c '%g' "$dev" 2>/dev/null || true)
        [ -n "$gid" ] && [ "$gid" != "0" ] || continue
        grp=$(getent group "$gid" | cut -d: -f1 || true)
        if [ -z "$grp" ]; then
            grp="hostgpu_${gid}"
            groupadd -g "$gid" "$grp" 2>/dev/null || true
        fi
        usermod -aG "$grp" "$USER_NAME" 2>/dev/null || true
    done

    # 3. Ensure persistent directories and /app/checkpoints -> /data/checkpoints symlink exist.
    mkdir -p "$DATA_DIR/checkpoints" "$DATA_DIR/hf_home" \
             "$DATA_DIR/torch_home" "$DATA_DIR/cuda_cache" "$DATA_DIR/cache"
    if [ ! -L "$APP_DIR/checkpoints" ]; then
        rm -rf "$APP_DIR/checkpoints"
        ln -sfn "$DATA_DIR/checkpoints" "$APP_DIR/checkpoints"
    fi

    chown -R "$USER_NAME:$USER_NAME" "$DATA_DIR" "/home/$USER_NAME" 2>/dev/null || true

    exec gosu "$USER_NAME" "$SELF" "$@"
fi

cd "$APP_DIR"

# If the caller passed an executable command (python, bash, nvidia-smi, rocminfo, ...), run it.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ] && command -v "$1" >/dev/null 2>&1; then
    exec "$@"
fi

# Non-fatal device sanity check.
backend="${CROWDGPT_BACKEND:-auto}"
case "$backend" in
    cuda) [ -e /dev/nvidia0 ] || log "WARNING: /dev/nvidia0 missing; CUDA may not be available" ;;
    rocm)
        [ -e /dev/kfd ] || log "WARNING: /dev/kfd missing; ROCm will not work"
        [ -e /dev/dri ] || log "WARNING: /dev/dri missing; ROCm will not work"
        ;;
    xpu)  [ -e /dev/dri ] || log "WARNING: /dev/dri missing; XPU will not work" ;;
esac

args=(python client.py --headless)

[ -n "${CROWDGPT_SERVER:-}" ] && args+=(--server "$CROWDGPT_SERVER")
if [ -n "${CROWDGPT_BACKEND:-}" ] && [ "$CROWDGPT_BACKEND" != "auto" ]; then
    args+=(--backend "$CROWDGPT_BACKEND")
fi
[ -n "${CROWDGPT_VRAM_GB:-}" ]          && args+=(--vram-gb "$CROWDGPT_VRAM_GB")
[ -n "${CROWDGPT_ROUNDS:-}" ]           && args+=(--rounds "$CROWDGPT_ROUNDS")
[ -n "${CROWDGPT_USER:-}" ]             && args+=(--user "$CROWDGPT_USER")
[ "${CROWDGPT_NO_UPDATE:-1}" = "1" ]    && args+=(--no-update)
[ "${CROWDGPT_NO_TOKENIZER:-1}" = "1" ] && args+=(--no-tokenizer)
[ "${CROWDGPT_NO_TUI:-0}" = "1" ]       && args+=(--no-tui)
[ "${CROWDGPT_DEBUG:-0}" = "1" ]        && args+=(--debug)

# Fall back to anonymous training only when no token, user, or saved session exists.
if [ -z "${CROWDGPT_TOKEN:-}" ] && [ -z "${CROWDGPT_USER:-}" ] \
   && [ "${CROWDGPT_ANON:-1}" = "1" ] \
   && [ ! -f "$DATA_DIR/checkpoints/auth_token.json" ]; then
    args+=(--anon)
fi

if [ -n "${CROWDGPT_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    args+=(${CROWDGPT_EXTRA_ARGS})
fi

exec "${args[@]}" "$@"
