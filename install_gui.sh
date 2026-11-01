#!/usr/bin/env bash
# ============================================================
# CrowdGPT GUI bootstrap installer — v3
#
# Flow:
#   1. Ensure git + uv
#   2. Clone / sync repo
#   3. Download Python 3.11 (python-build-standalone, no compile)
#   4. Installer venv + WebView (PyQt6 + WebEngine + qtpy)
#   5. Verify the Qt backend actually imports
#   6. Launch installer.py (GUI) or fall back to headless
#
# v3 vs v2:
#   - install qtpy (pywebview's Qt backend imports it, pip does not
#     pull it in automatically) and pywebview[qt]
#   - import-check QtWebEngine BEFORE launching installer.py, so a
#     broken backend doesn't crash the GUI at start()
#   - if the backend can't load, degrade to --headless automatically
#     (with a Debian/Ubuntu lib hint) instead of crashing
#   - headless decision made once, before the launch command
# ============================================================

# Guard 1: must be bash, not sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: This script requires bash, not sh."
    echo "  Run:  bash $0 $*"
    exit 1
fi

# Guard 2: CRLF line endings break every shell parse. Fix in place.
if grep -q $'\r' "$0" 2>/dev/null; then
    echo "WARNING: $0 has Windows (CRLF) line endings — fixing..."
    sed -i 's/\r$//' "$0" 2>/dev/null || {
        echo "ERROR: Could not auto-fix CRLF. Run: sed -i 's/\\r$//' $0"
        exit 1
    }
    echo "Fixed. Re-executing..."
    exec bash "$0" "$@"
fi

set -Eeuo pipefail

REPO_URL="https://github.com/Vxtzq/CrowdGPT.git"
REPO_DIR="CrowdGPT"
BACKEND_OVERRIDE=""
LAUNCH=1
INTEGRATE=1
HEADLESS=0
DEBUG=0
KEEP_OPEN=0

if [ -t 1 ]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

# Defined before trap/die because they both call it.
pause_if_interactive() {
    if [[ -t 0 ]]; then
        echo
        read -r -p "Press Enter to close..." _ || true
    fi
}

log()  { printf '\n%s[INFO]%s %s\n' "$BLUE" "$RESET" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%s[FAIL]%s %s\n' "$RED" "$RESET" "$*" >&2; pause_if_interactive; exit 1; }

trap 'rc=$?; printf "\n%s[FAIL]%s failed near line %s (exit %s)\n" "$RED" "$RESET" "$LINENO" "$rc" >&2; pause_if_interactive; exit $rc' ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ -n "${2:-}" ]] || die "--backend requires a value (nvidia|amd|intel|apple|cpu)"
            BACKEND_OVERRIDE="$2"; shift 2 ;;
        --dir)
            [[ -n "${2:-}" ]] || die "--dir requires a path"
            REPO_DIR="$2"; shift 2 ;;
        --no-launch)    LAUNCH=0; shift ;;
        --no-integrate) INTEGRATE=0; shift ;;
        --headless)     HEADLESS=1; shift ;;
        --debug)        DEBUG=1; shift ;;
        --keep-open)    KEEP_OPEN=1; shift ;;
        --help|-h)
            cat <<EOF
CrowdGPT installer (GUI bootstrap)

Options:
  --backend <name>   Force: nvidia | amd | intel | apple | cpu
  --dir <path>       Directory to install into (default: ./CrowdGPT)
  --no-launch        Install only, don't start the client
  --no-integrate     Skip desktop / command integration
  --headless         Skip GUI, use text-mode installer
  --debug            Enable set -x tracing
  --keep-open        Pause before exit even on success
EOF
            exit 0
            ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

if [[ "$DEBUG" -eq 1 ]]; then
    set -x
fi

echo
echo "============================================================"
echo "               CrowdGPT Installer (GUI)"
echo "============================================================"
echo

# ── 1. git ──
if ! command -v git >/dev/null 2>&1; then
    log "Installing git..."
    if [[ "$(uname -s)" == "Darwin" ]]; then
        command -v brew >/dev/null 2>&1 && brew install git || die "Install git and rerun"
    else
        if   command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y git
        elif command -v dnf     >/dev/null 2>&1; then sudo dnf install -y git
        elif command -v pacman  >/dev/null 2>&1; then sudo pacman -Sy --noconfirm git
        elif command -v zypper  >/dev/null 2>&1; then sudo zypper --non-interactive install git
        else die "Unsupported package manager. Install git manually and rerun."
        fi
    fi
fi
ok "git: $(git --version | awk '{print $3}')"

# ── 2. clone / sync ──
if [[ -f "client.py" && -d ".git" ]]; then
    REPO_DIR="."
elif [[ -d "$REPO_DIR/.git" ]]; then
    :
else
    [[ -e "$REPO_DIR" ]] && die "$REPO_DIR exists but is not a git repo."
    log "Cloning $REPO_URL into $REPO_DIR ..."
    git clone "$REPO_URL" "$REPO_DIR" || die "git clone failed"
fi
cd "$REPO_DIR"

BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo main)"
log "Syncing to origin/$BRANCH..."
git fetch --all --prune --tags || die "git fetch failed"
git reset --hard "origin/$BRANCH" || die "git reset failed"
ok "Synced to $(git log -1 --format='%h %s')"

# ── 3. installer.py must exist ──
if [[ ! -f installer.py ]]; then
    die "installer.py not found in $PWD

  The repo must contain installer.py. Fix with one of:
    git -C '$PWD' pull origin $BRANCH
    git -C '$PWD' log --oneline origin/$BRANCH | head    # verify it's committed upstream
    rm -rf '$PWD' && bash $0 --dir '$REPO_DIR'          # fresh clone"
fi
ok "installer.py found"

# ── 4. uv ──
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Need curl or wget to bootstrap uv"
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH"
ok "uv: $(uv --version | awk '{print $2}')"

# ── 5. Python 3.11 (python-build-standalone tarball, no compile) ──
log "Fetching Python 3.11 ..."
uv python install 3.11 >/dev/null 2>&1 || die "Failed to download Python 3.11 via uv"
ok "Python 3.11 ready"

# ── 6. Installer venv + WebView ──
INSTALLER_VENV="$PWD/.installer-venv"
log "Creating installer venv at $INSTALLER_VENV ..."
rm -rf "$INSTALLER_VENV"
uv venv --python 3.11 --clear "$INSTALLER_VENV" >/dev/null 2>&1 \
    || die "Failed to create installer venv"

log "Downloading WebView runtime (PyQt6 + WebEngine + qtpy, ~120 MB)..."
uv pip install --python "$INSTALLER_VENV/bin/python" --quiet \
    pywebview "pywebview[qt]" qtpy pyqt6 pyqt6-webengine \
    || die "Failed to install WebView dependencies"

# ── 7. Verify the Qt backend actually imports ──
# pywebview[qt] pulls qtpy transitively, but if anything ever regresses
# (a wheel resolution change, a bad tag), catch it here with a clear
# message instead of letting installer.py crash on webview.start().
WEBVIEW_OK=0
if "$INSTALLER_VENV/bin/python" - <<'PY' >/dev/null 2>&1
from qtpy import QtCore
from PyQt6 import QtWidgets, QtWebEngineWidgets
import webview.platforms.qt
print("qt-ok")
PY
then
    WEBVIEW_OK=1
    ok "Qt WebEngine OK"
else
    warn "Qt WebEngine could not load in this environment."
    if [[ "$(uname -s)" == "Linux" ]]; then
        warn "Common missing system libs on Debian / Ubuntu:"
        warn "  sudo apt install -y \\"
        warn "      libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \\"
        warn "      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \\"
        warn "      libxcb-xinerama0 libxcb-xkb1 libegl1 libgl1"
        warn "On Fedora: sudo dnf install xcb-util-cursor xcb-util-wm xcb-util-image \\"
        warn "      xcb-util-keysyms xcb-util-renderutil libxkbcommon-x11 mesa-libEGL"
    fi
    warn "Continuing in headless mode (full install still runs)."
fi

# ── 8. Decide GUI vs headless ──
ARGS=( --repo-dir "$PWD" )
[[ -n "$BACKEND_OVERRIDE" ]] && ARGS+=( --backend "$BACKEND_OVERRIDE" )
[[ "$LAUNCH"    -eq 0 ]] && ARGS+=( --no-launch )
[[ "$INTEGRATE" -eq 0 ]] && ARGS+=( --no-integrate )

if [[ "$HEADLESS" -eq 0 ]]; then
    if [[ "$WEBVIEW_OK" -ne 1 ]]; then
        warn "No working WebView backend — switching to headless mode."
        HEADLESS=1
    elif [[ "$(uname -s)" != "Darwin" ]] \
        && [[ -z "${DISPLAY:-}" ]] \
        && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
        warn "No display detected — switching to headless mode."
        HEADLESS=1
    fi
fi
[[ "$HEADLESS" -eq 1 ]] && ARGS+=( --headless )

# ── 9. Launch ──
echo
log "Launching: $INSTALLER_VENV/bin/python $PWD/installer.py ${ARGS[*]}"
echo

# CRITICAL: no exec. If python fails, bash stays alive to print + pause.
"$INSTALLER_VENV/bin/python" "$PWD/installer.py" "${ARGS[@]}"
rc=$?

if [[ $rc -ne 0 ]]; then
    echo
    warn "installer.py exited with code $rc"
    warn "Re-run in debug mode for a detailed trace:"
    warn "  bash $0 --debug --keep-open ${ARGS[*]}"
    warn "Or headless (writes readable log even without a display):"
    warn "  bash $0 --headless ${ARGS[*]} 2>&1 | tee /tmp/crowdgpt-install.log"
fi

if [[ $rc -ne 0 ]] || [[ "$KEEP_OPEN" -eq 1 ]]; then
    pause_if_interactive
fi

exit $rc
