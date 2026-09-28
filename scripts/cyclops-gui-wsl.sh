#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cyclops GUI launcher for WSL (Windows Subsystem for Linux).
#
# WSL2 on Windows 11 (and recent Windows 10) ships WSLg, which provides a
# Wayland + X11 display server automatically. But the virtual GPU often
# can't run the OpenGL that eframe/egui wants, so we force Mesa's pure-CPU
# "llvmpipe" software renderer. This is slower to draw but rock-solid and
# needs no GPU passthrough.
#
# Usage:   ./scripts/cyclops-gui-wsl.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the binary -----------------------------------------------------
# Prefer an installed copy on PATH, fall back to a local release build.
if command -v cyclops-gui >/dev/null 2>&1; then
    BIN="$(command -v cyclops-gui)"
elif [[ -x "$HOME/.cargo/bin/cyclops-gui" ]]; then
    BIN="$HOME/.cargo/bin/cyclops-gui"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    for cand in \
        "$ROOT/target/release/cyclops-gui" \
        "$ROOT/target/release-small/cyclops-gui" \
        "$ROOT/target/debug/cyclops-gui"; do
        if [[ -x "$cand" ]]; then BIN="$cand"; break; fi
    done
fi

if [[ -z "${BIN:-}" ]]; then
    echo "error: could not find the cyclops-gui binary." >&2
    echo "       build it first with:  cargo build --release -p cyclops-gui" >&2
    echo "       or install it with:   cargo install --path cyclops-gui --locked" >&2
    exit 1
fi

# --- check we actually have a display --------------------------------------
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    cat >&2 <<'MSG'
error: no display server detected (neither DISPLAY nor WAYLAND_DISPLAY is set).

  On WSL this almost always means WSLg is not active. Fix it from a
  *Windows* PowerShell / Command Prompt (not inside WSL):

      wsl --update
      wsl --shutdown

  then reopen your Ubuntu terminal and try again. WSLg requires WSL2 on
  Windows 11, or Windows 10 build 19044+ with the latest WSL.

  Check your WSL version from PowerShell with:  wsl --version
MSG
    exit 1
fi

# --- force software OpenGL (the WSL fix) -----------------------------------
export LIBGL_ALWAYS_SOFTWARE=1     # Mesa: use CPU rendering, ignore the vGPU
export GALLIUM_DRIVER=llvmpipe     # Mesa: pick the llvmpipe software driver
export WINIT_UNIX_BACKEND=x11      # prefer X11 over Wayland (more reliable on WSLg)

# Some WSLg setups also need this to stop a Wayland probe from hanging:
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"

echo "Launching Cyclops GUI (software rendering) ..."
echo "  binary  : $BIN"
echo "  DISPLAY : ${DISPLAY:-<wayland>}"
exec "$BIN" "$@"
