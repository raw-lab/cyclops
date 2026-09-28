#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install a desktop launcher (application-menu entry + icon) for Cyclops.
#
# Works on regular Linux desktops and on WSLg (where it adds Cyclops to the
# Windows Start Menu under "Ubuntu"). On WSL the launcher uses the
# software-rendering wrapper so it Just Works.
#
# Usage:   ./scripts/install-desktop.sh
# Remove:  rm ~/.local/share/applications/cyclops.desktop
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- find the GUI binary ---------------------------------------------------
if command -v cyclops-gui >/dev/null 2>&1; then
    GUI_BIN="$(command -v cyclops-gui)"
elif [[ -x "$HOME/.cargo/bin/cyclops-gui" ]]; then
    GUI_BIN="$HOME/.cargo/bin/cyclops-gui"
elif [[ -x "$ROOT/target/release/cyclops-gui" ]]; then
    GUI_BIN="$ROOT/target/release/cyclops-gui"
else
    echo "error: cyclops-gui not found. Build or install it first." >&2
    exit 1
fi

# --- detect WSL and choose the right Exec line -----------------------------
WSL_LAUNCHER="$ROOT/scripts/cyclops-gui-wsl.sh"
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null && [[ -x "$WSL_LAUNCHER" ]]; then
    EXEC_LINE="$WSL_LAUNCHER"
    echo "Detected WSL — using the software-rendering launcher."
else
    EXEC_LINE="$GUI_BIN"
fi

# --- icon ------------------------------------------------------------------
ICON_SRC="$ROOT/assets/cyclops.svg"
ICON_DIR="$HOME/.local/share/icons"
mkdir -p "$ICON_DIR"
if [[ -f "$ICON_SRC" ]]; then
    cp "$ICON_SRC" "$ICON_DIR/cyclops.svg"
    ICON_PATH="$ICON_DIR/cyclops.svg"
else
    ICON_PATH="applications-science"   # fall back to a stock themed icon
fi

# --- write the .desktop entry ----------------------------------------------
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"
DESKTOP_FILE="$APP_DIR/cyclops.desktop"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Cyclops
GenericName=Epifluorescence Particle Quantifier
Comment=Size and count viruses, bacteria, archaea and protists in epifluorescence microscopy
Exec=$EXEC_LINE
Icon=$ICON_PATH
Terminal=false
Categories=Science;Biology;Education;ImageProcessing;
Keywords=microscopy;epifluorescence;VLP;phage;virus;bacteria;
StartupNotify=true
StartupWMClass=cyclops-gui
EOF

chmod +x "$DESKTOP_FILE"

# --- refresh the menu database ---------------------------------------------
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APP_DIR" 2>/dev/null || true
fi

echo "Installed desktop launcher:"
echo "  $DESKTOP_FILE"
echo "  Exec = $EXEC_LINE"
echo "  Icon = $ICON_PATH"
echo
echo "Cyclops should now appear in your application menu under Science/Education."
echo "On WSLg it also shows up in the Windows Start Menu under 'Ubuntu'."
echo
echo "To remove it later:  rm \"$DESKTOP_FILE\""
