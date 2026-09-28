#!/usr/bin/env bash
# Cyclops demo runner — Rust equivalent of EpiVirQuant's `runDemo.sh`.
#
# Builds the release binaries (once) then invokes `cyclops` against the
# bundled GSL example data with the original EpiVirQuant defaults.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found. Install Rust from https://rustup.rs and retry." >&2
  exit 1
fi

echo "▶ building Cyclops (release)…"
cargo build --release --bin cyclops

BIN="$ROOT/target/release/cyclops"

DAPI_DIR="${CYCLOPS_DAPI_DIR:-data/GSL/tiff/dapi}"
FITC_DIR="${CYCLOPS_FITC_DIR:-data/GSL/tiff/fitc}"
CAL="${CYCLOPS_CAL:-$DAPI_DIR/GSL_+_blue_beads_1_(dapi).tiff}"
OUT="${CYCLOPS_OUT:-Cyclops_Output_demo}"

echo "▶ running Cyclops on the GSL demo data…"
"$BIN" \
  --dapi        "$DAPI_DIR" \
  --fitc        "$FITC_DIR" \
  --calibration "$CAL" \
  --scaleLength 585 \
  --scaleMetric 20000 \
  --sphereSize  175 \
  --psfMethod   gam \
  --domains     virus,bacteria \
  --outDir      "$OUT" \
  -v

echo "✓ done. results in $OUT/"
