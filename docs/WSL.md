# Running Cyclops on WSL (Windows Subsystem for Linux)

The Cyclops **CLI** runs on WSL with zero fuss. The **GUI** needs a couple
of one-time setup steps because WSL's virtual GPU usually can't run the
OpenGL that the desktop app wants. This guide gets both working.

---

## TL;DR

```bash
# 1. system libraries (one time)
sudo apt-get update
sudo apt-get install -y \
    pkg-config build-essential \
    libgtk-3-dev libxkbcommon-dev libxcb1-dev \
    libfontconfig1 \
    mesa-utils libgl1-mesa-dri libglu1-mesa

# 2. build (from the cyclops source dir)
cd ~/cyclops
cargo build --release

# 3. run the GUI through the WSL launcher (sets software rendering for you)
./scripts/cyclops-gui-wsl.sh

# the CLI just runs directly — no launcher needed:
./target/release/cyclops --help
```

If the GUI window appears, you're done. If not, read on.

---

## Step 0 — Confirm you have WSL2 + WSLg

The GUI needs a display server. On WSL that's provided by **WSLg**, which
ships with WSL2 on Windows 11 and recent Windows 10 (build 19044+).

Open a **Windows PowerShell** (not the Ubuntu terminal) and run:

```powershell
wsl --version
```

You want to see a `WSLg version:` line. If `wsl --version` isn't even a
valid command, your WSL is too old. Update it (still in PowerShell):

```powershell
wsl --update
wsl --shutdown
```

Then reopen Ubuntu. To double-check WSLg is live, from inside Ubuntu:

```bash
echo "$DISPLAY $WAYLAND_DISPLAY"
```

You should see something like `:0 wayland-0`. If both are empty, WSLg
isn't running — re-do the `wsl --update && wsl --shutdown` dance above.

---

## Step 1 — System libraries

```bash
sudo apt-get update
sudo apt-get install -y \
    pkg-config build-essential \
    libgtk-3-dev libxkbcommon-dev libxcb1-dev \
    libfontconfig1 \
    mesa-utils libgl1-mesa-dri libglu1-mesa
```

What each group is for:

| packages | why |
|---|---|
| `pkg-config build-essential` | the C toolchain Rust's `-sys` crates need |
| `libgtk-3-dev` | the native file-picker dialog (`rfd`) |
| `libxkbcommon-dev libxcb1-dev` | X11 windowing for `winit` |
| `libfontconfig1` | runtime font discovery (the dev package is *not* required — Cyclops uses `fontconfig-dlopen`) |
| `mesa-utils libgl1-mesa-dri libglu1-mesa` | Mesa's `llvmpipe` software OpenGL — the key to GUI rendering on WSL |

---

## Step 2 — Build

```bash
cd ~/cyclops
cargo build --release
```

First build takes 5–10 minutes (it compiles ~600 crates). Rebuilds are
seconds. The binaries land in `target/release/cyclops` and
`target/release/cyclops-gui`.

Optionally install them onto your `PATH`:

```bash
cargo install --path cyclops-cli --locked
cargo install --path cyclops-gui --locked
# → ~/.cargo/bin/cyclops  and  ~/.cargo/bin/cyclops-gui
```

---

## Step 3 — Run the GUI

Use the bundled launcher — it sets the software-rendering environment
variables for you and prints a clear error if WSLg is missing:

```bash
./scripts/cyclops-gui-wsl.sh
```

Or, if you installed it on PATH and want to set the env yourself:

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe WINIT_UNIX_BACKEND=x11 cyclops-gui
```

A window titled **"Cyclops v0.1.0"** should open within a second or two.

---

## Step 4 — Run the CLI

The CLI never needs the launcher or any display:

```bash
cyclops --help

cyclops \
  --dapi        ~/cyclops/data/GSL/tiff/dapi \
  --fitc        ~/cyclops/data/GSL/tiff/fitc \
  --calibration "~/cyclops/data/GSL/tiff/dapi/GSL_+_blue_beads_1_(dapi).tiff" \
  --outDir      ~/cyclops_output \
  -v
```

---

## Troubleshooting by exact error

### `Error: WinitEventLoop(... "neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set.")`

No display server. WSLg isn't active. Fix from **PowerShell**:

```powershell
wsl --update
wsl --shutdown
```

Reopen Ubuntu and confirm `echo $DISPLAY` is non-empty.

### Window opens but is **black**, or the app crashes on launch

The vGPU is being used and failing. Force software rendering:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
cyclops-gui
```

(The `cyclops-gui-wsl.sh` launcher does this automatically.)

### `Failed to initialize any backend! Wayland status: ... X11 status: ...`

`winit` couldn't pick a backend. Force X11:

```bash
export WINIT_UNIX_BACKEND=x11
cyclops-gui
```

### `libGL error: failed to load driver: swrast` / `libGL error: MESA-LOADER`

Mesa's software driver isn't installed. Install it:

```bash
sudo apt-get install -y libgl1-mesa-dri libglu1-mesa mesa-utils
```

Verify with `glxinfo | grep "OpenGL renderer"` — you want to see
`llvmpipe` (software) listed.

### The build itself fails on `yeslogic-fontconfig-sys` / `Package fontconfig was not found`

You're on an older checkout without the `fontconfig-dlopen` fix. Either:

```bash
sudo apt-get install -y libfontconfig1-dev   # quick fix
```

or update to the current Cyclops source, where `Cargo.toml` already sets
plotters' `fontconfig-dlopen` feature so only the *runtime* library
(`libfontconfig1`) is needed.

### GUI runs but is **slow / laggy**

That's expected with `llvmpipe` (CPU rendering). The Cyclops UI is
lightweight so it's usually fine, but if you have a real GPU and want
hardware acceleration, that requires WSL GPU passthrough setup (NVIDIA
CUDA-on-WSL or the `d3d12` Mesa driver) which is beyond this guide —
software rendering is the reliable default.

---

## Performance note

The pipeline's heavy lifting (PSF sweep, Richardson–Lucy on full-res
microscopy TIFFs) runs on the **CPU regardless** of the GUI rendering
path, and is parallelised across cores via rayon. WSL2 gives Linux
processes access to all your CPU cores, so analysis speed on WSL is
essentially native — only the window *drawing* uses software rendering,
and that's a negligible fraction of total runtime.
