# Prebuilt binaries

## linux-x86_64/cyclops

The **command-line** Cyclops binary, built on Ubuntu 24.04 (glibc 2.39)
for x86-64 Linux (including WSL2 Ubuntu). This is a release build with
the critical data-safety fix.

Run it directly:

```bash
./prebuilt/linux-x86_64/cyclops --version
./prebuilt/linux-x86_64/cyclops --help
```

Or copy it onto your PATH:

```bash
cp prebuilt/linux-x86_64/cyclops ~/.local/bin/cyclops   # or ~/.cargo/bin
```

### Compatibility

This binary needs glibc ≥ 2.39. If you get a
`GLIBC_2.39 not found` error on an older distro, build from source
instead (see the top-level README) — the source is identical.

### GUI

The desktop GUI (`cyclops-gui`) is **not** prebuilt here because it links
a large graphics stack that is slow to compile in constrained CI. Build
it on your own machine in a couple of minutes:

```bash
cargo build --release -p cyclops-gui
./target/release/cyclops-gui
```

On WSL, launch it via `./scripts/cyclops-gui-wsl.sh` (see docs/WSL.md).
