# Cyclops

<p align="center">
  <img src="assets/cyclops.svg" width="180" alt="Cyclops mascot"/>
</p>

<p align="center">
  <em>One eye on the field of view, one count per particle.</em><br/>
  <strong>Cyclops</strong> sizes and counts viral-like particles, bacteria, archaea, and protists
  in epifluorescence microscopy — a complete <strong>Rust</strong> rewrite of
  <a href="https://github.com/raw-lab/EpiVirQuant">EpiVirQuant</a> with a desktop GUI,
  Polars-backed object tables, and an optional ML classifier.
</p>

<p align="center">
  <a href="#install"><img alt="install" src="https://img.shields.io/badge/install-cargo-blue"/></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-green"/></a>
  <img alt="rust" src="https://img.shields.io/badge/rust-1.75%2B-orange"/>
</p>

---

## Why Cyclops?

Cyclops is a fully reimplemented, statically-compiled successor to
**EpiVirQuant v0.1.1** (Figueroa III, Hollenack & White III, 2026). The original
Python pipeline pairs scale-bar beads, performs blind-deconvolution maximum-
likelihood estimation of a **tunable point-spread function** (γ-sinc, Gaussian,
or hybrid), calibrates against DAPI microspheres, then quantifies FITC-stained
samples. Cyclops preserves that scientific core *bit-for-bit* and adds:

- **Desktop GUI** (`cyclops-gui`) built on **egui/eframe** — file pickers,
  live parameter sliders, run-on-thread, log/result tabs. Same paradigm
  as our DeGenPrime 2.0 desktop tool.
- **CLI** (`cyclops`) with every flag from `epivirquant.py` plus new
  Cyclops-only switches (`--domains`, `--no-gmm`, `--onnx`).
- **Polars** dataframes throughout — Parquet + TSV object tables out of
  the box, ready for downstream analysis in any language.
- **Seaborn-style plotting** via `plotters` with the viridis colour ramp.
- **ML organism classifier** — rule-based prior on size band /
  eccentricity / DAPI-FITC ratio, optionally refined by a Gaussian
  Mixture Model (linfa). Optional ONNX hook (Cellpose, StarDist exports)
  behind a build feature for future deep-learning integration.
- **Multi-domain support** — virus / bacteria / archaea / protist size
  ranges in the same run.
- **Native parallelism** via `rayon` across the PSF sweep, calibration,
  and quantification stages.
- **Small footprint** — single ~10 MB statically-linked binary on
  release builds (LTO + strip), no Python environment, no `mamba`,
  no NumPy ABI mismatches.

## Status

| crate          | description                                  | status |
|----------------|----------------------------------------------|--------|
| `cyclops-core` | algorithms — FFT, PSF sweep, MLE blind decon, calibration, quantification, ML classifier | ✅ builds clean, 10 unit tests + 1 integration test passing |
| `cyclops` (bin) | command-line driver matching `epivirquant.py` flags | ✅ default build |
| `cyclops-gui` (bin) | desktop application (egui + rfd file dialogs) | ✅ behind `--features gui` |
| ONNX backend | optional pure-Rust `tract` classifier hook | ✅ behind `--features onnx` |

### Verifying your build

The end-to-end integration test generates a synthetic DAPI/FITC stack
and runs the full four-stage pipeline (pairing → blind decon →
calibration → quantify → classify) in under a second:

```bash
cargo test --lib -p cyclops-core           # 10 unit tests
cargo test --test end_to_end -p cyclops-core -- --nocapture
```

Expected output from the integration test:

```text
n_dapi      : 3
n_fitc      : 2
min_dist_nm : 1500.00
CORR        : 0.7736
n_objects   : 2
mean_size   : 663.7 nm
```

The synthetic beads were placed with a known radius and pixel scale; the
recovered mean size lands within ≈ 2 % of the geometric prediction, and
every promised output file (`cyclops_report.json`,
`cyclops_objects.parquet`, `cyclops_objects.tsv`, `sizeCoords.tsv`, and
the three per-step log files) is verified to exist and be non-empty.

## Install

### From source

You need a recent stable Rust toolchain (1.85 or newer):

Cyclops is a **single crate** that provides a library plus two binaries
(`cyclops`, the CLI, and `cyclops-gui`, the desktop app). The GUI and the
optional ONNX backend are Cargo **features**, so a plain install stays
lightweight.

```bash
# install rustup if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# --- straight from crates.io ---
# CLI only (library + `cyclops` binary; no heavy GUI deps):
cargo install cyclops --locked
# CLI + desktop GUI:
cargo install cyclops --locked --features gui
# add the optional ONNX classifier backend as well:
cargo install cyclops --locked --features "gui onnx"

# --- or from a clone ---
git clone https://github.com/raw-lab/cyclops
cd cyclops
cargo install --path . --locked                    # CLI only
cargo install --path . --locked --features gui     # CLI + GUI
```

The binaries land in `~/.cargo/bin/` (`cyclops`, and `cyclops-gui` when
built with `--features gui`); add it to your `$PATH` if it isn't already.

| feature | adds | default? |
|---|---|---|
| *(none)* | library + `cyclops` CLI | ✅ on |
| `gui` | `cyclops-gui` desktop app (eframe/egui/rfd) | off |
| `onnx` | pure-Rust ONNX classifier backend (tract) | off |

### System dependencies

Cyclops builds out-of-the-box on a stock macOS or Windows toolchain. On
**Linux** the GUI links against a small handful of system libraries for
window creation, font discovery, and the native file picker:

| distribution | install command |
|---|---|
| Debian / Ubuntu / Pop!_OS / Mint | `sudo apt-get install -y pkg-config libgtk-3-dev libxkbcommon-dev libxcb1-dev libfontconfig1` |
| Fedora / RHEL / Rocky | `sudo dnf install -y pkg-config gtk3-devel libxkbcommon-devel libxcb-devel fontconfig` |
| Arch / Manjaro | `sudo pacman -S pkg-config gtk3 libxkbcommon libxcb fontconfig` |
| openSUSE | `sudo zypper install pkg-config gtk3-devel libxkbcommon-devel libxcb-devel fontconfig` |

The CLI (`cyclops`) has **no system-library requirements** — only Rust.
The GUI system packages above are needed **only** when you build with
`--features gui`; a default `cargo install cyclops` skips them entirely.

### Running on WSL (Windows Subsystem for Linux)

The GUI works on WSL2 with WSLg, but the virtual GPU usually can't run
hardware OpenGL, so you launch it through the bundled software-rendering
wrapper:

```bash
./scripts/cyclops-gui-wsl.sh
```

See [`docs/WSL.md`](docs/WSL.md) for the full WSL setup and a
troubleshooting table keyed to the exact error messages.

### Desktop launcher / application-menu icon

To add Cyclops to your application menu (and the Windows Start Menu on
WSLg) with the Cyclops icon:

```bash
./scripts/install-desktop.sh
```

This auto-detects WSL and wires the menu entry to the software-rendering
launcher when appropriate. Remove it later with
`rm ~/.local/share/applications/cyclops.desktop`.

Note: `fontconfig` itself can be installed as the runtime library
(`libfontconfig1` / `fontconfig`) rather than the dev package
(`libfontconfig1-dev` / `fontconfig-devel`) — Cyclops uses plotters'
`fontconfig-dlopen` feature, which loads fontconfig at runtime via
`dlopen` instead of linking against it at build time. If fontconfig
isn't installed at runtime either, charts fall back to plotters'
bundled bitmap font.

### Bioconda

A Bioconda recipe will follow the first tagged release alongside the existing
[raw-lab](https://anaconda.org/bioconda/repo) channel tooling (MetaCerberus,
MerCat2, NFixDB, DeGenPrime, Pathview-web).

## Quick start

The Cyclops repository ships with the same calibration scan and demo TIFFs as
EpiVirQuant, so any existing analysis recipe transfers directly:

```bash
cyclops \
  --dapi        data/GSL/tiff/dapi \
  --fitc        data/GSL/tiff/fitc \
  --calibration data/GSL/tiff/dapi/GSL_+_blue_beads_1_\(dapi\).tiff \
  --scaleLength 585 \
  --scaleMetric 20000 \
  --sphereSize  175 \
  --psfMethod   gam \
  --domains     virus,bacteria \
  --outDir      Cyclops_Output
```

For the desktop experience:

```bash
# built with --features gui
cyclops-gui
```

…then point the **DAPI / FITC / Calibration / Output** fields at your data
and click **▶ Run Cyclops**.

> **Important — keep your output directory separate from your data.**
> Always set `--outDir` (or the GUI's Output field) to a folder that is
> **not** inside your DAPI/FITC image folders — for example
> `~/cyclops_runs/run1`. Cyclops writes its `Step-1` … `Step-4`
> results and object tables there. As a safety measure Cyclops now
> **refuses to run** (with a clear message) if the output path overlaps
> any of your input folders, so it can never touch your source images —
> but choosing a clean, separate output folder keeps everything tidy.

## How it works

The pipeline keeps the four EpiVirQuant stages, now orchestrated by
`cyclops_core::pipeline::run_pipeline`:

1. **Pair detection.** Otsu-threshold + 8-connectivity labelling locates
   every bead in the calibration scan; pairwise distance picks the
   closest valid VP pair. (`pairing.rs`)
2. **Blind-deconvolution PSF sweep.** `f_size × τ × v` are swept with
   maximum-likelihood blind deconvolution; the kernel with minimum
   image entropy wins. The PSF family is **tunable** (`gam`/`gau`/`hyb`)
   and the sweep runs in parallel via `rayon`. (`deconv.rs`, `psf.rs`)
3. **Calibration.** Richardson-Lucy deconvolution + region-props on
   DAPI microsphere fields yields the bead diameter; CORR =
   `sphere_size_nm / mean_measured_size`. (`calibration.rs`)
4. **Quantification.** FITC images are deconvolved with the same PSF,
   thresholded, labelled, and reported with diameter × axis lengths ×
   eccentricity × intensity × area. False positives are pruned by
   eccentricity > 0.9999 and a configurable `SM_constraint`.
   (`quantify.rs`)
5. **Classification (new in Cyclops).** Each object gets a domain call
   (`virus`, `bacteria`, `archaea`, `protist`) from a rule-based prior
   (Gaussian membership on size band + eccentricity boost + DAPI/FITC
   intensity ratio band), optionally refined by a linfa Gaussian
   Mixture Model whose clusters are relabelled by the majority
   rule-based call. (`classify.rs`)

### Outputs

```
Cyclops_Output/
  Step-1_VP/                       PNGs of the calibration scan
  Step-2_Decon/
    PSF_final.png
    deconLog.txt
    optimization/EntVsIter.png …
  Step-3_Corr/
    CORR_<image>/…                 per-image diagnostic PNGs
    XB_SizeHistogram.png
    corrLog.txt
  Step-4_genMasks/
    genMask_<image>/…              per-image diagnostic PNGs
    sizeCoords.tsv                 legacy EpiVirQuant TSV
    cyclops_objects.parquet        full Polars frame incl. domain calls
    cyclops_objects.tsv            same as above, TSV
    countLog.txt
  cyclops_report.json              machine-readable summary
```

Every column you got from EpiVirQuant is still there. The new
`cyclops_objects.{parquet,tsv}` adds:

| column             | description                                              |
|--------------------|----------------------------------------------------------|
| `file_name`        | source FITC image                                        |
| `object_id`        | integer label within the image                           |
| `size_nm`          | equivalent diameter (or average axes, per `--szMetric`)   |
| `axis_major_nm`    | semi-major axis × 2 in nm                                 |
| `axis_minor_nm`    | semi-minor axis × 2 in nm                                 |
| `x_px`, `y_px`     | centroid in pixels                                        |
| `intensity`        | mean masked intensity                                     |
| `eccentricity`     | region-props eccentricity                                 |
| `area_px`          | pixel count                                              |
| `domain`           | `virus` / `bacteria` / `archaea` / `protist`              |
| `confidence`       | classifier posterior in [0, 1]                            |
| `dapi_fitc_ratio`  | per-object intensity ratio used by the classifier         |

## CLI reference

```text
cyclops [OPTIONS] --dapi <DIR> --fitc <DIR> --calibration <FILE>
```

| flag                  | default          | original `epivirquant.py` | meaning |
|-----------------------|------------------|---------------------------|---------|
| `--dapi`              | —                | `--dapi`                  | DAPI directory |
| `--fitc`              | —                | `--fitc`                  | FITC directory |
| `--calibration`       | —                | `--calibration`           | calibration image |
| `--outDir`            | `Cyclops_Output` | `--outDir`                | output directory |
| `--scaleLength`       | 585              | `--scaleLength`           | scale-bar length (px) |
| `--scaleMetric`       | 20000            | `--scaleMetric`           | scale-bar length (nm) |
| `--sphereSize`        | 175              | `--sphereSize`            | bead diameter (nm) |
| `--pad`               | 14               | `--pad`                   | VP crop padding |
| `--dConstraint`       | 30               | `--dConstraint`           | max paired-bead distance |
| `--fSize`             | 0                | `--fSize`                 | PSF kernel size (0 = sweep) |
| `--psfMethod`         | gam              | `--psfMethod`             | `gam` \| `gau` \| `hyb` |
| `--a/--b/--sig/--r/--tau/--v/--s` | …    | same                      | PSF hyper-parameters |
| `--nMLE_iter`         | 10               | `--nMLE_iter`             | blind decon iterations |
| `--nLR_iter`          | 80               | `--nLR_iter`              | Richardson-Lucy iterations |
| `--szMetric`          | 1                | `--szMetric`              | 1 = equiv. diameter, 2 = avg axes |
| `--SM_constraint`     | 8000             | `--SM_constraint`         | reject above this nm |
| `--genFigs`           | false            | `--genFigs`               | save every diagnostic figure |
| `--cpus`              | -2               | `--cpus`                  | joblib convention |
| `--domains`           | all              | *(new)*                   | `virus,bacteria,archaea,protist` |
| `--no-gmm`            | off              | *(new)*                   | disable linfa GMM refinement |
| `--onnx`              | —                | *(new)*                   | optional ONNX segmentation model |
| `--noIntermediates`   | off              | *(new)*                   | skip per-image PNG diagnostics |

## Programmatic use

```rust
use cyclops_core::{
    config::{Config, OrganismDomain, ClassifierConfig},
    pipeline::run_pipeline,
};

let cfg = Config {
    dapi_dir:    "data/dapi".into(),
    fitc_dir:    "data/fitc".into(),
    calibration: "data/dapi/blue_beads_1.tiff".into(),
    out_dir:     "Cyclops_Output".into(),
    classifier:  ClassifierConfig {
        domains: vec![OrganismDomain::Virus, OrganismDomain::Bacteria],
        gmm_refine: true,
        onnx_model: None,
    },
    ..Default::default()
};

let report = run_pipeline(&cfg)?;
println!("{} objects (mean {:.1} nm)", report.n_objects, report.mean_size_nm);
```

## Build options

```bash
# debug build, fast compile
cargo build

# optimised release (LTO, strip, single codegen unit)
cargo build --release

# enable the (still experimental) ONNX classifier hook
cargo build --release -p cyclops-cli --features onnx
```

## Citation

If you use Cyclops in published work, please cite both the original
EpiVirQuant article and this software:

> **EpiVirQuant** — Figueroa III JL, Hollenack SM, White III RAW.
> *Direct counting and sizing of viral-like particles by tunable
> blind-deconvolution of epifluorescence microscopy.*
> **BMC Methods 3:10**, 2026. <https://doi.org/10.1186/s44330-026-00060-z>

> **Cyclops** — White III RAW, Figueroa III JL, Hollenack SM. *Cyclops:
> a Rust desktop application for sizing and counting viral-like
> particles, bacteria, archaea, and protists in epifluorescence
> microscopy.* (in preparation, 2026).

## License

Cyclops is released under **Creative Commons Attribution-NonCommercial
4.0 (CC BY-NC 4.0)** — identical to upstream EpiVirQuant. Academic and
non-commercial use is free; commercial licensing inquiries should be
directed to Richard Allen White III (`rwhit101@charlotte.edu`).

## Acknowledgements

Cyclops is developed at the **RAW lab**, College of Computing &
Informatics, **UNC Charlotte**, with support from the **Charlotte AI
Institute** and the **CIPHER center**. We thank the maintainers of
`ndarray`, `rustfft`, `polars`, `plotters`, `linfa`, `egui` and `rfd`
— the Rust crates that make a project this size feasible for a single
team.
