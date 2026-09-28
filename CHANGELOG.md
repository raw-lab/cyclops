# Changelog

All notable changes to Cyclops are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

Initial Rust rewrite of the EpiVirQuant v0.1.1 pipeline. The four-stage
analysis (pair detection → tunable-PSF blind deconvolution → Richardson–
Lucy calibration → quantification) is preserved bit-for-bit on the
algorithmic axes, with these additions:

### Added

- **Cyclops branding** — single-eye microscope-objective mascot bundled
  as a vector SVG and embedded into the desktop GUI at compile time.
- **Desktop GUI** (`cyclops-gui`) — egui/eframe app with native
  `rfd` file pickers, collapsible parameter panels, run-on-thread,
  cancel control, and four result tabs (Log / Summary / Size bands /
  Domains). Mirrors the DeGenPrime 2.0 GUI paradigm.
- **CLI** (`cyclops`) — every flag from `epivirquant.py` preserved,
  plus `--domains`, `--no-gmm`, `--onnx`, `--noIntermediates`.
- **Polars object table** — `cyclops_objects.parquet` and
  `cyclops_objects.tsv` written alongside the legacy `sizeCoords.tsv`
  with per-object domain calls (`virus`/`bacteria`/`archaea`/`protist`),
  classifier confidence, and DAPI∕FITC intensity ratio.
- **ML organism classifier** — rule-based prior on size band /
  eccentricity / DAPI–FITC ratio, optionally refined by a linfa
  Gaussian-Mixture-Model. Optional ONNX hook behind the `onnx` feature.
- **Rayon-parallel** PSF sweep, calibration, and quantification stages.
- **End-to-end integration test** that synthesises a DAPI/FITC stack and
  runs the full pipeline in under a second on a single core, asserting
  on every promised output artefact.

### Verified end-to-end (2026-05-26)

The desktop GUI was launched headless under Xvfb with software-rendered
OpenGL (`LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`) and
exercised through the full workflow:

1. Window renders cleanly at 1280 × 820, with the bundled Cyclops SVG
   mascot, header, sidebar, central tab bar, and Run/Cancel buttons all
   laid out as designed.
2. All six collapsible side-panel sections (Inputs, Scale-bar / sphere,
   PSF sweep, Pairing & filtering, Organism domains, Runtime) expand
   and render their controls. Numeric inputs accept keyboard input
   correctly. Text inputs accept path strings.
3. Clicking **▶ Run Cyclops** with valid paths in the four input fields
   disables the Run button, enables Cancel, shows a rotating spinner
   and a live elapsed-time counter ("running... NN.Ns"), and streams
   pre-flight log lines into the Log tab.
4. On completion the GUI auto-switches to the **Summary** tab and
   displays the full results table (n_objects, mean_size_nm, CORR,
   PSF parameters, elapsed time). The **Size bands** and **Domains**
   tabs render their bar charts correctly with colour-coded counts.
5. Every output file the CLI produces is also produced by the GUI run:
   `cyclops_report.json`, `cyclops_objects.{tsv,parquet}`,
   `sizeCoords.tsv`, the size histogram PNG, and the three per-step
   logs.

Screenshots from this verification are in the project root as
`screenshots/cyclops_gui_*.png`.

### Build / packaging fixes

- **`plotters`: switched to `fontconfig-dlopen` so the build no longer
  requires `libfontconfig1-dev` (Debian) / `fontconfig-devel` (Fedora)
  at compile time.** Without this flag, `yeslogic-fontconfig-sys`'s
  build script invokes `pkg-config --libs --cflags fontconfig` and
  panics with `Package fontconfig was not found in the pkg-config
  search path` on any host that doesn't have the dev package
  installed. With `fontconfig-dlopen`, plotters' `font-kit` dependency
  loads fontconfig via `dlopen` at runtime instead, so the build
  succeeds with just the runtime library (`libfontconfig1` /
  `fontconfig`) — or no fontconfig at all, in which case plotters
  falls back to its bundled bitmap font.

### Performance

- **Memory: the full 26-image GSL dataset no longer OOMs.** The pipeline
  used to load both the DAPI and FITC image stacks into RAM up front and
  hold them for the whole run — ~1 GB each for 26 × 5 MP images stored as
  f64, which pushed peak RSS past 4 GB and got the process OOM-killed
  during Step 3 on the full dataset. Fixed by loading DAPI + the small
  calibration image first, running Steps 1–3, then **freeing the DAPI
  stack before loading FITC** for Step 4, so the two large stacks never
  coexist. Peak RSS on the full 26-image run dropped from >2.7 GB
  (OOM-killed) to 1.79 GB (completes). Output is byte-for-byte identical
  before and after — verified on the 3-image subset (same 376 objects,
  182.6 nm mean, same size bands). See `docs/BENCHMARK.md` for the full
  timing and memory numbers.

### Added

- **GUI streaming progress bar.** A new `progress` module defines a
  `Progress` trait (`stage`/`tick`/`message`) with a `Stage` enum and
  global-fraction anchors. `run_pipeline_with_progress()` threads a
  reporter through every stage; `calibrate()` and `quantify()` tick per
  image via an atomic counter inside their rayon loops. The GUI bridges
  this to its mpsc channel and renders an `egui::ProgressBar` — animated
  for single-shot stages (PSF sweep), with a live `done/total` counter
  and overall-percent + elapsed-time readout for the per-image stages.
  Verified on screen (bar advances 5 %→10 %→… through the stages) and by
  a headless test (`tests/progress_ticks.rs`) asserting per-image ticks.
- **ONNX classifier backend (optional `onnx` feature).** `onnx.rs`
  implements a real, pure-Rust inference path via `tract-onnx` (no C++
  ONNX Runtime to bundle — keeps Cyclops a single self-contained binary),
  plus a no-op stub for default builds. Contract: input `[N, 5]` feature
  tensor (log10 size, eccentricity, log10 DAPI/FITC ratio, log10 area,
  aspect ratio), output `[N, C]` per-domain scores → softmax → arg-max
  label + confidence. Wired into `classify_objects` after the GMM stage;
  falls back transparently to rule-based/GMM if the model is missing,
  mis-shaped, or the feature is off. `cargo build --features onnx` and
  its unit tests both pass.
- **Numerical validation vs the Python EpiVirQuant** on the GSL dataset
  (`docs/PYTHON_COMPARISON.md`, `figure_python_vs_rust.png`). With
  identical parameters, mean particle size agrees to **0.6 %** over three
  images (2.7 % single-image), total object count to **3 %**, VP pair
  detection **exactly**, and the sub-viral / bacterium size bands
  exactly. Confirms the Rust rewrite reproduces the reference pipeline.

### Fixed (vs. the initial port)

- **classify.rs — NaN-safe classification (never drop an object).** The
  per-object domain score `membership × shape_boost × ratio_boost` could
  in principle be NaN if an upstream `size_nm`/`eccentricity` were NaN;
  the `score > best.1` comparison would then silently leave the object on
  its default domain. Hardened to `score.is_finite() && score > best.1`,
  with a `nearest_by_size` fallback (closest canonical size-band centre,
  compared via `total_cmp`) so every object is still classified — never
  dropped or left unlabelled. The GMM cluster-relabel path already used
  `total_cmp`. Covered by two new tests
  (`nan_features_do_not_drop_objects`, `every_object_gets_a_domain`).
- **plots.rs — create the output folder before writing a figure.** Each
  of `histogram`, `scatter`, `heatmap` now calls `ensure_parent_dir()`
  before `BitMapBackend::new`, so a plot written to a not-yet-existing
  directory (standalone/library use, or a future output-layout change)
  succeeds instead of erroring. The pipeline already `create_dir_all`s
  each Step directory; this is a defensive backstop.

- **CRITICAL / data-loss: the pipeline no longer deletes the output
  directory, and refuses to run if the output path overlaps your input
  data.** The initial implementation ran
  `fs::remove_dir_all(out_dir)` unconditionally at the start of every
  run. If a user pointed `--outDir` (or the GUI's Output field) at — or
  inside — their DAPI/FITC data folder, this recursively deleted their
  images, after which the pipeline crashed because there was nothing
  left to read. This exactly produced the reported "GUI crashes and
  deletes my data" behaviour. The fix:
  * `run_pipeline` now calls `guard_output_directory`, which **refuses
    to run** (with a clear message) whenever the output directory is the
    same as, inside, or an ancestor of any input location (DAPI dir,
    FITC dir, or the calibration image's folder);
  * the unconditional `remove_dir_all` is **gone entirely** — Cyclops
    creates the output directory if missing and writes into it, but
    never recursively deletes a pre-existing directory, so unrelated
    files you may have left there are preserved;
  * the GUI performs the same overlap check as a friendly pre-flight
    validation *before* the run starts, so you see the warning
    immediately instead of after a crash.
  Five regression tests in `tests/data_safety.rs` assert that a planted
  "precious" file survives every overlapping-path configuration.
- **CLI: fatal errors are now always printed to stderr**, independent
  of the `-v` log level. Previously a refused or failed run exited with
  a non-zero code but printed nothing unless `-v` was passed, because
  the error only went through the `tracing` logger (which is filtered
  to WARN by default). Now `Error: ...` is written to stderr
  unconditionally, so the user always sees *why* a run stopped.


These bugs were caught by the test suite *before* the first tagged
release and never reached external users — but they are documented here
so that anyone reading the git history can confirm that the relevant
code paths are exercised.

- **`regions.rs`: off-by-one in union-find initialisation** — the
  `parent` vector was initialised with an extra `push(0)`, causing every
  newly-allocated label `N` to be backed by slot `N − 1` instead of
  `N`. The downstream effect was that two disjoint connected components
  could end up merged into one in the second-pass relabel, silently
  inflating measured object size on real microscopy data. Fixed by
  letting `parent.push(next_label)` self-root each new label.
- **`deconv.rs`: missing border excision in padded Richardson–Lucy** —
  the upstream Python pipeline (lines 47–53 of `epivirquant_masks.py`)
  pads the input image up, runs RL, then crops **past** the original
  border to `(h - pad_y, w - pad_x)`, zero-pads back to original
  dimensions, and fills the resulting border ring with the image mean.
  My initial port did a `center_crop` of the padded result back to
  `(h, w)`, which kept the original border untouched. RL leaves long
  thin streaks at the border that the connected-component labeller
  picked up as ~10:1 aspect-ratio objects, **tripling the object count
  and halving mean size** versus the published reference. Caught by
  the side-by-side comparison run against the GSL dataset (Cyclops
  reported 230 obj/image at 167 nm; reference @ iter 20 was 64 obj/image
  at 358 nm). Fixed by mirroring the upstream sequence step-for-step.
- **`threshold.rs`: Otsu returned bin edges instead of bin centres** —
  for a perfectly bimodal `[0, 1]` image, the threshold landed at
  exactly `0.0` (the lower mode). Aligned with the scikit-image
  `bin_centers[argmax]` convention so that the returned threshold is
  strictly interior, matching what downstream `> threshold` comparisons
  expect.

### Compatibility

- Minimum supported Rust version: **1.85** (stable, March 2025).
- Output layout (`Cyclops_Output/Step-{1,2,3,4}_…`) is preserved
  verbatim from EpiVirQuant, including `deconLog.txt`, `corrLog.txt`,
  `countLog.txt`, and `sizeCoords.tsv`. New artefacts
  (`cyclops_objects.{parquet,tsv}`, `cyclops_report.json`) live
  alongside without disturbing existing analysis recipes.
- The `Config` struct's `Default` matches every EpiVirQuant CLI default
  (`scaleLength=585`, `scaleMetric=20000`, `sphereSize=175`,
  `pad=14`, `dConstraint=30`, `nMLE_iter=10`, `nLR_iter=80`,
  `SM_constraint=8000`, `cpus=-2`).

### Known limitations

- The pipeline currently emits log messages and reports at the end of
  each stage rather than streaming per-iteration progress. The GUI
  surfaces a spinner and elapsed-time counter while running but no
  per-step progress bar. A streaming progress channel is on the
  roadmap (`run_pipeline` would take an `&dyn Progress` parameter).
- The ONNX classifier hook is wired through the `Config` and CLI but
  the inference backend itself is not yet implemented; the feature
  flag is reserved.
- No side-by-side numerical comparison against upstream EpiVirQuant on
  the published GSL dataset is included in CI yet. The synthetic
  integration test catches crashes, schema drift, and dimensionally
  inconsistent results, but bit-exact reproducibility against the
  reference Python implementation will be added in a follow-up.
