# Cyclops — Bug Audit Checklist

Every item from `Cyclops_debug_prompt.docx`, audited against the actual
Cyclops source and resolved. Verification: `cargo check` clean; **18 tests
pass** (12 unit + 5 data-safety + 1 end-to-end).

> Context: the debug prompt was written against a generic / 3D-volume
> template of the pipeline. Cyclops is a **2D** implementation built on
> `ndarray` (bounds-checked indexing), `rustfft` (arbitrary-size FFT), and
> `rayon` map-reduce (race-free by construction). Several listed bugs
> therefore describe code paths that never existed here; those are marked
> **N/A** with the reason. Two items were genuine latent risks and were
> fixed this pass.

| # | Module | Bug (as described) | Status | Resolution / evidence |
|---|--------|--------------------|--------|-----------------------|
| 1 | regions.rs | Recursive-DFS stack overflow on large blobs | ✅ N/A — never present | Labeling is two-pass **union-find**; `find()` is an iterative `while` loop with path compression. No `label_components_dfs`, no `explore_neighbors`, no self-recursion. Structurally immune. |
| 2 | threshold.rs | Otsu zero-division / NaN (`w_b`,`w_f`=0) | ✅ Already fixed | Guards: `total==0 → return`; `w_b==0 → continue`; `w_f==0 → break`. Uses multiply form `w_b·w_f·(m_b−m_f)²` — no division by weights. |
| 3 | calibration.rs | Zero-division on empty bead sets | ✅ Already fixed | `.count().max(1)`; `correction = if mean>0 {…} else {1.0}`; `if sizes.is_empty()` guard; `std_dev` guards empty. No NaN path. |
| 4 | deconv.rs | RL `observed/blurred` → Inf/NaN | ✅ Already fixed | `EPS` clamp on `blur` before division in **both** RL routines (blind-MLE + FFT-RL); `+EPS.sqrt()` on scale denominators. |
| 5 | psf.rs (+regions) | OOB 3D flat-index; un-normalized PSF | ✅ N/A + already fixed | Pure 2D `ndarray`, bounds-checked `[(i,j)]`, zero flat-index sites. PSF sum-normalized to 1.0 (`if sum.is_finite() && sum>0`), NaN-guarded per element. |
| 6 | image_io.rs | 16-bit → 8-bit truncation | ✅ Already fixed | Casts to **f64**, divides by type max (`/65535` u16, `/255` u8, + u32/u64/f32/i8–i64). No truncation. |
| 7 | fft.rs | Non-power-of-two panic | ✅ N/A | `rustfft::FftPlanner` handles arbitrary sizes (mixed-radix/Bluestein). 2448×2048 runs fine; padding to 2ⁿ would be wrong. |
| 8 | quantify.rs | Cross-channel stride/dim mismatch | ✅ N/A + guarded | One FITC image at a time; no pixel-wise channel zip to mismatch. Both mean divisions guarded by `is_empty()`. |
| 9 | classify.rs | NaN float comparisons drop valid objects | ✅ **Fixed this pass** | `if score.is_finite() && score > best.1` + `nearest_by_size` fallback (via `total_cmp`). No object dropped. 2 new regression tests pass. |
| 10 | pairing.rs | O(N²) slowdown on dense fields | ✅ N/A | O(N²) over ~4–60 regions of **one** calibration image (measured 4 candidates, 0.10 s). Slice bounds fully guarded. k-d tree unwarranted. |
| 11 | main_2.rs / pipeline.rs | Concurrent data races | ✅ N/A — safe by design | No `main_2.rs`. All rayon use is `par_iter().map(pure_fn).collect()` with immutable captures — the compiler forbids a race. Mutex would only serialize. |
| 12 | plots.rs / config.rs | Missing output dirs; unvalidated config | ✅ **Fixed this pass** + present | Added `ensure_parent_dir()` before all three `BitMapBackend::new`. `Config::validate()` already exists and is called at `run_pipeline` (guards scale/sphere/iterations → no px2nm div-by-zero). |

## Also fixed in earlier passes (beyond the docx list)

- **Union-find init off-by-one** (regions.rs) — merged disjoint blobs.
- **Otsu bin-edge vs bin-centre** (threshold.rs).
- **RL border-excision step missing** (deconv.rs) — had tripled counts / halved sizes vs reference.
- **CRITICAL data-loss:** pipeline used to `remove_dir_all(out_dir)`, deleting input data when output pointed at the data folder. Removed; now refuses on any output/input path overlap (5 data-safety tests).
- **CLI swallowed fatal errors** unless `-v`. Now always prints to stderr.
- **Memory OOM on full dataset:** DAPI+FITC stacks were both resident. Now DAPI is freed before FITC loads; 26-image peak dropped from >2.7 GB (OOM) to 1.79 GB.

## Test inventory

- **Unit (12):** psf (4), deconv (2), threshold (2), regions (2), classify (2 — NaN-safety).
- **Data-safety (5):** output/input overlap refusal in every configuration.
- **End-to-end (1):** synthetic pipeline runs and writes all artifacts.
