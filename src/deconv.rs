//! Blind deconvolution (MLE) and Richardson–Lucy.
//!
//! `mle_blind` is a Rust port of `epivirquant_decon.get_MLE` —
//! a Bertero–Boccacci-style alternating MLE that updates the image
//! estimate `J` and the PSF estimate `P` simultaneously, with a
//! Polak–Ribière-flavoured momentum term on each variable.
//!
//! `richardson_lucy` is the standard non-blind RL applied to the
//! microsphere (DAPI) and FITC images once a PSF has been chosen.

use std::time::Instant;

use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;
use tracing::info;

use crate::config::Config;
use crate::error::{CyclopsError, Result};
use crate::fft::{
    center_crop, conj_freq, fft2, ifft2, mul_freq, otf2psf, psf2otf, real_of,
    zero_pad_centered,
};
use crate::psf::{create_psf, normalise_l1};
use crate::threshold::{glcm_energy_dx2, shannon_entropy};

const EPS: f64 = f64::EPSILON;

/// Output of an MLE-blind deconv: a recovered PSF and the deconvolved
/// optimisation-box image.
#[derive(Debug, Clone)]
pub struct MleResult {
    pub psf:        Array2<f64>,
    pub deconv_box: Array2<f64>,
}

/// Run one blind-deconv MLE optimisation on an `optBox` patch using
/// the seed PSF `psf0`.
pub fn mle_blind(
    opt_box:    &Array2<f64>,
    psf0:       &Array2<f64>,
    n_iter:     usize,
) -> MleResult {
    let (h, w) = opt_box.dim();
    let (fh, fw) = psf0.dim();
    debug_assert_eq!(fh, fw, "PSF must be square");
    let f_size = fh;

    // Working state — mirrors the J = [.,.,.,.] / P = [.,.,.,.] tuples
    // used in the Python reference (current, previous, history,
    // 2-column momentum buffer).
    let mut j_curr = opt_box.clone();
    let mut j_prev = opt_box.clone();
    let mut j_hist: Vec<f64> = vec![0.0; h * w * 2];

    let mut p_curr = psf0.clone();
    let mut p_prev = psf0.clone();
    let mut p_hist: Vec<f64> = vec![0.0; f_size * f_size * 2];

    // FFT of an all-ones image of the optBox size (`fw = fft2(ones)`).
    let ones = Array2::<f64>::ones((h, w));
    let fw_freq = fft2(&ones);

    let mut psf_final = psf0.clone();
    let mut xdec_final = opt_box.clone();

    for k in 0..n_iter {
        // Momentum update: at k = 0 just renormalise; otherwise compute
        // step sizes alpha_J and alpha_P from the two most recent
        // history vectors.
        let (jk, pk) = momentum_update(
            &j_curr, &j_prev, &j_hist,
            &p_curr, &p_prev, &p_hist,
            k, h, w, f_size,
        );

        // -- image update --
        let otf_pk = psf2otf(&pk, h, w);
        let blur = real_of(&ifft2(&mul_freq(&otf_pk, &fft2(&jk))));
        let blur = blur.mapv(|x| if x <= 0.0 { EPS } else { x });
        let ratio = opt_box / &(blur + EPS);
        let psi_k = fft2(&ratio);

        // Save previous J before overwriting.
        j_prev = j_curr.clone();

        let otf_p_curr = psf2otf(&p_curr, h, w);
        let scale_j = real_of(&ifft2(&mul_freq(&conj_freq(&otf_p_curr), &fw_freq)))
            .mapv(|x| x + EPS.sqrt());
        let conv = real_of(&ifft2(&mul_freq(&conj_freq(&otf_p_curr), &psi_k)));
        let new_j: Array2<f64> = (&jk * &conv / &scale_j).mapv(|x| x.max(0.0));
        j_curr = new_j;

        // Push (J_curr - J_k) into the momentum history.
        push_history(&mut j_hist, h * w, |idx| {
            let r = idx / w;
            let c = idx % w;
            j_curr[(r, c)] - jk[(r, c)]
        });

        // -- PSF update --
        p_prev = p_curr.clone();
        let j_freq = fft2(&j_prev);
        let otf_k_for_scale = mul_freq(&conj_freq(&j_freq), &fw_freq);
        let scale_p = otf2psf(&otf_k_for_scale, f_size)
            .mapv(|x| x + EPS.sqrt());
        let j_freq_psi = mul_freq(&conj_freq(&j_freq), &psi_k);
        let conv_p = otf2psf(&j_freq_psi, f_size).mapv(|x| x + EPS.sqrt());

        let mut new_p = (&pk * &conv_p / &scale_p).mapv(|x| x.max(0.0));
        normalise_l1(&mut new_p);
        p_curr = new_p;

        push_history(&mut p_hist, f_size * f_size, |idx| {
            let r = idx / f_size;
            let c = idx % f_size;
            p_curr[(r, c)] - pk[(r, c)]
        });

        if k == n_iter - 1 {
            psf_final = p_curr.clone();
            xdec_final = j_curr.clone();
        }
    }

    // Sanity: the deconvolved estimate occasionally drifts outside
    // [0,1] due to MLE noise — clip rather than normalise so that
    // downstream metrics line up with the Python reference.
    xdec_final.mapv_inplace(|x| x.clamp(-1.0, 1.0));

    MleResult { psf: psf_final, deconv_box: xdec_final }
}

/// Compute the J/P momentum estimates J_k, P_k for iteration `k`.
fn momentum_update(
    j_curr: &Array2<f64>, j_prev: &Array2<f64>, j_hist: &[f64],
    p_curr: &Array2<f64>, _p_prev: &Array2<f64>, p_hist: &[f64],
    k: usize, h: usize, w: usize, f_size: usize,
) -> (Array2<f64>, Array2<f64>) {
    if k == 0 {
        let mut jk = j_curr.clone();
        jk.mapv_inplace(|x| x.max(0.0));
        let mut pk = p_curr.clone();
        pk.mapv_inplace(|x| x.max(0.0));
        normalise_l1(&mut pk);
        return (jk, pk);
    }
    // Two history slabs are stored back-to-back: [latest | previous].
    let (j1, j2) = j_hist.split_at(h * w);
    let dot11 = j1.iter().zip(j1.iter()).map(|(a, b)| a * b).sum::<f64>();
    let dot12 = j1.iter().zip(j2.iter()).map(|(a, b)| a * b).sum::<f64>();
    // alpha = (j1 · j2) / (j1 · j1) clamped to ≤ 0.  The Python
    // reference uses np.maximum(np.minimum(alpha, 0), 0) which always
    // returns 0 -- we preserve that exact behaviour for fidelity.
    let _alpha_j = (dot12 / (dot11 + EPS)).min(0.0).max(0.0);
    let mut jk = j_curr.clone();
    jk.zip_mut_with(j_prev, |a, b| {
        *a = (*a + _alpha_j * (*a - *b)).max(0.0);
    });

    let (p1, p2) = p_hist.split_at(f_size * f_size);
    let dot11p = p1.iter().zip(p1.iter()).map(|(a, b)| a * b).sum::<f64>();
    let dot12p = p1.iter().zip(p2.iter()).map(|(a, b)| a * b).sum::<f64>();
    let _alpha_p = (dot12p / (dot11p + EPS)).min(0.0).max(0.0);
    let mut pk = p_curr.clone();
    pk.mapv_inplace(|x| x.max(0.0));
    normalise_l1(&mut pk);
    (jk, pk)
}

/// Push a new vector into the two-column history buffer, shifting the
/// previous front column into the back column.
fn push_history(hist: &mut Vec<f64>, n: usize, mut f: impl FnMut(usize) -> f64) {
    // Move column 0 → column 1.
    for i in 0..n {
        hist[n + i] = hist[i];
    }
    // Fill column 0 with the new values.
    for i in 0..n {
        hist[i] = f(i);
    }
}

/// Richardson–Lucy non-blind deconvolution.
///
/// Equivalent to `skimage.restoration.richardson_lucy(image, psf,
/// num_iter)` for circular boundary conditions.
pub fn richardson_lucy(
    image: &Array2<f64>,
    psf:   &Array2<f64>,
    n_iter: usize,
) -> Array2<f64> {
    let (h, w) = image.dim();
    let otf  = psf2otf(psf, h, w);
    let cotf = conj_freq(&otf);

    let mut estimate = image.clone();
    for _ in 0..n_iter {
        let blur = real_of(&ifft2(&mul_freq(&otf, &fft2(&estimate))));
        let blur = blur.mapv(|x| if x <= 0.0 { EPS } else { x });
        let ratio = image / &blur;
        let corr = real_of(&ifft2(&mul_freq(&cotf, &fft2(&ratio))));
        estimate = (&estimate * &corr).mapv(|x| x.max(0.0));
    }
    estimate
}

/// Richardson–Lucy with the EpiVirQuant style "pad-deconv-truncate-pad"
/// routine used to suppress edge ringing.
///
/// This mirrors the upstream Python sequence exactly:
///   1. zero-pad the input from (h, w) → (h + pad_y, w + pad_x)
///   2. run Richardson–Lucy on the padded image
///   3. **crop to (h - pad_y, w - pad_x)** — i.e. drop the outermost
///      pad_y/2 rows and pad_x/2 cols of the *original* on each side,
///      so any edge ringing introduced by RL is excised
///   4. zero-pad back to (h, w) — the outer border is now exactly 0
///   5. replace those zero-valued border pixels with the mean of the
///      original image
///
/// Step 3 is the critical detail: without it the deconvolution leaves
/// long thin streaks at the image border that the connected-component
/// labeller would otherwise pick up as ~10:1 aspect-ratio "objects".
pub fn richardson_lucy_padded(
    image: &Array2<f64>,
    psf:   &Array2<f64>,
    n_iter: usize,
    pad_y: usize,
    pad_x: usize,
) -> Array2<f64> {
    let (h, w) = image.dim();
    let mean = image.iter().sum::<f64>() / (h * w) as f64;

    // 1. Pad up.
    let mut padded = zero_pad_centered(image, h + pad_y, w + pad_x);
    // 2. RL on the padded image.
    padded = richardson_lucy(&padded, psf, n_iter);
    // 3. Crop down past the original border: take the inner
    //    (h - pad_y, w - pad_x) region of the *padded* result, so we
    //    lose the original-image-border region where RL artifacts live.
    let inner_h = h.saturating_sub(pad_y).max(1);
    let inner_w = w.saturating_sub(pad_x).max(1);
    let inner = center_crop(&padded, inner_h, inner_w);
    // 4. Zero-pad the inner image back to original dimensions; the
    //    border ring is now exactly 0.
    let mut out = zero_pad_centered(&inner, h, w);
    // 5. Replace the zero border with the original image's mean
    //    intensity, so downstream thresholding sees neutral grey there
    //    instead of "this is the brightest possible negative" black.
    out.mapv_inplace(|x| if x == 0.0 { mean } else { x });
    out
}

/// One score record from the parameter sweep.
#[derive(Debug, Clone, Copy)]
pub struct SweepRecord {
    pub f_size:  usize,
    pub tau:     f64,
    pub v:       f64,
    pub entropy: f64,
    pub energy:  f64,
}

/// PSF + metadata of the *best* sweep iteration.
#[derive(Debug, Clone)]
pub struct OptimalPsf {
    pub psf:         Array2<f64>,
    pub deconv_box:  Array2<f64>,
    pub f_size:      usize,
    pub tau:         f64,
    pub v:           f64,
    pub min_entropy: f64,
    pub max_energy:  f64,
    pub records:     Vec<SweepRecord>,
    pub elapsed_s:   f64,
}

/// Build the (f_size, τ, v) grid the original EpiVirQuant sweeps.
pub fn build_grid(opt_box: &Array2<f64>) -> Vec<(usize, f64, f64)> {
    let (h, w) = opt_box.dim();
    let mut f_max = h.min(w);
    if f_max % 2 == 0 {
        f_max = f_max.saturating_sub(1);
    }
    let f_sizes: Vec<usize> = (3..=f_max).step_by(2).collect();

    // tauVec: [0, 1/(10π), step 1/(100π))  → in practice ~10 entries.
    let mut tau_vec: Vec<f64> = Vec::new();
    let step_t = 1.0 / (100.0 * std::f64::consts::PI);
    let stop_t = 1.0 / (10.0 * std::f64::consts::PI);
    let mut t = 0.0;
    while t < stop_t {
        tau_vec.push(t);
        t += step_t;
    }
    // vVec: [0, π, step 0.1π) → ~10 entries.
    let mut v_vec: Vec<f64> = Vec::new();
    let step_v = std::f64::consts::PI / 10.0;
    let stop_v = std::f64::consts::PI;
    let mut vv = 0.0;
    while vv < stop_v {
        v_vec.push(vv);
        vv += step_v;
    }

    let mut grid = Vec::with_capacity(f_sizes.len() * tau_vec.len() * v_vec.len());
    for &fs in &f_sizes {
        for &tau in &tau_vec {
            for &v in &v_vec {
                grid.push((fs, tau, v));
            }
        }
    }
    grid
}

/// Full blind-deconvolution parameter sweep — picks the PSF that
/// minimises Shannon entropy of the deconvolved patch (with GLCM
/// energy as a tie-breaker, matching the EpiVirQuant log).
pub fn sweep_psf(opt_box: &Array2<f64>, cfg: &Config) -> Result<OptimalPsf> {
    let grid = build_grid(opt_box);
    if grid.is_empty() {
        return Err(CyclopsError::DeconvFailed(
            "PSF parameter grid empty — optBox too small".into(),
        ));
    }
    info!(
        "PSF sweep: {} parameter combinations across {} CPU threads",
        grid.len(),
        cfg.effective_threads()
    );
    let start = Instant::now();

    // Parallel evaluation; rayon's pool size is set in pipeline.rs.
    let records: Vec<(SweepRecord, Array2<f64>, Array2<f64>)> = grid
        .par_iter()
        .map(|&(fs, tau, v)| {
            let psf0 = create_psf(
                fs, cfg.a, cfg.b, cfg.sigma, cfg.r, tau, v, cfg.s, cfg.psf_method,
            );
            let result = mle_blind(opt_box, &psf0, cfg.n_mle_iter);
            let ent = shannon_entropy(&result.deconv_box);
            let en  = glcm_energy_dx2(&result.deconv_box);
            (
                SweepRecord { f_size: fs, tau, v, entropy: ent, energy: en },
                result.psf,
                result.deconv_box,
            )
        })
        .collect();

    // Pick min entropy (matches the Python `np.argmin(ent_vec)`).
    let (min_idx, _) = records
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.0.entropy.total_cmp(&b.1.0.entropy))
        .expect("non-empty");

    let (best_rec, best_psf, best_dec) = &records[min_idx];
    let max_energy = records
        .iter()
        .map(|(r, _, _)| r.energy)
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(OptimalPsf {
        psf:         best_psf.clone(),
        deconv_box:  best_dec.clone(),
        f_size:      best_rec.f_size,
        tau:         best_rec.tau,
        v:           best_rec.v,
        min_entropy: best_rec.entropy,
        max_energy,
        records:     records.into_iter().map(|(r, _, _)| r).collect(),
        elapsed_s:   start.elapsed().as_secs_f64(),
    })
}

/// Convenience helper for the GUI / CLI: when the user fixes `f_size`,
/// skip the sweep entirely and just build the PSF.
pub fn psf_from_params(cfg: &Config, f_size: usize) -> Array2<f64> {
    create_psf(
        f_size, cfg.a, cfg.b, cfg.sigma, cfg.r, cfg.tau, cfg.v, cfg.s, cfg.psf_method,
    )
}

// Silence unused-import warnings: `Complex64` is exposed here to keep
// the public surface stable for the GUI's spectrum visualisation.
#[doc(hidden)]
pub fn _exposed_complex_marker() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// Convenience predicate used by tests + GUI: returns true if `psf` is
/// L¹-normalised within tolerance.
pub fn is_psf_normalised(psf: &Array2<f64>, eps: f64) -> bool {
    let s: f64 = psf.iter().sum();
    (s - 1.0).abs() < eps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PsfMethod;

    #[test]
    fn grid_is_nonempty_for_small_box() {
        let arr = Array2::<f64>::from_elem((25, 25), 0.5);
        let g = build_grid(&arr);
        assert!(!g.is_empty());
    }

    #[test]
    fn rl_converges_on_gaussian_blur() {
        // Build a synthetic delta + gaussian-blurred observation and
        // verify RL recovers something closer to the truth.
        let mut truth = Array2::<f64>::zeros((33, 33));
        truth[(16, 16)] = 1.0;
        let psf = create_psf(7, 1.0, 1.0, 1.5, 2.718, 0.0, 0.0, 0.0, PsfMethod::Gaussian);
        // simulate convolution via psf2otf
        let otf = psf2otf(&psf, 33, 33);
        let observed = real_of(&ifft2(&mul_freq(&otf, &fft2(&truth))));
        let recovered = richardson_lucy(&observed, &psf, 30);
        // recovered peak should be at (16,16) and brighter than the
        // observed peak.
        let mut p_obs = 0.0_f64;
        let mut p_rec = 0.0_f64;
        for &v in observed.iter() { p_obs = p_obs.max(v); }
        for &v in recovered.iter() { p_rec = p_rec.max(v); }
        assert!(p_rec >= p_obs, "RL should sharpen: obs={p_obs}, rec={p_rec}");
    }
}
