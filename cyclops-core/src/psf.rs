//! Tunable point-spread function construction.
//!
//! Port of `epivirquant_decon.create_PSF`. The three families correspond
//! to equations 5 / 6 / hybrid in Figueroa III *et al.* (2026):
//!
//! * **γ-sinc** (`gam`)  — `PSF(x,y) = a · exp(v / (Γ(1+τR) · Γ(1-τR))) + s`
//!   with `R = sqrt(x² + y²)`. This is the EpiVirQuant default and the
//!   PSF whose `(fSize, τ, v)` triple is swept during blind
//!   deconvolution.
//! * **Gaussian** (`gau`) — standard 2-D isotropic Gaussian with stdev
//!   `sigma`.
//! * **Hybrid** (`hyb`)  — flat-top kernel (1/N²) used as a numerically
//!   stable seed for difficult images.
//!
//! All three are returned **L¹-normalised** (sum to 1) which is what
//! the downstream MLE update requires.

use ndarray::Array2;
// Float trait import removed — gamma/normalisation use f64 inherent methods.

use crate::config::PsfMethod;

/// Build a PSF of size `f_size × f_size`.
///
/// Parameters mirror the EpiVirQuant CLI flags:
/// * `a`  — gaussian-component amplitude (γ-sinc, gaussian)
/// * `_b` — sinc-component amplitude (hybrid; currently unused, kept
///   for parameter-compatibility with the Python reference)
/// * `sigma` — gaussian stdev
/// * `_r`    — sinc width (hybrid; reserved)
/// * `tau`   — γ-sinc periodicity τ
/// * `v`     — γ-sinc vertical stretch v
/// * `s`     — γ-sinc vertical shift s
pub fn create_psf(
    f_size: usize,
    a:      f64,
    _b:     f64,
    sigma:  f64,
    _r:     f64,
    tau:    f64,
    v:      f64,
    s:      f64,
    method: PsfMethod,
) -> Array2<f64> {
    assert!(f_size >= 1, "PSF size must be ≥ 1");
    let n = f_size as f64;
    let mut psf = Array2::<f64>::zeros((f_size, f_size));

    match method {
        PsfMethod::GammaSinc => {
            // Symmetric coordinate grid x,y ∈ linspace(-f_size, f_size, f_size).
            for i in 0..f_size {
                let xi = lerp(-n, n, (i as f64) / (n - 1.0).max(1.0));
                for j in 0..f_size {
                    let yj = lerp(-n, n, (j as f64) / (n - 1.0).max(1.0));
                    let r = (xi * xi + yj * yj).sqrt();
                    let denom = (gamma_safe(1.0 + tau * r))
                        * (gamma_safe(1.0 - tau * r));
                    let val = if denom.is_finite() && denom.abs() > f64::EPSILON {
                        a * (v / denom).exp() + s
                    } else {
                        s
                    };
                    // Guard against floating overflow far from the
                    // origin: cap to a finite envelope.
                    psf[(i, j)] = if val.is_finite() { val } else { 0.0 };
                }
            }
        }
        PsfMethod::Gaussian => {
            let half = (n - 1.0) * 0.5;
            for i in 0..f_size {
                let xi = (i as f64) - half;
                for j in 0..f_size {
                    let yj = (j as f64) - half;
                    psf[(i, j)] = (-(xi * xi + yj * yj) / (2.0 * sigma * sigma)).exp();
                }
            }
            let eps_cut = f64::EPSILON * max_abs(&psf);
            psf.mapv_inplace(|x| if x < eps_cut { 0.0 } else { x });
        }
        PsfMethod::Hybrid => {
            // Flat top — used as a robust seed for the MLE sweep.
            let inv = 1.0 / (n * n);
            psf.fill(inv);
        }
    }

    let sum: f64 = psf.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        psf.mapv_inplace(|x| x / sum);
    }
    psf
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
fn max_abs(arr: &Array2<f64>) -> f64 {
    arr.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Continuous Γ(x) using Lanczos g=7. Returns a *very* large finite
/// number rather than ±∞ for poles so the PSF construction never
/// produces NaNs.
fn gamma_safe(x: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9), from Wikipedia / NR
    const P: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if !x.is_finite() {
        return 0.0;
    }
    if x < 0.5 {
        // Reflection: Γ(1-x) = π / (sin(πx) Γ(x))
        let denom = (std::f64::consts::PI * x).sin();
        if denom.abs() < 1e-300 {
            return 1.0e300; // pole → huge value, denom in PSF kills the term
        }
        return std::f64::consts::PI / (denom * gamma_safe(1.0 - x));
    }
    let xc = x - 1.0;
    let mut acc = 0.99999999999980993_f64;
    for (i, &p) in P.iter().enumerate() {
        acc += p / (xc + (i as f64) + 1.0);
    }
    let t = xc + 7.0 + 0.5;
    let res = (2.0_f64 * std::f64::consts::PI).sqrt()
        * t.powf(xc + 0.5)
        * (-t).exp()
        * acc;
    if res.is_finite() {
        res
    } else {
        1.0e300
    }
}

/// L1-normalise a PSF in place. No-op if the sum is non-positive.
pub fn normalise_l1(psf: &mut Array2<f64>) {
    let s: f64 = psf.iter().sum();
    if s.is_finite() && s > 0.0 {
        psf.mapv_inplace(|x| x / s);
    }
}

/// Coordinate-grid helper exposed mainly for plotting (the original
/// `[X, Y] = meshgrid(...)` return values).
pub fn meshgrid(f_size: usize) -> (Array2<f64>, Array2<f64>) {
    let n = f_size as f64;
    let mut x = Array2::<f64>::zeros((f_size, f_size));
    let mut y = Array2::<f64>::zeros((f_size, f_size));
    for i in 0..f_size {
        let xi = lerp(-n, n, (i as f64) / (n - 1.0).max(1.0));
        for j in 0..f_size {
            let yj = lerp(-n, n, (j as f64) / (n - 1.0).max(1.0));
            x[(i, j)] = xi;
            y[(i, j)] = yj;
        }
    }
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_sinc_is_normalised() {
        let psf = create_psf(11, 1.0, 1.0, 1.0, 2.718, 0.5, 2.25, 0.0, PsfMethod::GammaSinc);
        let s: f64 = psf.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "PSF must sum to 1, got {s}");
    }

    #[test]
    fn gaussian_is_normalised() {
        let psf = create_psf(9, 1.0, 1.0, 1.5, 2.718, 0.0, 0.0, 0.0, PsfMethod::Gaussian);
        let s: f64 = psf.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hybrid_is_uniform() {
        let psf = create_psf(5, 1.0, 1.0, 1.0, 2.718, 0.0, 0.0, 0.0, PsfMethod::Hybrid);
        for &v in psf.iter() {
            assert!((v - 1.0 / 25.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gamma_safe_known_values() {
        // Γ(1) = 1, Γ(5) = 24, Γ(0.5) = √π
        assert!((gamma_safe(1.0) - 1.0).abs() < 1e-9);
        assert!((gamma_safe(5.0) - 24.0).abs() < 1e-7);
        assert!((gamma_safe(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-9);
    }
}
