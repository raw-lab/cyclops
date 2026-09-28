//! Otsu thresholding + light morphology used throughout the pipeline.
//!
//! Direct port of `skimage.filters.threshold_otsu` and
//! `scipy.ndimage.binary_erosion` (3×3 cross / square structuring
//! element).

use ndarray::Array2;

/// Compute the Otsu threshold for an image whose values are in
/// `[0, 1]`. Internally bins values into 256 levels.
pub fn threshold_otsu(image: &Array2<f64>) -> f64 {
    const NBINS: usize = 256;
    let mut hist = [0u64; NBINS];
    for &v in image.iter() {
        let bin = (v.clamp(0.0, 1.0) * (NBINS as f64 - 1.0)).round() as usize;
        hist[bin] += 1;
    }
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;

    // Pre-compute cumulative weight and weighted mean.
    let mut sum_total = 0.0_f64;
    for (i, &h) in hist.iter().enumerate() {
        sum_total += (i as f64) * (h as f64);
    }

    let mut w_b = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut max_var = 0.0_f64;
    let mut threshold_bin = 0_usize;

    for (t, &h) in hist.iter().enumerate() {
        w_b += h as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total_f - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += (t as f64) * (h as f64);
        let m_b = sum_b / w_b;
        let m_f = (sum_total - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f).powi(2);
        if var_between > max_var {
            max_var = var_between;
            threshold_bin = t;
        }
    }

    // Return the *center* of the chosen bin (matches skimage's
    // `bin_centers[idx]` convention) so that for a perfectly bimodal
    // [0, 1] image, the threshold lands strictly in (0, 1) — not on the
    // bin edge at 0 where it would coincide with the lower mode.
    (threshold_bin as f64 + 0.5) / NBINS as f64
}

/// Binarise an image using the supplied threshold (`> threshold`).
pub fn binarise(image: &Array2<f64>, threshold: f64) -> Array2<bool> {
    image.mapv(|x| x > threshold)
}

/// `n_iters` rounds of binary erosion with a 3×3 cross structuring
/// element (matching `scipy.ndimage.binary_erosion` defaults).
pub fn binary_erosion(mask: &Array2<bool>, n_iters: usize) -> Array2<bool> {
    let mut out = mask.clone();
    for _ in 0..n_iters {
        out = erode_once(&out);
    }
    out
}

fn erode_once(mask: &Array2<bool>) -> Array2<bool> {
    let (h, w) = mask.dim();
    let mut out = Array2::<bool>::default((h, w));
    for y in 0..h {
        for x in 0..w {
            // 3×3 cross: centre + 4-neighbours all true ⇒ true.
            let here = mask[(y, x)];
            if !here {
                out[(y, x)] = false;
                continue;
            }
            let up    = if y == 0       { false } else { mask[(y - 1, x)] };
            let down  = if y == h - 1   { false } else { mask[(y + 1, x)] };
            let left  = if x == 0       { false } else { mask[(y, x - 1)] };
            let right = if x == w - 1   { false } else { mask[(y, x + 1)] };
            out[(y, x)] = here && up && down && left && right;
        }
    }
    out
}

/// Image-wide Shannon entropy in bits over a 256-level histogram.
///
/// Used as the optimisation target inside the blind-deconvolution
/// sweep.
pub fn shannon_entropy(image: &Array2<f64>) -> f64 {
    const NBINS: usize = 256;
    let mut hist = [0u64; NBINS];
    for &v in image.iter() {
        let bin = (v.clamp(-1.0, 1.0).abs() * (NBINS as f64 - 1.0)).round() as usize;
        hist[bin] += 1;
    }
    let total: f64 = image.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let mut h = 0.0_f64;
    for &c in &hist {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total;
        h -= p * p.log2();
    }
    h
}

/// Gray-Level Co-occurrence Matrix energy for a horizontal offset of
/// `dx = 2`, matching the EpiVirQuant call
/// `graycomatrix(.., [2], [0], symmetric=True, normed=True)`.
pub fn glcm_energy_dx2(image: &Array2<f64>) -> f64 {
    const LEVELS: usize = 256;
    let (h, w) = image.dim();
    if w < 3 {
        return 0.0;
    }
    let mut glcm = vec![0.0_f64; LEVELS * LEVELS];
    let quantize = |v: f64| -> usize {
        ((v.clamp(-1.0, 1.0).abs()) * (LEVELS as f64 - 1.0)).round() as usize
    };
    for y in 0..h {
        for x in 0..(w - 2) {
            let a = quantize(image[(y, x)]);
            let b = quantize(image[(y, x + 2)]);
            glcm[a * LEVELS + b] += 1.0;
            glcm[b * LEVELS + a] += 1.0; // symmetric=True
        }
    }
    let sum: f64 = glcm.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    glcm.iter().map(|p| (p / sum).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn otsu_separates_bimodal() {
        let arr = array![
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let t = threshold_otsu(&arr);
        assert!(t > 0.0 && t < 1.0, "Otsu threshold should land between modes: {t}");
    }

    #[test]
    fn erosion_shrinks() {
        let mask = array![
            [false, false, false, false, false],
            [false, true,  true,  true,  false],
            [false, true,  true,  true,  false],
            [false, true,  true,  true,  false],
            [false, false, false, false, false],
        ];
        let eroded = binary_erosion(&mask, 1);
        assert!(eroded[(2, 2)]);
        assert!(!eroded[(1, 1)]);
    }
}
