//! Small 2-D FFT toolbox used by the deconvolution kernels.
//!
//! Wraps [`rustfft`] to provide:
//!
//! * [`fft2`] / [`ifft2`]                   — complex 2-D transforms,
//! * [`psf2otf`] / [`otf2psf`]              — MATLAB-style PSF↔OTF
//!   conversions matching the `pypher` reference implementation used
//!   in EpiVirQuant,
//! * [`zero_pad_centered`]                  — symmetric zero-padding
//!   (port of `pypher.pypher.zero_pad(..., position='center')`).

use ndarray::{s, Array2};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Forward 2-D FFT (row-major). Returns a complex matrix the same
/// shape as the input.
pub fn fft2(input: &Array2<f64>) -> Array2<Complex64> {
    let (h, w) = input.dim();
    let mut out = Array2::<Complex64>::from_shape_fn((h, w), |(y, x)| {
        Complex64::new(input[(y, x)], 0.0)
    });
    fft2_inplace(&mut out, /*inverse=*/ false);
    out
}

/// Inverse 2-D FFT. Returns the complex result; callers typically take
/// the real part.
pub fn ifft2(input: &Array2<Complex64>) -> Array2<Complex64> {
    let mut out = input.clone();
    fft2_inplace(&mut out, /*inverse=*/ true);
    out
}

fn fft2_inplace(buf: &mut Array2<Complex64>, inverse: bool) {
    let (h, w) = buf.dim();
    let mut planner = FftPlanner::<f64>::new();
    let fft_row = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    // Row pass
    for mut row in buf.outer_iter_mut() {
        let row_slice = row.as_slice_mut().expect("row should be contiguous");
        fft_row.process(row_slice);
    }
    // Column pass
    let fft_col = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col_buf = vec![Complex64::new(0.0, 0.0); h];
    for x in 0..w {
        for y in 0..h {
            col_buf[y] = buf[(y, x)];
        }
        fft_col.process(&mut col_buf);
        for y in 0..h {
            buf[(y, x)] = col_buf[y];
        }
    }
    if inverse {
        let scale = 1.0 / ((h as f64) * (w as f64));
        buf.mapv_inplace(|c| c * scale);
    }
}

/// Real-part extractor (used after `ifft2`).
pub fn real_of(arr: &Array2<Complex64>) -> Array2<f64> {
    arr.mapv(|c| c.re)
}

/// Centred zero-pad to `(out_h, out_w)`. Mirrors
/// `pypher.pypher.zero_pad(image, (out_h, out_w), position='center')`.
pub fn zero_pad_centered(input: &Array2<f64>, out_h: usize, out_w: usize) -> Array2<f64> {
    let (h, w) = input.dim();
    assert!(out_h >= h && out_w >= w, "target shape smaller than input");
    let mut out = Array2::<f64>::zeros((out_h, out_w));
    let y0 = (out_h - h) / 2;
    let x0 = (out_w - w) / 2;
    out.slice_mut(s![y0..y0 + h, x0..x0 + w]).assign(input);
    out
}

/// Centre-crop back to `(h, w)`.
pub fn center_crop(input: &Array2<f64>, h: usize, w: usize) -> Array2<f64> {
    let (in_h, in_w) = input.dim();
    let y0 = in_h.saturating_sub(h) / 2;
    let x0 = in_w.saturating_sub(w) / 2;
    input.slice(s![y0..y0 + h, x0..x0 + w]).to_owned()
}

/// MATLAB-style `psf2otf` — circularly shifts the PSF so its centre is
/// at `(0,0)`, zero-pads to the target shape, then takes the FFT.
///
/// This matches the convention used by `pypher.psf2otf` and therefore
/// the original EpiVirQuant pipeline.
pub fn psf2otf(psf: &Array2<f64>, out_h: usize, out_w: usize) -> Array2<Complex64> {
    let (h, w) = psf.dim();
    // Place PSF in the top-left of a zero-padded array.
    let mut padded = Array2::<f64>::zeros((out_h, out_w));
    padded.slice_mut(s![..h, ..w]).assign(psf);
    // Centre is floor(h/2), floor(w/2) in numpy/MATLAB.
    let shift_y = h / 2;
    let shift_x = w / 2;
    let shifted = circshift(&padded, shift_y as isize * -1, shift_x as isize * -1);
    fft2(&shifted)
}

/// Inverse of [`psf2otf`]. Returns the real part of the IFFT, cropped
/// to the requested PSF size, with the centre shifted back.
pub fn otf2psf(otf: &Array2<Complex64>, f_size: usize) -> Array2<f64> {
    let (h, w) = otf.dim();
    let ifft = ifft2(otf);
    let real = real_of(&ifft);
    let shifted = circshift(&real, (h / 2) as isize, (w / 2) as isize);
    // Crop the centre.
    let y0 = (h - f_size) / 2;
    let x0 = (w - f_size) / 2;
    shifted.slice(s![y0..y0 + f_size, x0..x0 + f_size]).to_owned()
}

/// Toroidal shift by `(dy, dx)` (positive shifts move the image down /
/// right). Equivalent to `numpy.roll`.
pub fn circshift(input: &Array2<f64>, dy: isize, dx: isize) -> Array2<f64> {
    let (h, w) = input.dim();
    let mut out = Array2::<f64>::zeros((h, w));
    let h_i = h as isize;
    let w_i = w as isize;
    for y in 0..h_i {
        let sy = ((y + dy) % h_i + h_i) % h_i;
        for x in 0..w_i {
            let sx = ((x + dx) % w_i + w_i) % w_i;
            out[(sy as usize, sx as usize)] = input[(y as usize, x as usize)];
        }
    }
    out
}

/// Frequency-domain multiplication.
#[inline]
pub fn mul_freq(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (h, w) = a.dim();
    Array2::from_shape_fn((h, w), |(y, x)| a[(y, x)] * b[(y, x)])
}

/// Element-wise complex conjugate.
#[inline]
pub fn conj_freq(a: &Array2<Complex64>) -> Array2<Complex64> {
    a.mapv(|c| c.conj())
}
