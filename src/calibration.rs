//! Step 3 — calibration: derive the size-correction factor `CORR`.
//!
//! Port of `epivirquant_calibration.get_corrolation`.  For every DAPI
//! microsphere image we:
//!
//! 1. apply `richardson_lucy_padded` with the optimal PSF,
//! 2. threshold (Otsu) and label,
//! 3. measure each object's diameter (in nm),
//! 4. average per-image,
//!
//! then return `CORR = sphere_size / mean_measured_size`.

use std::time::Instant;

use ndarray::Array2;
use rayon::prelude::*;
use tracing::{debug, info};

use crate::config::{Config, SizeMetric};
use crate::deconv::richardson_lucy_padded;
use crate::error::Result;
use crate::image_io::GrayImage;
use crate::regions::{label, region_props};
use crate::threshold::{binarise, threshold_otsu};

#[derive(Debug, Clone)]
pub struct PerImageStats {
    pub name:           String,
    pub n_objects:      usize,
    pub mean_size_nm:   f64,
    pub sizes_nm:       Vec<f64>,
    pub intensities:    Vec<f64>,
    pub eccentricities: Vec<f64>,
    pub elapsed_s:      f64,
}

#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub per_image:        Vec<PerImageStats>,
    pub mean_size_nm:     f64,
    pub correction:       f64,
    pub measurement_err:  f64,
    pub n_images:         usize,
    pub elapsed_s:        f64,
}

pub fn calibrate(
    dapi:    &[GrayImage],
    psf:     &Array2<f64>,
    cfg:     &Config,
    prog:    &dyn crate::progress::Progress,
) -> Result<CalibrationReport> {
    info!("Step 3 — Richardson-Lucy + calibration on {} images", dapi.len());
    let start = Instant::now();

    // Per-image work parallelised across CPU cores. Rayon thread pool
    // configured in `pipeline::run_pipeline`. An atomic counter lets each
    // finished image tick the progress bar from whichever worker thread
    // completed it (order-independent, contention-free).
    let total = dapi.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let per_image: Vec<PerImageStats> = dapi
        .par_iter()
        .map(|img| {
            let r = process_single(img, psf, cfg);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            prog.tick(n, total);
            r
        })
        .collect::<Result<Vec<_>>>()?;

    // Aggregate.
    let mean_size_nm = per_image
        .iter()
        .filter(|s| s.n_objects > 0)
        .map(|s| s.mean_size_nm)
        .sum::<f64>()
        / per_image.iter().filter(|s| s.n_objects > 0).count().max(1) as f64;

    let mut all_sizes: Vec<f64> = per_image
        .iter()
        .flat_map(|p| p.sizes_nm.clone())
        .collect();
    all_sizes.sort_by(|a, b| a.total_cmp(b));
    let sigma = std_dev(&all_sizes);
    let correction = if mean_size_nm > 0.0 {
        cfg.sphere_size_nm / mean_size_nm
    } else {
        1.0
    };

    let elapsed = start.elapsed().as_secs_f64();
    info!(
        "Step 3 — mean diameter {:.2} nm ⇒ CORR = {:.4} (σ = {:.2} nm) in {:.2}s",
        mean_size_nm, correction, sigma, elapsed
    );

    Ok(CalibrationReport {
        per_image,
        mean_size_nm,
        correction,
        measurement_err: sigma,
        n_images: dapi.len(),
        elapsed_s: elapsed,
    })
}

fn process_single(
    image: &GrayImage,
    psf:   &Array2<f64>,
    cfg:   &Config,
) -> Result<PerImageStats> {
    let start = Instant::now();
    debug!("calibrating {}", image.short_name());
    let (h, w) = image.shape();
    // Original used pad = int(L * (16/L)) which is just 16 in practice;
    // keep the same behaviour.
    let pad_y = 16.min(h);
    let pad_x = 16.min(w);
    let deconv = richardson_lucy_padded(&image.data, psf, cfg.n_lr_iter, pad_y, pad_x);

    let t = threshold_otsu(&deconv);
    let mask = binarise(&deconv, t);
    let (labels, n) = label(&mask);
    let props = region_props(&labels, &deconv, n);

    let mut sizes = Vec::with_capacity(props.len());
    let mut intensities = Vec::with_capacity(props.len());
    let mut eccs = Vec::with_capacity(props.len());
    for p in &props {
        let dia_px = match cfg.size_metric {
            SizeMetric::EquivalentDiameter => p.equivalent_diameter,
            SizeMetric::AverageAxes        => 0.5 * (p.axis_major_length + p.axis_minor_length),
        };
        sizes.push(dia_px * cfg.px2nm());
        intensities.push(p.intensity_mean);
        eccs.push(p.eccentricity);
    }

    let mean_size_nm = if sizes.is_empty() {
        0.0
    } else {
        sizes.iter().copied().sum::<f64>() / sizes.len() as f64
    };

    Ok(PerImageStats {
        name:           image.short_name(),
        n_objects:      sizes.len(),
        mean_size_nm:   round2(mean_size_nm),
        sizes_nm:       sizes,
        intensities,
        eccentricities: eccs,
        elapsed_s:      start.elapsed().as_secs_f64(),
    })
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}
