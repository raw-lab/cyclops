//! Step 4 — quantify viral-like particles (and other objects) on the
//! FITC images.
//!
//! Port of `epivirquant_masks.generate_masks`. For every FITC image we
//! Richardson–Lucy deconvolve, threshold (Otsu + 0.01), label, measure
//! object properties, drop obvious false positives (eccentricity → 1
//! or scaled semi-major beyond `SM_constraint`), then return the
//! per-object table.

use std::time::Instant;

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::{Config, SizeMetric};
use crate::deconv::richardson_lucy_padded;
use crate::error::Result;
use crate::image_io::GrayImage;
use crate::regions::{label, region_props};
use crate::threshold::{binarise, threshold_otsu};

/// One row of the per-object output table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRecord {
    pub file_name:     String,
    pub object_id:     u32,
    pub size_nm:       f64,
    pub axis_major_nm: f64,
    pub axis_minor_nm: f64,
    pub x_px:          f64,
    pub y_px:          f64,
    pub intensity:     f64,
    pub eccentricity:  f64,
    pub area_px:       u64,
}

#[derive(Debug, Clone)]
pub struct PerImageQuant {
    pub name:         String,
    pub n_objects:    usize,
    pub mean_size_nm: f64,
    pub records:      Vec<ObjectRecord>,
    pub elapsed_s:    f64,
}

#[derive(Debug, Clone)]
pub struct QuantReport {
    pub per_image:     Vec<PerImageQuant>,
    pub mean_size_nm:  f64,
    /// Histogram-style size breakdown, in nm bands.
    pub size_bands:    SizeBands,
    pub elapsed_s:     f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SizeBands {
    pub lt_100:           u64, // < 100 nm        — putative tiny viruses, free DNA
    pub vlp_100_220:      u64, // 100 – 220 nm    — VLPs
    pub small_220_500:    u64, // 220 – 500 nm    — small bacteria, large VLPs (jumbophages)
    pub bact_500_1200:    u64, // 500 – 1200 nm   — typical bacteria/archaea
    pub large_1200_3000:  u64, // 1200 – 3000 nm  — large bacteria, small protists
    pub protist_gt_3000:  u64, // > 3000 nm       — protists
}

pub fn quantify(
    fitc:        &[GrayImage],
    psf:         &Array2<f64>,
    correction:  f64,
    cfg:         &Config,
) -> Result<QuantReport> {
    info!("Step 4 — quantifying {} FITC images (CORR = {:.4})", fitc.len(), correction);
    let start = Instant::now();

    let per_image: Vec<PerImageQuant> = fitc
        .par_iter()
        .map(|img| quantify_single(img, psf, correction, cfg))
        .collect::<Result<Vec<_>>>()?;

    // Aggregate.
    let all_sizes: Vec<f64> = per_image
        .iter()
        .flat_map(|p| p.records.iter().map(|r| r.size_nm))
        .collect();
    let mean = if all_sizes.is_empty() {
        0.0
    } else {
        all_sizes.iter().sum::<f64>() / all_sizes.len() as f64
    };

    let mut bands = SizeBands::default();
    for s in &all_sizes {
        match *s {
            x if x < 100.0                   => bands.lt_100          += 1,
            x if (100.0..=220.0).contains(&x)=> bands.vlp_100_220     += 1,
            x if (220.0..500.0).contains(&x) => bands.small_220_500   += 1,
            x if (500.0..=1200.0).contains(&x) => bands.bact_500_1200 += 1,
            x if (1200.0..3000.0).contains(&x) => bands.large_1200_3000 += 1,
            _                                => bands.protist_gt_3000 += 1,
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    info!(
        "Step 4 — {} objects, mean {:.2} nm, in {:.2}s",
        all_sizes.len(), mean, elapsed
    );
    Ok(QuantReport {
        per_image,
        mean_size_nm: mean,
        size_bands:   bands,
        elapsed_s:    elapsed,
    })
}

fn quantify_single(
    image:      &GrayImage,
    psf:        &Array2<f64>,
    correction: f64,
    cfg:        &Config,
) -> Result<PerImageQuant> {
    let start = Instant::now();
    let (h, w) = image.shape();
    // pad_L = int(L*(32/L)) = 32, pad_dim = 24 in original.
    let pad_y = 32.min(h);
    let pad_x = 24.min(w);
    let deconv = richardson_lucy_padded(&image.data, psf, cfg.n_lr_iter, pad_y, pad_x);

    // Otsu + 0.01 offset (matches `threshold_otsu(XG) + 0.01`).
    let t = threshold_otsu(&deconv) + 0.01;
    let mask = binarise(&deconv, t);
    let (labels, n) = label(&mask);
    let props = region_props(&labels, &deconv, n);

    let px2nm = cfg.px2nm();
    let mut records = Vec::with_capacity(props.len());
    let mut obj_id: u32 = 0;
    for p in &props {
        let semi_major_nm = p.axis_major_length * px2nm * correction;
        // Drop obvious false positives:
        if semi_major_nm == 0.0 || p.eccentricity > 0.9999 {
            continue;
        }
        if semi_major_nm > cfg.sm_constraint {
            continue;
        }
        let dia_nm = match cfg.size_metric {
            SizeMetric::EquivalentDiameter => p.equivalent_diameter * px2nm * correction,
            SizeMetric::AverageAxes => {
                0.5 * (p.axis_major_length + p.axis_minor_length) * px2nm * correction
            }
        };
        let (cy, cx) = p.centroid;
        obj_id += 1;
        records.push(ObjectRecord {
            file_name:     image.short_name(),
            object_id:     obj_id,
            size_nm:       round2(dia_nm),
            axis_major_nm: round2(p.axis_major_length * px2nm * correction),
            axis_minor_nm: round2(p.axis_minor_length * px2nm * correction),
            x_px:          cx,
            y_px:          cy,
            intensity:     p.intensity_mean,
            eccentricity:  p.eccentricity,
            area_px:       p.area,
        });
    }

    let mean = if records.is_empty() {
        0.0
    } else {
        records.iter().map(|r| r.size_nm).sum::<f64>() / records.len() as f64
    };

    Ok(PerImageQuant {
        name:         image.short_name(),
        n_objects:    records.len(),
        mean_size_nm: round2(mean),
        records,
        elapsed_s:    start.elapsed().as_secs_f64(),
    })
}

#[inline]
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
