//! Step 1 — find the optimisation box.
//!
//! Port of `epivirquant_pairing.getVP`. Given a DAPI calibration image
//! containing fluorescent microspheres, locate the closest pair of
//! objects, crop a padded bounding box around them, and return that
//! crop as the "optimisation box" used by the blind-deconv sweep.

use std::time::Instant;

use ndarray::{s, Array2};
use tracing::info;

use crate::config::Config;
use crate::error::{CyclopsError, Result};
use crate::regions::{label, region_props, RegionProp};
use crate::threshold::{binarise, binary_erosion, threshold_otsu};

/// Result of pairing.
#[derive(Debug, Clone)]
pub struct OptBox {
    pub patch:           Array2<f64>,
    pub min_distance_px: f64,
    pub min_distance_nm: f64,
    pub n_pairs:         usize,
    pub elapsed_s:       f64,
}

/// Find the optimisation box. Errors out if no qualifying pair exists.
pub fn find_opt_box(calibration: &Array2<f64>, cfg: &Config) -> Result<OptBox> {
    info!("Step 1 — scanning calibration image for VP candidates");
    let start = Instant::now();

    let t = threshold_otsu(calibration);
    let mask = binarise(calibration, t);
    let mask = binary_erosion(&mask, 1);

    let (labels, n_labels) = label(&mask);
    let props = region_props(&labels, calibration, n_labels);
    if props.len() < 2 {
        let dist_nm = (cfg.d_constraint as f64) * cfg.px2nm();
        return Err(CyclopsError::NoPairFound { dist_nm });
    }

    let centroids: Vec<(f64, f64)> = props.iter().map(|p| p.centroid).collect();
    let bboxes:    Vec<(usize, usize, usize, usize)> = props.iter().map(|p| p.bbox).collect();

    let (h, w) = calibration.dim();
    let mut opt_patch: Option<Array2<f64>> = None;
    let mut min_dist = f64::INFINITY;
    let mut n_pairs = 0_usize;
    let d_constraint = cfg.d_constraint as f64;
    let pad = cfg.pad as isize;

    for i in 0..props.len() {
        let (yi, xi) = centroids[i];
        for j in (i + 1)..props.len() {
            let (yj, xj) = centroids[j];
            let dy = yi - yj;
            let dx = xi - xj;
            let d = (dy * dy + dx * dx).sqrt();
            if d <= 0.0 || d >= d_constraint {
                continue;
            }
            let (a_min_r, a_min_c, a_max_r, a_max_c) = bboxes[i];
            let (b_min_r, b_min_c, b_max_r, b_max_c) = bboxes[j];

            let y_min = a_min_r.min(b_min_r) as isize - pad;
            let y_max = a_max_r.max(b_max_r) as isize + pad;
            let x_min = a_min_c.min(b_min_c) as isize - pad;
            let x_max = a_max_c.max(b_max_c) as isize + pad;
            if y_min < 0 || x_min < 0 || y_max as usize > h || x_max as usize > w {
                continue;
            }
            let crop = calibration
                .slice(s![y_min as usize..y_max as usize, x_min as usize..x_max as usize])
                .to_owned();

            // Confirm the crop still has exactly two objects after a
            // local Otsu + erosion.
            if !crop_has_pair(&crop) {
                continue;
            }
            n_pairs += 1;
            if d < min_dist {
                min_dist = d;
                opt_patch = Some(crop);
            }
        }
    }

    let patch = opt_patch.ok_or(CyclopsError::NoPairFound {
        dist_nm: d_constraint * cfg.px2nm(),
    })?;

    let elapsed = start.elapsed().as_secs_f64();
    let min_dist_nm = min_dist * cfg.px2nm();
    info!(
        "Step 1 — selected closest pair at {min_dist_nm:.1} nm ({n_pairs} candidates) \
         in {elapsed:.2} s"
    );

    Ok(OptBox {
        patch,
        min_distance_px: min_dist,
        min_distance_nm: min_dist_nm,
        n_pairs,
        elapsed_s: elapsed,
    })
}

fn crop_has_pair(crop: &Array2<f64>) -> bool {
    let t = threshold_otsu(crop);
    let mask = binarise(crop, t);
    let mask = binary_erosion(&mask, 1);
    let (_, n) = label(&mask);
    n == 2
}

/// Helper for diagnostics: return all detected regions in the
/// calibration image so the GUI can render them as overlays.
pub fn detect_regions(calibration: &Array2<f64>) -> Vec<RegionProp> {
    let t = threshold_otsu(calibration);
    let mask = binarise(calibration, t);
    let mask = binary_erosion(&mask, 1);
    let (labels, n_labels) = label(&mask);
    region_props(&labels, calibration, n_labels)
}
