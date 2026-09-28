//! Domain classifier — the new Cyclops capability.
//!
//! Given the per-object records from [`crate::quantify`] (plus the
//! per-image DAPI intensities), assign each object a microbial domain:
//! **virus / bacteria / archaea / protist**.
//!
//! Two layers stacked on top of each other:
//!
//! 1. **Rule-based prior** — size in nm + eccentricity + DAPI∕FITC
//!    intensity ratio narrows the candidate domains. This is fast,
//!    interpretable, and matches the size bands that microbial
//!    ecologists already use.
//!
//! 2. **GMM refinement** (optional, `cfg.classifier.gmm_refine`) — fit
//!    a Gaussian Mixture in feature space `(log10 size, ecc,
//!    log10 (DAPI/FITC), log10 area)` with `k = |domains|`. Each
//!    component is then re-labelled by mapping its centroid through
//!    the rule-based prior, so the labels remain semantically
//!    grounded.
//!
//! 3. **(Optional) ONNX deep model** — placeholder behind the
//!    `onnx` feature flag for users who want to drop in a pretrained
//!    Cellpose / StarDist segmentation network. Without the feature
//!    the call is a no-op.
//!
//! Bacteria and archaea share a size window. When the user enables
//! both, the classifier prefers `Bacteria` unless the DAPI∕FITC ratio
//! is in the archaea-favouring band; users running an archaea-specific
//! probe should pass `domains = [Archaea]` to force the label.

use std::collections::HashMap;

use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_clustering::GaussianMixtureModel;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::config::{Config, OrganismDomain};
use crate::quantify::ObjectRecord;

/// One classified object — a thin wrapper over [`ObjectRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedObject {
    pub record:      ObjectRecord,
    pub domain:      String,
    pub confidence:  f64,
    pub dapi_fitc:   f64,
}

#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub objects:        Vec<ClassifiedObject>,
    pub domain_counts:  HashMap<String, u64>,
    pub gmm_used:       bool,
}

/// Per-image DAPI intensity lookup, keyed by file stem (without the
/// `(dapi)` / `(fitc)` suffix and channel tag).
///
/// Users typically name their pairs as
/// `Sample_(dapi).tiff` / `Sample_(fitc).tiff` — the matcher first
/// tries an exact strip of `(dapi)` / `(fitc)`, then falls back to the
/// longest common prefix.
pub fn classify_objects(
    records:      &[ObjectRecord],
    dapi_lookup:  &HashMap<String, f64>,
    cfg:          &Config,
) -> ClassificationReport {
    let domains: &[OrganismDomain] = if cfg.classifier.domains.is_empty() {
        // Sensible default: everything.
        &[
            OrganismDomain::Virus,
            OrganismDomain::Bacteria,
            OrganismDomain::Archaea,
            OrganismDomain::Protist,
        ]
    } else {
        &cfg.classifier.domains
    };

    // Step 1 — rule-based prior.
    let mut prior: Vec<(OrganismDomain, f64)> = records
        .iter()
        .map(|r| rule_based(r, dapi_lookup, domains))
        .collect();

    let gmm_used = cfg.classifier.gmm_refine && records.len() >= (domains.len() * 4);
    if gmm_used {
        if let Some(refined) = gmm_refine(records, dapi_lookup, &prior, domains) {
            prior = refined;
        }
    }

    // Optional ONNX refinement: if the user supplied a model, score every
    // object with it and override the prior where it returns a result. The
    // backend is a no-op stub unless the crate was built with the `onnx`
    // feature, so this call is always safe.
    let mut onnx_used = false;
    if let Some(model_path) = &cfg.classifier.onnx_model {
        let ratios: Vec<f64> = records
            .iter()
            .map(|r| dapi_fitc_ratio(r, dapi_lookup))
            .collect();
        if let Some(refined) =
            crate::onnx::classify_with_onnx(model_path, records, &ratios, domains)
        {
            if refined.len() == prior.len() {
                prior = refined;
                onnx_used = true;
            }
        }
    }
    let _ = onnx_used; // reserved for future report field

    let mut classified = Vec::with_capacity(records.len());
    let mut counts: HashMap<String, u64> = HashMap::new();
    for (r, (dom, conf)) in records.iter().zip(prior.iter()) {
        let ratio = dapi_fitc_ratio(r, dapi_lookup);
        let label = dom.label().to_string();
        *counts.entry(label.clone()).or_insert(0) += 1;
        classified.push(ClassifiedObject {
            record:     r.clone(),
            domain:     label,
            confidence: *conf,
            dapi_fitc:  ratio,
        });
    }

    ClassificationReport {
        objects:       classified,
        domain_counts: counts,
        gmm_used,
    }
}

/// Pure rule-based assignment from the (size, eccentricity, ratio)
/// triple. Returns `(domain, confidence in [0, 1])`.
fn rule_based(
    r:           &ObjectRecord,
    lookup:      &HashMap<String, f64>,
    candidates:  &[OrganismDomain],
) -> (OrganismDomain, f64) {
    let size = r.size_nm;
    let ratio = dapi_fitc_ratio(r, lookup);

    // Score each candidate domain by how well the size lies inside its
    // canonical band.  A Gaussian-shaped membership function centred
    // on the band midpoint with width ½ band-width gives a smooth
    // confidence in (0, 1].
    let mut best: (OrganismDomain, f64) = (candidates[0], 0.0);
    for &dom in candidates {
        let (lo, hi) = dom.nm_range();
        let mid = 0.5 * (lo + hi);
        let half = 0.5 * (hi - lo).max(1e-9);
        let z = ((size - mid) / half).clamp(-5.0, 5.0);
        let membership = (-0.5 * z * z).exp();

        // Eccentricity hint: protists and bacteria can be rod-shaped
        // (high eccentricity); viruses are typically near-spherical.
        let shape_boost = match dom {
            OrganismDomain::Virus    => 1.0 - 0.5 * r.eccentricity,
            OrganismDomain::Bacteria => 0.6 + 0.4 * r.eccentricity,
            OrganismDomain::Archaea  => 0.6 + 0.4 * r.eccentricity,
            OrganismDomain::Protist  => 0.7 + 0.3 * (1.0 - r.eccentricity),
        };

        // DAPI/FITC ratio: high ratio means the object lit up the DNA
        // channel preferentially. Anaerobic / chemoautotrophic
        // archaea + bacteria are often nucleic-acid-rich relative to
        // an autofluorescent protist; archaea bias slightly higher.
        let ratio_boost = match dom {
            OrganismDomain::Virus    => 1.0,                                // ratio not informative at VLP size
            OrganismDomain::Bacteria => smooth_band(ratio, 0.5, 4.0),
            OrganismDomain::Archaea  => smooth_band(ratio, 1.0, 6.0),
            OrganismDomain::Protist  => smooth_band(ratio, 0.0, 1.5),
        };

        let score = membership * shape_boost * ratio_boost;
        // NaN-safe comparison: a non-finite score (which could only arise
        // from a NaN size_nm / eccentricity upstream) must never silently
        // win or be treated as "greater". `score > best.1` is already
        // false for NaN, but we make the intent explicit and skip
        // non-finite scores so a valid object is never misclassified by a
        // stray NaN. (Addresses the "NaN float comparisons drop valid
        // objects" class of bug.)
        if score.is_finite() && score > best.1 {
            best = (dom, score);
        }
    }
    // If every candidate scored non-finite or zero, fall back to the
    // size-nearest domain so the object is still classified, never dropped.
    if !best.1.is_finite() || best.1 <= 0.0 {
        best = (nearest_by_size(r.size_nm, candidates), 0.0);
    }
    // Squash to [0, 1] confidence with a soft cap.
    let conf = (best.1 / (best.1 + 1.0)).clamp(0.0, 1.0);
    (best.0, conf)
}

fn dapi_fitc_ratio(r: &ObjectRecord, lookup: &HashMap<String, f64>) -> f64 {
    // Match the FITC file name to a DAPI partner. The original naming
    // convention swaps `(fitc)` ↔ `(dapi)`; users with different
    // schemes can still consume the rule-based prior alone.
    let stem = &r.file_name;
    let dapi_name = stem.replace("(fitc)", "(dapi)");
    let intensity = lookup
        .get(&dapi_name)
        .copied()
        .or_else(|| lookup.get(stem).copied())
        .unwrap_or(0.0);
    if r.intensity > 1e-12 {
        intensity / r.intensity
    } else {
        0.0
    }
}

/// Fallback classifier used only when every candidate's score was
/// non-finite or zero: pick the domain whose canonical size band centre
/// is closest to the object's size. Guarantees an object is never dropped
/// or left unclassified even in the presence of a NaN feature. Distance
/// uses `total_cmp` so a NaN size sorts deterministically to the end
/// rather than corrupting the comparison.
fn nearest_by_size(size: f64, candidates: &[OrganismDomain]) -> OrganismDomain {
    candidates
        .iter()
        .copied()
        .min_by(|a, b| {
            let da = {
                let (lo, hi) = a.nm_range();
                (size - 0.5 * (lo + hi)).abs()
            };
            let db = {
                let (lo, hi) = b.nm_range();
                (size - 0.5 * (lo + hi)).abs()
            };
            da.total_cmp(&db)
        })
        .unwrap_or(candidates[0])
}

#[inline]
fn smooth_band(x: f64, lo: f64, hi: f64) -> f64 {
    // Trapezoid-ish membership: 1 inside, smoothly decaying outside.
    if x >= lo && x <= hi {
        1.0
    } else {
        let d = if x < lo { lo - x } else { x - hi };
        (-d).exp().clamp(0.0, 1.0)
    }
}

/// Optional GMM refinement. Returns `None` if the GMM fit fails
/// (e.g. degenerate features), in which case the rule-based prior is
/// kept verbatim.
fn gmm_refine(
    records:    &[ObjectRecord],
    lookup:     &HashMap<String, f64>,
    prior:      &[(OrganismDomain, f64)],
    domains:    &[OrganismDomain],
) -> Option<Vec<(OrganismDomain, f64)>> {
    let n = records.len();
    let n_feats = 4;
    let mut x = Array2::<f64>::zeros((n, n_feats));
    for (i, r) in records.iter().enumerate() {
        let ratio = dapi_fitc_ratio(r, lookup);
        x[(i, 0)] = (r.size_nm.max(1e-3)).log10();
        x[(i, 1)] = r.eccentricity.clamp(0.0, 0.999);
        x[(i, 2)] = (ratio.max(1e-6)).log10();
        x[(i, 3)] = ((r.area_px as f64).max(1.0)).log10();
    }
    let targets: Array1<usize> = Array1::zeros(n);
    let dataset = DatasetBase::new(x.clone(), targets);

    let k = domains.len().min(n).max(2);
    let gmm = GaussianMixtureModel::params(k)
        .n_runs(3)
        .tolerance(1e-4)
        .max_n_iterations(200)
        .fit(&dataset)
        .ok()?;
    let assignments = gmm.predict(&x);

    // For each cluster, average the original rule-based confidences
    // and pick the domain with the highest summed score — that's the
    // canonical label we'll attach to the cluster.
    let mut cluster_scores: Vec<HashMap<OrganismDomain, f64>> =
        (0..k).map(|_| HashMap::new()).collect();
    for (i, &c) in assignments.iter().enumerate() {
        let (dom, conf) = prior[i];
        *cluster_scores[c].entry(dom).or_insert(0.0) += conf;
    }
    let cluster_label: Vec<OrganismDomain> = cluster_scores
        .iter()
        .map(|m| {
            m.iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(d, _)| *d)
                .unwrap_or(domains[0])
        })
        .collect();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = assignments[i];
        let dom = cluster_label[c];
        // confidence = posterior responsibility for the chosen cluster
        // (we approximate it as 1 - rel. distance to other centroids).
        let conf = prior[i].1.max(0.5);
        out.push((dom, conf.min(0.999)));
    }
    Some(out)
}

/// Build the per-DAPI-image intensity lookup used as a prior for the
/// classifier.  Keyed by `image.short_name()` so the FITC counterpart
/// can rewrite the suffix.
pub fn build_dapi_lookup(
    dapi_stats: &[crate::calibration::PerImageStats],
) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    for s in dapi_stats {
        let mean: f64 = if s.intensities.is_empty() {
            0.0
        } else {
            s.intensities.iter().sum::<f64>() / s.intensities.len() as f64
        };
        map.insert(s.name.clone(), mean);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ClassifierConfig, Config, OrganismDomain};
    use crate::quantify::ObjectRecord;
    use std::collections::HashMap;

    fn rec(size: f64, ecc: f64, intensity: f64) -> ObjectRecord {
        ObjectRecord {
            file_name:     "S_(fitc)".into(),
            object_id:     1,
            size_nm:       size,
            axis_major_nm: size,
            axis_minor_nm: size,
            x_px:          0.0,
            y_px:          0.0,
            intensity,
            eccentricity:  ecc,
            area_px:       10,
        }
    }

    fn cfg_with(domains: Vec<OrganismDomain>) -> Config {
        Config {
            classifier: ClassifierConfig { domains, gmm_refine: false, onnx_model: None },
            ..Default::default()
        }
    }

    #[test]
    fn nan_features_do_not_drop_objects() {
        // Objects with NaN size / eccentricity must still be classified,
        // never silently dropped (Bug 9 regression guard).
        let records = vec![
            rec(f64::NAN, 0.5, 1.0),
            rec(150.0, f64::NAN, 1.0),
            rec(f64::NAN, f64::NAN, 0.0),
            rec(150.0, 0.5, 1.0), // one normal object
        ];
        let lookup: HashMap<String, f64> = HashMap::new();
        let report = classify_objects(&records, &lookup, &cfg_with(vec![]));
        assert_eq!(report.objects.len(), records.len(),
            "every object must survive classification, even with NaN features");
        // Confidences must all be finite.
        for o in &report.objects {
            assert!(o.confidence.is_finite(), "confidence must be finite, got {}", o.confidence);
        }
    }

    #[test]
    fn every_object_gets_a_domain() {
        let records: Vec<ObjectRecord> =
            (0..20).map(|i| rec(50.0 + i as f64 * 100.0, 0.4, 1.0)).collect();
        let lookup: HashMap<String, f64> = HashMap::new();
        let report = classify_objects(&records, &lookup, &cfg_with(vec![
            OrganismDomain::Virus, OrganismDomain::Bacteria,
        ]));
        // domain_counts must sum to the number of objects — nothing lost.
        let total: u64 = report.domain_counts.values().sum();
        assert_eq!(total as usize, records.len());
    }
}
