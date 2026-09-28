//! ONNX classifier backend (optional, behind the `onnx` feature).
//!
//! The rule-based prior and the linfa GMM refinement in `classify.rs`
//! need no external model. This module adds a *third* option: score each
//! detected object with a user-supplied ONNX model (e.g. a small MLP
//! exported from scikit-learn/PyTorch, or a morphology classifier). It is
//! deliberately generic so a lab can drop in their own trained model
//! without recompiling anything but this crate's feature flag.
//!
//! ## Contract with the model
//!
//! * **Input:**  a float32 tensor of shape `[N, F]` — N objects, F
//!   features per object. Cyclops supplies `F = 5` features per object,
//!   in this fixed order:
//!     0. `log10(size_nm)`
//!     1. `eccentricity`
//!     2. `log10(dapi_fitc_ratio + 1)`
//!     3. `log10(area_px + 1)`
//!     4. `axis_major_nm / axis_minor_nm` (aspect ratio; 1.0 if minor≈0)
//! * **Output:** a float32 tensor of shape `[N, C]` — per-object scores
//!   (logits or probabilities) over C classes. C must equal the number
//!   of candidate domains passed in, and column `c` corresponds to
//!   `candidates[c]`. Cyclops applies a softmax if the row doesn't already
//!   sum to ≈1, then takes the arg-max as the label and the softmax
//!   probability as the confidence.
//!
//! The input/output tensor *names* are discovered from the model graph
//! (first input, first output), so the exporter doesn't need to match a
//! magic name.
//!
//! ## Building
//!
//! ```bash
//! cargo build -p cyclops-cli --features onnx
//! cyclops ... --onnx path/to/model.onnx
//! ```
//!
//! Without the feature, [`classify_with_onnx`] is a stub that returns
//! `None` and logs a one-line hint, so the pipeline transparently falls
//! back to the rule-based / GMM path.

use std::path::Path;

use crate::config::OrganismDomain;
use crate::quantify::ObjectRecord;

/// The fixed feature vector Cyclops extracts per object for the ONNX
/// model. Kept public so downstream tooling (and the model's training
/// script) can depend on the exact ordering.
pub const N_FEATURES: usize = 5;

/// Build the `[N, F]` feature matrix (row-major) for a batch of objects.
/// Shared by both the real and stub paths so the contract is identical
/// regardless of whether the `onnx` feature is on.
pub fn feature_matrix(
    records: &[ObjectRecord],
    ratios:  &[f64],
) -> Vec<f32> {
    debug_assert_eq!(records.len(), ratios.len());
    let mut m = Vec::with_capacity(records.len() * N_FEATURES);
    for (r, &ratio) in records.iter().zip(ratios.iter()) {
        let size = r.size_nm.max(0.0);
        let aspect = if r.axis_minor_nm.abs() > 1e-9 {
            (r.axis_major_nm / r.axis_minor_nm) as f32
        } else {
            1.0
        };
        m.push((size + 1.0).log10() as f32);          // 0
        m.push(r.eccentricity as f32);                 // 1
        m.push((ratio.max(0.0) + 1.0).log10() as f32); // 2
        m.push(((r.area_px as f64) + 1.0).log10() as f32); // 3
        m.push(aspect);                                // 4
    }
    m
}

/// Numerically-stable softmax over one row, in place.
pub fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

// ===========================================================================
//  Real backend (feature = "onnx")
// ===========================================================================
#[cfg(feature = "onnx")]
pub fn classify_with_onnx(
    model_path: &Path,
    records:    &[ObjectRecord],
    ratios:     &[f64],
    candidates: &[OrganismDomain],
) -> Option<Vec<(OrganismDomain, f64)>> {
    use tract_onnx::prelude::*;

    if records.is_empty() || candidates.is_empty() {
        return None;
    }
    let n = records.len();
    let c = candidates.len();

    // Load & optimise the graph, fixing the batch dimension to N and the
    // feature dimension to N_FEATURES.
    let model = match (|| -> TractResult<_> {
        let m = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(n, N_FEATURES)),
            )?
            .into_optimized()?
            .into_runnable()?;
        Ok(m)
    })() {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                "ONNX model at {} could not be loaded ({e}); \
                 falling back to rule-based/GMM classification",
                model_path.display()
            );
            return None;
        }
    };

    // Build the input tensor [N, F]. tract re-exports ndarray.
    let feats = feature_matrix(records, ratios);
    let input = match tract_onnx::prelude::tract_ndarray::Array2::from_shape_vec(
        (n, N_FEATURES), feats,
    ) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("ONNX feature tensor shape error ({e}); falling back");
            return None;
        }
    };
    let input_tensor: Tensor = input.into();

    let outputs = match model.run(tvec!(input_tensor.into())) {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("ONNX inference failed ({e}); falling back");
            return None;
        }
    };

    // Read the first output as an [N, C'] f32 array. `outputs[0]` is a
    // TValue in tract 0.23; deref to the inner Tensor for to_array_view.
    let out_tensor: &Tensor = &outputs[0];
    let arr = match out_tensor.to_plain_array_view::<f32>() {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("ONNX output not f32 ({e}); falling back");
            return None;
        }
    };
    let shape = arr.shape();
    let cols = *shape.last().unwrap_or(&0);
    if cols != c {
        tracing::warn!(
            "ONNX model outputs {cols} classes but {c} domains were requested; \
             falling back to rule-based/GMM"
        );
        return None;
    }
    let flat: Vec<f32> = arr.iter().cloned().collect();

    // Per-row softmax → arg-max label + probability confidence.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f32> = flat[i * cols..(i + 1) * cols].to_vec();
        // Only softmax if it isn't already a probability distribution.
        let row_sum: f32 = row.iter().sum();
        if (row_sum - 1.0).abs() > 1e-3 {
            softmax_inplace(&mut row);
        }
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() && v > best_val {
                best_val = v;
                best_idx = j;
            }
        }
        out.push((candidates[best_idx], best_val as f64));
    }
    tracing::info!("ONNX classifier scored {n} objects from {}", model_path.display());
    Some(out)
}

// ===========================================================================
//  Stub backend (feature disabled) — keeps the call site identical
// ===========================================================================
#[cfg(not(feature = "onnx"))]
pub fn classify_with_onnx(
    model_path: &Path,
    _records:   &[ObjectRecord],
    _ratios:    &[f64],
    _candidates: &[OrganismDomain],
) -> Option<Vec<(OrganismDomain, f64)>> {
    tracing::warn!(
        "an ONNX model was supplied ({}) but this build lacks the `onnx` \
         feature; rebuild with `--features onnx` to enable it. \
         Using rule-based/GMM classification for now.",
        model_path.display()
    );
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut r = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut r);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // monotonic: larger logit → larger prob
        assert!(r[2] > r[1] && r[1] > r[0]);
    }

    #[test]
    fn softmax_handles_empty() {
        let mut r: Vec<f32> = vec![];
        softmax_inplace(&mut r); // must not panic
        assert!(r.is_empty());
    }

    #[test]
    fn feature_matrix_shape_and_order() {
        use crate::quantify::ObjectRecord;
        let rec = ObjectRecord {
            file_name: "x".into(), object_id: 1,
            size_nm: 100.0, axis_major_nm: 120.0, axis_minor_nm: 80.0,
            x_px: 0.0, y_px: 0.0, intensity: 1.0,
            eccentricity: 0.5, area_px: 42,
        };
        let m = feature_matrix(&[rec], &[2.0]);
        assert_eq!(m.len(), N_FEATURES);
        assert!((m[0] - (101.0f32).log10()).abs() < 1e-5); // log10(size+1)
        assert!((m[1] - 0.5).abs() < 1e-6);                // eccentricity
        assert!((m[4] - 1.5).abs() < 1e-5);                // 120/80 aspect
    }
}
