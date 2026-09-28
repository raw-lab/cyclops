//! Run-time configuration for the Cyclops pipeline.
//!
//! Mirrors every CLI flag in the original `epivirquant.py`, plus a few
//! Cyclops-only additions: organism domain selection and the ML
//! classifier toggle.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{CyclopsError, Result};

/// Point-spread function family. The original EpiVirQuant called these
/// `gam`, `gau`, `hyb`; we keep those short codes for backwards
/// compatibility on the CLI but expose readable names internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PsfMethod {
    /// γ-sinc PSF (eq. 6). The classic EpiVirQuant default.
    GammaSinc,
    /// Pure isotropic Gaussian PSF.
    Gaussian,
    /// Flat-top hybrid (uniform inside, smoothed edges).
    Hybrid,
}

impl PsfMethod {
    pub fn from_short(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gam" | "gamma" | "gamma-sinc" | "γ" => Ok(Self::GammaSinc),
            "gau" | "gaussian"                   => Ok(Self::Gaussian),
            "hyb" | "hybrid"                     => Ok(Self::Hybrid),
            other => Err(CyclopsError::Config(format!(
                "unknown PSF method `{other}` — expected gam|gau|hyb"
            ))),
        }
    }

    pub fn short(&self) -> &'static str {
        match self {
            Self::GammaSinc => "gam",
            Self::Gaussian  => "gau",
            Self::Hybrid    => "hyb",
        }
    }
}

/// Size metric used to summarise a detected object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SizeMetric {
    /// Equivalent diameter of a circle with the same area as the
    /// detected object (EpiVirQuant default).
    EquivalentDiameter,
    /// Average of the semi-major and semi-minor axes.
    AverageAxes,
}

impl SizeMetric {
    pub fn from_int(n: i64) -> Result<Self> {
        match n {
            1 => Ok(Self::EquivalentDiameter),
            2 => Ok(Self::AverageAxes),
            other => Err(CyclopsError::Config(format!(
                "size metric must be 1 or 2 (got {other})"
            ))),
        }
    }
}

/// A microbial domain the pipeline can resolve. Diameter ranges in nm
/// follow ranges from White et al. (2024) and standard microbiology
/// references (e.g. *Brock Biology of Microorganisms*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganismDomain {
    /// Viral-like particles (≈ 20–200 nm).
    Virus,
    /// Bacteria (≈ 200–5 000 nm).
    Bacteria,
    /// Archaea (≈ 200–5 000 nm; size-overlapping with bacteria).
    Archaea,
    /// Protists (≈ 5 000–200 000 nm; eukaryotic single cells).
    Protist,
}

impl OrganismDomain {
    /// Inclusive nm range used as the default size band.
    pub fn nm_range(&self) -> (f64, f64) {
        match self {
            Self::Virus    => (20.0,         220.0),
            Self::Bacteria => (220.0,       5_000.0),
            Self::Archaea  => (220.0,       5_000.0),
            Self::Protist  => (5_000.0, 200_000.0),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Virus    => "virus",
            Self::Bacteria => "bacteria",
            Self::Archaea  => "archaea",
            Self::Protist  => "protist",
        }
    }
}

/// Optional ML classifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    /// Domains the user *expects* in the field of view. The classifier
    /// only ever assigns these labels. Default is *all* domains.
    pub domains: Vec<OrganismDomain>,
    /// Use a Gaussian-Mixture-Model on (size, intensity, eccentricity)
    /// to refine the rule-based assignment.
    pub gmm_refine: bool,
    /// Optional path to a pretrained ONNX segmentation model
    /// (e.g. exported Cellpose / StarDist). Behind the `onnx` feature.
    pub onnx_model: Option<PathBuf>,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            domains: vec![
                OrganismDomain::Virus,
                OrganismDomain::Bacteria,
                OrganismDomain::Archaea,
                OrganismDomain::Protist,
            ],
            gmm_refine: true,
            onnx_model: None,
        }
    }
}

/// Top-level Cyclops configuration.
///
/// Defaults match the EpiVirQuant 0.1.1 CLI so that existing analysis
/// recipes reproduce bit-for-bit on the same inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // --- inputs ------------------------------------------------------
    pub dapi_dir:    PathBuf,
    pub fitc_dir:    PathBuf,
    pub calibration: PathBuf,
    pub out_dir:     PathBuf,

    // --- sizing -----------------------------------------------------
    pub scale_length_px:    f64, // 585
    pub scale_metric_nm:    f64, // 20_000
    pub sphere_size_nm:     f64, // 175

    // --- pairing ----------------------------------------------------
    pub pad:           usize, // 14
    pub d_constraint:  usize, // 30

    // --- PSF parameter sweep ---------------------------------------
    pub f_size:    usize,      // 0 → auto
    pub psf_method: PsfMethod, // GammaSinc
    pub a:         f64,        // 1
    pub b:         f64,        // 1
    pub sigma:     f64,        // 1
    pub r:         f64,        // e ≈ 2.7182818
    pub tau:       f64,        // 0.5
    pub v:         f64,        // 2.25
    pub s:         f64,        // 0.0
    pub n_mle_iter: usize,     // 10
    pub n_lr_iter:  usize,     // 80

    // --- post-processing -------------------------------------------
    pub size_metric:     SizeMetric, // EquivalentDiameter
    pub sm_constraint:   f64,        // 8000 nm
    pub gen_figures:     bool,       // false

    // --- runtime ----------------------------------------------------
    /// Negative values follow the joblib convention (-1 → all cores,
    /// -2 → all but one). Positive values are taken verbatim.
    pub cpus: i32,

    // --- Cyclops additions -----------------------------------------
    pub classifier:    ClassifierConfig,
    /// Save intermediate PNG plots for every image.
    pub keep_intermediates: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dapi_dir:    PathBuf::new(),
            fitc_dir:    PathBuf::new(),
            calibration: PathBuf::new(),
            out_dir:     PathBuf::from("Cyclops_Output"),

            scale_length_px: 585.0,
            scale_metric_nm: 20_000.0,
            sphere_size_nm:  175.0,

            pad:          14,
            d_constraint: 30,

            f_size:     0,
            psf_method: PsfMethod::GammaSinc,
            a:          1.0,
            b:          1.0,
            sigma:      1.0,
            r:          std::f64::consts::E,
            tau:        0.5,
            v:          2.25,
            s:          0.0,
            n_mle_iter: 10,
            n_lr_iter:  80,

            size_metric:   SizeMetric::EquivalentDiameter,
            sm_constraint: 8_000.0,
            gen_figures:   false,

            cpus: -2,

            classifier: ClassifierConfig::default(),
            keep_intermediates: true,
        }
    }
}

impl Config {
    /// Pixel-to-nanometre conversion derived from the scale-bar fields.
    #[inline]
    pub fn px2nm(&self) -> f64 {
        self.scale_metric_nm / self.scale_length_px
    }

    /// Translate the joblib-style `cpus` value into the actual thread
    /// count this machine will use.
    pub fn effective_threads(&self) -> usize {
        let total = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        match self.cpus {
            n if n > 0  => (n as usize).min(total),
            -1          => total,
            -2          => total.saturating_sub(1).max(1),
            n           => total.saturating_sub((-n) as usize - 1).max(1),
        }
    }

    /// Light sanity-checks before kicking off a pipeline run.
    pub fn validate(&self) -> Result<()> {
        if self.scale_length_px <= 0.0 {
            return Err(CyclopsError::Config("scale_length_px must be > 0".into()));
        }
        if self.scale_metric_nm <= 0.0 {
            return Err(CyclopsError::Config("scale_metric_nm must be > 0".into()));
        }
        if self.sphere_size_nm <= 0.0 {
            return Err(CyclopsError::Config("sphere_size_nm must be > 0".into()));
        }
        if self.n_mle_iter == 0 || self.n_lr_iter == 0 {
            return Err(CyclopsError::Config(
                "iteration counts must be > 0".into(),
            ));
        }
        Ok(())
    }
}
