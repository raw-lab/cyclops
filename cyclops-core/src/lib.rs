//! Cyclops core — formerly EpiVirQuant.
//!
//! Sizes and counts viral-like particles, bacteria, archaea, and protists
//! in epifluorescence microscopy images via tunable blind deconvolution.
//!
//! Pipeline:
//!  1. [`pairing`]      — find an "optimization box" containing a pair of
//!                        closely-spaced calibration objects.
//!  2. [`deconv`]       — sweep a tunable point-spread function (γ-sinc,
//!                        Gaussian, hybrid) and pick the PSF that minimises
//!                        Shannon entropy / maximises GLCM energy.
//!  3. [`calibration`]  — apply Richardson–Lucy on DAPI microsphere images
//!                        of known diameter to derive the CORR coefficient.
//!  4. [`quantify`]     — apply the same PSF + CORR to the FITC images and
//!                        emit per-object size / position / intensity.
//!  5. [`classify`]     — optional ML pass that assigns each detected
//!                        object to a microbial domain
//!                        (virus / bacteria / archaea / protist) based on
//!                        size, DAPI∕FITC ratio, eccentricity, and shape.
//!
//! All modules are deterministic given the same `Config` and inputs.

pub mod config;
pub mod error;
pub mod image_io;
pub mod fft;
pub mod psf;
pub mod threshold;
pub mod regions;
pub mod deconv;
pub mod pairing;
pub mod calibration;
pub mod quantify;
pub mod classify;
pub mod plots;
pub mod progress;
pub mod onnx;
pub mod pipeline;

pub use config::{Config, OrganismDomain, PsfMethod, SizeMetric};
pub use error::{CyclopsError, Result};
pub use pipeline::{run_pipeline, PipelineReport};

/// Library name as it appears in logs and reports.
pub const NAME: &str = "Cyclops";

/// Cyclops version (kept in sync with the workspace).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build provenance — formerly known as `EpiVirQuant` v0.1.1 (2026-03-18).
pub const FORMERLY: &str = "EpiVirQuant v0.1.1 (2026-03-18)";
