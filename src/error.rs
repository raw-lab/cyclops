//! Cyclops error type.

use std::path::PathBuf;

use thiserror::Error;

/// Result alias for Cyclops APIs.
pub type Result<T> = std::result::Result<T, CyclopsError>;

#[derive(Debug, Error)]
pub enum CyclopsError {
    #[error("I/O error at {path:?}: {source}")]
    Io {
        path:   PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Generic I/O error reached via `?` — when we don't have a path to attach.
    #[error("I/O error: {0}")]
    IoBare(#[from] std::io::Error),

    #[error("image decoding error at {path:?}: {message}")]
    ImageDecode { path: PathBuf, message: String },

    #[error("no images found in directory {0:?}")]
    EmptyDirectory(PathBuf),

    #[error("calibration failed: {0}")]
    Calibration(String),

    #[error("no viral-particle candidate found within {dist_nm:.1} nm — \
             increase --d-constraint or pick a new calibration image")]
    NoPairFound { dist_nm: f64 },

    #[error("invalid PSF parameters: {0}")]
    InvalidPsf(String),

    #[error("deconvolution did not converge: {0}")]
    DeconvFailed(String),

    #[error("dataframe error: {0}")]
    Polars(String),

    #[error("plot error: {0}")]
    Plot(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<polars::error::PolarsError> for CyclopsError {
    fn from(err: polars::error::PolarsError) -> Self {
        Self::Polars(err.to_string())
    }
}

impl From<image::ImageError> for CyclopsError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageDecode {
            path:    PathBuf::from("<unknown>"),
            message: err.to_string(),
        }
    }
}
