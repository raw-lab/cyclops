//! Cyclops command-line interface.
//!
//! Mirrors every flag from the original `epivirquant.py` driver, adds a
//! handful of Cyclops-only knobs (`--domains`, `--no-gmm`, `--onnx`),
//! and emits the same on-disk artefacts plus a Polars-backed Parquet/TSV
//! object table.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use cyclops_core::config::{
    ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric,
};
use cyclops_core::pipeline::run_pipeline;
use cyclops_core::{FORMERLY, NAME, VERSION};

#[derive(Debug, Clone, Copy, ValueEnum)]
#[clap(rename_all = "lower")]
enum PsfArg { Gam, Gau, Hyb }

impl From<PsfArg> for PsfMethod {
    fn from(p: PsfArg) -> Self {
        match p {
            PsfArg::Gam => PsfMethod::GammaSinc,
            PsfArg::Gau => PsfMethod::Gaussian,
            PsfArg::Hyb => PsfMethod::Hybrid,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[clap(rename_all = "lower")]
enum DomainArg { Virus, Bacteria, Archaea, Protist, All }

#[derive(Parser, Debug)]
#[command(
    name    = "cyclops",
    version = VERSION,
    about   = "Cyclops — formerly EpiVirQuant. Sizes and counts VLPs, bacteria, archaea and protists in epifluorescence microscopy.",
    long_about = None,
)]
struct Cli {
    // --- inputs (original short flags preserved) -----------------------------
    /// Directory of DAPI (calibration / bead) images.
    #[arg(long = "dapi", value_name = "DIR")]
    dapi: PathBuf,

    /// Directory of FITC (sample) images.
    #[arg(long = "fitc", value_name = "DIR")]
    fitc: PathBuf,

    /// Single calibration image (a DAPI bead field with the VP pair).
    #[arg(long = "calibration", value_name = "FILE")]
    calibration: PathBuf,

    /// Output directory (default: `Cyclops_Output`).
    #[arg(long = "outDir", default_value = "Cyclops_Output", value_name = "DIR")]
    out_dir: PathBuf,

    // --- sizing / scale-bar --------------------------------------------------
    /// Scale-bar length in pixels.
    #[arg(long = "scaleLength", default_value_t = 585.0)]
    scale_length: f64,

    /// Scale-bar length in nanometres.
    #[arg(long = "scaleMetric", default_value_t = 20_000.0)]
    scale_metric: f64,

    /// Nominal sphere (bead) diameter in nm used for the correction factor.
    #[arg(long = "sphereSize", default_value_t = 175.0)]
    sphere_size: f64,

    // --- pairing -------------------------------------------------------------
    /// Padding (px) around each candidate VP crop.
    #[arg(long = "pad", default_value_t = 14)]
    pad: usize,

    /// Distance constraint (px) between paired beads.
    #[arg(long = "dConstraint", default_value_t = 30)]
    d_constraint: usize,

    // --- PSF sweep -----------------------------------------------------------
    /// Fixed PSF kernel size (0 = sweep).
    #[arg(long = "fSize", default_value_t = 0)]
    f_size: usize,

    /// PSF family (gam = γ-sinc, gau = Gaussian, hyb = hybrid).
    #[arg(long = "psfMethod", value_enum, default_value_t = PsfArg::Gam)]
    psf_method: PsfArg,

    #[arg(long, default_value_t = 1.0)]  a:     f64,
    #[arg(long, default_value_t = 1.0)]  b:     f64,
    #[arg(long, default_value_t = 1.0)]  sig:   f64,
    #[arg(long, default_value_t = std::f64::consts::E)]
    r:     f64,
    #[arg(long, default_value_t = 0.5)]  tau:   f64,
    #[arg(long, default_value_t = 2.25)] v:     f64,
    #[arg(long, default_value_t = 0.0)]  s:     f64,

    /// MLE blind-deconv iteration count.
    #[arg(long = "nMLE_iter", default_value_t = 10)]
    n_mle_iter: usize,

    /// Richardson-Lucy iteration count.
    #[arg(long = "nLR_iter", default_value_t = 80)]
    n_lr_iter: usize,

    // --- post-processing -----------------------------------------------------
    /// Size metric: 1 = equivalent diameter, 2 = average axes.
    #[arg(long = "szMetric", default_value_t = 1)]
    sz_metric: i64,

    /// Reject objects with semi-major axis above this many nm.
    #[arg(long = "SM_constraint", default_value_t = 8_000.0)]
    sm_constraint: f64,

    /// Save every intermediate diagnostic figure.
    #[arg(long = "genFigs", default_value_t = false, action = ArgAction::Set)]
    gen_figs: bool,

    /// Suppress per-image PNG diagnostics (faster, smaller output).
    #[arg(long = "noIntermediates", default_value_t = false, action = ArgAction::SetTrue)]
    no_intermediates: bool,

    // --- runtime -------------------------------------------------------------
    /// Thread count. Negative values follow the joblib convention
    /// (-1 = all cores, -2 = all-but-one).
    #[arg(long = "cpus", default_value_t = -2)]
    cpus: i32,

    // --- Cyclops-only --------------------------------------------------------
    /// Restrict the ML classifier to one or more domains.
    /// Repeat the flag or pass a comma-separated list.
    #[arg(long = "domains", value_enum, value_delimiter = ',', default_values_t = [DomainArg::All])]
    domains: Vec<DomainArg>,

    /// Disable the GMM refinement on top of the rule-based classifier.
    #[arg(long = "no-gmm", default_value_t = false, action = ArgAction::SetTrue)]
    no_gmm: bool,

    /// Optional ONNX segmentation model (requires the `onnx` build feature).
    #[arg(long = "onnx", value_name = "MODEL")]
    onnx: Option<PathBuf>,

    /// Verbose logging (`-v` = info, `-vv` = debug, `-vvv` = trace).
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

impl Cli {
    fn into_config(self) -> Result<Config> {
        let domains = if self.domains.iter().any(|d| matches!(d, DomainArg::All)) {
            vec![
                OrganismDomain::Virus,
                OrganismDomain::Bacteria,
                OrganismDomain::Archaea,
                OrganismDomain::Protist,
            ]
        } else {
            self.domains
                .iter()
                .filter_map(|d| match d {
                    DomainArg::Virus    => Some(OrganismDomain::Virus),
                    DomainArg::Bacteria => Some(OrganismDomain::Bacteria),
                    DomainArg::Archaea  => Some(OrganismDomain::Archaea),
                    DomainArg::Protist  => Some(OrganismDomain::Protist),
                    DomainArg::All      => None,
                })
                .collect()
        };

        Ok(Config {
            dapi_dir:        self.dapi,
            fitc_dir:        self.fitc,
            calibration:     self.calibration,
            out_dir:         self.out_dir,

            scale_length_px: self.scale_length,
            scale_metric_nm: self.scale_metric,
            sphere_size_nm:  self.sphere_size,

            pad:          self.pad,
            d_constraint: self.d_constraint,

            f_size:     self.f_size,
            psf_method: self.psf_method.into(),
            a:          self.a,
            b:          self.b,
            sigma:      self.sig,
            r:          self.r,
            tau:        self.tau,
            v:          self.v,
            s:          self.s,
            n_mle_iter: self.n_mle_iter,
            n_lr_iter:  self.n_lr_iter,

            size_metric:   SizeMetric::from_int(self.sz_metric)
                .context("invalid --szMetric")?,
            sm_constraint: self.sm_constraint,
            gen_figures:   self.gen_figs,

            cpus: self.cpus,

            classifier: ClassifierConfig {
                domains,
                gmm_refine: !self.no_gmm,
                onnx_model: self.onnx,
            },
            keep_intermediates: !self.no_intermediates,
        })
    }
}

fn init_logging(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("cyclops_core={level},cyclops_cli={level}")));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .compact()
        .init();
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose);

    info!("{NAME} v{VERSION}  ({FORMERLY})");

    let cfg = match cli.into_config() {
        Ok(c)  => c,
        Err(e) => {
            error!("invalid configuration: {e:#}");
            return ExitCode::from(2);
        }
    };

    match run_pipeline(&cfg) {
        Ok(report) => {
            // pretty-print a one-screen summary for humans
            println!("\n=== Cyclops summary ===");
            println!("output dir       : {}", report.output_dir.display());
            println!("DAPI / FITC      : {} / {}", report.n_dapi, report.n_fitc);
            println!("PSF              : f={} τ={:.4} v={:.4}",
                     report.psf_f_size, report.psf_tau, report.psf_v);
            println!("VP min distance  : {:.1} nm", report.min_distance_nm);
            println!("correction (CORR): {:.4}", report.correction);
            println!("objects detected : {}", report.n_objects);
            println!("mean size        : {:.1} nm", report.mean_size_nm);
            println!("size bands       : {:?}", report.size_bands);
            println!("domain counts    : {:?}", report.domain_counts);
            println!("elapsed          : {:.2} s", report.elapsed_seconds);
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!("pipeline failed: {e:#}");
            ExitCode::FAILURE
        }
    }
}
