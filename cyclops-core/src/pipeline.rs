//! Top-level pipeline orchestration.
//!
//! `run_pipeline(cfg)` reproduces the four EpiVirQuant steps and writes
//! Cyclops outputs to `cfg.out_dir`:
//!
//! ```text
//! Cyclops_Output/
//!   Step-1_VP/                       PNGs of the calibration scan
//!   Step-2_Decon/
//!       PSF_final.png
//!       deconLog.txt
//!       optimization/EntVsIter.png …
//!   Step-3_Corr/
//!       CORR_<image>/…               per-image diagnostic PNGs
//!       XB_SizeHistogram.png
//!       corrLog.txt
//!   Step-4_genMasks/
//!       genMask_<image>/…            per-image diagnostic PNGs
//!       sizeCoords.tsv               legacy TSV (filename, size, x, y, intensity)
//!       cyclops_objects.parquet      full Polars frame including domain calls
//!       countLog.txt
//!   cyclops_report.json              machine-readable summary
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::calibration::{calibrate, CalibrationReport};
use crate::classify::{build_dapi_lookup, classify_objects, ClassificationReport};
use crate::config::Config;
use crate::deconv::{psf_from_params, sweep_psf, OptimalPsf};
use crate::error::{CyclopsError, Result};
use crate::image_io::{list_image_files, load_gray, save_png, GrayImage};
use crate::pairing::{find_opt_box, OptBox};
use crate::plots;
use crate::quantify::{quantify, QuantReport};
use crate::{FORMERLY, NAME, VERSION};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub name:             String,
    pub version:          String,
    pub formerly:         String,
    pub started_at:       String,
    pub elapsed_seconds:  f64,
    pub config_summary:   ConfigSummary,
    pub n_dapi:           usize,
    pub n_fitc:           usize,
    pub min_distance_nm:  f64,
    pub psf_f_size:       usize,
    pub psf_tau:          f64,
    pub psf_v:            f64,
    pub correction:       f64,
    pub n_objects:        usize,
    pub mean_size_nm:     f64,
    pub size_bands:       crate::quantify::SizeBands,
    pub domain_counts:    HashMap<String, u64>,
    pub output_dir:       PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSummary {
    pub psf_method:   String,
    pub size_metric:  String,
    pub px2nm:        f64,
    pub n_mle_iter:   usize,
    pub n_lr_iter:    usize,
    pub sphere_nm:    f64,
    pub threads:      usize,
}

/// Normalise a path for safe comparison: make it absolute (relative to the
/// current working directory if needed) and collapse `.`/`..` lexically.
/// We deliberately avoid `canonicalize()` because it requires the path to
/// already exist and resolves symlinks — here we only need a stable textual
/// form for the containment checks below.
fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::ParentDir => { out.pop(); }
            Component::CurDir    => {}
            other                => out.push(other.as_os_str()),
        }
    }
    out
}

/// Returns true if `child` is equal to, or nested inside, `ancestor`.
fn is_inside(child: &Path, ancestor: &Path) -> bool {
    let c = normalize(child);
    let a = normalize(ancestor);
    c == a || c.starts_with(&a)
}

/// Guard against the catastrophic case where the output directory overlaps
/// the user's input data. Without this, creating output inside the data
/// folder (an easy mistake in the GUI) used to be destructive.
///
/// We refuse to run if the output directory:
///   * is the same as, or inside, the DAPI / FITC directory or the folder
///     containing the calibration image, OR
///   * contains (is an ancestor of) any of those input locations.
///
/// In either direction the two trees overlap and running the pipeline
/// could read-then-clobber the user's own data, so we stop with a clear
/// message instead.
fn guard_output_directory(cfg: &Config) -> Result<()> {
    let out = &cfg.out_dir;

    let calib_parent = cfg
        .calibration
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let inputs: [(&str, &Path); 3] = [
        ("DAPI directory",         &cfg.dapi_dir),
        ("FITC directory",         &cfg.fitc_dir),
        ("calibration image folder", calib_parent.as_path()),
    ];

    for (label, input) in inputs {
        // Output sits inside (or equals) an input location.
        if is_inside(out, input) {
            return Err(CyclopsError::Config(format!(
                "refusing to run: the output directory ({}) is the same as or \
                 inside your {} ({}). Choose an output directory OUTSIDE your \
                 data folders so Cyclops cannot overwrite your images.",
                normalize(out).display(),
                label,
                normalize(input).display(),
            )));
        }
        // An input location sits inside the output directory (so writing/
        // managing output could engulf the inputs).
        if is_inside(input, out) {
            return Err(CyclopsError::Config(format!(
                "refusing to run: your {} ({}) is inside the output directory \
                 ({}). Choose an output directory that does NOT contain your \
                 input data.",
                label,
                normalize(input).display(),
                normalize(out).display(),
            )));
        }
    }

    Ok(())
}

/// Run the full Cyclops pipeline.
pub fn run_pipeline(cfg: &Config) -> Result<PipelineReport> {
    run_pipeline_with_progress(cfg, &crate::progress::NoProgress)
}

/// Like [`run_pipeline`] but reports progress through `prog`. Front-ends
/// that want a progress bar (the GUI, the CLI's indicatif bar) call this;
/// everything else uses [`run_pipeline`] which passes a no-op reporter.
pub fn run_pipeline_with_progress(
    cfg:  &Config,
    prog: &dyn crate::progress::Progress,
) -> Result<PipelineReport> {
    use crate::progress::Stage;
    cfg.validate()?;
    let started = Instant::now();
    let started_at = chrono::Utc::now().to_rfc3339();

    // --- Output directory (with data-safety guards) ------------------------
    //
    // SAFETY: earlier versions ran `remove_dir_all(out_dir)` unconditionally,
    // which could delete a user's input data if they pointed --outDir at (or
    // inside) their image folder. We now:
    //   1. refuse to run if the output path overlaps any input path, and
    //   2. NEVER recursively delete a pre-existing directory — we create the
    //      output dir if missing and write into it, leaving any unrelated
    //      files alone.
    guard_output_directory(cfg)?;
    fs::create_dir_all(&cfg.out_dir).map_err(|e| CyclopsError::Io {
        path:   cfg.out_dir.clone(),
        source: e,
    })?;

    // --- Thread pool -------------------------------------------------------
    let threads = cfg.effective_threads();
    // Setting the global pool is allowed once; subsequent calls return an
    // error which we deliberately swallow (in a long-lived GUI process).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    info!("{NAME} v{VERSION} — pipeline start ({threads} threads)");
    info!("({FORMERLY})");

    // --- Step 0: load images ----------------------------------------------
    //
    // MEMORY: we deliberately do NOT load the FITC stack yet. The DAPI stack
    // is only needed through Step 3 (calibration) and the FITC stack only in
    // Step 4 (quantification). Holding both simultaneously doubles peak RAM
    // (each full-res 16-bit stack is ~1 GB as f64 for 26×5 MP images), so we
    // load DAPI now, free it after Step 3, then load FITC. The calibration
    // image is small and kept throughout.
    prog.stage(Stage::Loading, 0);
    let dapi = load_dir(&cfg.dapi_dir)?;
    let fitc_paths = list_image_files(&cfg.fitc_dir)?;
    let calib_img = load_gray(&cfg.calibration)?;
    info!("loaded {} DAPI / {} FITC (deferred) / 1 calibration image",
          dapi.len(), fitc_paths.len());
    prog.message(&format!("Loaded {} DAPI / {} FITC images",
                          dapi.len(), fitc_paths.len()));

    // --- Step 1: pairing ---------------------------------------------------
    prog.stage(Stage::Pairing, 0);
    let step1_dir = cfg.out_dir.join("Step-1_VP");
    fs::create_dir_all(&step1_dir)?;
    let opt_box: OptBox = find_opt_box(&calib_img.data, cfg)?;
    if cfg.keep_intermediates {
        save_png(&calib_img.data, step1_dir.join("calibration.png")).ok();
        save_png(&opt_box.patch,  step1_dir.join("optimization_box.png")).ok();
    }

    // --- Step 2: blind deconvolution ---------------------------------------
    prog.stage(Stage::PsfSweep, 0);
    let step2_dir = cfg.out_dir.join("Step-2_Decon");
    fs::create_dir_all(&step2_dir)?;

    let optimal: OptimalPsf = if cfg.f_size > 0 {
        // User has overridden the sweep — just build the PSF.
        let psf = psf_from_params(cfg, cfg.f_size);
        info!("user-fixed PSF (f_size = {})", cfg.f_size);
        OptimalPsf {
            psf:         psf.clone(),
            deconv_box:  opt_box.patch.clone(),
            f_size:      cfg.f_size,
            tau:         cfg.tau,
            v:           cfg.v,
            min_entropy: f64::NAN,
            max_energy:  f64::NAN,
            records:     Vec::new(),
            elapsed_s:   0.0,
        }
    } else {
        sweep_psf(&opt_box.patch, cfg)?
    };

    if cfg.keep_intermediates {
        plots::heatmap(&optimal.psf,        "Final point-spread function",
            step2_dir.join("PSF_final.png")).ok();
        plots::heatmap(&optimal.deconv_box, "Final optimisation box",
            step2_dir.join("optBox_final.png")).ok();
        if !optimal.records.is_empty() {
            let ent: Vec<(f64, f64)> = optimal.records.iter().enumerate()
                .map(|(i, r)| (i as f64, r.entropy)).collect();
            plots::scatter(&ent, "Shannon entropy vs. iteration", "iteration", "entropy (bits)",
                step2_dir.join("EntVsIter.png")).ok();
            let en: Vec<(f64, f64)> = optimal.records.iter().enumerate()
                .map(|(i, r)| (i as f64, r.energy)).collect();
            plots::scatter(&en, "GLCM energy vs. iteration", "iteration", "energy",
                step2_dir.join("EnergyVsIter.png")).ok();
        }
    }
    write_decon_log(&step2_dir, &optimal)?;

    // --- Step 3: calibration ----------------------------------------------
    prog.stage(Stage::Calibration, dapi.len());
    let calib_report: CalibrationReport = calibrate(&dapi, &optimal.psf, cfg, prog)?;
    let step3_dir = cfg.out_dir.join("Step-3_Corr");
    fs::create_dir_all(&step3_dir)?;
    if cfg.keep_intermediates {
        let all_sizes: Vec<f64> = calib_report.per_image.iter()
            .flat_map(|p| p.sizes_nm.clone()).collect();
        plots::histogram(&all_sizes, 21,
            "DAPI: calibration object size distribution",
            "object diameter (nm)", "frequency",
            step3_dir.join("XB_SizeHistogram.png")).ok();
    }
    write_corr_log(&step3_dir, &calib_report, cfg)?;

    // --- Step 4: quantification -------------------------------------------
    // Free the DAPI stack now that calibration is done, THEN load the FITC
    // stack — this keeps peak memory to roughly one stack, not two.
    drop(dapi);
    let n_dapi_loaded = calib_report.per_image.len();
    let fitc: Vec<GrayImage> = fitc_paths
        .iter()
        .map(|p| load_gray(p))
        .collect::<Result<Vec<_>>>()?;
    info!("loaded {} FITC images for quantification", fitc.len());

    prog.stage(Stage::Quantification, fitc.len());
    let quant_report: QuantReport =
        quantify(&fitc, &optimal.psf, calib_report.correction, cfg, prog)?;
    let step4_dir = cfg.out_dir.join("Step-4_genMasks");
    fs::create_dir_all(&step4_dir)?;
    if cfg.keep_intermediates {
        let all_sizes: Vec<f64> = quant_report.per_image.iter()
            .flat_map(|p| p.records.iter().map(|r| r.size_nm)).collect();
        plots::histogram(&all_sizes, 21,
            "FITC: object size distribution",
            "object diameter (nm)", "frequency",
            step4_dir.join("XG_SizeHistogram.png")).ok();
    }
    write_count_log(&step4_dir, &quant_report, cfg)?;
    write_legacy_tsv(&step4_dir.join("sizeCoords.tsv"), &quant_report)?;

    // --- Cyclops: ML classification ---------------------------------------
    prog.stage(Stage::Classification, 0);
    let all_records: Vec<crate::quantify::ObjectRecord> = quant_report
        .per_image
        .iter()
        .flat_map(|p| p.records.clone())
        .collect();
    let dapi_lookup = build_dapi_lookup(&calib_report.per_image);
    let classification: ClassificationReport =
        classify_objects(&all_records, &dapi_lookup, cfg);

    prog.stage(Stage::Writing, 0);
    write_objects_parquet(&step4_dir.join("cyclops_objects.parquet"), &classification)?;
    write_objects_csv(&step4_dir.join("cyclops_objects.tsv"), &classification)?;

    // --- Final report -----------------------------------------------------
    let report = PipelineReport {
        name:             NAME.to_string(),
        version:          VERSION.to_string(),
        formerly:         FORMERLY.to_string(),
        started_at,
        elapsed_seconds:  started.elapsed().as_secs_f64(),
        config_summary:   ConfigSummary {
            psf_method:  format!("{:?}", cfg.psf_method),
            size_metric: format!("{:?}", cfg.size_metric),
            px2nm:       cfg.px2nm(),
            n_mle_iter:  cfg.n_mle_iter,
            n_lr_iter:   cfg.n_lr_iter,
            sphere_nm:   cfg.sphere_size_nm,
            threads,
        },
        n_dapi:          n_dapi_loaded,
        n_fitc:          fitc.len(),
        min_distance_nm: opt_box.min_distance_nm,
        psf_f_size:      optimal.f_size,
        psf_tau:         optimal.tau,
        psf_v:           optimal.v,
        correction:      calib_report.correction,
        n_objects:       classification.objects.len(),
        mean_size_nm:    quant_report.mean_size_nm,
        size_bands:      quant_report.size_bands.clone(),
        domain_counts:   classification.domain_counts.clone(),
        output_dir:      cfg.out_dir.clone(),
    };
    let f = File::create(cfg.out_dir.join("cyclops_report.json"))
        .map_err(|e| CyclopsError::Io { path: cfg.out_dir.clone(), source: e })?;
    serde_json::to_writer_pretty(f, &report)
        .map_err(|e| CyclopsError::Other(anyhow::anyhow!(e)))?;

    info!("Cyclops complete: {:.1}s elapsed", report.elapsed_seconds);
    prog.stage(Stage::Done, 0);
    Ok(report)
}

fn load_dir(dir: &Path) -> Result<Vec<GrayImage>> {
    let paths = list_image_files(dir)?;
    paths.into_iter().map(load_gray).collect()
}

fn write_decon_log(dir: &Path, opt: &OptimalPsf) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("deconLog.txt"))?;
    writeln!(f, "|>=============================================<|")?;
    writeln!(f, "|>============ Cyclops Decon Log ==============<|")?;
    writeln!(f, "Sweeps          : {}",  opt.records.len())?;
    if opt.elapsed_s > 0.0 {
        writeln!(f, "Sweep rate      : {:.2} sweeps/s",
                 (opt.records.len() as f64) / opt.elapsed_s)?;
    }
    writeln!(f, "Optimal PSF:")?;
    writeln!(f, "  Min entropy   : {:.6} bits", opt.min_entropy)?;
    writeln!(f, "  Max GLCM E    : {:.6}",      opt.max_energy)?;
    writeln!(f, "  Filter size   : {}x{}",      opt.f_size, opt.f_size)?;
    writeln!(f, "  tau           : {:.6}",      opt.tau)?;
    writeln!(f, "  v             : {:.6}",      opt.v)?;
    writeln!(f, "|>=============================================<|")?;
    Ok(())
}

fn write_corr_log(dir: &Path, c: &CalibrationReport, cfg: &Config) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("corrLog.txt"))?;
    writeln!(f, "|>=============================================<|")?;
    writeln!(f, "|>============ Cyclops CORR Log ===============<|")?;
    for s in &c.per_image {
        writeln!(f, "  {}", s.name)?;
        writeln!(f, "    n_objects       : {}", s.n_objects)?;
        writeln!(f, "    mean size (nm)  : {:.2}", s.mean_size_nm)?;
    }
    writeln!(f, "Across {} images:", c.n_images)?;
    writeln!(f, "  Mean size       : {:.2} nm", c.mean_size_nm)?;
    writeln!(f, "  σ               : {:.2} nm", c.measurement_err)?;
    writeln!(f, "  Sphere diameter : {:.2} nm", cfg.sphere_size_nm)?;
    writeln!(f, "  CORR            : {:.6}",    c.correction)?;
    writeln!(f, "|>=============================================<|")?;
    Ok(())
}

fn write_count_log(dir: &Path, q: &QuantReport, _cfg: &Config) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(dir.join("countLog.txt"))?;
    writeln!(f, "|>=============================================<|")?;
    writeln!(f, "|>=========== Cyclops Count Log ==============<|")?;
    let mut total = 0u64;
    for s in &q.per_image {
        writeln!(f, "  {}", s.name)?;
        writeln!(f, "    n_objects       : {}",   s.n_objects)?;
        writeln!(f, "    mean size (nm)  : {:.2}", s.mean_size_nm)?;
        total += s.n_objects as u64;
    }
    writeln!(f, "Across {} images:", q.per_image.len())?;
    writeln!(f, "  Total objects   : {}", total)?;
    writeln!(f, "  Mean size       : {:.2} nm", q.mean_size_nm)?;
    writeln!(f, "Size bands:")?;
    writeln!(f, "  < 100 nm        : {}", q.size_bands.lt_100)?;
    writeln!(f, "  100–220 nm      : {}", q.size_bands.vlp_100_220)?;
    writeln!(f, "  220–500 nm      : {}", q.size_bands.small_220_500)?;
    writeln!(f, "  500–1200 nm     : {}", q.size_bands.bact_500_1200)?;
    writeln!(f, "  1200–3000 nm    : {}", q.size_bands.large_1200_3000)?;
    writeln!(f, "  > 3000 nm       : {}", q.size_bands.protist_gt_3000)?;
    writeln!(f, "|>=============================================<|")?;
    Ok(())
}

fn write_legacy_tsv(path: &Path, q: &QuantReport) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "fileName\tsize\txcoord\tycoord\tintensity(arb)")?;
    for img in &q.per_image {
        for r in &img.records {
            writeln!(f, "{}\t{}\t{}\t{}\t{}", r.file_name, r.size_nm, r.x_px, r.y_px, r.intensity)?;
        }
    }
    Ok(())
}

fn classified_to_dataframe(c: &ClassificationReport) -> Result<DataFrame> {
    let n = c.objects.len();
    let mut file_name    = Vec::with_capacity(n);
    let mut object_id    = Vec::with_capacity(n);
    let mut size_nm      = Vec::with_capacity(n);
    let mut major_nm     = Vec::with_capacity(n);
    let mut minor_nm     = Vec::with_capacity(n);
    let mut x            = Vec::with_capacity(n);
    let mut y            = Vec::with_capacity(n);
    let mut intensity    = Vec::with_capacity(n);
    let mut eccentricity = Vec::with_capacity(n);
    let mut area         = Vec::with_capacity(n);
    let mut domain       = Vec::with_capacity(n);
    let mut conf         = Vec::with_capacity(n);
    let mut ratio        = Vec::with_capacity(n);

    for o in &c.objects {
        file_name.push(o.record.file_name.clone());
        object_id.push(o.record.object_id as i64);
        size_nm.push(o.record.size_nm);
        major_nm.push(o.record.axis_major_nm);
        minor_nm.push(o.record.axis_minor_nm);
        x.push(o.record.x_px);
        y.push(o.record.y_px);
        intensity.push(o.record.intensity);
        eccentricity.push(o.record.eccentricity);
        area.push(o.record.area_px as i64);
        domain.push(o.domain.clone());
        conf.push(o.confidence);
        ratio.push(o.dapi_fitc);
    }
    let df = df!(
        "file_name"      => file_name,
        "object_id"      => object_id,
        "size_nm"        => size_nm,
        "axis_major_nm"  => major_nm,
        "axis_minor_nm"  => minor_nm,
        "x_px"           => x,
        "y_px"           => y,
        "intensity"      => intensity,
        "eccentricity"   => eccentricity,
        "area_px"        => area,
        "domain"         => domain,
        "confidence"     => conf,
        "dapi_fitc_ratio"=> ratio,
    )?;
    Ok(df)
}

fn write_objects_parquet(path: &Path, c: &ClassificationReport) -> Result<()> {
    let mut df = classified_to_dataframe(c)?;
    let f = std::fs::File::create(path)?;
    ParquetWriter::new(f).finish(&mut df)?;
    Ok(())
}

fn write_objects_csv(path: &Path, c: &ClassificationReport) -> Result<()> {
    let mut df = classified_to_dataframe(c)?;
    let f = std::fs::File::create(path)?;
    CsvWriter::new(f).with_separator(b'\t').finish(&mut df)?;
    Ok(())
}
