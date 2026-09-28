//! End-to-end integration test for Cyclops.
//!
//! Generates synthetic DAPI calibration + FITC sample TIFFs in a
//! tempdir, runs the full four-stage pipeline through
//! [`cyclops_core::pipeline::run_pipeline`], and asserts that the
//! expected output artefacts exist and are non-empty.
//!
//! This is a *smoke* test — it does not validate numerical fidelity
//! against the upstream EpiVirQuant reference data (that needs the
//! GSL TIFFs and a benchmark machine). Its job is to catch crashes,
//! divergent file paths, missing-feature panics, and broken Polars
//! schema changes.

use std::path::PathBuf;

use image::{ImageBuffer, Luma};
use tempfile::TempDir;

use cyclops_core::config::{
    ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric,
};
use cyclops_core::pipeline::run_pipeline;

/// Make a tiny `16 × 16` u16 grayscale TIFF with two bright spots of
/// `radius_px` at the given centroid offsets from the image centre.
fn write_bead_pair(
    path:     &PathBuf,
    size:     u32,
    spot1:    (i32, i32),
    spot2:    (i32, i32),
    radius:   f32,
    bg_noise: u16,
) {
    let mut img = ImageBuffer::<Luma<u16>, Vec<u16>>::new(size, size);
    let cx = (size as i32) / 2;
    let cy = (size as i32) / 2;
    for (x, y, px) in img.enumerate_pixels_mut() {
        let xi = x as i32;
        let yi = y as i32;
        let d1 = (((xi - cx - spot1.0).pow(2) + (yi - cy - spot1.1).pow(2)) as f32).sqrt();
        let d2 = (((xi - cx - spot2.0).pow(2) + (yi - cy - spot2.1).pow(2)) as f32).sqrt();
        let v1 = (50_000.0 * (-(d1 / radius).powi(2)).exp()) as u32;
        let v2 = (50_000.0 * (-(d2 / radius).powi(2)).exp()) as u32;
        let v  = (v1 + v2 + bg_noise as u32).min(65_535);
        *px = Luma([v as u16]);
    }
    img.save(path).expect("write TIFF");
}

/// Make a single-bead field at the image centre (used for the FITC
/// sample stack).
fn write_single_bead(path: &PathBuf, size: u32, radius: f32) {
    let mut img = ImageBuffer::<Luma<u16>, Vec<u16>>::new(size, size);
    let cx = (size as i32) / 2;
    let cy = (size as i32) / 2;
    for (x, y, px) in img.enumerate_pixels_mut() {
        let d = (((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32).sqrt();
        let v = (45_000.0 * (-(d / radius).powi(2)).exp()) as u32;
        *px = Luma([v.min(65_535) as u16]);
    }
    img.save(path).expect("write TIFF");
}

#[test]
fn pipeline_runs_end_to_end_on_synthetic_data() {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path().to_path_buf();

    let dapi_dir = root.join("dapi");
    let fitc_dir = root.join("fitc");
    std::fs::create_dir_all(&dapi_dir).unwrap();
    std::fs::create_dir_all(&fitc_dir).unwrap();

    // -- DAPI calibration scan: one image with two paired beads --
    let calibration = dapi_dir.join("cal_001.tiff");
    write_bead_pair(&calibration, 64, (-6, 0), (6, 0), 2.5, 100);

    // Another DAPI sphere field for the calibration stage to average
    // measurements over.
    write_single_bead(&dapi_dir.join("dapi_002.tiff"), 32, 3.0);
    write_single_bead(&dapi_dir.join("dapi_003.tiff"), 32, 3.0);

    // -- FITC sample stack: two single-bead fields --
    write_single_bead(&fitc_dir.join("fitc_001.tiff"), 32, 3.5);
    write_single_bead(&fitc_dir.join("fitc_002.tiff"), 32, 3.5);

    let out_dir = root.join("Cyclops_Output");

    let cfg = Config {
        dapi_dir:        dapi_dir.clone(),
        fitc_dir:        fitc_dir.clone(),
        calibration:     calibration.clone(),
        out_dir:         out_dir.clone(),

        // generous scale: 8 px = 1000 nm → 125 nm/px
        scale_length_px: 8.0,
        scale_metric_nm: 1_000.0,
        sphere_size_nm:  500.0,

        pad:          4,
        d_constraint: 40,

        f_size:     5,                  // fixed, skip the sweep
        psf_method: PsfMethod::Gaussian,
        sigma:      1.0,
        n_mle_iter: 2,                  // keep the test snappy
        n_lr_iter:  10,

        size_metric:        SizeMetric::EquivalentDiameter,
        sm_constraint:      100_000.0,
        gen_figures:        false,
        keep_intermediates: false,

        cpus: 1,

        classifier: ClassifierConfig {
            domains:    vec![OrganismDomain::Virus, OrganismDomain::Bacteria],
            gmm_refine: false,
            onnx_model: None,
        },

        ..Default::default()
    };

    let report = run_pipeline(&cfg).expect("pipeline should succeed on synthetic data");

    eprintln!("\n=== Pipeline report ===");
    eprintln!("output      : {}", report.output_dir.display());
    eprintln!("n_dapi      : {}", report.n_dapi);
    eprintln!("n_fitc      : {}", report.n_fitc);
    eprintln!("min_dist_nm : {:.2}", report.min_distance_nm);
    eprintln!("psf         : f={} τ={:.3} v={:.3}", report.psf_f_size, report.psf_tau, report.psf_v);
    eprintln!("CORR        : {:.4}", report.correction);
    eprintln!("n_objects   : {}", report.n_objects);
    eprintln!("mean_size   : {:.1} nm", report.mean_size_nm);
    eprintln!("size_bands  : {:?}", report.size_bands);
    eprintln!("domains     : {:?}", report.domain_counts);
    eprintln!("elapsed     : {:.2} s\n", report.elapsed_seconds);

    // Basic sanity checks on the report itself.
    assert_eq!(report.n_dapi, 3, "DAPI count should be 3");
    assert_eq!(report.n_fitc, 2, "FITC count should be 2");
    assert!(report.elapsed_seconds > 0.0, "elapsed should be positive");
    assert!(report.correction.is_finite(), "CORR factor must be finite");
    assert!(report.correction > 0.0, "CORR factor must be positive");
    assert_eq!(report.output_dir, out_dir);

    // Output files we promise users.
    let report_json = out_dir.join("cyclops_report.json");
    assert!(report_json.exists(), "cyclops_report.json must exist");

    let objects_tsv = out_dir.join("Step-4_genMasks").join("cyclops_objects.tsv");
    let objects_pq  = out_dir.join("Step-4_genMasks").join("cyclops_objects.parquet");
    assert!(objects_tsv.exists(), "cyclops_objects.tsv must exist");
    assert!(objects_pq.exists(),  "cyclops_objects.parquet must exist");

    // Legacy compatibility files.
    let legacy_tsv = out_dir.join("Step-4_genMasks").join("sizeCoords.tsv");
    assert!(legacy_tsv.exists(), "legacy sizeCoords.tsv must exist");

    let decon_log = out_dir.join("Step-2_Decon").join("deconLog.txt");
    let corr_log  = out_dir.join("Step-3_Corr") .join("corrLog.txt");
    let count_log = out_dir.join("Step-4_genMasks").join("countLog.txt");
    for p in [&decon_log, &corr_log, &count_log] {
        assert!(p.exists(), "log {} must exist", p.display());
        assert!(std::fs::metadata(p).unwrap().len() > 0, "{} should be non-empty", p.display());
    }

    // The TSV header must include the new ML-classifier columns.
    let tsv = std::fs::read_to_string(&objects_tsv).unwrap();
    let header = tsv.lines().next().unwrap_or("");
    for col in ["file_name", "size_nm", "domain", "confidence"] {
        assert!(header.contains(col), "header missing `{col}`: {header}");
    }
}
