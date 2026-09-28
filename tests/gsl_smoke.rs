//! GSL-subset smoke test — runs the full pipeline on a 4-image slice of
//! the published GSL dataset and asserts the numerical output is in the
//! same order of magnitude as the EpiVirQuant reference. This test is
//! *gated* on the presence of the GSL TIFFs at a well-known path and
//! is silently skipped otherwise.

use std::path::PathBuf;

use cyclops_core::config::{
    ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric,
};
use cyclops_core::pipeline::run_pipeline;

#[test]
fn gsl_subset_pipeline_runs() {
    let gsl_root = PathBuf::from("/home/claude/epivirquant-main/data/GSL/tiff");
    if !gsl_root.exists() {
        eprintln!("Skipping GSL smoke test — no GSL TIFFs present at {gsl_root:?}");
        return;
    }

    // Build a tempdir-style subset by symlinking 4 paired DAPI/FITC images
    // into /tmp/cyclops_gsl_subset_test/ (idempotent).
    let work = PathBuf::from("/tmp/cyclops_gsl_subset_test");
    let dapi = work.join("dapi");
    let fitc = work.join("fitc");
    let out  = work.join("out");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&dapi).unwrap();
    std::fs::create_dir_all(&fitc).unwrap();
    for i in 1..=4 {
        let dap_name = format!("GSL_+_blue_beads_{i}_(dapi).tiff");
        let fit_name = format!("GSL_+_blue_beads_{i}_(fitc).tiff");
        let src_dapi = gsl_root.join("dapi").join(&dap_name);
        let src_fitc = gsl_root.join("fitc").join(&fit_name);
        let dst_dapi = dapi.join(&dap_name);
        let dst_fitc = fitc.join(&fit_name);
        if !dst_dapi.exists() {
            std::fs::copy(&src_dapi, &dst_dapi).unwrap();
        }
        if !dst_fitc.exists() {
            std::fs::copy(&src_fitc, &dst_fitc).unwrap();
        }
    }

    let cal = dapi.join("GSL_+_blue_beads_1_(dapi).tiff");

    let cfg = Config {
        dapi_dir:        dapi.clone(),
        fitc_dir:        fitc.clone(),
        calibration:     cal,
        out_dir:         out.clone(),

        scale_length_px: 585.0,
        scale_metric_nm: 20_000.0,
        sphere_size_nm:  175.0,

        pad:          14,
        d_constraint: 30,

        // Use the PSF parameters Cyclops's own sweep already discovered
        // on this dataset, so we skip the 25 s sweep stage:
        f_size:     19,
        psf_method: PsfMethod::GammaSinc,
        sigma:      1.0,
        tau:        0.0318,
        v:          2.5133,
        n_mle_iter: 5,
        n_lr_iter:  20,

        size_metric:        SizeMetric::EquivalentDiameter,
        sm_constraint:      8_000.0,
        gen_figures:        false,
        keep_intermediates: false,

        cpus: 1,

        classifier: ClassifierConfig {
            domains:    vec![
                OrganismDomain::Virus,
                OrganismDomain::Bacteria,
            ],
            gmm_refine: false,
            onnx_model: None,
        },

        ..Default::default()
    };

    let report = run_pipeline(&cfg).expect("GSL subset pipeline should succeed");

    eprintln!("\n=== GSL subset Cyclops report ===");
    eprintln!("VP min distance : {:.1} nm (reference: 457.9 nm)", report.min_distance_nm);
    eprintln!("CORR            : {:.4}", report.correction);
    eprintln!("n_objects       : {} (= {:.1}/image)",
              report.n_objects, report.n_objects as f64 / report.n_fitc as f64);
    eprintln!("mean size       : {:.1} nm", report.mean_size_nm);
    eprintln!("size bands      : {:?}", report.size_bands);
    eprintln!("elapsed         : {:.1} s\n", report.elapsed_seconds);

    // ---- sanity bounds (NOT bit-exact — RL is iterative and we run nLR_iter=20) ----
    //
    // EpiVirQuant cs2 default sweep @ panel C @ iteration 20 reports
    // mean_size = 358.0 nm and mean_count = 63.6 / image.
    // We assert Cyclops lands within ±30 % of those numbers.

    let per_image = report.n_objects as f64 / report.n_fitc as f64;
    assert!(report.mean_size_nm > 250.0 && report.mean_size_nm < 465.0,
        "mean_size {:.1} nm out of expected band [250, 465] (ref 358)", report.mean_size_nm);
    assert!(per_image > 45.0 && per_image < 85.0,
        "per-image count {:.1} out of expected band [45, 85] (ref 63.6)", per_image);
    assert!((report.min_distance_nm - 457.9).abs() < 5.0,
        "VP min distance {:.1} ≠ 457.9 nm", report.min_distance_nm);
}
