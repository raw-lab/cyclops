//! Data-safety tests for the output-directory guard.
//!
//! These verify the fix for the bug where pointing `--outDir` at (or inside)
//! the input data folder would cause the pipeline to recursively delete the
//! user's images. The guard must REFUSE to run in any overlapping-path
//! configuration, and must NOT delete anything.

use std::fs;
use std::path::PathBuf;

use cyclops_core::config::{
    ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric,
};
use cyclops_core::pipeline::run_pipeline;
use tempfile::TempDir;

/// Build a minimal config rooted at `root`, with the given output dir.
fn make_cfg(root: &std::path::Path, dapi: PathBuf, fitc: PathBuf, cal: PathBuf, out: PathBuf) -> Config {
    let _ = root;
    Config {
        dapi_dir:        dapi,
        fitc_dir:        fitc,
        calibration:     cal,
        out_dir:         out,
        scale_length_px: 585.0,
        scale_metric_nm: 20_000.0,
        sphere_size_nm:  175.0,
        pad:          14,
        d_constraint: 30,
        f_size:     5,
        psf_method: PsfMethod::Gaussian,
        sigma:      1.0,
        n_mle_iter: 2,
        n_lr_iter:  5,
        size_metric:        SizeMetric::EquivalentDiameter,
        sm_constraint:      8_000.0,
        gen_figures:        false,
        keep_intermediates: false,
        cpus: 1,
        classifier: ClassifierConfig {
            domains:    vec![OrganismDomain::Virus],
            gmm_refine: false,
            onnx_model: None,
        },
        ..Default::default()
    }
}

/// A precious "image" file we will assert is never deleted.
fn plant_data_file(dir: &std::path::Path) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let f = dir.join("precious_image.tiff");
    fs::write(&f, b"PRETEND TIFF BYTES - DO NOT DELETE").unwrap();
    f
}

#[test]
fn refuses_when_output_equals_dapi_dir() {
    let tmp = TempDir::new().unwrap();
    let dapi = tmp.path().join("data");
    let precious = plant_data_file(&dapi);

    // Output points AT the DAPI directory — the old bug would nuke it.
    let cfg = make_cfg(
        tmp.path(),
        dapi.clone(),
        dapi.clone(),
        precious.clone(),
        dapi.clone(), // out_dir == dapi_dir
    );

    let result = run_pipeline(&cfg);
    assert!(result.is_err(), "pipeline must refuse when output == data dir");
    assert!(
        precious.exists(),
        "DATA LOSS: precious file was deleted even though the run was refused"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("refusing to run"),
        "error should explain the refusal, got: {msg}"
    );
}

#[test]
fn refuses_when_output_inside_dapi_dir() {
    let tmp = TempDir::new().unwrap();
    let dapi = tmp.path().join("data");
    let precious = plant_data_file(&dapi);

    // Output is a SUBfolder of the data dir.
    let out = dapi.join("Cyclops_Output");
    let cfg = make_cfg(tmp.path(), dapi.clone(), dapi.clone(), precious.clone(), out);

    let result = run_pipeline(&cfg);
    assert!(result.is_err(), "pipeline must refuse when output is inside data dir");
    assert!(precious.exists(), "DATA LOSS: precious file deleted on refused run");
}

#[test]
fn refuses_when_dapi_inside_output_dir() {
    let tmp = TempDir::new().unwrap();
    let out  = tmp.path().join("everything");
    let dapi = out.join("data"); // data nested under the output dir
    let precious = plant_data_file(&dapi);

    let cfg = make_cfg(tmp.path(), dapi.clone(), dapi.clone(), precious.clone(), out);

    let result = run_pipeline(&cfg);
    assert!(result.is_err(), "pipeline must refuse when data is inside output dir");
    assert!(precious.exists(), "DATA LOSS: precious file deleted on refused run");
}

#[test]
fn refuses_when_output_inside_fitc_dir() {
    let tmp = TempDir::new().unwrap();
    let dapi = tmp.path().join("dapi");
    let fitc = tmp.path().join("fitc");
    let cal  = plant_data_file(&dapi);
    let precious_fitc = plant_data_file(&fitc);

    // Output nested under the FITC dir.
    let out = fitc.join("out");
    let cfg = make_cfg(tmp.path(), dapi, fitc.clone(), cal, out);

    let result = run_pipeline(&cfg);
    assert!(result.is_err(), "pipeline must refuse when output is inside FITC dir");
    assert!(precious_fitc.exists(), "DATA LOSS: FITC file deleted on refused run");
}

#[test]
fn does_not_delete_unrelated_files_in_a_separate_output_dir() {
    // When the output dir is genuinely separate, the pipeline may write into
    // it — but it must NOT recursively delete a pre-existing unrelated file
    // that happens to already be in that output dir.
    let tmp = TempDir::new().unwrap();
    let dapi = tmp.path().join("dapi");
    let fitc = tmp.path().join("fitc");
    plant_data_file(&dapi);
    plant_data_file(&fitc);
    let cal = dapi.join("precious_image.tiff");

    let out = tmp.path().join("results");
    fs::create_dir_all(&out).unwrap();
    // A pre-existing file the user might have left in the output dir.
    let keep = out.join("my_notes.txt");
    fs::write(&keep, b"keep me").unwrap();

    let cfg = make_cfg(tmp.path(), dapi, fitc, cal, out.clone());

    // This will likely Err during image loading (our planted files aren't
    // real TIFFs), but the key assertion is that the guard didn't delete the
    // pre-existing file before failing.
    let _ = run_pipeline(&cfg);
    assert!(
        keep.exists(),
        "pre-existing file in a separate output dir must not be deleted"
    );
}
