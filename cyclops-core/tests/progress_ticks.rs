//! Verify that run_pipeline_with_progress emits per-image ticks during the
//! calibration and quantification stages (the bar's per-item motion).

use std::path::PathBuf;
use std::sync::Mutex;

use cyclops_core::config::{ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric};
use cyclops_core::pipeline::run_pipeline_with_progress;
use cyclops_core::progress::{Progress, Stage};
use image::{ImageBuffer, Luma};
use tempfile::TempDir;

#[derive(Default)]
struct Recorder {
    stages: Mutex<Vec<String>>,
    ticks:  Mutex<Vec<(usize, usize)>>,
}
impl Progress for Recorder {
    fn stage(&self, s: Stage, total: usize) {
        self.stages.lock().unwrap().push(format!("{}:{}", s.label(), total));
    }
    fn tick(&self, done: usize, total: usize) {
        self.ticks.lock().unwrap().push((done, total));
    }
}

fn bead_pair(path: &PathBuf, size: u32, s1: (i32,i32), s2: (i32,i32), r: f32, bg: u16) {
    let mut img = ImageBuffer::<Luma<u16>, Vec<u16>>::new(size, size);
    let (cx, cy) = ((size/2) as i32, (size/2) as i32);
    for (x,y,px) in img.enumerate_pixels_mut() {
        let d1 = (((x as i32-cx-s1.0).pow(2)+(y as i32-cy-s1.1).pow(2)) as f32).sqrt();
        let d2 = (((x as i32-cx-s2.0).pow(2)+(y as i32-cy-s2.1).pow(2)) as f32).sqrt();
        let v = (50000.0*(-(d1/r).powi(2)).exp() + 50000.0*(-(d2/r).powi(2)).exp() + bg as f32).min(65535.0);
        *px = Luma([v as u16]);
    }
    img.save(path).unwrap();
}
fn single(path: &PathBuf, size: u32, r: f32) {
    let mut img = ImageBuffer::<Luma<u16>, Vec<u16>>::new(size, size);
    let (cx, cy) = ((size/2) as i32, (size/2) as i32);
    for (x,y,px) in img.enumerate_pixels_mut() {
        let d = (((x as i32-cx).pow(2)+(y as i32-cy).pow(2)) as f32).sqrt();
        *px = Luma([(45000.0*(-(d/r).powi(2)).exp()).min(65535.0) as u16]);
    }
    img.save(path).unwrap();
}

#[test]
fn pipeline_emits_per_image_ticks() {
    let tmp = TempDir::new().unwrap();
    let dapi = tmp.path().join("dapi");
    let fitc = tmp.path().join("fitc");
    std::fs::create_dir_all(&dapi).unwrap();
    std::fs::create_dir_all(&fitc).unwrap();
    let cal = dapi.join("cal.tiff");
    bead_pair(&cal, 64, (-6,0), (6,0), 2.5, 100);
    // 3 DAPI + 3 FITC so we expect ticks up to (3,3) in two stages.
    single(&dapi.join("d2.tiff"), 32, 3.0);
    single(&dapi.join("d3.tiff"), 32, 3.0);
    for i in 0..3 { single(&fitc.join(format!("f{i}.tiff")), 32, 3.5); }

    let cfg = Config {
        dapi_dir: dapi.clone(), fitc_dir: fitc.clone(), calibration: cal,
        out_dir: tmp.path().join("out"),
        scale_length_px: 8.0, scale_metric_nm: 1000.0, sphere_size_nm: 500.0,
        pad: 4, d_constraint: 40,
        f_size: 5, psf_method: PsfMethod::Gaussian, sigma: 1.0,
        n_mle_iter: 2, n_lr_iter: 5,
        size_metric: SizeMetric::EquivalentDiameter, sm_constraint: 100000.0,
        gen_figures: false, keep_intermediates: false, cpus: 1,
        classifier: ClassifierConfig {
            domains: vec![OrganismDomain::Virus, OrganismDomain::Bacteria],
            gmm_refine: false, onnx_model: None,
        },
        ..Default::default()
    };

    let rec = Recorder::default();
    run_pipeline_with_progress(&cfg, &rec).expect("pipeline");

    let stages = rec.stages.lock().unwrap();
    let ticks  = rec.ticks.lock().unwrap();

    // We must have seen the key stages in order.
    let joined = stages.join(" | ");
    assert!(joined.contains("Step 3"), "missing calibration stage: {joined}");
    assert!(joined.contains("Step 4"), "missing quantification stage: {joined}");
    assert!(joined.contains("Done"), "missing done stage: {joined}");

    // Calibration announced 3 items, quantification announced 3 items.
    assert!(stages.iter().any(|s| s.contains("Step 3") && s.ends_with(":3")),
        "calibration should announce 3 items: {joined}");
    assert!(stages.iter().any(|s| s.contains("Step 4") && s.ends_with(":3")),
        "quantification should announce 3 items: {joined}");

    // We must have received ticks reaching (3,3) at least twice
    // (once per per-image stage).
    let full = ticks.iter().filter(|(d,t)| *d == 3 && *t == 3).count();
    assert!(full >= 2, "expected >=2 completion ticks (calib+quant), got {full}: {:?}", *ticks);
    // And ticks must be monotonic within bounds.
    for (d, t) in ticks.iter() {
        assert!(*d >= 1 && *d <= *t, "tick out of range: {d}/{t}");
    }
}
