//! Cyclops GUI — top-level `eframe::App`.
//!
//! Layout
//! ------
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ [logo]  Cyclops  v0.1.0      formerly EpiVirQuant               │
//! ├──────── side panel ─────────┬─────────── central panel ─────────┤
//! │ Inputs                       │ Status / Log                      │
//! │   DAPI dir       [Browse]    │                                   │
//! │   FITC dir       [Browse]    │ ─ Results ─                       │
//! │   Calibration    [Browse]    │   ▣ Object table (Polars-backed)  │
//! │   Output dir     [Browse]    │   ▣ Size histogram                │
//! │ Scale  ─────                 │   ▣ Domain breakdown              │
//! │   px / nm / sphereSize       │                                   │
//! │ PSF sweep                    │                                   │
//! │   fSize / method / τ / v …   │                                   │
//! │ Domains                      │                                   │
//! │   ☑ virus  ☑ bacteria …      │                                   │
//! │ [Run Cyclops]   [Cancel]     │                                   │
//! └─────────────────────────────┴───────────────────────────────────┘
//! ```

use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
    Arc,
};
use std::thread;
use std::time::Instant;

use eframe::egui::{self, Color32, RichText, Stroke, Ui, Vec2};
use egui_extras::{Column, TableBuilder};

use cyclops_core::config::{
    ClassifierConfig, Config, OrganismDomain, PsfMethod, SizeMetric,
};
use cyclops_core::pipeline::{run_pipeline_with_progress, PipelineReport};
use cyclops_core::{FORMERLY, NAME, VERSION};

// ---------------------------------------------------------------------------
//  Bundled assets
// ---------------------------------------------------------------------------

const CYCLOPS_LOGO_SVG: &[u8] = include_bytes!("../../assets/cyclops.svg");

// ---------------------------------------------------------------------------
//  Worker messages
// ---------------------------------------------------------------------------

enum WorkerMsg {
    Log(String),
    /// A new pipeline stage began; carries its label and the number of
    /// per-item ticks expected (0 if unknown).
    StageChange { label: String, start_frac: f32, end_frac: f32, total: usize },
    /// One item within the current stage finished.
    Tick { done: usize, total: usize },
    Done(Box<PipelineReport>),
    Failed(String),
}

// ---------------------------------------------------------------------------
//  GUI form-state (1:1 with [`Config`], but uses `String` for path fields so
//  the user can type freely before resolving to a [`PathBuf`])
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FormState {
    dapi_dir:    String,
    fitc_dir:    String,
    calibration: String,
    out_dir:     String,

    scale_length_px: f64,
    scale_metric_nm: f64,
    sphere_size_nm:  f64,

    pad:          i32,
    d_constraint: i32,

    f_size:     i32,
    psf_method: PsfMethod,
    tau:        f64,
    v:          f64,
    sigma:      f64,
    n_mle_iter: i32,
    n_lr_iter:  i32,

    size_metric_idx: usize, // 0 = EquivalentDiameter, 1 = AverageAxes
    sm_constraint:   f64,
    gen_figs:        bool,
    keep_intermed:   bool,

    cpus: i32,

    // domains
    use_virus:    bool,
    use_bacteria: bool,
    use_archaea:  bool,
    use_protist:  bool,
    gmm_refine:   bool,
    onnx_model:   String,
}

impl Default for FormState {
    fn default() -> Self {
        Self {
            dapi_dir:    String::new(),
            fitc_dir:    String::new(),
            calibration: String::new(),
            out_dir:     "Cyclops_Output".into(),

            scale_length_px: 585.0,
            scale_metric_nm: 20_000.0,
            sphere_size_nm:  175.0,

            pad:          14,
            d_constraint: 30,

            f_size:     0,
            psf_method: PsfMethod::GammaSinc,
            tau:        0.5,
            v:          2.25,
            sigma:      1.0,
            n_mle_iter: 10,
            n_lr_iter:  80,

            size_metric_idx: 0,
            sm_constraint:   8_000.0,
            gen_figs:        false,
            keep_intermed:   true,

            cpus: -2,

            use_virus:    true,
            use_bacteria: true,
            use_archaea:  true,
            use_protist:  true,
            gmm_refine:   true,
            onnx_model:   String::new(),
        }
    }
}

impl FormState {
    fn to_config(&self) -> Result<Config, String> {
        let dapi = PathBuf::from(self.dapi_dir.trim());
        let fitc = PathBuf::from(self.fitc_dir.trim());
        let cal  = PathBuf::from(self.calibration.trim());
        let out  = PathBuf::from(self.out_dir.trim());

        if self.dapi_dir.trim().is_empty()    { return Err("DAPI directory is required.".into()); }
        if self.fitc_dir.trim().is_empty()    { return Err("FITC directory is required.".into()); }
        if self.calibration.trim().is_empty() { return Err("Calibration image is required.".into()); }
        if self.out_dir.trim().is_empty()     { return Err("Output directory is required.".into()); }

        // --- DATA-SAFETY pre-flight check -----------------------------------
        // Catch the dangerous "output dir overlaps data dir" case here, before
        // the pipeline runs, with a friendly message. (The pipeline enforces
        // this again as a hard backstop.)
        {
            let norm = |p: &str| -> PathBuf {
                let pb = PathBuf::from(p.trim());
                let abs = if pb.is_absolute() {
                    pb
                } else {
                    std::env::current_dir().unwrap_or_default().join(pb)
                };
                let mut out = PathBuf::new();
                for c in abs.components() {
                    use std::path::Component;
                    match c {
                        Component::ParentDir => { out.pop(); }
                        Component::CurDir    => {}
                        other                => out.push(other.as_os_str()),
                    }
                }
                out
            };
            let out_n  = norm(&self.out_dir);
            let dapi_n = norm(&self.dapi_dir);
            let fitc_n = norm(&self.fitc_dir);
            let cal_parent = PathBuf::from(self.calibration.trim());
            let cal_n = norm(cal_parent.parent().map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".into()).as_str());

            for (label, inp) in [("DAPI directory", &dapi_n),
                                 ("FITC directory", &fitc_n),
                                 ("calibration image's folder", &cal_n)] {
                let overlap = out_n == *inp
                    || out_n.starts_with(inp)
                    || inp.starts_with(&out_n);
                if overlap {
                    return Err(format!(
                        "Output directory overlaps your {label}.\n\n\
                         Output: {}\n{}: {}\n\n\
                         Pick an output folder OUTSIDE your data folders so \
                         Cyclops cannot touch your images (e.g. \
                         ~/cyclops_runs/run1).",
                        out_n.display(), label, inp.display(),
                    ));
                }
            }
        }

        let mut domains = Vec::with_capacity(4);
        if self.use_virus    { domains.push(OrganismDomain::Virus); }
        if self.use_bacteria { domains.push(OrganismDomain::Bacteria); }
        if self.use_archaea  { domains.push(OrganismDomain::Archaea); }
        if self.use_protist  { domains.push(OrganismDomain::Protist); }
        if domains.is_empty() {
            return Err("Select at least one organism domain.".into());
        }

        let size_metric = if self.size_metric_idx == 0 {
            SizeMetric::EquivalentDiameter
        } else {
            SizeMetric::AverageAxes
        };

        let onnx_model = if self.onnx_model.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(self.onnx_model.trim()))
        };

        Ok(Config {
            dapi_dir:        dapi,
            fitc_dir:        fitc,
            calibration:     cal,
            out_dir:         out,

            scale_length_px: self.scale_length_px,
            scale_metric_nm: self.scale_metric_nm,
            sphere_size_nm:  self.sphere_size_nm,

            pad:          self.pad.max(0) as usize,
            d_constraint: self.d_constraint.max(0) as usize,

            f_size:     self.f_size.max(0) as usize,
            psf_method: self.psf_method,
            a:          1.0,
            b:          1.0,
            sigma:      self.sigma,
            r:          std::f64::consts::E,
            tau:        self.tau,
            v:          self.v,
            s:          0.0,
            n_mle_iter: self.n_mle_iter.max(1) as usize,
            n_lr_iter:  self.n_lr_iter.max(1) as usize,

            size_metric,
            sm_constraint: self.sm_constraint,
            gen_figures:   self.gen_figs,

            cpus: self.cpus,

            classifier: ClassifierConfig {
                domains,
                gmm_refine: self.gmm_refine,
                onnx_model,
            },
            keep_intermediates: self.keep_intermed,
        })
    }
}

// ---------------------------------------------------------------------------
//  App
// ---------------------------------------------------------------------------

#[derive(PartialEq, Eq)]
enum CentralTab {
    Status,
    Objects,
    Bands,
    Domains,
}

pub struct CyclopsApp {
    form:           FormState,
    log_lines:      Vec<String>,
    last_report:    Option<PipelineReport>,
    last_error:     Option<String>,
    is_running:     bool,
    cancel_flag:    Arc<AtomicBool>,
    rx:             Option<Receiver<WorkerMsg>>,
    started_at:     Option<Instant>,
    central_tab:    CentralTab,
    // --- live progress state ---
    stage_label:    String,   // current stage's human label
    stage_start:    f32,      // global fraction at stage start
    stage_end:      f32,      // global fraction at stage end
    stage_done:     usize,    // items finished in this stage
    stage_total:    usize,    // items expected in this stage (0 = indeterminate)
}

impl CyclopsApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Install the SVG/image loaders so we can render the bundled logo.
        egui_extras::install_image_loaders(&cc.egui_ctx);

        // Slightly punchier default fonts/spacing.
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 6.0);
        style.spacing.button_padding = Vec2::new(10.0, 6.0);
        cc.egui_ctx.set_style(style);

        Self {
            form:        FormState::default(),
            log_lines:   vec![format!("{NAME} v{VERSION} — ready ({FORMERLY})")],
            last_report: None,
            last_error:  None,
            is_running:  false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            rx:          None,
            started_at:  None,
            central_tab: CentralTab::Status,
            stage_label: String::new(),
            stage_start: 0.0,
            stage_end:   0.0,
            stage_done:  0,
            stage_total: 0,
        }
    }

    // ----- worker plumbing -------------------------------------------------

    fn launch_pipeline(&mut self) {
        let cfg = match self.form.to_config() {
            Ok(c) => c,
            Err(e) => {
                self.last_error = Some(e);
                return;
            }
        };

        self.last_error  = None;
        self.last_report = None;
        self.is_running  = true;
        self.started_at  = Some(Instant::now());
        self.stage_label = "Starting…".into();
        self.stage_start = 0.0;
        self.stage_end   = 0.05;
        self.stage_done  = 0;
        self.stage_total = 0;
        self.log_lines.push(format!(
            "▶ launching pipeline — output → {}",
            cfg.out_dir.display()
        ));
        self.cancel_flag.store(false, Ordering::SeqCst);

        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);

        thread::spawn(move || {
            // Pipe a couple of pre-flight log lines back to the GUI.
            let _ = tx.send(WorkerMsg::Log(format!(
                "DAPI ⇒ {}", cfg.dapi_dir.display()
            )));
            let _ = tx.send(WorkerMsg::Log(format!(
                "FITC ⇒ {}", cfg.fitc_dir.display()
            )));

            // Bridge the core's Progress trait to our mpsc channel. The
            // parallel stages tick from rayon worker threads, so the
            // closure must be Sync — wrap the (Send-only) Sender in a Mutex.
            let tx_prog = std::sync::Mutex::new(tx.clone());
            let prog = cyclops_core::progress::FnProgress::new(move |ev| {
                use cyclops_core::progress::ProgressEvent;
                let msg = match ev {
                    ProgressEvent::Stage { stage, total } => WorkerMsg::StageChange {
                        label:      stage.label().to_string(),
                        start_frac: stage.start_fraction(),
                        end_frac:   stage.end_fraction(),
                        total,
                    },
                    ProgressEvent::Tick { done, total } => WorkerMsg::Tick { done, total },
                    ProgressEvent::Message(m) => WorkerMsg::Log(m),
                };
                if let Ok(s) = tx_prog.lock() {
                    let _ = s.send(msg);
                }
            });

            match run_pipeline_with_progress(&cfg, &prog) {
                Ok(report) => {
                    let _ = tx.send(WorkerMsg::Done(Box::new(report)));
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Failed(format!("{e:#}")));
                }
            }
        });
    }

    fn pump_worker(&mut self, ctx: &egui::Context) {
        // Take the receiver out so the loop body can freely mutate &mut self,
        // including reassigning self.rx = None when the worker reports Done/Failed.
        let Some(rx) = self.rx.take() else { return };
        let mut keep_rx = true;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                WorkerMsg::Log(line) => {
                    self.log_lines.push(line);
                }
                WorkerMsg::StageChange { label, start_frac, end_frac, total } => {
                    self.log_lines.push(format!("▸ {label}"));
                    self.stage_label = label;
                    self.stage_start = start_frac;
                    self.stage_end   = end_frac;
                    self.stage_total = total;
                    self.stage_done  = 0;
                }
                WorkerMsg::Tick { done, total } => {
                    self.stage_done  = done;
                    self.stage_total = total;
                }
                WorkerMsg::Done(report) => {
                    let elapsed = report.elapsed_seconds;
                    self.log_lines.push(format!(
                        "✓ pipeline complete in {elapsed:.2} s — {} objects",
                        report.n_objects
                    ));
                    self.last_report = Some(*report);
                    self.is_running  = false;
                    self.central_tab = CentralTab::Objects;
                    self.stage_label = "Done".into();
                    self.stage_start = 1.0;
                    self.stage_end   = 1.0;
                    self.stage_done  = 0;
                    self.stage_total = 0;
                    keep_rx = false;
                }
                WorkerMsg::Failed(err) => {
                    self.log_lines.push(format!("✗ pipeline failed: {err}"));
                    self.last_error  = Some(err);
                    self.is_running  = false;
                    keep_rx = false;
                }
            }
        }
        if keep_rx {
            self.rx = Some(rx);
        }
        if self.is_running {
            // Keep the UI ticking while the worker runs.
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }

    // ----- panels ---------------------------------------------------------

    fn header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            // Render the bundled SVG logo via egui_extras' image loaders.
            let logo = egui::Image::from_bytes("bytes://cyclops.svg", CYCLOPS_LOGO_SVG)
                .max_height(56.0)
                .max_width(56.0)
                .fit_to_original_size(1.0);
            ui.add(logo);

            ui.vertical(|ui| {
                ui.heading(RichText::new(NAME).size(26.0).strong());
                ui.label(
                    RichText::new(format!("v{VERSION}  •  {FORMERLY}"))
                        .small()
                        .color(Color32::from_rgb(140, 140, 160)),
                );
            });

            ui.add_space(ui.available_width() - 320.0);

            ui.vertical(|ui| {
                ui.label(RichText::new("epifluorescence sizing & counting").italics());
                ui.label(
                    RichText::new("virus • bacteria • archaea • protist")
                        .small()
                        .color(Color32::from_rgb(100, 160, 200)),
                );
            });
        });
        ui.separator();
    }

    fn side_panel_inputs(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("Inputs").strong(), |ui| {
            path_picker_dir (ui, "DAPI directory",   &mut self.form.dapi_dir);
            path_picker_dir (ui, "FITC directory",   &mut self.form.fitc_dir);
            path_picker_file(ui, "Calibration image",&mut self.form.calibration);
            path_picker_dir (ui, "Output directory", &mut self.form.out_dir);
        });
    }

    fn side_panel_scale(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("Scale-bar / sphere").strong(), |ui| {
            number_row(ui, "Scale length (px)",  &mut self.form.scale_length_px, 1.0, 5_000.0);
            number_row(ui, "Scale metric (nm)",  &mut self.form.scale_metric_nm, 1.0, 1_000_000.0);
            number_row(ui, "Sphere size (nm)",   &mut self.form.sphere_size_nm,  1.0, 100_000.0);
            ui.label(
                RichText::new(format!(
                    "→ {:.3} nm/px",
                    self.form.scale_metric_nm / self.form.scale_length_px.max(1e-6)
                ))
                .small()
                .italics(),
            );
        });
    }

    fn side_panel_psf(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("PSF sweep").strong(), |ui| {
            ui.horizontal(|ui| {
                ui.label("Method");
                egui::ComboBox::from_id_source("psf_method")
                    .selected_text(match self.form.psf_method {
                        PsfMethod::GammaSinc => "γ-sinc (gam)",
                        PsfMethod::Gaussian  => "Gaussian (gau)",
                        PsfMethod::Hybrid    => "Hybrid (hyb)",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.form.psf_method, PsfMethod::GammaSinc, "γ-sinc");
                        ui.selectable_value(&mut self.form.psf_method, PsfMethod::Gaussian,  "Gaussian");
                        ui.selectable_value(&mut self.form.psf_method, PsfMethod::Hybrid,    "Hybrid");
                    });
            });

            ui.horizontal(|ui| {
                ui.label("f-size");
                ui.add(egui::DragValue::new(&mut self.form.f_size).clamp_range(0..=99).speed(1.0));
                ui.label(RichText::new("(0 = sweep)").small().italics());
            });
            ui.horizontal(|ui| {
                ui.label("τ");
                ui.add(egui::DragValue::new(&mut self.form.tau).clamp_range(0.0..=10.0).speed(0.01));
                ui.label("v");
                ui.add(egui::DragValue::new(&mut self.form.v).clamp_range(0.0..=10.0).speed(0.05));
                ui.label("σ");
                ui.add(egui::DragValue::new(&mut self.form.sigma).clamp_range(0.0..=10.0).speed(0.05));
            });
            ui.horizontal(|ui| {
                ui.label("MLE iters");
                ui.add(egui::DragValue::new(&mut self.form.n_mle_iter).clamp_range(1..=200).speed(1.0));
                ui.label("LR iters");
                ui.add(egui::DragValue::new(&mut self.form.n_lr_iter).clamp_range(1..=1000).speed(1.0));
            });
        });
    }

    fn side_panel_pairing(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("Pairing & filtering").strong(), |ui| {
            ui.horizontal(|ui| {
                ui.label("pad");
                ui.add(egui::DragValue::new(&mut self.form.pad).clamp_range(0..=100).speed(1.0));
                ui.label("dConstraint");
                ui.add(egui::DragValue::new(&mut self.form.d_constraint).clamp_range(0..=500).speed(1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Size metric");
                egui::ComboBox::from_id_source("size_metric")
                    .selected_text(["equivalent diameter", "average axes"][self.form.size_metric_idx])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.form.size_metric_idx, 0, "equivalent diameter");
                        ui.selectable_value(&mut self.form.size_metric_idx, 1, "average axes");
                    });
            });
            number_row(ui, "SM constraint (nm)", &mut self.form.sm_constraint, 0.0, 1_000_000.0);
            ui.checkbox(&mut self.form.gen_figs,      "Save every diagnostic figure");
            ui.checkbox(&mut self.form.keep_intermed, "Keep per-image PNGs");
        });
    }

    fn side_panel_domains(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("Organism domains").strong(), |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.form.use_virus,    "virus");
                ui.checkbox(&mut self.form.use_bacteria, "bacteria");
                ui.checkbox(&mut self.form.use_archaea,  "archaea");
                ui.checkbox(&mut self.form.use_protist,  "protist");
            });
            ui.checkbox(&mut self.form.gmm_refine, "GMM refinement (linfa)");
            ui.horizontal(|ui| {
                ui.label("ONNX model");
                ui.add(
                    egui::TextEdit::singleline(&mut self.form.onnx_model)
                        .hint_text("(optional, requires `onnx` feature)")
                        .desired_width(180.0),
                );
                if ui.button("…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("ONNX model", &["onnx"])
                        .pick_file()
                    {
                        self.form.onnx_model = p.display().to_string();
                    }
                }
            });
        });
    }

    fn side_panel_runtime(&mut self, ui: &mut Ui) {
        ui.collapsing(RichText::new("Runtime").strong(), |ui| {
            ui.horizontal(|ui| {
                ui.label("CPUs");
                ui.add(egui::DragValue::new(&mut self.form.cpus).clamp_range(-32..=256).speed(1.0));
                ui.label(RichText::new("(-1 = all, -2 = all but one)").small().italics());
            });
        });
    }

    fn side_panel_actions(&mut self, ui: &mut Ui) {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let run_btn = egui::Button::new(
                RichText::new("▶  Run Cyclops").size(15.0).strong(),
            )
            .fill(Color32::from_rgb(60, 110, 170))
            .stroke(Stroke::new(1.0, Color32::from_rgb(80, 140, 200)));

            if ui
                .add_enabled(!self.is_running, run_btn)
                .on_hover_text("Start the four-step pipeline")
                .clicked()
            {
                self.launch_pipeline();
            }

            if ui
                .add_enabled(self.is_running, egui::Button::new("⏹  Cancel"))
                .on_hover_text("Request cancellation (effective between stages)")
                .clicked()
            {
                self.cancel_flag.store(true, Ordering::SeqCst);
                self.log_lines.push("⏹ cancel requested".into());
            }
        });

        if self.is_running {
            ui.add_space(6.0);

            // Compute the global fraction: anchor at the current stage's
            // start, then interpolate across the stage span by the per-item
            // done/total ratio. Stages with no item count (total == 0) sit
            // at their start anchor and show an animated bar instead.
            let within = if self.stage_total > 0 {
                (self.stage_done as f32 / self.stage_total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let frac = (self.stage_start
                + (self.stage_end - self.stage_start) * within)
                .clamp(0.0, 1.0);

            // Bar text: stage label + per-item counter when known.
            let bar_text = if self.stage_total > 0 {
                format!("{}  ·  {}/{}", self.stage_label, self.stage_done, self.stage_total)
            } else {
                self.stage_label.clone()
            };

            let mut bar = egui::ProgressBar::new(frac)
                .desired_width(ui.available_width())
                .text(bar_text);
            // Indeterminate stages (no item count) get the animated stripe
            // so the user sees the app is alive during long single-shot work
            // like the PSF sweep.
            if self.stage_total == 0 {
                bar = bar.animate(true);
            }
            ui.add(bar);

            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0));
                if let Some(t) = self.started_at {
                    ui.label(
                        RichText::new(format!(
                            "{:.0}% overall  ·  {:.1}s elapsed",
                            frac * 100.0,
                            t.elapsed().as_secs_f64()
                        ))
                        .small()
                        .italics(),
                    );
                }
            });
        }

        if let Some(err) = &self.last_error {
            ui.add_space(6.0);
            ui.colored_label(Color32::from_rgb(220, 80, 80), format!("✗ {err}"));
        }
    }

    fn central_status(&self, ui: &mut Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.log_lines {
                    ui.label(RichText::new(line).monospace().size(12.5));
                }
            });
    }

    fn central_objects(&self, ui: &mut Ui) {
        let Some(report) = &self.last_report else {
            ui.label(RichText::new("No results yet — run the pipeline.").italics());
            return;
        };
        ui.label(
            RichText::new(format!(
                "{} objects   •   mean size {:.1} nm   •   CORR = {:.4}",
                report.n_objects, report.mean_size_nm, report.correction
            ))
            .strong(),
        );
        ui.add_space(6.0);

        TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .column(Column::auto().at_least(80.0))    // metric
            .column(Column::remainder().at_least(120.0)) // value
            .header(22.0, |mut h| {
                h.col(|ui| { ui.strong("metric"); });
                h.col(|ui| { ui.strong("value"); });
            })
            .body(|mut body| {
                let rows: &[(&str, String)] = &[
                    ("output dir",       report.output_dir.display().to_string()),
                    ("DAPI / FITC",      format!("{} / {}", report.n_dapi, report.n_fitc)),
                    ("VP min distance",  format!("{:.1} nm", report.min_distance_nm)),
                    ("PSF f-size",       report.psf_f_size.to_string()),
                    ("PSF τ",            format!("{:.4}", report.psf_tau)),
                    ("PSF v",            format!("{:.4}", report.psf_v)),
                    ("correction (CORR)",format!("{:.4}", report.correction)),
                    ("n objects",        report.n_objects.to_string()),
                    ("mean size (nm)",   format!("{:.1}", report.mean_size_nm)),
                    ("elapsed (s)",      format!("{:.2}", report.elapsed_seconds)),
                ];
                for (k, v) in rows {
                    body.row(20.0, |mut row| {
                        row.col(|ui| { ui.monospace(*k); });
                        row.col(|ui| { ui.monospace(v); });
                    });
                }
            });
    }

    fn central_bands(&self, ui: &mut Ui) {
        let Some(report) = &self.last_report else {
            ui.label(RichText::new("No results yet — run the pipeline.").italics());
            return;
        };
        let b = &report.size_bands;
        ui.label(RichText::new("Size bands (counts)").strong());
        ui.add_space(4.0);
        let entries = [
            ("< 100 nm (sub-viral)",        b.lt_100),
            ("100–220 nm (VLP)",             b.vlp_100_220),
            ("220–500 nm (small microbe)",   b.small_220_500),
            ("500–1 200 nm (bacterium)",     b.bact_500_1200),
            ("1.2–3 µm (large microbe)",     b.large_1200_3000),
            ("> 3 µm (protist-like)",        b.protist_gt_3000),
        ];
        let max = entries.iter().map(|(_, n)| *n).max().unwrap_or(1).max(1);
        for (label, n) in entries {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{n:>6}")).monospace());
                let frac = n as f32 / max as f32;
                let w = (ui.available_width() - 220.0).max(40.0) * frac;
                let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, Color32::from_rgb(80, 150, 200));
                ui.label(label);
            });
        }
    }

    fn central_domains(&self, ui: &mut Ui) {
        let Some(report) = &self.last_report else {
            ui.label(RichText::new("No results yet — run the pipeline.").italics());
            return;
        };
        ui.label(RichText::new("Domain classification").strong());
        ui.add_space(4.0);
        let mut entries: Vec<(&String, &u64)> = report.domain_counts.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1));
        let total: u64 = entries.iter().map(|(_, n)| **n).sum::<u64>().max(1);
        for (name, n) in entries {
            let frac = *n as f32 / total as f32;
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{n:>6}")).monospace());
                let w = (ui.available_width() - 200.0).max(40.0) * frac;
                let color = match name.as_str() {
                    "virus"    => Color32::from_rgb( 80, 150, 200),
                    "bacteria" => Color32::from_rgb(220, 140,  60),
                    "archaea"  => Color32::from_rgb(170,  90, 180),
                    "protist"  => Color32::from_rgb( 80, 180, 120),
                    _          => Color32::GRAY,
                };
                let (rect, _) = ui.allocate_exact_size(Vec2::new(w, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, color);
                ui.label(format!("{name}  ({:.1}%)", frac * 100.0));
            });
        }
    }
}

impl eframe::App for CyclopsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump_worker(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            self.header(ui);
        });

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(360.0)
            .min_width(320.0)
            .max_width(480.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.side_panel_inputs(ui);
                    self.side_panel_scale(ui);
                    self.side_panel_psf(ui);
                    self.side_panel_pairing(ui);
                    self.side_panel_domains(ui);
                    self.side_panel_runtime(ui);
                    self.side_panel_actions(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.central_tab, CentralTab::Status,  "Log");
                ui.selectable_value(&mut self.central_tab, CentralTab::Objects, "Summary");
                ui.selectable_value(&mut self.central_tab, CentralTab::Bands,   "Size bands");
                ui.selectable_value(&mut self.central_tab, CentralTab::Domains, "Domains");
            });
            ui.separator();
            match self.central_tab {
                CentralTab::Status  => self.central_status(ui),
                CentralTab::Objects => self.central_objects(ui),
                CentralTab::Bands   => self.central_bands(ui),
                CentralTab::Domains => self.central_domains(ui),
            }
        });
    }
}

// ---------------------------------------------------------------------------
//  small UI helpers
// ---------------------------------------------------------------------------

fn path_picker_dir(ui: &mut Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::TextEdit::singleline(value)
                .desired_width(ui.available_width() - 60.0),
        );
        if ui.button("…").on_hover_text("Browse for a folder").clicked() {
            if let Some(p) = rfd::FileDialog::new().pick_folder() {
                *value = p.display().to_string();
            }
        }
    });
}

fn path_picker_file(ui: &mut Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::TextEdit::singleline(value)
                .desired_width(ui.available_width() - 60.0),
        );
        if ui.button("…").on_hover_text("Browse for a file").clicked() {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter("Microscopy images", &["tif", "tiff", "png", "jpg", "jpeg"])
                .pick_file()
            {
                *value = p.display().to_string();
            }
        }
    });
}

fn number_row(ui: &mut Ui, label: &str, value: &mut f64, min: f64, max: f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).clamp_range(min..=max).speed(1.0));
    });
}
