//! Pipeline progress reporting.
//!
//! `run_pipeline` takes a `&dyn Progress` so any front-end can observe
//! stage transitions and fractional progress without the core crate
//! knowing anything about GUIs, terminals, or channels.
//!
//! Three implementations ship here:
//!   * [`NoProgress`]      — silent (the default, e.g. library/tests)
//!   * [`FnProgress`]      — forwards every event to a user closure
//!     (the GUI wraps its mpsc sender in one of these)
//!   * [`LoggingProgress`] — logs each stage via `tracing::info!`
//!
//! The CLI builds an indicatif bar on top of `FnProgress`.

/// The coarse pipeline stages, in execution order. `fraction()` gives a
/// rough completed-fraction anchor so a front-end can render a global bar
/// even before per-item ticks arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Loading,
    Pairing,
    PsfSweep,
    Calibration,
    Quantification,
    Classification,
    Writing,
    Done,
}

impl Stage {
    /// A human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Loading        => "Loading images",
            Stage::Pairing        => "Step 1 · pair detection",
            Stage::PsfSweep       => "Step 2 · PSF sweep",
            Stage::Calibration    => "Step 3 · calibration (Richardson–Lucy)",
            Stage::Quantification => "Step 4 · quantification",
            Stage::Classification => "Classification",
            Stage::Writing        => "Writing outputs",
            Stage::Done           => "Done",
        }
    }

    /// Global fraction (0.0–1.0) at which this stage *begins*. Anchors a
    /// coarse overall progress bar; within a stage the front-end can
    /// interpolate using the per-item `(done, total)` from [`Progress::tick`].
    pub fn start_fraction(self) -> f32 {
        match self {
            Stage::Loading        => 0.00,
            Stage::Pairing        => 0.05,
            Stage::PsfSweep       => 0.10,
            Stage::Calibration    => 0.40,
            Stage::Quantification => 0.70,
            Stage::Classification => 0.95,
            Stage::Writing        => 0.98,
            Stage::Done           => 1.00,
        }
    }

    /// Global fraction at which this stage *ends* (== the next stage's start).
    pub fn end_fraction(self) -> f32 {
        match self {
            Stage::Loading        => Stage::Pairing.start_fraction(),
            Stage::Pairing        => Stage::PsfSweep.start_fraction(),
            Stage::PsfSweep       => Stage::Calibration.start_fraction(),
            Stage::Calibration    => Stage::Quantification.start_fraction(),
            Stage::Quantification => Stage::Classification.start_fraction(),
            Stage::Classification => Stage::Writing.start_fraction(),
            Stage::Writing        => Stage::Done.start_fraction(),
            Stage::Done           => 1.00,
        }
    }
}

/// Observer for pipeline progress. All methods have default no-op bodies
/// so implementors override only what they care about. Implementations
/// must be `Sync` because the parallel stages may tick from worker
/// threads.
pub trait Progress: Sync {
    /// A new stage has begun. `total` is the number of `tick`s expected
    /// for this stage when known (e.g. image count), else 0.
    fn stage(&self, _stage: Stage, _total: usize) {}

    /// One unit of work within the current stage finished. `done` is the
    /// running count, `total` mirrors the value from [`Progress::stage`].
    fn tick(&self, _done: usize, _total: usize) {}

    /// A free-form status line (mirrors what used to be a log line).
    fn message(&self, _msg: &str) {}
}

/// The silent default.
pub struct NoProgress;
impl Progress for NoProgress {}

/// Forwards every event to a single closure as a [`ProgressEvent`].
/// `Send + Sync` closure so it can be shared across rayon threads; the
/// GUI's closure just pushes onto an `mpsc::Sender` (which is `Send`),
/// wrapped in a `Mutex` for `Sync`.
pub struct FnProgress<F: Fn(ProgressEvent) + Sync> {
    f: F,
}

impl<F: Fn(ProgressEvent) + Sync> FnProgress<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: Fn(ProgressEvent) + Sync> Progress for FnProgress<F> {
    fn stage(&self, stage: Stage, total: usize) {
        (self.f)(ProgressEvent::Stage { stage, total });
    }
    fn tick(&self, done: usize, total: usize) {
        (self.f)(ProgressEvent::Tick { done, total });
    }
    fn message(&self, msg: &str) {
        (self.f)(ProgressEvent::Message(msg.to_string()));
    }
}

/// The event a [`FnProgress`] closure receives.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Stage { stage: Stage, total: usize },
    Tick  { done: usize, total: usize },
    Message(String),
}

impl ProgressEvent {
    /// Best-effort global fraction (0.0–1.0) for this event, for driving a
    /// single overall bar. `Message` events return `None`.
    pub fn global_fraction(&self) -> Option<f32> {
        match self {
            ProgressEvent::Stage { stage, .. } => Some(stage.start_fraction()),
            ProgressEvent::Tick { done, total } => {
                // Caller doesn't know the stage here; the GUI tracks the
                // current stage separately and interpolates. This helper
                // is only a coarse fallback: fraction of the current stage.
                if *total > 0 {
                    Some((*done as f32 / *total as f32).clamp(0.0, 1.0))
                } else {
                    None
                }
            }
            ProgressEvent::Message(_) => None,
        }
    }
}

/// Logs each stage through `tracing`.
pub struct LoggingProgress;
impl Progress for LoggingProgress {
    fn stage(&self, stage: Stage, total: usize) {
        if total > 0 {
            tracing::info!("{} ({} items)", stage.label(), total);
        } else {
            tracing::info!("{}", stage.label());
        }
    }
    fn message(&self, msg: &str) {
        tracing::info!("{msg}");
    }
}
