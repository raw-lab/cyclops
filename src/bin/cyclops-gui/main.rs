//! Cyclops desktop GUI — entry point.
//!
//! A thin shim that creates the eframe window and launches the
//! `CyclopsApp` defined in `app.rs`. The Cyclops logo is rendered
//! *inside* the app surface from a bundled SVG, so the binary does not
//! depend on any external PNG assets at compile time.

use eframe::egui;
use tracing_subscriber::EnvFilter;

mod app;

use app::CyclopsApp;
use cyclops_core::{NAME, VERSION};

fn main() -> eframe::Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,cyclops_core=info,cyclops_gui=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    let viewport = egui::ViewportBuilder::default()
        .with_title(format!("{NAME} v{VERSION}"))
        .with_inner_size([1280.0, 820.0])
        .with_min_inner_size([960.0, 640.0]);

    let native_options = eframe::NativeOptions {
        viewport,
        vsync: true,
        ..Default::default()
    };

    eframe::run_native(
        &format!("{NAME} v{VERSION}"),
        native_options,
        Box::new(|cc| Box::new(CyclopsApp::new(cc))),
    )
}
