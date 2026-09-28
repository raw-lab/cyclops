//! Plotting helpers.
//!
//! Uses [`plotters`] with a deliberately seaborn-flavoured default
//! style: white background, light grid, viridis colour ramp, sans-serif
//! axes. All plots are rendered as PNG (`bitmap_backend`) or SVG
//! (`svg_backend`) so the GUI can embed either format.

use std::path::Path;

use plotters::prelude::*;
// Pull a few specific named colors from full_palette without re-globbing
// (full_palette also defines WHITE/BLACK which collide with prelude::*).
use plotters::style::full_palette::GREY_300;

use crate::error::{CyclopsError, Result};

/// Seaborn-like palette (10 colours from the deep palette).
pub const DEEP: [RGBColor; 10] = [
    RGBColor(76, 114, 176),
    RGBColor(221, 132, 82),
    RGBColor(85, 168, 104),
    RGBColor(196, 78, 82),
    RGBColor(129, 114, 178),
    RGBColor(147, 120, 96),
    RGBColor(218, 139, 195),
    RGBColor(140, 140, 140),
    RGBColor(204, 185, 116),
    RGBColor(100, 181, 205),
];

/// Ensure the parent directory of an output path exists before we try to
/// create a file there. Defensive: the pipeline already creates each Step
/// directory, but a plot function called standalone (tests, library users)
/// must not fail just because the folder isn't there yet.
fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| CyclopsError::Io {
                path:   parent.to_path_buf(),
                source: e,
            })?;
        }
    }
    Ok(())
}

fn map_err(err: impl std::fmt::Display) -> CyclopsError {
    CyclopsError::Plot(err.to_string())
}

/// Histogram of `values` with `nbins` bins, saved as PNG.
pub fn histogram<P: AsRef<Path>>(
    values: &[f64],
    nbins:  usize,
    title:  &str,
    xlabel: &str,
    ylabel: &str,
    path:   P,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in values {
        if v.is_finite() {
            if v < min { min = v; }
            if v > max { max = v; }
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return Ok(());
    }
    if (max - min).abs() < 1e-12 {
        max = min + 1.0;
    }
    let bin_width = (max - min) / nbins as f64;
    let mut bins = vec![0u64; nbins];
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        let mut idx = ((v - min) / bin_width).floor() as isize;
        if idx >= nbins as isize { idx = nbins as isize - 1; }
        if idx < 0 { idx = 0; }
        bins[idx as usize] += 1;
    }
    let y_max = (*bins.iter().max().unwrap_or(&1) as f64) * 1.1;

    ensure_parent_dir(path.as_ref())?;
    let root = BitMapBackend::new(path.as_ref(), (900, 540)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0..y_max)
        .map_err(map_err)?;

    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc(ylabel)
        .light_line_style(GREY_300)
        .axis_desc_style(("sans-serif", 14).into_font())
        .draw()
        .map_err(map_err)?;

    // Bars coloured along the viridis ramp by frequency.
    let palette = ViridisRGB {};
    chart
        .draw_series(bins.iter().enumerate().map(|(i, &c)| {
            let x0 = min + (i as f64) * bin_width;
            let x1 = x0 + bin_width;
            let frac = (c as f64) / y_max.max(1.0);
            let color = palette.get_color(frac);
            Rectangle::new([(x0, 0.0), (x1, c as f64)], color.filled())
        }))
        .map_err(map_err)?;

    // Vertical line at the mean.
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(mean, 0.0), (mean, y_max)],
            BLACK.stroke_width(2),
        )))
        .map_err(map_err)?
        .label(format!("μ = {:.2}", mean))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(map_err)?;

    root.present().map_err(map_err)?;
    Ok(())
}

/// Scatter of `(x, y)` points coloured by `y` (viridis ramp).
pub fn scatter<P: AsRef<Path>>(
    xy:     &[(f64, f64)],
    title:  &str,
    xlabel: &str,
    ylabel: &str,
    path:   P,
) -> Result<()> {
    if xy.is_empty() { return Ok(()); }
    let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in xy {
        if x.is_finite() { if x < x_min { x_min = x; } if x > x_max { x_max = x; } }
        if y.is_finite() { if y < y_min { y_min = y; } if y > y_max { y_max = y; } }
    }
    if !x_max.is_finite() || !y_max.is_finite() { return Ok(()); }
    if (x_max - x_min).abs() < 1e-12 { x_max = x_min + 1.0; }
    if (y_max - y_min).abs() < 1e-12 { y_max = y_min + 1.0; }
    let pad_x = (x_max - x_min) * 0.05;
    let pad_y = (y_max - y_min) * 0.05;

    ensure_parent_dir(path.as_ref())?;
    let root = BitMapBackend::new(path.as_ref(), (900, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min - pad_x..x_max + pad_x, y_min - pad_y..y_max + pad_y)
        .map_err(map_err)?;
    chart
        .configure_mesh()
        .x_desc(xlabel)
        .y_desc(ylabel)
        .light_line_style(GREY_300)
        .draw()
        .map_err(map_err)?;

    let palette = ViridisRGB {};
    chart
        .draw_series(xy.iter().map(|&(x, y)| {
            let frac = ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
            let color = palette.get_color(frac);
            Circle::new((x, y), 4, color.filled())
        }))
        .map_err(map_err)?;
    root.present().map_err(map_err)?;
    Ok(())
}

/// Render a 2-D float image (e.g. PSF or deconvolved patch) as a
/// pseudo-coloured PNG.
pub fn heatmap<P: AsRef<Path>>(
    img:    &ndarray::Array2<f64>,
    title:  &str,
    path:   P,
) -> Result<()> {
    let (h, w) = img.dim();
    if h == 0 || w == 0 {
        return Ok(());
    }
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in img.iter() {
        if v.is_finite() {
            if v < min { min = v; }
            if v > max { max = v; }
        }
    }
    if (max - min).abs() < 1e-12 { max = min + 1.0; }
    ensure_parent_dir(path.as_ref())?;
    let root = BitMapBackend::new(path.as_ref(), (700, 700)).into_drawing_area();
    root.fill(&WHITE).map_err(map_err)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(0..w, 0..h)
        .map_err(map_err)?;
    chart.configure_mesh().disable_mesh().draw().map_err(map_err)?;
    let palette = ViridisRGB {};
    chart
        .draw_series((0..h).flat_map(|y| {
            (0..w).map(move |x| {
                let v = img[(y, x)];
                let frac = ((v - min) / (max - min)).clamp(0.0, 1.0);
                let color = palette.get_color(frac);
                Rectangle::new([(x, h - 1 - y), (x + 1, h - y)], color.filled())
            })
        }))
        .map_err(map_err)?;
    root.present().map_err(map_err)?;
    Ok(())
}

/// Viridis colour ramp shim — `plotters` ships one, but pinning the
/// implementation here lets us swap palettes without touching every
/// callsite.
#[derive(Clone, Copy)]
struct ViridisRGB;
impl ViridisRGB {
    fn get_color(&self, t: f64) -> RGBColor {
        // Five-stop linear interpolation over the canonical viridis
        // anchors (RGB sampled from matplotlib).
        const STOPS: [(f64, [u8; 3]); 5] = [
            (0.00, [68,  1,   84]),
            (0.25, [59,  82,  139]),
            (0.50, [33,  144, 140]),
            (0.75, [94,  201, 97]),
            (1.00, [253, 231, 36]),
        ];
        let t = t.clamp(0.0, 1.0);
        for w in STOPS.windows(2) {
            let (t0, c0) = w[0];
            let (t1, c1) = w[1];
            if t <= t1 {
                let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
                let r = (c0[0] as f64 + f * (c1[0] as f64 - c0[0] as f64)) as u8;
                let g = (c0[1] as f64 + f * (c1[1] as f64 - c0[1] as f64)) as u8;
                let b = (c0[2] as f64 + f * (c1[2] as f64 - c0[2] as f64)) as u8;
                return RGBColor(r, g, b);
            }
        }
        let last = STOPS.last().unwrap().1;
        RGBColor(last[0], last[1], last[2])
    }
}

// Keep the namespace imports above happy in case the user disables
// some plotters palettes.
#[allow(dead_code)]
fn _palette_marker() -> RGBColor {
    GREY_300
}
