//! Image I/O.
//!
//! Loads 8- or 16-bit TIFF/PNG/JPG as a normalised `f32`/`f64` ndarray
//! mirroring `skimage.io.imread(..., as_gray=True)` followed by
//! `img_as_float`. Floats are in `[0.0, 1.0]`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use image::{io::Reader as ImageReader, GenericImageView};
use ndarray::Array2;
use walkdir::WalkDir;

use crate::error::{CyclopsError, Result};

/// Supported image extensions.
const SUPPORTED_EXT: &[&str] = &["tif", "tiff", "png", "jpg", "jpeg"];

/// A single grayscale image, normalised to `[0.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub data: Array2<f64>,
    pub path: PathBuf,
}

impl GrayImage {
    /// File stem with extension removed — equivalent to the Python
    /// `name[:-len(fExtension)]` slicing in the original.
    pub fn short_name(&self) -> String {
        self.path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string())
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn mean(&self) -> f64 {
        let n = self.data.len();
        if n == 0 { 0.0 } else { self.data.sum() / n as f64 }
    }
}

/// Read a single image file as a normalised grayscale `f64` matrix.
///
/// 8-bit pixels are scaled by `1/255`, 16-bit by `1/65535`, floats are
/// passed through clipped to `[0,1]`.
pub fn load_gray<P: AsRef<Path>>(path: P) -> Result<GrayImage> {
    let p = path.as_ref();
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "tif" | "tiff" => load_tiff(p),
        "png" | "jpg" | "jpeg" => load_via_image_crate(p),
        other => Err(CyclopsError::ImageDecode {
            path:    p.to_path_buf(),
            message: format!("unsupported extension `.{other}`"),
        }),
    }
}

fn load_via_image_crate(p: &Path) -> Result<GrayImage> {
    let img = ImageReader::open(p)
        .map_err(|e| CyclopsError::Io {
            path:   p.to_path_buf(),
            source: e,
        })?
        .with_guessed_format()
        .map_err(|e| CyclopsError::Io {
            path:   p.to_path_buf(),
            source: e,
        })?
        .decode()
        .map_err(|e| CyclopsError::ImageDecode {
            path:    p.to_path_buf(),
            message: e.to_string(),
        })?;

    let (w, h) = img.dimensions();
    let gray = img.into_luma16();
    let mut out = Array2::<f64>::zeros((h as usize, w as usize));
    for (y, row) in out.outer_iter_mut().enumerate() {
        let row_slice = &gray.as_raw()
            [(y * w as usize)..((y + 1) * w as usize)];
        for (x, col) in row.into_iter().enumerate() {
            *col = row_slice[x] as f64 / 65_535.0;
        }
    }
    Ok(GrayImage { data: out, path: p.to_path_buf() })
}

/// TIFF loader: keeps full 16-bit dynamic range. Multi-channel TIFFs
/// are reduced to grayscale via luminosity weighting; multi-sample
/// floats are clipped to `[0,1]`.
fn load_tiff(p: &Path) -> Result<GrayImage> {
    let file = File::open(p).map_err(|e| CyclopsError::Io {
        path:   p.to_path_buf(),
        source: e,
    })?;
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file))
        .map_err(|e| CyclopsError::ImageDecode {
            path:    p.to_path_buf(),
            message: e.to_string(),
        })?;

    let (w, h) = decoder.dimensions().map_err(|e| CyclopsError::ImageDecode {
        path:    p.to_path_buf(),
        message: e.to_string(),
    })?;

    let result = decoder.read_image().map_err(|e| CyclopsError::ImageDecode {
        path:    p.to_path_buf(),
        message: e.to_string(),
    })?;

    use tiff::decoder::DecodingResult;
    let flat: Vec<f64> = match result {
        DecodingResult::U8(v)    => v.into_iter().map(|x| x as f64 / 255.0).collect(),
        DecodingResult::U16(v)   => v.into_iter().map(|x| x as f64 / 65_535.0).collect(),
        DecodingResult::U32(v)   => v.into_iter().map(|x| x as f64 / u32::MAX as f64).collect(),
        DecodingResult::U64(v)   => v.into_iter().map(|x| x as f64 / u64::MAX as f64).collect(),
        DecodingResult::F32(v)   => v.into_iter().map(|x| x as f64).collect(),
        DecodingResult::F64(v)   => v,
        DecodingResult::I8(v)    => v.into_iter().map(|x| (x as f64 + 128.0) / 255.0).collect(),
        DecodingResult::I16(v)   => v.into_iter().map(|x| (x as f64 + 32_768.0) / 65_535.0).collect(),
        DecodingResult::I32(v)   => v.into_iter().map(|x| (x as f64 - i32::MIN as f64) / (u32::MAX as f64)).collect(),
        DecodingResult::I64(v)   => v.into_iter().map(|x| x as f64 / i64::MAX as f64).collect(),
    };

    let expected = (w as usize) * (h as usize);
    let arr = if flat.len() == expected {
        Array2::from_shape_vec((h as usize, w as usize), flat)
            .map_err(|e| CyclopsError::ImageDecode {
                path:    p.to_path_buf(),
                message: format!("reshape: {e}"),
            })?
    } else if flat.len() % expected == 0 {
        // Multi-sample TIFF (e.g. RGB). Average channels into grayscale.
        let ch = flat.len() / expected;
        let mut buf = Vec::with_capacity(expected);
        for px in 0..expected {
            let mut acc = 0.0_f64;
            for c in 0..ch {
                acc += flat[px * ch + c];
            }
            buf.push(acc / ch as f64);
        }
        Array2::from_shape_vec((h as usize, w as usize), buf)
            .map_err(|e| CyclopsError::ImageDecode {
                path:    p.to_path_buf(),
                message: format!("reshape (multi-channel): {e}"),
            })?
    } else {
        return Err(CyclopsError::ImageDecode {
            path:    p.to_path_buf(),
            message: format!(
                "decoded {} samples for a {}×{} image",
                flat.len(),
                w,
                h
            ),
        });
    };

    let arr = arr.mapv(|x| x.clamp(0.0, 1.0));
    Ok(GrayImage { data: arr, path: p.to_path_buf() })
}

/// List image files inside a directory (non-recursive). Sorts by path
/// for reproducibility.
pub fn list_image_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    if !dir.is_dir() {
        return Err(CyclopsError::EmptyDirectory(dir.to_path_buf()));
    }
    let mut out: Vec<PathBuf> = WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| SUPPORTED_EXT.contains(&x.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(CyclopsError::EmptyDirectory(dir.to_path_buf()));
    }
    Ok(out)
}

/// Save a normalised `[0,1]` float image as 8-bit PNG.
pub fn save_png<P: AsRef<Path>>(img: &Array2<f64>, path: P) -> Result<()> {
    let (h, w) = img.dim();
    let mut buf = Vec::<u8>::with_capacity(h * w);
    for &v in img.iter() {
        buf.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    image::save_buffer(
        path.as_ref(),
        &buf,
        w as u32,
        h as u32,
        image::ColorType::L8,
    )
    .map_err(|e| CyclopsError::ImageDecode {
        path:    path.as_ref().to_path_buf(),
        message: e.to_string(),
    })?;
    Ok(())
}
