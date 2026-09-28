//! Connected-component labelling and region property extraction.
//!
//! Equivalent to `skimage.measure.label` (8-connectivity by default
//! for 2-D) followed by `skimage.measure.regionprops(.., intensity_image=...)`.
//! We implement only the property subset Cyclops needs.

use ndarray::Array2;

/// A connected component plus its extracted properties.
#[derive(Debug, Clone)]
pub struct RegionProp {
    pub label:                  u32,
    pub area:                   u64,
    /// (row, col) centroid in pixels.
    pub centroid:               (f64, f64),
    /// (min_row, min_col, max_row, max_col), exclusive on max.
    pub bbox:                   (usize, usize, usize, usize),
    /// All `(row, col)` pixels in this region.
    pub coords:                 Vec<(usize, usize)>,
    pub axis_major_length:      f64,
    pub axis_minor_length:      f64,
    pub equivalent_diameter:    f64,
    pub eccentricity:           f64,
    pub intensity_mean:         f64,
    pub intensity_max:          f64,
    pub intensity_min:          f64,
}

/// 8-connected labelling using two-pass union-find. Returns a label
/// image (0 = background) and the number of labels.
pub fn label(mask: &Array2<bool>) -> (Array2<u32>, u32) {
    let (h, w) = mask.dim();
    let mut labels = Array2::<u32>::zeros((h, w));
    // parent[i] = the union-find parent of label i. Index 0 is the
    // background sentinel; label k (k≥1) is initially its own root, so
    // when we push for the first new label we end up with parent[1] = 1.
    let mut parent: Vec<u32> = vec![0];

    let mut next_label: u32 = 1;

    fn find(p: &mut [u32], x: u32) -> u32 {
        let mut r = x;
        while p[r as usize] != r {
            r = p[r as usize];
        }
        // path compression
        let mut cur = x;
        while p[cur as usize] != r {
            let nxt = p[cur as usize];
            p[cur as usize] = r;
            cur = nxt;
        }
        r
    }
    fn union(p: &mut Vec<u32>, a: u32, b: u32) {
        let ra = find(p, a);
        let rb = find(p, b);
        if ra != rb {
            // attach the larger to the smaller for determinism
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            p[hi as usize] = lo;
        }
    }

    // First pass.
    for y in 0..h {
        for x in 0..w {
            if !mask[(y, x)] {
                continue;
            }
            // 4 already-visited neighbours (8-connectivity).
            let mut neigh: Vec<u32> = Vec::with_capacity(4);
            if y > 0 && mask[(y - 1, x)] {
                neigh.push(labels[(y - 1, x)]);
            }
            if x > 0 && mask[(y, x - 1)] {
                neigh.push(labels[(y, x - 1)]);
            }
            if y > 0 && x > 0 && mask[(y - 1, x - 1)] {
                neigh.push(labels[(y - 1, x - 1)]);
            }
            if y > 0 && x + 1 < w && mask[(y - 1, x + 1)] {
                neigh.push(labels[(y - 1, x + 1)]);
            }
            if neigh.is_empty() {
                labels[(y, x)] = next_label;
                parent.push(next_label);
                next_label += 1;
            } else {
                let lo = *neigh.iter().min().unwrap();
                labels[(y, x)] = lo;
                for &n in &neigh {
                    if n != lo {
                        union(&mut parent, n, lo);
                    }
                }
            }
        }
    }

    // Second pass: replace with root labels, compacting numbering.
    let mut compact: Vec<u32> = vec![0; parent.len()];
    let mut current: u32 = 0;
    for v in 1..parent.len() as u32 {
        let r = find(&mut parent, v);
        if compact[r as usize] == 0 {
            current += 1;
            compact[r as usize] = current;
        }
        compact[v as usize] = compact[r as usize];
    }
    for y in 0..h {
        for x in 0..w {
            let v = labels[(y, x)];
            if v != 0 {
                labels[(y, x)] = compact[v as usize];
            }
        }
    }

    (labels, current)
}

/// Extract per-region properties given a label image and the original
/// intensity image used to derive it.
pub fn region_props(
    labels:    &Array2<u32>,
    intensity: &Array2<f64>,
    n_labels:  u32,
) -> Vec<RegionProp> {
    if n_labels == 0 {
        return Vec::new();
    }
    let (h, w) = labels.dim();

    // Accumulators.
    let nl = n_labels as usize;
    let mut area     = vec![0u64; nl + 1];
    let mut sum_y    = vec![0.0_f64; nl + 1];
    let mut sum_x    = vec![0.0_f64; nl + 1];
    let mut sum_yy   = vec![0.0_f64; nl + 1];
    let mut sum_xx   = vec![0.0_f64; nl + 1];
    let mut sum_xy   = vec![0.0_f64; nl + 1];
    let mut min_r    = vec![usize::MAX; nl + 1];
    let mut max_r    = vec![0usize; nl + 1];
    let mut min_c    = vec![usize::MAX; nl + 1];
    let mut max_c    = vec![0usize; nl + 1];
    let mut sum_i    = vec![0.0_f64; nl + 1];
    let mut max_i    = vec![f64::NEG_INFINITY; nl + 1];
    let mut min_i    = vec![f64::INFINITY; nl + 1];
    let mut coords: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nl + 1];

    for y in 0..h {
        for x in 0..w {
            let l = labels[(y, x)] as usize;
            if l == 0 {
                continue;
            }
            let yf = y as f64;
            let xf = x as f64;
            area[l]   += 1;
            sum_y[l]  += yf;
            sum_x[l]  += xf;
            sum_yy[l] += yf * yf;
            sum_xx[l] += xf * xf;
            sum_xy[l] += xf * yf;
            min_r[l]   = min_r[l].min(y);
            max_r[l]   = max_r[l].max(y);
            min_c[l]   = min_c[l].min(x);
            max_c[l]   = max_c[l].max(x);
            let i_val = intensity[(y, x)];
            sum_i[l] += i_val;
            if i_val > max_i[l] { max_i[l] = i_val; }
            if i_val < min_i[l] { min_i[l] = i_val; }
            coords[l].push((y, x));
        }
    }

    let mut out = Vec::with_capacity(nl);
    for l in 1..=nl {
        let a = area[l] as f64;
        if a == 0.0 {
            continue;
        }
        let cy = sum_y[l] / a;
        let cx = sum_x[l] / a;
        // Central moments → axis lengths (sklearn / skimage convention).
        let mu_yy = sum_yy[l] / a - cy * cy;
        let mu_xx = sum_xx[l] / a - cx * cx;
        let mu_xy = sum_xy[l] / a - cx * cy;
        // Eigenvalues of the 2×2 covariance matrix.
        let trace = mu_yy + mu_xx;
        let det   = mu_yy * mu_xx - mu_xy * mu_xy;
        let disc  = (trace * trace * 0.25 - det).max(0.0).sqrt();
        let lam1 = (trace * 0.5 + disc).max(0.0);
        let lam2 = (trace * 0.5 - disc).max(0.0);
        // Axis length convention: 4 √λ (matches skimage).
        let axis_major = 4.0 * lam1.sqrt();
        let axis_minor = 4.0 * lam2.sqrt();
        let eccentricity = if lam1 > 0.0 {
            (1.0 - lam2 / lam1).max(0.0).sqrt()
        } else {
            0.0
        };
        let equivalent_diameter = (4.0 * a / std::f64::consts::PI).sqrt();
        let intensity_mean = sum_i[l] / a;

        out.push(RegionProp {
            label:               l as u32,
            area:                area[l],
            centroid:            (cy, cx),
            bbox:                (min_r[l], min_c[l], max_r[l] + 1, max_c[l] + 1),
            coords:              std::mem::take(&mut coords[l]),
            axis_major_length:   axis_major,
            axis_minor_length:   axis_minor,
            equivalent_diameter,
            eccentricity,
            intensity_mean,
            intensity_max:       max_i[l].max(0.0),
            intensity_min:       min_i[l].max(0.0),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_blobs_get_two_labels() {
        let mask = array![
            [true,  true,  false, false, false],
            [true,  true,  false, false, false],
            [false, false, false, true,  true ],
            [false, false, false, true,  true ],
        ];
        let (lbl, n) = label(&mask);
        assert_eq!(n, 2, "expected 2 components, got {n}");
        let unique: std::collections::BTreeSet<u32> = lbl.iter().copied().collect();
        // {0, 1, 2}
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn region_props_for_square() {
        let mut mask = Array2::<bool>::default((10, 10));
        for y in 2..6 {
            for x in 3..7 {
                mask[(y, x)] = true;
            }
        }
        let (lbl, n) = label(&mask);
        assert_eq!(n, 1);
        let intensity = mask.mapv(|b| if b { 0.8 } else { 0.0 });
        let props = region_props(&lbl, &intensity, n);
        assert_eq!(props.len(), 1);
        let p = &props[0];
        assert_eq!(p.area, 16);
        assert!((p.intensity_mean - 0.8).abs() < 1e-12);
        // 4×4 square -> equivalent diameter = sqrt(64/π) ≈ 4.51
        assert!((p.equivalent_diameter - (64.0_f64 / std::f64::consts::PI).sqrt()).abs() < 1e-9);
    }
}
