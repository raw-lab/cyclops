#!/usr/bin/env python3
"""
Generate publication-quality figures from Cyclops pipeline output.

Reads the per-object table (cyclops_objects.tsv / .parquet) and the run
report (cyclops_report.json) and produces a multi-panel figure summarising
the size distribution, domain classification, per-image counts, and the
size-vs-shape relationship.

Usage:
    python3 make_figures.py <cyclops_output_dir> [<figure_output_dir>]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ---- seaborn / matplotlib house style -------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
})

# Domain → colour (matches the Rust GUI palette)
DOMAIN_COLORS = {
    "virus":    "#4c72b0",
    "bacteria": "#dd8452",
    "archaea":  "#aa5ab4",
    "protist":  "#50b478",
}
VIRIDIS = plt.get_cmap("viridis")


def load_results(out_dir: Path):
    """Load the object table (prefer parquet) and the JSON report."""
    tsv = out_dir / "Step-4_genMasks" / "cyclops_objects.tsv"
    pq  = out_dir / "Step-4_genMasks" / "cyclops_objects.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except Exception:
            df = pd.read_csv(tsv, sep="\t")
    elif tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
    else:
        raise FileNotFoundError(f"No object table found under {out_dir}")

    report = {}
    rp = out_dir / "cyclops_report.json"
    if rp.exists():
        report = json.loads(rp.read_text())
    return df, report


def make_summary_figure(df: pd.DataFrame, report: dict, fig_path: Path):
    """A 2×3 multi-panel summary figure."""
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.28)

    n_obj = len(df)
    mean_sz = df["size_nm"].mean() if n_obj else 0.0

    fig.suptitle(
        f"Cyclops — G_Phage GSL dataset  |  {n_obj} particles  |  "
        f"mean size {mean_sz:.1f} nm  |  CORR {report.get('correction', float('nan')):.3f}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Panel A: size distribution histogram + KDE ------------------------
    axA = fig.add_subplot(gs[0, 0])
    if n_obj:
        sizes = df["size_nm"].to_numpy()
        n, bins, patches = axA.hist(sizes, bins=40, edgecolor="white", linewidth=0.4)
        # colour bars by viridis along the size axis
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        norm = (bin_centers - bin_centers.min()) / (np.ptp(bin_centers) + 1e-9)
        for c, p in zip(norm, patches):
            p.set_facecolor(VIRIDIS(c))
        axA.axvline(sizes.mean(), color="crimson", ls="--", lw=1.5,
                    label=f"mean {sizes.mean():.0f} nm")
        axA.axvline(np.median(sizes), color="black", ls=":", lw=1.5,
                    label=f"median {np.median(sizes):.0f} nm")
        axA.legend()
    axA.set_title("A · Particle size distribution")
    axA.set_xlabel("equivalent diameter (nm)")
    axA.set_ylabel("count")

    # --- Panel B: domain classification bar --------------------------------
    axB = fig.add_subplot(gs[0, 1])
    if n_obj and "domain" in df.columns:
        counts = df["domain"].value_counts()
        colors = [DOMAIN_COLORS.get(d, "#888888") for d in counts.index]
        bars = axB.bar(counts.index, counts.values, color=colors, edgecolor="white")
        for b, v in zip(bars, counts.values):
            axB.text(b.get_x() + b.get_width()/2, v, f"{v}\n{100*v/n_obj:.0f}%",
                     ha="center", va="bottom", fontsize=9)
        axB.set_ylim(0, counts.values.max() * 1.18)
    axB.set_title("B · Domain classification")
    axB.set_xlabel("domain")
    axB.set_ylabel("count")

    # --- Panel C: per-image particle count ---------------------------------
    axC = fig.add_subplot(gs[0, 2])
    if n_obj and "file_name" in df.columns:
        per_img = df.groupby("file_name").size().sort_values(ascending=False)
        y = np.arange(len(per_img))
        axC.barh(y, per_img.values, color=VIRIDIS(np.linspace(0.15, 0.85, len(per_img))))
        axC.set_yticks(y)
        # shorten file names
        labels = [Path(f).stem.replace("GSL_+_blue_beads_", "img ").replace("_(fitc)", "")
                  for f in per_img.index]
        axC.set_yticklabels(labels, fontsize=6)
        axC.invert_yaxis()
        axC.axvline(per_img.mean(), color="crimson", ls="--", lw=1.2,
                    label=f"mean {per_img.mean():.0f}/img")
        axC.legend()
    axC.set_title("C · Particles per image")
    axC.set_xlabel("count")

    # --- Panel D: size vs eccentricity scatter -----------------------------
    axD = fig.add_subplot(gs[1, 0])
    if n_obj and "eccentricity" in df.columns:
        sc = axD.scatter(df["size_nm"], df["eccentricity"],
                         c=df["size_nm"], cmap="viridis", s=14,
                         alpha=0.6, edgecolors="none")
        plt.colorbar(sc, ax=axD, label="size (nm)")
    axD.set_title("D · Size vs shape")
    axD.set_xlabel("equivalent diameter (nm)")
    axD.set_ylabel("eccentricity")

    # --- Panel E: size bands (from report) ---------------------------------
    axE = fig.add_subplot(gs[1, 1])
    bands = report.get("size_bands") or {}
    if bands:
        band_labels = ["<100\nsub-viral", "100–220\nVLP", "220–500\nsm microbe",
                       "500–1200\nbacterium", "1.2–3µm\nlg microbe", ">3µm\nprotist"]
        band_keys = ["lt_100", "vlp_100_220", "small_220_500",
                     "bact_500_1200", "large_1200_3000", "protist_gt_3000"]
        vals = [bands.get(k, 0) for k in band_keys]
        bars = axE.bar(range(len(vals)), vals,
                       color=VIRIDIS(np.linspace(0.1, 0.9, len(vals))),
                       edgecolor="white")
        axE.set_xticks(range(len(band_labels)))
        axE.set_xticklabels(band_labels, fontsize=7)
        for b, v in zip(bars, vals):
            if v:
                axE.text(b.get_x()+b.get_width()/2, v, str(v),
                         ha="center", va="bottom", fontsize=8)
    axE.set_title("E · Size bands")
    axE.set_ylabel("count")

    # --- Panel F: cumulative size distribution -----------------------------
    axF = fig.add_subplot(gs[1, 2])
    if n_obj:
        s = np.sort(df["size_nm"].to_numpy())
        cdf = np.arange(1, len(s)+1) / len(s)
        axF.plot(s, cdf, color="#2c6fa6", lw=2)
        axF.fill_between(s, cdf, color="#2c6fa6", alpha=0.15)
        for q in (0.25, 0.5, 0.75):
            xq = np.quantile(s, q)
            axF.axvline(xq, color="grey", ls=":", lw=1)
            axF.text(xq, q, f" {q:.0%}: {xq:.0f}nm", fontsize=7, va="center")
    axF.set_title("F · Cumulative size distribution")
    axF.set_xlabel("equivalent diameter (nm)")
    axF.set_ylabel("cumulative fraction")
    axF.set_ylim(0, 1)

    fig.savefig(fig_path, bbox_inches="tight")
    print(f"  wrote {fig_path}")
    plt.close(fig)


def make_domain_size_figure(df: pd.DataFrame, fig_path: Path):
    """Size distribution split by domain (violin + strip)."""
    if "domain" not in df.columns or df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = [d for d in ["virus", "bacteria", "archaea", "protist"]
             if d in df["domain"].unique()]
    pal = {d: DOMAIN_COLORS.get(d, "#888") for d in order}
    sns.violinplot(data=df, x="domain", y="size_nm", order=order,
                   palette=pal, inner=None, cut=0, ax=ax, alpha=0.5)
    sns.stripplot(data=df, x="domain", y="size_nm", order=order,
                  palette=pal, size=3, alpha=0.5, ax=ax)
    ax.set_title("Size distribution by classified domain", fontweight="bold")
    ax.set_xlabel("domain")
    ax.set_ylabel("equivalent diameter (nm)")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"  wrote {fig_path}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    fig_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, report = load_results(out_dir)
    print(f"Loaded {len(df)} objects from {out_dir}")
    print(f"Columns: {list(df.columns)}")

    make_summary_figure(df, report, fig_dir / "cyclops_summary.png")
    make_domain_size_figure(df, fig_dir / "cyclops_domain_size.png")

    # Also dump a small text summary
    summary = fig_dir / "results_summary.txt"
    with open(summary, "w") as f:
        f.write("Cyclops results summary\n=======================\n\n")
        f.write(f"objects           : {len(df)}\n")
        if len(df):
            f.write(f"mean size (nm)    : {df['size_nm'].mean():.2f}\n")
            f.write(f"median size (nm)  : {df['size_nm'].median():.2f}\n")
            f.write(f"size std (nm)     : {df['size_nm'].std():.2f}\n")
            f.write(f"size range (nm)   : [{df['size_nm'].min():.1f}, {df['size_nm'].max():.1f}]\n")
            if "domain" in df.columns:
                f.write("\ndomain counts:\n")
                for d, c in df["domain"].value_counts().items():
                    f.write(f"  {d:10s}: {c:5d}  ({100*c/len(df):.1f}%)\n")
        for k in ("correction", "min_distance_nm", "psf_f_size", "psf_tau",
                  "psf_v", "n_dapi", "n_fitc", "elapsed_seconds"):
            if k in report:
                f.write(f"{k:18s}: {report[k]}\n")
    print(f"  wrote {summary}")


if __name__ == "__main__":
    main()
