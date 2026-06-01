"""Rich figures for the MAAD-Bench paper (GEO-Bench tradeoff style): multi-dim
encoding, shaded regions, annotated points, legends OUTSIDE the data area,
deliberate palette. Saved as PDFs into RESULTS_DIR.

  fig_overview.pdf : focused, large normal-vs-anomaly mel-spectrogram gallery for
                     four sub-datasets (two designated-class, two organic-label),
                     with a one-line caption carrying the pipeline and finding.
  fig_transfer.pdf : quadrant scatter of the 25 detectors (color = family,
                     size = rank shift); legends outside; labels decluttered.
  fig_repr.pdf     : grouped bars per sub-dataset, handcrafted / PANNs / AST.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

R = mb.RESULTS_DIR
DES = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORG = ["mimii", "physionet", "icbhi"]
LABEL = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
         "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet", "icbhi": "ICBHI"}

FAMILY = {
    "KNN": "Proximity", "LOF": "Proximity", "COF": "Proximity", "SOD": "Proximity",
    "CBLOF": "Proximity",
    "PCA": "Linear", "KPCA": "Linear", "OCSVM": "Linear", "MCD": "Linear", "LMDD": "Linear",
    "GMM": "Density", "KDE": "Density", "HBOS": "Density", "COPOD": "Density",
    "ECOD": "Density", "ABOD": "Density", "QMCD": "Density",
    "IForest": "Ensemble", "INNE": "Ensemble", "LODA": "Ensemble", "Sampling": "Ensemble",
    "AutoEncoder": "Neural", "VAE": "Neural", "DIF": "Neural", "LUNAR": "Neural",
}
FAMS = ["Proximity", "Linear", "Density", "Ensemble", "Neural"]
FAMCOLOR = {"Proximity": "#5B8FB9", "Linear": "#26A69A", "Density": "#EF5350",
            "Ensemble": "#F4A259", "Neural": "#9575CD"}
DESBLUE, ORGORANGE, ACCENT = "#3A6EA5", "#C2691F", "#1F4E79"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------- transfer
def load_means():
    rows = []
    for ds in DES + ORG:
        d = pd.read_csv(os.path.join(R, ds, "metrics.csv"))
        d = d[d["setting"] == "primary"][["detector", "auc_roc_mean"]].copy()
        d["dataset"] = ds
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    des = df[df.dataset.isin(DES)].groupby("detector")["auc_roc_mean"].mean()
    org = df[df.dataset.isin(ORG)].groupby("detector")["auc_roc_mean"].mean()
    common = [d for d in des.index if d in org.index and d in FAMILY]
    t = pd.DataFrame({"det": common, "des": des[common].values, "org": org[common].values})
    t["fam"] = t["det"].map(FAMILY)
    t["shift"] = (t["des"].rank(ascending=False) - t["org"].rank(ascending=False)).abs()
    return t


def fig_transfer():
    t = load_means()
    rho = t["des"].rank().corr(t["org"].rank(), method="spearman")
    mx, my = t["des"].median(), t["org"].median()
    x0, x1, y0, y1 = 0.55, 0.90, 0.27, 0.67

    fig, ax = plt.subplots(figsize=(7.3, 3.9))
    fig.subplots_adjust(left=0.085, right=0.80, top=0.88, bottom=0.13)
    ax.add_patch(Rectangle((mx, y0), x1 - mx, my - y0, facecolor="#FDECEA", ec="none", zorder=0))
    ax.add_patch(Rectangle((x0, my), mx - x0, y1 - my, facecolor="#E5F3F1", ec="none", zorder=0))
    ax.axvline(mx, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(my, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(x1 - 0.004, y0 + 0.010, "strong on designated,\nweak on organic", ha="right",
            va="bottom", fontsize=7.6, style="italic", color="#C0392B", zorder=1)
    ax.text(x0 + 0.004, y1 - 0.006, "overlooked on designated,\nstrong on organic", ha="left",
            va="top", fontsize=7.6, style="italic", color="#138D75", zorder=1)
    for _, r in t.iterrows():
        ax.scatter(r["des"], r["org"], s=36 + 22 * r["shift"], color=FAMCOLOR[r["fam"]],
                   edgecolor="white", linewidth=1.0, alpha=0.92, zorder=3)
    off = {"SOD": (-7, 6, "right", "bottom"), "DIF": (8, 3, "left", "center"),
           "MCD": (8, -7, "left", "top"), "QMCD": (8, 5, "left", "bottom"),
           "KNN": (7, -3, "left", "top"), "HBOS": (0, -9, "center", "top")}
    for _, r in t.iterrows():
        if r["det"] in off:
            dx, dy, ha, va = off[r["det"]]
            ax.annotate(r["det"], (r["des"], r["org"]), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=8.4, color="#222222", zorder=4)
    ax.text(x0 + 0.008, y0 + 0.045, r"Spearman $\rho = %.2f$ ($p = 0.58$)" % rho + "\nno significant transfer",
            ha="left", va="bottom", fontsize=9.2, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#999999", lw=0.8), zorder=5)
    ax.set_xlabel("Effectiveness on designated-class tasks   (mean AUC-ROC $\\rightarrow$)", fontsize=10)
    ax.set_ylabel("Effectiveness on\norganic-label tasks $\\rightarrow$", fontsize=10)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.grid(True, color="#ECECEC", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fam_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMCOLOR[f],
                          markeredgecolor="white", markersize=9, label=f) for f in FAMS]
    leg1 = ax.legend(handles=fam_handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                     ncol=5, fontsize=8.4, frameon=False, handletextpad=0.25, columnspacing=1.0)
    ax.add_artist(leg1)
    size_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
                           markeredgecolor="white", markersize=ms, label="%d" % s)
                    for s, ms in ((2, 6), (10, 9), (18, 13))]
    ax.legend(handles=size_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              labelspacing=1.1, title="rank shift\n(des. vs org.)", fontsize=8.2,
              title_fontsize=8.2, frameon=True, framealpha=0.95, borderpad=0.7)
    fig.savefig(os.path.join(R, "fig_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_transfer.pdf (rho=%.3f, %d detectors)" % (rho, len(t)))


# ----------------------------------------------------------------------------- repr
def fig_repr():
    backbones = [("Handcrafted-74", "metrics.csv", "#9AA0A6"),
                 ("PANNs-64", "panns_metrics.csv", "#5B8FB9"),
                 ("AST-64", "ast_metrics.csv", "#E8743B")]
    order = DES + ORG
    best = {lab: [] for lab, _, _ in backbones}
    for ds in order:
        for lab, fn, _ in backbones:
            p = os.path.join(R, ds, fn); v = np.nan
            if os.path.exists(p):
                d = pd.read_csv(p); d = d[d["setting"] == "primary"]
                if len(d):
                    v = float(d["auc_roc_mean"].max())
            best[lab].append(v)
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    ax.add_patch(Rectangle((-0.5, 0.4), 4.0, 0.8, facecolor="#EEF4FA", ec="none", zorder=0))
    ax.add_patch(Rectangle((3.5, 0.4), 3.0, 0.8, facecolor="#FDF1E8", ec="none", zorder=0))
    ax.text(1.5, 1.13, "designated-class", ha="center", va="center", fontsize=9, style="italic",
            color=DESBLUE, zorder=2)
    ax.text(5.0, 1.13, "organic-label", ha="center", va="center", fontsize=9, style="italic",
            color=ORGORANGE, zorder=2)
    for i, (lab, _, c) in enumerate(backbones):
        vals = best[lab]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, edgecolor="white", lw=0.6, zorder=3)
        for xi, v in zip(x + (i - 1) * w, vals):
            if not np.isnan(v):
                ax.annotate("%.2f" % v, (xi, v), textcoords="offset points", xytext=(0, 2),
                            ha="center", va="bottom", fontsize=6.2, color="#333333", zorder=4)
    ax.axvline(3.5, color="#BBBBBB", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in order], fontsize=9)
    ax.set_ylabel("Best AUC-ROC (primary)", fontsize=10.5)
    ax.set_ylim(0.4, 1.2); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, axis="y", color="#ECECEC", linewidth=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=9, frameon=False,
              columnspacing=1.6, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_repr.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_repr.pdf")


# ----------------------------------------------------------------------------- overview gallery
def _spec(name, anomaly, category=None):
    """Return (mel_dB 64xT, category) for the highest-energy clip among a handful of
    candidates, so the panel shows clear structure rather than near-silence."""
    import librosa
    df = mb.LOADERS[name]()
    sub = df[df["anomaly_label"] == (1 if anomaly else 0)]
    if category is not None and "category" in df.columns:
        hit = sub[sub["category"].astype(str).str.lower().str.contains(category.lower())]
        if len(hit):
            sub = hit
    step = max(1, len(sub) // 8)
    cands = sub.iloc[::step][:8] if len(sub) else sub
    best, best_e = None, -1.0
    for _, row in cands.iterrows():
        off = float(row["offset"]) if "offset" in df.columns and not pd.isna(row.get("offset")) else 0.0
        dur = 2.5
        if "duration" in df.columns and not pd.isna(row.get("duration")):
            dur = min(2.5, float(row["duration"]))
        try:
            y, sr = librosa.load(row["path"], sr=mb.SR, mono=True, offset=off, duration=dur)
        except Exception:
            continue
        if len(y) < 1024:
            y = np.pad(y, (0, 1024 - len(y)))
        e = float(np.sqrt(np.mean(y ** 2)))
        if e > best_e:
            best, best_e = (y, sr, str(row.get("category", ""))), e
    if best is None:
        raise RuntimeError("no loadable clip for %s" % name)
    y, sr, cat = best
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max), cat


def fig_overview():
    from matplotlib.colors import LinearSegmentedColormap
    des_blue, org_orange = DESBLUE, ORGORANGE
    muted, anomaly_red = "#6B7280", "#A33A32"
    cols = [("esc50", "ESC-50", des_blue, ("rain", "rain"), ("glass", "glass break")),
            ("ravdess", "RAVDESS", des_blue, ("neutral", "neutral"), ("angry", "angry")),
            ("mimii", "MIMII", org_orange, (None, "normal"), (None, "machine fault")),
            ("physionet", "PhysioNet", org_orange, (None, "normal"), (None, "abnormal"))]
    specs = []
    for name, _, _, (ncat, nlab), (acat, alab) in cols:
        try:
            sn, _ = _spec(name, False, ncat)
        except Exception as e:
            sn = None; print("spec fail %s normal: %s" % (name, e))
        try:
            sa, _ = _spec(name, True, acat)
        except Exception as e:
            sa = None; print("spec fail %s anomaly: %s" % (name, e))
        specs.append((sn, nlab, sa, alab))

    mel_cmap = LinearSegmentedColormap.from_list(
        "maad_ice", ["#0C1B2E", "#1B466F", "#2F7FA6", "#74C3C6", "#CFE7DB", "#F6F1DA"], N=256)

    with plt.rc_context({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.2, 3.35))
        bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
        bg.set_xlim(0, 1); bg.set_ylim(0, 1)
        x0, pw, g, gg = 0.072, 0.196, 0.012, 0.032
        cx = [x0, x0 + pw + g, x0 + 2 * (pw + g) + gg, x0 + 3 * (pw + g) + gg]
        y_top, y_bot, ph = 0.478, 0.108, 0.286

        for gc, lab, c in (((cx[0] + cx[1] + pw) / 2, "Designated-class", des_blue),
                           ((cx[2] + cx[3] + pw) / 2, "Organic-label", org_orange)):
            bg.text(gc, 0.912, lab, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=c)
            bg.plot([gc - 0.105, gc + 0.105], [0.876, 0.876], color=c, lw=1.6,
                    solid_capstyle="round", transform=bg.transAxes)
        dvx = (cx[1] + pw + cx[2]) / 2
        bg.plot([dvx, dvx], [0.055, 0.852], color="#D9DEE4", lw=1.0, transform=bg.transAxes)
        bg.text(0.034, y_top + ph / 2, "Normal", rotation=90, ha="center", va="center",
                fontsize=9.2, color=muted)
        bg.text(0.034, y_bot + ph / 2, "Anomaly", rotation=90, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=anomaly_red)
        for j, (name, disp, col, _, _) in enumerate(cols):
            bg.text(cx[j] + pw / 2, 0.823, disp, ha="center", va="center", fontsize=9.6,
                    fontweight="bold", color=col)
            sn, nlab, sa, alab = specs[j]
            for img, lab, yb, lc in ((sn, nlab, y_top, muted), (sa, alab, y_bot, anomaly_red)):
                ax = fig.add_axes([cx[j], yb, pw, ph])
                if img is not None:
                    ax.imshow(np.asarray(img, float), aspect="auto", origin="lower",
                              cmap=mel_cmap, vmin=-60, vmax=0, interpolation="bilinear")
                else:
                    ax.set_facecolor("#EEF1F4")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor("white"); s.set_linewidth(1.4)
                ax.text(0.5, -0.085, lab, transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.2, color=lc, style="italic")
        bg.text(0.5, 0.028,
                r"one 74-feature vector per clip  $\rightarrow$  25 PyOD detectors  "
                r"$\rightarrow$  no significant rank transfer ($\rho = 0.12$, $p = 0.58$)",
                ha="center", va="center", fontsize=8.8, color="#3A4654")
        fig.savefig(os.path.join(R, "fig_overview.pdf"), bbox_inches="tight")
        plt.close(fig)
    print("wrote fig_overview.pdf")


if __name__ == "__main__":
    fig_transfer()
    fig_repr()
    fig_overview()
