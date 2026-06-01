"""Performance-space acoustics case study for MAAD-Bench. Treats a measured room
impulse response (RIR) as one sample, extracts ISO 3382-1 room-acoustic parameters
via Schroeder backward integration, and runs the SAME PyOD detector idea on the
parameter vectors -- showing the benchmark's tabular pipeline carries over from
audio clips to room-acoustic measurements.

Two anomaly questions:
  (1) distributional: which spaces are acoustic outliers among the set (unsupervised
      detector consensus);
  (2) task-conditioned: a space is "anomalous" for a musical task if its reverberation
      time falls outside the range that task wants (e.g., orchestral/chamber music
      prefers mid-frequency T30 ~ 1.5-2.2 s). We then ask whether unsupervised
      detectors on the full ISO 3382-1 vector recover the task-unsuitable spaces (AUC).

Usage:  python acoustics.py --rir_dir <dir of *.wav RIRs>
Writes results/acoustics_params.csv and prints distributional + task-conditioned AD.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb


def edc_schroeder(h):
    """Energy decay curve (dB) by Schroeder backward integration of the squared IR,
    with a simple noise-floor truncation so the late tail does not flatten the slope."""
    e = h.astype(np.float64) ** 2
    # truncate at an estimated noise floor: last 10% of the IR is assumed noise
    tail = e[int(0.9 * len(e)):]
    nf = np.mean(tail) if len(tail) else 0.0
    cut = len(e)
    above = np.where(e > nf * 1.0)[0]
    if len(above):
        cut = min(len(e), above[-1] + 1)
    e = e[:cut]
    edc = np.cumsum(e[::-1])[::-1]
    edc = edc / (edc[0] + 1e-20)
    return 10.0 * np.log10(edc + 1e-20)


def rt_from_edc(edc_db, sr, d1, d2):
    """Reverberation time from the EDC slope between d1 and d2 dB, extrapolated to -60."""
    t = np.arange(len(edc_db)) / sr
    i1 = np.argmax(edc_db <= d1)
    i2 = np.argmax(edc_db <= d2)
    if i2 <= i1 + 2:
        return np.nan
    a = np.polyfit(t[i1:i2], edc_db[i1:i2], 1)
    slope = a[0]
    return float(-60.0 / slope) if slope < -1e-6 else np.nan


def iso3382(h, sr):
    h = h / (np.max(np.abs(h)) + 1e-12)
    onset = int(np.argmax(np.abs(h)))
    h = h[max(0, onset - int(0.001 * sr)):]  # align to a hair before the direct sound
    e = h.astype(np.float64) ** 2
    total = e.sum() + 1e-20
    edc_db = edc_schroeder(h)
    n50, n80 = int(0.05 * sr), int(0.08 * sr)
    t = np.arange(len(h)) / sr
    return {
        "EDT": rt_from_edc(edc_db, sr, 0.0, -10.0),
        "T20": rt_from_edc(edc_db, sr, -5.0, -25.0),
        "T30": rt_from_edc(edc_db, sr, -5.0, -35.0),
        "C50": 10.0 * np.log10(e[:n50].sum() / (e[n50:].sum() + 1e-20)),
        "C80": 10.0 * np.log10(e[:n80].sum() / (e[n80:].sum() + 1e-20)),
        "D50": float(e[:n50].sum() / total),
        "Ts": float((t * e).sum() / total),
    }


def load_rir(path):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rir_dir", required=True)
    ap.add_argument("--task_lo", type=float, default=1.5)  # orchestral/chamber T30 window
    ap.add_argument("--task_hi", type=float, default=2.2)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.rir_dir, "**", "*.wav"), recursive=True))
    print("[acoustics] %d RIR files under %s" % (len(paths), args.rir_dir), flush=True)
    rows = []
    for p in paths:
        try:
            y, sr = load_rir(p)
            if len(y) < sr // 10:
                continue
            par = iso3382(y, sr)
            par["rir"] = os.path.relpath(p, args.rir_dir)
            rows.append(par)
        except Exception as e:
            print("  skip %s: %s" % (os.path.basename(p), type(e).__name__), flush=True)
    df = pd.DataFrame(rows).dropna(subset=["EDT"]).reset_index(drop=True)
    cols = ["EDT", "T20", "T30", "C50", "C80", "D50", "Ts"]
    out = os.path.join(mb.RESULTS_DIR, "acoustics_params.csv")
    os.makedirs(mb.RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False)
    print("[acoustics] %d spaces with valid ISO 3382-1 params -> %s" % (len(df), out), flush=True)
    print(df[cols].describe().loc[["mean", "min", "max"]].round(3).to_string())

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    feat = df[cols].copy()
    feat = feat.fillna(feat.median())                      # short RIRs lack T20/T30
    X = StandardScaler().fit_transform(feat.to_numpy(dtype=float))
    facs = mb.detector_factories()

    # (1) distributional: which spaces are acoustic outliers (detector consensus)
    keep = ["IForest", "KNN", "GMM", "ECOD", "PCA", "HBOS"]
    votes = np.zeros(len(df))
    nv = 0
    for name in keep:
        if name in facs:
            d = facs[name](0.1, 0)
            d.fit(X)
            s = d.decision_function(X)
            thr = np.sort(s)[::-1][max(1, int(0.1 * len(s))) - 1]
            votes += (s >= thr).astype(int)
            nv += 1
    df["outlier_votes"] = votes.astype(int)
    print("\n=== distributional: most acoustically anomalous spaces (consensus of %d detectors) ===" % nv)
    print(df.sort_values("outlier_votes", ascending=False)
          [["rir", "EDT", "T30", "C80", "outlier_votes"]].head(8).round(3).to_string(index=False))

    # (2) task-conditioned: the most reverberant spaces are unsuitable for clear
    # speech/recording. Can unsupervised detectors recover them from CLARITY params
    # alone (C50/C80/D50/Ts), i.e., without seeing the decay time that defines the task?
    tau = float(df["EDT"].quantile(0.85))
    y_task = (df["EDT"] >= tau).astype(int).to_numpy()
    clar = df[["C50", "C80", "D50", "Ts"]].copy()
    clar = clar.fillna(clar.median())
    Xo = StandardScaler().fit_transform(clar.to_numpy(float))
    print("\n=== task-conditioned: 'too reverberant for clear speech' = EDT >= %.2f s "
          "(top 15%%, %d/%d spaces); detected from clarity params only ===" %
          (tau, int(y_task.sum()), len(y_task)))
    res = []
    for name, make in facs.items():
        try:
            d = make(0.15, 0)
            d.fit(Xo)
            res.append((name, roc_auc_score(y_task, d.decision_function(Xo))))
        except Exception:
            pass
    res.sort(key=lambda r: -r[1])
    for name, auc in res[:8]:
        print("  %-12s AUC=%.3f" % (name, auc))


if __name__ == "__main__":
    main()
