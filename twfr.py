"""MAAD-Bench TWFR-GMM arm (audio-native, CPU-only). Following TWFR-GMM
(Guan et al., ICASSP 2023): pool a log-mel spectrogram over time with Generalized
Weighted Rank Pooling (GWRP) into a frequency-domain clip vector, then run the
PyOD detector sweep. The named method is the GMM detector on this representation;
running the full roster also shows how other detectors fare on GWRP features.
Reuses the train-on-normal protocol and the summarize schema. Run from anywhere.

  python twfr.py --dataset all

Writes results/<name>/twfr_metrics.csv (+ _scores.parquet), same schema as the
tabular metrics.csv.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

N_MELS, N_FFT, HOP, DECAY = 128, 1024, 512, 0.7
ALLNAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def gwrp_time(logmel, decay):
    """Generalized Weighted Rank Pooling over time. For each mel band, sort its
    time series descending and weight by decay**rank (decay=1 -> mean, ->0 -> max),
    giving a time-weighted frequency-domain vector (n_mels,)."""
    xs = np.sort(logmel, axis=1)[:, ::-1]
    T = logmel.shape[1]
    w = decay ** np.arange(T)
    w = w / w.sum()
    return (xs * w[None, :]).sum(axis=1)


def twfr_one(path, offset=0.0, duration=None):
    import librosa
    y, sr = librosa.load(path, sr=mb.SR, mono=True, offset=offset, duration=duration)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel)
    return gwrp_time(logmel, DECAY).astype(np.float32)


def extract_twfr(name):
    df = mb.LOADERS[name]()
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(("twfr|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_twfr_%s.parquet" % (name, key))
    if os.path.exists(cache):
        print("[%s/twfr] cached %s" % (name, cache), flush=True)
        return df, pd.read_parquet(cache)
    from joblib import Parallel, delayed
    print("[%s/twfr] extracting GWRP log-mel for %d clips ..." % (name, len(df)), flush=True)
    if has_seg:
        rows = Parallel(n_jobs=-1, verbose=5)(
            delayed(twfr_one)(p, o, d)
            for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(twfr_one)(p) for p in df["path"])
    feat = pd.DataFrame(np.vstack(rows), columns=["m%d" % j for j in range(N_MELS)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    return df, feat


def run(name):
    cfg = mb.DATASETS[name]
    df, feat = extract_twfr(name)
    print("[%s/twfr] features %s" % (name, feat.shape), flush=True)
    scores = mb.run(df, feat, cfg["test_groups"])
    summ = mb.summarize(scores, name)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    scores.to_parquet(os.path.join(out, "twfr_scores.parquet"), index=False)
    summ.to_csv(os.path.join(out, "twfr_metrics.csv"), index=False)
    g = summ[(summ["setting"] == "primary") & (summ["detector"] == "GMM")]
    print("[%s/twfr] TWFR-GMM primary AUC = %.3f" %
          (name, float(g["auc_roc_mean"].iloc[0]) if len(g) else float("nan")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=ALLNAMES + ["all"])
    args = ap.parse_args()
    mb.SEEDS = [0, 1, 2]  # supplementary arms: fewer seeds for speed
    _DROP = {"LMDD", "KPCA", "ABOD", "COF", "SOD"}  # O(n^2) detectors, slow on big datasets
    _orig_fac = mb.detector_factories
    mb.detector_factories = lambda: {k: v for k, v in _orig_fac().items() if k not in _DROP}
    names = ALLNAMES if args.dataset == "all" else [args.dataset]
    for nm in names:
        try:
            run(nm)
        except Exception as e:
            print("[%s/twfr] FAILED %s: %s" % (nm, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    main()
