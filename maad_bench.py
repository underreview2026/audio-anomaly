"""MAAD-Bench multi-dataset runner (supersedes esc50_slice.py).

One feature schema, one group-safe split discipline, one detector sweep across
ESC-50, UrbanSound8K, and RAVDESS. Two settings per task: normal-only training
(primary) and contaminated ADBench-style training (secondary), over >= 5 seeds.
Writes leakage-safe artifacts (features / metadata / scores) and a metrics table
per dataset, plus a combined cross-dataset summary.

PRE-REGISTERED CLASS MAPPINGS (placeholder -- confirm with domain experts).
Split units: ESC-50 fold, UrbanSound8K fold (both group same source recording in
one fold), RAVDESS actor (actor-disjoint).

Usage:
  python maad_bench.py --dataset esc50
  python maad_bench.py --dataset ravdess
  python maad_bench.py --dataset urbansound8k
  python maad_bench.py --dataset all
  python maad_bench.py --dataset ravdess --stage features   # extract+cache only
"""
import argparse
import glob
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SR = 22050
N_MFCC = 20
CONTAMINATION = 0.10
SEEDS = [0, 1, 2, 3, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")

# ------------------------- pre-registered mappings -------------------------
DATASETS = {
    "esc50": dict(
        normal=["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
                "water_drops", "wind", "pouring_water", "thunderstorm", "insects"],
        anomaly=["glass_breaking", "siren", "fireworks"],
        test_groups=[5],
    ),
    "urbansound8k": dict(
        normal=["air_conditioner", "engine_idling", "jackhammer", "drilling",
                "street_music", "children_playing"],
        anomaly=["gun_shot", "car_horn", "siren"],
        test_groups=[10],
    ),
    "ravdess": dict(
        normal=["neutral", "calm"],
        anomaly=["angry", "fearful"],
        test_groups=[21, 22, 23, 24],
    ),
    "nsynth": dict(
        normal=["acoustic"],
        anomaly=["electronic", "synthetic"],
        test_groups=[0],   # group = instrument id % 5; instruments stay disjoint
    ),
    "mimii": dict(
        normal=["normal"],
        anomaly=["abnormal"],
        test_groups=[6],   # hold out machine id_06; real machine anomalies
    ),
    "physionet": dict(
        normal=["normal"],
        anomaly=["abnormal"],
        test_groups=[5],   # hold out subset 'f'; PhysioNet subsets are population-disjoint
    ),
    "icbhi": dict(
        normal=["normal"],
        anomaly=["crackle", "wheeze", "both"],
        test_groups=[0],   # group = patient id % 5; patients stay disjoint
    ),
}


# --------------------------------- loaders ---------------------------------
def _download(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_esc50():
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    root = os.path.join(DATA_DIR, "ESC-50-master")
    if not os.path.exists(os.path.join(root, "meta", "esc50.csv")):
        zp = _download(url, os.path.join(DATA_DIR, "esc50.zip"))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(DATA_DIR)
    meta = pd.read_csv(os.path.join(root, "meta", "esc50.csv"))
    cfg = DATASETS["esc50"]
    meta = meta[meta["category"].isin(set(cfg["normal"]) | set(cfg["anomaly"]))].copy()
    return pd.DataFrame({
        "path": meta["filename"].map(lambda f: os.path.join(root, "audio", f)),
        "group": meta["fold"].to_numpy(),
        "category": meta["category"].to_numpy(),
        "anomaly_label": meta["category"].isin(cfg["anomaly"]).astype(int).to_numpy(),
    })


def load_urbansound8k():
    root = os.path.join(DATA_DIR, "UrbanSound8K")
    csv = os.path.join(root, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            "UrbanSound8K not found at %s. Download requires a form/mirror; place the "
            "extracted dataset there (metadata/UrbanSound8K.csv + audio/fold*/)." % root)
    meta = pd.read_csv(csv)
    cfg = DATASETS["urbansound8k"]
    meta = meta[meta["class"].isin(set(cfg["normal"]) | set(cfg["anomaly"]))].copy()
    return pd.DataFrame({
        "path": [os.path.join(root, "audio", "fold%d" % f, n)
                 for f, n in zip(meta["fold"], meta["slice_file_name"])],
        "group": meta["fold"].to_numpy(),
        "category": meta["class"].to_numpy(),
        "anomaly_label": meta["class"].isin(cfg["anomaly"]).astype(int).to_numpy(),
    })


def load_ravdess():
    url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    root = os.path.join(DATA_DIR, "RAVDESS")
    if not glob.glob(os.path.join(root, "Actor_*", "*.wav")):
        zp = _download(url, os.path.join(DATA_DIR, "ravdess.zip"))
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(root)
    emo = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
           "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    cfg = DATASETS["ravdess"]
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "Actor_*", "*.wav"))):
        parts = os.path.splitext(os.path.basename(p))[0].split("-")
        if len(parts) != 7:
            continue
        cat = emo.get(parts[2])
        if cat not in set(cfg["normal"]) | set(cfg["anomaly"]):
            continue
        rows.append({"path": p, "group": int(parts[6]), "category": cat,
                     "anomaly_label": int(cat in cfg["anomaly"])})
    return pd.DataFrame(rows)


def load_nsynth():
    import tarfile
    root = os.path.join(DATA_DIR, "nsynth-valid")
    ex = os.path.join(root, "examples.json")
    if not os.path.exists(ex):
        tgz = os.path.join(DATA_DIR, "nsynth-valid.jsonwav.tar.gz")
        if not os.path.exists(tgz):
            raise FileNotFoundError("NSynth valid not found; expected %s" % ex)
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)
    with open(ex) as fh:
        meta = json.load(fh)
    src = {0: "acoustic", 1: "electronic", 2: "synthetic"}
    cfg = DATASETS["nsynth"]
    keep = set(cfg["normal"]) | set(cfg["anomaly"])
    rows = []
    for note, m in meta.items():
        cat = src.get(m["instrument_source"])
        if cat not in keep:
            continue
        rows.append({
            "path": os.path.join(root, "audio", note + ".wav"),
            "group": int(m["instrument"]) % 5,
            "category": cat,
            "anomaly_label": int(cat in cfg["anomaly"]),
        })
    return pd.DataFrame(rows)


def load_mimii():
    import re
    cfg = DATASETS["mimii"]
    norm = glob.glob(os.path.join(DATA_DIR, "**", "id_*", "normal", "*.wav"), recursive=True)
    abn = glob.glob(os.path.join(DATA_DIR, "**", "id_*", "abnormal", "*.wav"), recursive=True)
    if not norm and not abn:
        raise FileNotFoundError(
            "MIMII not found under %s. Extract 6_dB_pump.zip there "
            "(expects id_*/{normal,abnormal}/*.wav)." % DATA_DIR)
    rows = []
    for p in sorted(norm) + sorted(abn):
        mobj = re.search(r"id_(\d+)", p)
        gid = int(mobj.group(1)) if mobj else -1
        is_abn = os.path.basename(os.path.dirname(p)) == "abnormal"
        rows.append({"path": p, "group": gid,
                     "category": "abnormal" if is_abn else "normal",
                     "anomaly_label": int(is_abn)})
    return pd.DataFrame(rows)


def load_physionet():
    # PhysioNet/CinC 2016 heart sound. Organic label: -1 normal, +1 abnormal.
    # Group by training subset (a..f), which are population-disjoint (Codex gate:
    # use the official population-disjoint split, no patient key needed).
    root = os.path.join(DATA_DIR, "physionet2016")
    refs = glob.glob(os.path.join(root, "**", "REFERENCE.csv"), recursive=True)
    if not refs:
        zp = os.path.join(DATA_DIR, "physionet2016_training.zip")
        if not os.path.exists(zp):
            raise FileNotFoundError("PhysioNet/CinC 2016 not found; expected %s" % zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(root)
        refs = glob.glob(os.path.join(root, "**", "REFERENCE.csv"), recursive=True)
    letters = "abcdefghij"
    rows = []
    for ref in sorted(refs):
        sub = os.path.dirname(ref)
        letter = os.path.basename(sub).split("-")[-1][:1]
        gid = letters.index(letter) if letter in letters else 0
        with open(ref) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                name, lab = parts[0].strip(), parts[1].strip()
                p = os.path.join(sub, name + ".wav")
                if not os.path.exists(p):
                    continue
                is_abn = (lab == "1")
                rows.append({"path": p, "group": gid,
                             "category": "abnormal" if is_abn else "normal",
                             "anomaly_label": int(is_abn)})
    return pd.DataFrame(rows)


def load_icbhi():
    # ICBHI 2017 respiratory sounds. Organic label: a respiratory cycle is
    # anomalous if it carries a crackle or a wheeze. Cycle-level samples via
    # (offset, duration) into each recording; grouped by patient (id % 5) so a
    # patient never spans train and test.
    root = os.path.join(DATA_DIR, "ICBHI", "ICBHI_final_database")
    rows = []
    for wav in sorted(glob.glob(os.path.join(root, "*.wav"))):
        base = os.path.basename(wav)[:-4]
        txt = os.path.join(root, base + ".txt")
        if not os.path.exists(txt):
            continue
        try:
            patient = int(base.split("_")[0])
        except ValueError:
            continue
        ann = pd.read_csv(txt, sep="\t", header=None,
                          names=["start", "end", "crackle", "wheeze"])
        for _, r in ann.iterrows():
            dur = float(r["end"]) - float(r["start"])
            if dur <= 0:
                continue
            crk, whz = int(r["crackle"]), int(r["wheeze"])
            cat = ("both" if crk and whz else "crackle" if crk
                   else "wheeze" if whz else "normal")
            rows.append({"path": wav, "offset": float(r["start"]), "duration": dur,
                         "group": patient % 5, "category": cat,
                         "anomaly_label": int(crk or whz)})
    return pd.DataFrame(rows)


LOADERS = {"esc50": load_esc50, "urbansound8k": load_urbansound8k,
           "ravdess": load_ravdess, "nsynth": load_nsynth, "mimii": load_mimii,
           "physionet": load_physionet, "icbhi": load_icbhi}


# --------------------------- feature extraction ---------------------------
def _summarize(name, arr):
    m, s = arr.mean(axis=1), arr.std(axis=1)
    out = {}
    for i in range(arr.shape[0]):
        out["%s%d_mean" % (name, i)] = float(m[i])
        out["%s%d_std" % (name, i)] = float(s[i])
    return out


def extract_one(path, offset=0.0, duration=None):
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    f = {}
    f.update(_summarize("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)))
    f.update(_summarize("chroma", librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0)))
    f.update(_summarize("cent", librosa.feature.spectral_centroid(y=y, sr=sr)))
    f.update(_summarize("bw", librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    f.update(_summarize("roll", librosa.feature.spectral_rolloff(y=y, sr=sr)))
    f.update(_summarize("zcr", librosa.feature.zero_crossing_rate(y)))
    f.update(_summarize("rms", librosa.feature.rms(y=y)))
    return f


def extract_features(name, df):
    import hashlib
    os.makedirs(CACHE_DIR, exist_ok=True)
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5(keysrc.encode()).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, "%s_feat_%s.parquet" % (name, key))
    if os.path.exists(cache):
        print("cached features:", cache, flush=True)
        return pd.read_parquet(cache)
    from joblib import Parallel, delayed
    print("extracting features for %d clips ..." % len(df), flush=True)
    if has_seg:
        rows = Parallel(n_jobs=-1, verbose=5)(
            delayed(extract_one)(p, o, d)
            for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        rows = Parallel(n_jobs=-1, verbose=5)(delayed(extract_one)(p) for p in df["path"])
    feat = pd.DataFrame(rows)
    feat.to_parquet(cache, index=False)
    return feat


# --------------------------------- splits ---------------------------------
def make_split(df, seed, setting, test_groups):
    rng = np.random.RandomState(seed)
    in_test = df["group"].isin(test_groups).to_numpy()
    is_anom = (df["anomaly_label"] == 1).to_numpy()
    idx = np.arange(len(df))

    test_normal = idx[in_test & ~is_anom]
    n_anom = max(1, int(round(CONTAMINATION / (1 - CONTAMINATION) * len(test_normal))))
    pool = idx[in_test & is_anom]
    test_anom = rng.choice(pool, size=min(n_anom, len(pool)), replace=False)
    test_idx = np.concatenate([test_normal, test_anom])

    train_normal = idx[~in_test & ~is_anom]
    if setting == "primary":
        train_idx = train_normal
    else:
        pool_tr = idx[~in_test & is_anom]
        n_tr = min(len(pool_tr),
                   max(1, int(round(CONTAMINATION / (1 - CONTAMINATION) * len(train_normal)))))
        train_idx = np.concatenate([train_normal, rng.choice(pool_tr, size=n_tr, replace=False)])
    return train_idx, test_idx


# -------------------------------- detectors --------------------------------
def detector_factories():
    """Build the PyOD detector roster. Constructor args are passed by signature
    introspection (contamination / random_state only when accepted), so detectors
    with incompatible APIs are skipped at build time rather than polluting results.
    """
    import importlib
    import inspect
    specs = [
        # proximity / density
        ("KNN", "pyod.models.knn", "KNN"),
        ("LOF", "pyod.models.lof", "LOF"),
        ("COF", "pyod.models.cof", "COF"),
        ("CBLOF", "pyod.models.cblof", "CBLOF"),
        ("ABOD", "pyod.models.abod", "ABOD"),
        ("SOD", "pyod.models.sod", "SOD"),
        ("KDE", "pyod.models.kde", "KDE"),
        ("Sampling", "pyod.models.sampling", "Sampling"),
        # linear / subspace
        ("PCA", "pyod.models.pca", "PCA"),
        ("KPCA", "pyod.models.kpca", "KPCA"),
        ("MCD", "pyod.models.mcd", "MCD"),
        ("OCSVM", "pyod.models.ocsvm", "OCSVM"),
        ("LMDD", "pyod.models.lmdd", "LMDD"),
        # probabilistic / statistical
        ("HBOS", "pyod.models.hbos", "HBOS"),
        ("COPOD", "pyod.models.copod", "COPOD"),
        ("ECOD", "pyod.models.ecod", "ECOD"),
        ("QMCD", "pyod.models.qmcd", "QMCD"),
        ("GMM", "pyod.models.gmm", "GMM"),
        # ensemble / tree
        ("INNE", "pyod.models.inne", "INNE"),
        ("IForest", "pyod.models.iforest", "IForest"),
        ("LODA", "pyod.models.loda", "LODA"),
        ("FeatureBagging", "pyod.models.feature_bagging", "FeatureBagging"),
        # neural
        ("AutoEncoder", "pyod.models.auto_encoder", "AutoEncoder"),
        ("VAE", "pyod.models.vae", "VAE"),
        ("DIF", "pyod.models.dif", "DIF"),
        ("LUNAR", "pyod.models.lunar", "LUNAR"),
    ]
    facs = {}
    for name, mod, cls in specs:
        try:
            klass = getattr(importlib.import_module(mod), cls)
            params = inspect.signature(klass.__init__).parameters
        except Exception:
            continue

        def mk(klass=klass, params=params, name=name):
            def build(c, rs):
                kw = {}
                if "contamination" in params:
                    kw["contamination"] = c
                if "random_state" in params:
                    kw["random_state"] = rs
                if name == "ABOD" and "method" in params:
                    kw["method"] = "fast"  # O(n^2) exact ABOD is too slow at scale
                return klass(**kw)
            return build

        b = mk()
        try:
            b(0.1, 0)  # test-construct; skip detectors whose API rejects these args
            facs[name] = b
        except Exception:
            pass
    return facs


def metrics(y, scores, contamination):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    out = {"auc_roc": float(roc_auc_score(y, scores)),
           "avg_precision": float(average_precision_score(y, scores))}
    try:
        out["pauc_0.1"] = float(roc_auc_score(y, scores, max_fpr=0.1))
    except Exception:
        out["pauc_0.1"] = float("nan")
    k = max(1, int(round(contamination * len(y))))
    thresh = np.sort(scores)[::-1][k - 1]
    out["f1_top_q"] = float(f1_score(y, (scores >= thresh).astype(int), zero_division=0))
    return out


def run(df, feat, test_groups):
    from sklearn.preprocessing import StandardScaler
    X = feat.to_numpy(dtype=float)
    y = df["anomaly_label"].to_numpy()
    facs = detector_factories()
    print("detectors:", ", ".join(facs), flush=True)
    rec = []
    for setting in ("primary", "secondary"):
        for seed in SEEDS:
            tr, te = make_split(df, seed, setting, test_groups)
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte, yte = sc.transform(X[tr]), sc.transform(X[te]), y[te]
            for name, make in facs.items():
                try:
                    det = make(CONTAMINATION, seed)
                    det.fit(Xtr)
                    m = metrics(yte, det.decision_function(Xte), CONTAMINATION)
                except Exception as e:
                    m = {"auc_roc": float("nan"), "avg_precision": float("nan"),
                         "pauc_0.1": float("nan"), "f1_top_q": float("nan"),
                         "error": type(e).__name__}
                m.update({"detector": name, "setting": setting, "seed": seed,
                          "n_train": len(tr), "n_test": len(te), "n_test_anom": int(yte.sum())})
                rec.append(m)
    return pd.DataFrame(rec)


def summarize(scores_df, dataset):
    agg = (scores_df.groupby(["setting", "detector"])
           [["auc_roc", "avg_precision", "pauc_0.1", "f1_top_q"]].agg(["mean", "std"]))
    agg.columns = ["%s_%s" % (a, b) for a, b in agg.columns]
    agg = agg.reset_index()
    agg.insert(0, "dataset", dataset)
    return agg.sort_values(["setting", "auc_roc_mean"], ascending=[True, False])


def run_dataset(name):
    cfg = DATASETS[name]
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    df = LOADERS[name]()
    # group-safety: no group straddles the train/test line (it cannot, groups are atomic)
    print("[%s] %d clips, %d anomalies, groups=%d, test_groups=%s" % (
        name, len(df), int(df["anomaly_label"].sum()), df["group"].nunique(), cfg["test_groups"]),
        flush=True)
    feat = extract_features(name, df)
    feat.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    df[["path", "group", "category", "anomaly_label"]].to_parquet(
        os.path.join(out_dir, "metadata.parquet"), index=False)
    scores = run(df, feat, cfg["test_groups"])
    scores.to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
    summ = summarize(scores, name)
    summ.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({**cfg, "contamination": CONTAMINATION, "seeds": SEEDS,
                   "n_features": feat.shape[1]}, fh, indent=2)
    print("\n=== %s summary (mean over seeds) ===" % name, flush=True)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ.to_string(index=False), flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["esc50", "urbansound8k", "ravdess", "nsynth", "mimii",
                             "physionet", "icbhi", "all"])
    ap.add_argument("--stage", choices=["all", "features"], default="all")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ALL = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
    names = ALL if args.dataset == "all" else [args.dataset]
    summaries = []
    for name in names:
        try:
            if args.stage == "features":
                df = LOADERS[name]()
                extract_features(name, df)
                print("[%s] features done" % name, flush=True)
            else:
                summaries.append(run_dataset(name))
        except Exception as e:
            print("[%s] FAILED: %s: %s" % (name, type(e).__name__, e), flush=True)
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "metrics_combined.csv"), index=False)
        print("\nwrote combined summary to", os.path.join(RESULTS_DIR, "metrics_combined.csv"), flush=True)


if __name__ == "__main__":
    main()
