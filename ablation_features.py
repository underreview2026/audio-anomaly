"""MAAD-Bench feature-family ablation. Reuses cached features and the maad_bench
split / detector / metric code. Incremental schema: MFCC only -> + chroma ->
+ spectral (full). CPU-only and fast (no audio re-extraction). Run from
~/projects/maad-bench/ so `import maad_bench` resolves.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

DATASETS = ["esc50", "urbansound8k", "ravdess"]

# Incremental feature families, identified by the column-name prefixes the
# extractor emits (mfcc / chroma / cent / bw / roll / zcr / rms).
FEATURE_SETS = {
    "mfcc": ("mfcc",),
    "mfcc+chroma": ("mfcc", "chroma"),
    "full": ("mfcc", "chroma", "cent", "bw", "roll", "zcr", "rms"),
}


def cols_for(feat, prefixes):
    keep = [c for c in feat.columns if any(c.startswith(p) for p in prefixes)]
    return feat[keep]


def main():
    rows = []
    for name in DATASETS:
        df = mb.LOADERS[name]()
        feat = mb.extract_features(name, df)  # cache hit
        for setname, prefixes in FEATURE_SETS.items():
            sub = cols_for(feat, prefixes)
            sc = mb.run(df, sub, mb.DATASETS[name]["test_groups"])
            s = mb.summarize(sc, name)
            s["feature_set"] = setname
            s["n_features"] = sub.shape[1]
            rows.append(s)
            print("[%s | %s] %d features done" % (name, setname, sub.shape[1]), flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(mb.RESULTS_DIR, "ablation_features.csv"), index=False)

    prim = out[out["setting"] == "primary"]
    mean_piv = prim.pivot_table(index="dataset", columns="feature_set", values="auc_roc_mean")
    print("\n=== feature ablation: primary AUC-ROC, mean over detectors & seeds ===")
    print(mean_piv[["mfcc", "mfcc+chroma", "full"]].round(3).to_string())

    best = prim.loc[prim.groupby(["dataset", "feature_set"])["auc_roc_mean"].idxmax()]
    print("\n=== feature ablation: best detector per set (primary) ===")
    print(best[["dataset", "feature_set", "detector", "auc_roc_mean", "n_features"]]
          .sort_values(["dataset", "feature_set"]).to_string(index=False))


if __name__ == "__main__":
    main()
