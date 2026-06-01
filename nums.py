"""Print best primary AUC + detector per dataset for each arm, plus judge breakdown.
Used to fill the temporal/representation tables and the judge/acoustics sections."""
import os
import pandas as pd

RES = "results"
DS = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]
ARMS = {"tabular": "metrics.csv", "panns": "panns_metrics.csv", "ast": "ast_metrics.csv",
        "temporal": "temporal_metrics.csv", "twfr": "twfr_metrics.csv"}


def best(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d["setting"] == "primary"]
    if not len(d):
        return None
    i = d["auc_roc_mean"].idxmax()
    return str(d.loc[i, "detector"]), float(d.loc[i, "auc_roc_mean"])


print("%-13s | %s" % ("dataset", " | ".join("%-14s" % a for a in ARMS)))
for ds in DS:
    cells = []
    for arm, fn in ARMS.items():
        b = best(os.path.join(RES, ds, fn))
        cells.append("%-14s" % ("--" if b is None else "%.3f %s" % (b[1], b[0])))
    print("%-13s | %s" % (ds, " | ".join(cells)))

print("\n=== temporal detectors: per-detector primary AUC (all rows) ===")
for ds in DS:
    p = os.path.join(RES, ds, "temporal_metrics.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[d["setting"] == "primary"][["detector", "auc_roc_mean"]]
        row = " ".join("%s=%.3f" % (r.detector, r.auc_roc_mean) for r in d.itertuples())
        print("%-13s %s" % (ds, row))

jp = "judge_results.csv"
if os.path.exists(jp):
    j = pd.read_csv(jp)
    print("\n=== judge: per-dataset confusion (TP=anomaly flagged, TN=normal kept) ===")
    for ds in j["dataset"].unique():
        s = j[j.dataset == ds]
        tp = int((s.true_anom & s.judge_oop).sum())
        fn = int((s.true_anom & ~s.judge_oop).sum())
        fp = int((~s.true_anom & s.judge_oop).sum())
        tn = int((~s.true_anom & ~s.judge_oop).sum())
        print("%-13s TP=%d FN=%d FP=%d TN=%d" % (ds, tp, fn, fp, tn))
