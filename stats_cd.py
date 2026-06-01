"""MAAD-Bench cross-dataset statistics: average detector ranks and a Friedman
test over (dataset x seed) blocks, primary setting. This is the rigorous
"no single detector dominates" evidence. scipy-only (no extra deps). Run from
~/projects/maad-bench/.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

rows = []
for sp in glob.glob(os.path.join(mb.RESULTS_DIR, "*", "scores.parquet")):
    ds = os.path.basename(os.path.dirname(sp))
    d = pd.read_parquet(sp)
    d["dataset"] = ds
    rows.append(d)
df = pd.concat(rows, ignore_index=True)

for setting in ("primary", "secondary"):
    sub = df[df["setting"] == setting]
    piv = sub.pivot_table(index=["dataset", "seed"], columns="detector", values="auc_roc")
    piv = piv.dropna(axis=1, how="any")          # detectors scored on every block
    dets = list(piv.columns)
    mat = piv.to_numpy()
    N, k = mat.shape
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, mat)   # best AUC = rank 1
    avg = ranks.mean(0)
    chi, p = friedmanchisquare(*[mat[:, i] for i in range(k)])
    print("\n=== %s: %d blocks (dataset x seed), %d detectors ===" % (setting, N, k))
    print("Friedman chi2=%.2f, p=%.3e" % (chi, p))
    order = np.argsort(avg)
    for i in order:
        print("  %-14s avg_rank=%5.2f   mean_auc=%.3f" % (dets[i], avg[i], mat[:, i].mean()))
