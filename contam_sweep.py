"""Contamination-level robustness sweep for MAAD-Bench.

Re-runs the detector sweep at 5% and 20% contamination (10% is the main setting)
reusing the CACHED features in results/<name>/features.parquet, so no audio is
re-read. The point is to show that AUC-ROC is roughly invariant to the
contamination level while average precision scales with it, which justifies
reporting AUC-ROC as the primary metric. Writes results/<name>/sweep_cNN.csv per
level and a combined results/sweep_combined.csv. Neural detectors use the GPU, so
run this when the GPU is free. Run from ~/projects/maad-bench/.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

LEVELS = [0.05, 0.20]
NAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]

rows = []
for c in LEVELS:
    mb.CONTAMINATION = c
    tag = "c%02d" % int(round(c * 100))
    for name in NAMES:
        d = os.path.join(mb.RESULTS_DIR, name)
        feat = pd.read_parquet(os.path.join(d, "features.parquet"))
        df = pd.read_parquet(os.path.join(d, "metadata.parquet"))
        scores = mb.run(df, feat, mb.DATASETS[name]["test_groups"])
        summ = mb.summarize(scores, name)
        summ["contamination"] = c
        summ.to_csv(os.path.join(d, "sweep_%s.csv" % tag), index=False)
        rows.append(summ)
        print("[%s @ %.2f] done" % (name, c), flush=True)

out = pd.concat(rows, ignore_index=True)
out.to_csv(os.path.join(mb.RESULTS_DIR, "sweep_combined.csv"), index=False)
print("wrote sweep_combined.csv (%d rows)" % len(out), flush=True)
