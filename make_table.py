"""Emit LaTeX table bodies for the MAAD-Bench paper from per-dataset metrics.csv.

Primary and secondary AUC-ROC, formatted as "mean (rank)" per cell (rank within
each sub-dataset column, 1 = best), best per column in bold, an "Avg. rank"
column, and detectors sorted by average rank (best on top). This matches the
value-(rank) convention used by ADBench and MetaOD. Run from ~/projects/maad-bench/.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

DESIGNATED = ["esc50", "urbansound8k", "ravdess", "nsynth"]
ORGANIC = ["mimii", "physionet", "icbhi"]
ORDER = DESIGNATED + ORGANIC
LABEL = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "ravdess": "RAVDESS",
         "nsynth": "NSynth", "mimii": "MIMII", "physionet": "PhysioNet",
         "icbhi": "ICBHI"}

rows = []
for mc in glob.glob(os.path.join(mb.RESULTS_DIR, "*", "metrics.csv")):
    rows.append(pd.read_csv(mc))
df = pd.concat(rows, ignore_index=True)


def emit(setting):
    sub = df[df["setting"] == setting]
    m = sub.pivot_table(index="detector", columns="dataset", values="auc_roc_mean")
    present = [d for d in ORDER if d in m.columns]
    m = m[present]
    ranks = m.rank(ascending=False, method="min")
    avg_rank = ranks.mean(axis=1)
    print("\n%% ===== %s AUC-ROC, mean (rank); cols: %s & Avg. rank =====" %
          (setting, " & ".join(LABEL[d] for d in present)))
    for det in avg_rank.sort_values().index:
        cells = []
        for d in present:
            mv, rk = m.loc[det, d], ranks.loc[det, d]
            if pd.isna(mv):
                cells.append("--")
            elif rk == 1:
                cells.append("\\textbf{%.3f (1)}" % mv)
            else:
                cells.append("%.3f (%d)" % (mv, int(rk)))
        cells.append("%.1f" % avg_rank.loc[det])
        print("%-12s & %s \\\\" % (det, " & ".join(cells)))


emit("primary")
emit("secondary")
