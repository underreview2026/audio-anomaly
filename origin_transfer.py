"""MAAD-Bench origin-transfer analysis. Do detector rankings on DESIGNATED-CLASS
tasks transfer to ORGANIC-LABEL tasks? Reports the Spearman rank correlation
between the two partitions' mean-AUC detector orderings (primary setting), plus
the per-detector rank shift. Framed as transfer of detector SELECTION across
benchmark partitions, not as causal proof (the organic side is confounded with
domain). scipy-only. Run from ~/projects/maad-bench/.
"""
import glob
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

DESIGNATED = {"esc50", "urbansound8k", "ravdess", "nsynth"}
ORGANIC = {"mimii", "physionet", "icbhi"}

rows = []
for sp in glob.glob(os.path.join(mb.RESULTS_DIR, "*", "scores.parquet")):
    ds = os.path.basename(os.path.dirname(sp))
    d = pd.read_parquet(sp)
    d["dataset"] = ds
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
prim = df[df["setting"] == "primary"]
present = set(prim["dataset"].unique())


def partition_mean(names):
    sub = prim[prim["dataset"].isin(names)]
    return sub.groupby("detector")["auc_roc"].mean()


des_sets = sorted(DESIGNATED & present)
org_sets = sorted(ORGANIC & present)
print("designated datasets:", des_sets)
print("organic datasets:", org_sets)
if not org_sets:
    print("no organic datasets present yet; rerun after MIMII/PhysioNet/ICBHI land")
    sys.exit(0)

des = partition_mean(des_sets)
org = partition_mean(org_sets)
common = sorted(set(des.index) & set(org.index))
des, org = des[common], org[common]
rho, p = spearmanr(des.values, org.values)
print("n detectors:", len(common))
print("Spearman rho (designated rank vs organic rank): %.3f (p=%.3f)" % (rho, p))

tab = pd.DataFrame({"designated": des, "organic": org})
tab["des_rank"] = tab["designated"].rank(ascending=False)
tab["org_rank"] = tab["organic"].rank(ascending=False)
tab["rank_shift"] = (tab["des_rank"] - tab["org_rank"]).abs()
print("\n=== detector mean AUC: designated vs organic (sorted by designated) ===")
print(tab.sort_values("designated", ascending=False).round(3).to_string())
print("\n=== biggest rank shifts (worst transfer) ===")
print(tab.sort_values("rank_shift", ascending=False).head(8).round(3).to_string())
