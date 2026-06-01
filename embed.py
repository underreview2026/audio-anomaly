"""MAAD-Bench pretrained-embedding arm, generalized over backbones. Replaces the
74-dim handcrafted vector with a pretrained audio embedding, then runs the SAME
PyOD detector sweep after a leakage-safe PCA reduction. Extends the PANNs result
(panns_embed.py) to more backbones so the representation axis spans handcrafted ->
PANNs -> AST -> CLAP. Needs a GPU + network for the first model download.

  python embed.py --backbone ast  --dataset all
  python embed.py --backbone clap --dataset all     # requires laion_clap

Writes results/<name>/<backbone>_metrics.csv (+ _scores.parquet), same schema as
the tabular metrics.csv so make_table-style aggregation can read it.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maad_bench as mb

ALLNAMES = ["esc50", "urbansound8k", "ravdess", "nsynth", "mimii", "physionet", "icbhi"]


def build_ast():
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe = ASTFeatureExtractor.from_pretrained(name)
    model = ASTModel.from_pretrained(name).to("cuda").eval()

    def embed_batch(wavs):
        inp = fe(list(wavs), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model(inp["input_values"].to("cuda"))
        return out.pooler_output.cpu().numpy()

    return embed_batch, 16000, 768, None


def build_clap():
    import numpy as np
    import torch
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    tgt = 480000  # 10 s at 48 kHz

    def embed_batch(wavs):
        x = np.stack([w[:tgt] if len(w) >= tgt else np.pad(w, (0, tgt - len(w)))
                      for w in wavs]).astype(np.float32)
        with torch.no_grad():
            return model.get_audio_embedding_from_data(x=x, use_tensor=False)

    return embed_batch, 48000, 512, tgt


BUILDERS = {"ast": build_ast, "clap": build_clap}


def embed_dataset(name, backbone, batch=16):
    import librosa
    df = mb.LOADERS[name]()
    has_seg = "offset" in df.columns
    if has_seg:
        keysrc = "\n".join("%s|%s|%s" % (p, o, d)
                           for p, o, d in zip(df["path"], df["offset"], df["duration"]))
    else:
        keysrc = "\n".join(df["path"].astype(str))
    key = hashlib.md5((backbone + "|" + keysrc).encode()).hexdigest()[:10]
    cache = os.path.join(mb.CACHE_DIR, "%s_%s_%s.parquet" % (name, backbone, key))
    if os.path.exists(cache):
        print("[%s/%s] cached embeddings %s" % (name, backbone, cache), flush=True)
        return df, pd.read_parquet(cache)
    embed_batch, sr, dim, tgt = BUILDERS[backbone]()
    embs = np.empty((len(df), dim), dtype=np.float32)
    bi, bw = [], []

    def flush():
        if bw:
            e = embed_batch(bw)
            for j, idx in enumerate(bi):
                embs[idx] = e[j]
        bi.clear()
        bw.clear()

    print("[%s/%s] extracting embeddings for %d clips ..." % (name, backbone, len(df)), flush=True)
    for i in range(len(df)):
        r = df.iloc[i]
        off = float(r["offset"]) if has_seg else 0.0
        dur = float(r["duration"]) if has_seg else None
        y, _ = librosa.load(r["path"], sr=sr, mono=True, offset=off, duration=dur)
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))
        bi.append(i)
        bw.append(y)
        if len(bw) >= batch:
            flush()
        if i % 2000 == 0 and i:
            print("  %d/%d" % (i, len(df)), flush=True)
    flush()
    feat = pd.DataFrame(embs, columns=["e%d" % j for j in range(dim)])
    os.makedirs(mb.CACHE_DIR, exist_ok=True)
    feat.to_parquet(cache, index=False)
    return df, feat


def run(name, backbone):
    from sklearn.decomposition import PCA
    cfg = mb.DATASETS[name]
    df, feat = embed_dataset(name, backbone)
    print("[%s/%s] embeddings %s" % (name, backbone, feat.shape), flush=True)
    k = min(64, feat.shape[1], feat.shape[0])
    red = PCA(n_components=k, random_state=0).fit_transform(feat.to_numpy(dtype=float))
    feat = pd.DataFrame(red, columns=["pc%d" % j for j in range(k)])
    print("[%s/%s] reduced to %s via PCA" % (name, backbone, feat.shape), flush=True)
    scores = mb.run(df, feat, cfg["test_groups"])
    summ = mb.summarize(scores, name)
    out = os.path.join(mb.RESULTS_DIR, name)
    os.makedirs(out, exist_ok=True)
    scores.to_parquet(os.path.join(out, "%s_scores.parquet" % backbone), index=False)
    summ.to_csv(os.path.join(out, "%s_metrics.csv" % backbone), index=False)
    print("\n=== %s %s-embedding sweep (primary AUC-ROC) ===" % (name, backbone), flush=True)
    print(summ[summ["setting"] == "primary"][["detector", "auc_roc_mean"]].to_string(index=False),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BUILDERS))
    ap.add_argument("--dataset", required=True, choices=ALLNAMES + ["all"])
    args = ap.parse_args()
    mb.SEEDS = [0, 1, 2]  # supplementary arms: fewer seeds for speed
    _orig_fac = mb.detector_factories
    mb.detector_factories = lambda: {k: v for k, v in _orig_fac().items() if k != "LMDD"}
    names = ALLNAMES if args.dataset == "all" else [args.dataset]
    for nm in names:
        try:
            run(nm, args.backbone)
        except Exception as e:
            print("[%s/%s] FAILED %s: %s" % (nm, args.backbone, type(e).__name__, e), flush=True)


if __name__ == "__main__":
    main()
