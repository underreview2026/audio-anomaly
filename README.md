# MAAD-Bench

A model-agnostic benchmark for audio anomaly detection. MAAD-Bench reduces each
audio clip to a single fixed-length tabular feature vector, so any standard
outlier detector applies without audio-specific modeling, and it compares
detectors across two kinds of anomaly: human-designated classes and organic
(real-world) fault or clinical labels.

## Overview

- **Seven sub-datasets in two groups.**
  - *Designated-class* (a held-out class is the anomaly): ESC-50 (environmental
    sound), UrbanSound8K (urban sound), RAVDESS (emotional speech), NSynth
    (musical timbre).
  - *Organic-label* (the label is a real fault or clinical abnormality): MIMII
    (industrial machine faults), PhysioNet 2016 (heart-sound abnormality),
    ICBHI 2017 (respiratory abnormality).
- **One feature schema.** Each clip becomes a 74-dimensional vector: 20 MFCCs,
  12 chroma bins, and 5 spectral descriptors, each summarized by its mean and
  standard deviation over frames (audio resampled to 22.05 kHz).
- **Twenty-five detectors plus a deep anchor.** 25 unsupervised detectors from
  PyOD across the proximity, linear, density, ensemble, and neural families,
  plus a DCASE-style log-mel reconstruction autoencoder reported separately.
- **Two settings.** Normal-only training (primary) and contaminated training
  (secondary), five seeds, and group-safe splits that prevent train/test
  leakage.

## Installation

Python 3.10 or newer is recommended.

```
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, scipy, scikit-learn, pyod (tested with 3.5.1),
librosa, soundfile, joblib, matplotlib, pyarrow, and torch (for the neural
detectors and the autoencoder). The optional pretrained-embedding arms also need
panns_inference (PANNs), transformers (AST), or laion_clap (CLAP).

## Data

The seven corpora are public and are not redistributed here. Download each and
place it under `data/<name>/`.

| Sub-dataset | Source |
|---|---|
| ESC-50 | https://github.com/karolpiczak/ESC-50 |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html |
| RAVDESS | https://zenodo.org/record/1188976 |
| NSynth | https://magenta.tensorflow.org/datasets/nsynth |
| MIMII | https://zenodo.org/record/3384388 |
| PhysioNet 2016 | https://physionet.org/content/challenge-2016/ |
| ICBHI 2017 | https://bhichallenge.med.auth.gr/ |

The folder layout each corpus expects is defined by its loader in
`maad_bench.py` (for example `load_esc50`, `load_mimii`). `us8k_fetch.py` helps
arrange UrbanSound8K.

## Precomputed Artifacts

To let the benchmark reproduce without re-extracting features from gigabytes of
audio, this release bundles the computed artifacts under `results/<sub-dataset>/`
for all seven sub-datasets:

- `features.parquet`: the 74-dimensional handcrafted feature matrix (20 MFCCs, 12
  chroma bins, 5 spectral descriptors, each as mean and standard deviation), one row
  per clip.
- `metadata.parquet`: evaluation-only fields, row-aligned to the features: `path`
  (relative to `data/`), `group` (the native grouping that defines the leakage-safe
  split), `category` (original class), and `anomaly_label` (0 or 1).
- `scores.parquet`: per-detector metrics (AUC-ROC, average precision, partial AUC, and
  F1 at the top-q threshold) for the 25-detector sweep across both settings and five
  seeds. `ae_scores.parquet`, `ast_scores.parquet`, `panns_scores.parquet`, and
  `twfr_scores.parquet` hold the autoencoder anchor and the representation and temporal
  arms; `metrics.csv` is a flat summary.

The raw audio is **not** redistributed (download each corpus from the sources above);
the relative `path` column maps each feature row back to its source clip. With the
bundled `results/`, the table scripts in the next section run directly, skipping the
sweep.

## Reproduce

All scripts read and write under the repository root (`data/`, `cache/`,
`results/`). The pre-registered class mappings, sampling rate, contamination
level, and seeds are set at the top of `maad_bench.py`.

1. **Main detector sweep** (writes `results/<name>/metrics.csv`):
   ```
   python maad_bench.py --dataset all
   ```
2. **Autoencoder anchor** (writes `results/<name>/ae_scores.parquet`):
   ```
   python ae_baseline.py --dataset all
   ```
3. **Tables and analysis:**
   - `python make_table.py` and `python tabulate_results.py`: primary and
     secondary result tables.
   - `python origin_transfer.py`: the designated-to-organic rank correlation and
     Friedman analysis (the headline transfer result).
   - `python extra_tables.py`: representation, robustness, and runtime tables.
   - `python ablation_features.py`: the feature-family ablation.
   - `python contam_sweep.py` and `python runtime.py`: the contamination
     robustness and per-detector runtime appendices.
4. **Representation and temporal arms** (optional, need the extra packages):
   - `python panns_embed.py` and `python embed.py`: PANNs and AST embeddings.
   - `python temporal.py`, `python twfr.py`: frame-level temporal detectors.
5. **Room-acoustics case study:**
   ```
   python acoustics.py
   ```
6. **Figures** (writes PDFs under `results/`):
   ```
   python fig_make.py
   ```

## Repository Layout

| File | Role |
|---|---|
| `maad_bench.py` | Core: loaders, group-safe splits, the 74-feature schema, the 25-detector sweep, metrics |
| `ae_baseline.py` | DCASE-style log-mel autoencoder anchor |
| `make_table.py`, `tabulate_results.py` | Primary and secondary result tables |
| `origin_transfer.py` | Designated-versus-organic rank correlation and Friedman test |
| `extra_tables.py`, `stats_cd.py` | Representation, robustness, runtime tables and rank statistics |
| `embed.py`, `panns_embed.py` | Pretrained-embedding arms (AST, PANNs) |
| `temporal.py`, `twfr.py` | Frame-level temporal detectors and TWFR-GMM |
| `ablation_features.py` | Feature-family ablation |
| `acoustics.py` | Room-acoustics (ISO 3382-1) case study |
| `contam_sweep.py`, `runtime.py` | Contamination robustness and runtime |
| `fig_make.py` | Figures |
| `count_meta.py`, `nums.py`, `us8k_fetch.py` | Dataset statistics and helpers |

## License

Released under the MIT License (see `LICENSE`).

## Citation

A paper describing MAAD-Bench is under review. Until it is published, please cite
it as a manuscript submitted for publication:

```
@unpublished{maadbench2026,
  title  = {MAAD-Bench: A Model-Agnostic Benchmark for Audio Anomaly Detection},
  author = {Qian Hao and Liang Zhao},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```
