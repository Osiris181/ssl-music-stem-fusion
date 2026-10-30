# Stem Fusion for Music Classification

This project investigates whether instrument and vocal representations extracted
from separated music stems improve classification beyond representations of
the original mixture. It studies genre classification, multi-label mood/theme
tagging, and binary valence/arousal classification using frozen pretrained
music representations and downstream neural classifiers.

The central question is whether stem features provide additional predictive
value when the mixture already has an informative representation, given the
effectiveness of modern pretrained music SSL representations. Comparisons
include mixture-only models with matched classifier capacity or ensemble size,
so improvements can be assessed alongside the extra model resources used by
fusion.

## Research purpose

The research follows three objectives:

1. **Explicit mixture-plus-stem fusion using pretrained representations.**
   Study embeddings extracted separately from the original mixture and selected
   stems using frozen, general-purpose pretrained music feature extractors.
   The MSS model is likewise pretrained/frozen; train the downstream classifier
   only. The scientifically important contrast is against strong mixture-only
   feature baselines.
2. **Incremental value of selected stem combinations.** Test whether
   incorporating selected instrument-stem subsets in addition to the original
   mixture adds predictive value, which combinations are beneficial, and
   whether including more stems necessarily improves results.
3. **Cross-task effectiveness.** Investigate whether explicit mixture-plus-stem
   fusion gives different results across genre classification, mood/theme
   tagging, and binary valence/arousal emotion recognition, including neutral
   or negative effects. Apparent task differences are exploratory unless
   dataset, label, feature-extractor, classifier, and other confounders are
   addressed.

## Datasets

The study covers the following dataset and representation settings:

| Dataset | Classification task | Representation |
| --- | --- | --- |
| FMA-Medium | Genre, 16 classes | MusicFM |
| GTZAN | Genre, 10 classes | MusicFM |
| ISMIR04 | Genre, 6 classes | MusicFM |
| MTG-Jamendo mood/theme | Multi-label tagging, 56 tags | MusicFM and MERT + chord/key, evaluated separately |
| PMEmo | Binary valence and binary arousal, evaluated separately | MERT + chord/key |
| EmoMusic | Binary valence and binary arousal, evaluated separately | MERT + chord/key |
| DEAM | Binary valence and binary arousal, evaluated separately | MERT + chord/key |

The MERT setting combines pretrained audio embeddings with estimated chord
and global-key descriptors. Dataset-specific labels, partitions, and
representation settings are described in the
[experimental design](context/PHASE2_EXPERIMENTAL_DESIGN.md).
Comparisons are made within each dataset, representation, and target.

## Implementation

Runnable extraction and training scripts are provided for
[FMA-Medium](src/stemfusion/experiments/fma_medium/README.md). Dataset scripts
load YAML settings and call shared modules for data loading, feature
extraction, model training, evaluation, and reporting.

| Component | Implementation |
| --- | --- |
| Source separation | Frozen HT-Demucs `htdemucs_6s`; retain the original mixture and five selected stems. |
| FMA feature extraction | Frozen MusicFM-MSD, layer 7, temporal mean pooling, 1,024-dimensional embeddings; full-duration mono audio at 24 kHz. |
| Mid-fusion | Concatenate selected embeddings and train one MLP. Compare against the standard mixture-only MLP and capacity-matched mixture-only MLPs. |
| Late-fusion | Train and select source MLPs independently, then sum their logits equally. Compare against independently trained mixture-only ensembles with the same branch counts. |
| Evaluation | Select checkpoints using validation data, evaluate the selected models on the test split, and save per-track predictions and aggregate metrics. |

The original mixture remains an independent input. The experiments compare
the following selected input configurations:

| Configuration | Inputs |
| --- | --- |
| `M+G+D+P` | Mixture, guitar, drums, piano |
| `M+G+D+P+B` | Mixture, guitar, drums, piano, bass |
| `M+G+D+P+B+V` | Mixture, guitar, drums, piano, bass, vocals |

Here `M` denotes the original mixture. “All stems” means the five selected
instrument/vocal stems; the separator's native `other` output is excluded.

For FMA-Medium, the standard MLP has two hidden layers of 1,024 and 512 units,
with ReLU and dropout `0.2`. Training uses cross-entropy, Adam at `1e-4`,
batch size `128`, and `20` epochs. Capacity-matched mixture controls widen
the first hidden layer to approximately match their corresponding fusion
model's trainable parameter count.

All compared models use the same retained tracks, labels, splits, and
evaluation functions. Checkpoint selection maximizes validation accuracy +
macro ROC-AUC + macro average precision, with each component on the 0–1 scale.
The test split is evaluated after checkpoint selection.

The five outer training seeds are **32, 64, 128, 256, and 512**. Summaries
report the mean and sample standard deviation (`ddof=1`), along with paired
differences between fusion models and their matched controls. PR-AUC denotes
macro average precision. For late fusion, branch seeds are derived
deterministically from each outer seed, and compatible selected branches are
shared across nested ensembles.

## Directory guide

```text
stem-fusion-music-classification/
├── configs/                         # YAML settings and experiment conditions
│   ├── separators/                  # HT-Demucs settings
│   ├── features/                    # MusicFM settings and checkout location
│   ├── training/                    # Shared classifier settings and paths
│   └── experiments/fma_medium/      # Extraction and training matrices
├── context/                         # Research motivation and experimental design notes
├── data/                            # Local manifests and cached features
├── docs/                            # Detailed configuration and usage guides
├── outputs/                         # Result summaries and generated run artifacts
├── src/stemfusion/
│   ├── cli.py                       # Unified command interface
│   ├── configuration.py             # YAML loading, validation, and CLI overrides
│   ├── data/                        # Dataset manifests and aligned feature loading
│   ├── separation/                  # Source-separation workflows
│   ├── features/                    # Frozen representation extraction
│   ├── models/                      # MLP and ensemble architectures
│   ├── training/                    # Shared trainer, fusion, metrics, and reporting
│   └── experiments/fma_medium/      # Runnable dataset scripts and scheduling
├── tests/                           # Synthetic tests for data, training, and reporting
├── STEM_Separation.pdf               # Research manuscript
└── pyproject.toml                   # Package metadata and dependencies
```

The `data/` folder at the repository root holds local inputs;
`src/stemfusion/data/` contains the code that prepares and reads them.
Dataset entry scripts belong in `experiments/`, while reusable implementation
belongs in the shared source modules. Directory READMEs explain the individual
files and their responsibilities.

## Usage

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install -e ".[htdemucs,musicfm,training]"
```

Feature extraction also requires a local MusicFM checkout and the MSD
checkpoint/statistics files. Configure the paths in:

- [MusicFM settings](configs/features/musicfm_msd_layer7.yaml).
- [FMA extraction settings](configs/experiments/fma_medium/extraction.yaml).
- [Shared FMA training settings](configs/training/fma_medium_genre.yaml).

Prepare the aligned manifest and features using the
[extraction guide](docs/fma_medium_extraction.md). Then run:

```bash
python -m stemfusion train fma-medium mid-fusion
python -m stemfusion train fma-medium late-fusion
```

Alternatively, run the dataset scripts directly:

```bash
cd src/stemfusion/experiments/fma_medium
python extraction.py
python train_mid.py
python train_late.py
```

Both interfaces use the same YAML settings and shared functions. Explicit
command-line options override YAML. Use `--help` to inspect options,
`--dry-run` to inspect a training plan, or `--validate-only` to check cached
features without training. Complete, compatible runs are verified and reused
automatically.

Training includes test evaluation and prints a results table when finished.
The complete reports are saved in `outputs/fma_medium/`:

```text
summary_mid_fusion.txt
summary_mid_fusion.json
summary_late_fusion.txt
summary_late_fusion.json
```

TXT files provide readable tables and metric explanations. JSON files contain
detailed per-seed results and checkpoint references. Both summary formats can
be tracked by Git; raw audio, feature caches, checkpoints, prediction arrays,
and intermediate run artifacts remain excluded.

For more detail, see the [separation guide](docs/htdemucs_separation.md),
[training guide](docs/fma_htdemucs_training.md), and
[configuration guide](docs/configuration.md).

Run the synthetic test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
