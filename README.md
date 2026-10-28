# Stem Fusion for Music Classification

This repository is the staging area for cleaning and consolidating the research
code associated with the paper **"Enhancing Music Classification via Source
Separation and Feature Fusion."**

The intended repository name is:

```text
stem-fusion-music-classification
```

The intended Python package and command-line name is `stemfusion`.

> Status (2026-10-28): the active project now contains HT-Demucs separation,
> FMA-Medium feature extraction, and classifier workflows. Banquet-specific
> code, configurations, documentation, and Table 2 outputs have been removed
> from this folder with a recovery archive outside the project. See
> [the cleanup record](docs/cleanup_20261028.md). Original research directories
> outside this project remain unchanged.
>
> The current revision protocol is in
> [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md). Its matched
> controls and five-seed evaluation have not yet been implemented. Existing
> RMS extraction and weighted-fusion code remain historical functionality,
> outside the revised primary matrix.

## Migration progress

- [x] Select the repository and package names.
- [x] Record the target repository structure and reproduction requirements.
- [x] Consolidate the HT-Demucs batch scripts into one configurable wrapper.
- [ ] Validate cleaned HT-Demucs outputs against legacy outputs.
- [x] Migrate the FMA-Medium HT-Demucs manifest, vanilla MusicFM extraction,
  and separate RMS extraction stage.
- [ ] Validate cleaned MusicFM embeddings against legacy feature caches.
- [x] Consolidate the FMA-Medium single-source genre classifier into one
  configurable MLP trainer with legacy cache adapters.
- [x] Remove the retired separator's workflows and archive its experiment outputs.
- [x] Migrate distinct HT-Demucs Table 3 mid-fusion and Table 4 late-fusion
  training workflows with separate RMS-cache support.
- [ ] Migrate feature extraction for the remaining datasets and backends.
- [ ] Migrate classifiers, training, and evaluation table by table.

See [docs/legacy_map.md](docs/legacy_map.md) for the source-to-replacement
mapping, [docs/fma_htdemucs_features.md](docs/fma_htdemucs_features.md) for
the FMA-Medium feature workflow, and
[docs/fma_htdemucs_training.md](docs/fma_htdemucs_training.md) for classifier usage.

## Research objective

The project evaluates whether features extracted from source-separated music
stems improve downstream music classification. The common pipeline is:

```text
mixture audio
    -> music source separation
    -> frozen feature extraction for the mixture and stems
    -> feature fusion
    -> trainable classifier
    -> task-specific evaluation
```

Only the downstream classifier is trained. The source-separation models and
pretrained feature extractors remain frozen.

## Current revision scope

### Source-separation systems

- HT-Demucs `htdemucs_6s`, which produces vocals, drums, bass, piano,
  guitar, and other.

The revised fusion configurations retain the original mixture and add guitar,
drums and piano, then bass, then vocals. The separator's `other` output is
excluded from these comparisons.

### Feature extractors

- MusicFM-MSD features with temporal average pooling.
- The reproduced Kang and Herremans feature set: frozen MERT-95M features,
  BTC chord estimates, and an inferred global key.

RMS extraction remains available for historical weighted-fusion experiments.

### Fusion models

- Mid-fusion: concatenate source features and classify them with a
  three-layer MLP.
- Late fusion with an unweighted sum: classify each source independently and
  sum the resulting logits.

The revised design adds parameter-matched mixture-only MLPs and
ensemble-matched mixture-only controls. These are planned comparisons;
the existing RMS-weighted implementation is outside the revised primary matrix.

### Datasets and tasks

| Task | Datasets |
| --- | --- |
| Genre classification | FMA-Medium, GTZAN, ISMIR04 |
| Mood/theme tagging | MTG-Jamendo |
| Valence and arousal classification | PMEmo, EmoMusic, DEAM |

The revised question is whether stems add predictive value beyond pretrained
mixture representations after controlling for classifier capacity and
ensembling. Conclusions depend on the planned controlled experiments; apparent
differences across tasks remain conditional on datasets and representations.

## Repository organization

The cleaned repository should be organized by reusable pipeline component,
not by duplicated dataset-specific training scripts.

```text
stem-fusion-music-classification/
├── README.md
├── pyproject.toml
├── .gitignore
├── paper/
│   └── STEM_Separation.pdf
├── configs/
│   ├── paths.example.yaml
│   ├── datasets/
│   ├── separators/
│   ├── features/
│   ├── fusion/
│   ├── training/
│   └── experiments/
│       └── paper/
├── src/
│   └── stemfusion/
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── reproducibility.py
│       ├── data/
│       │   └── datasets/
│       ├── separation/
│       ├── features/
│       ├── models/
│       ├── tasks/
│       ├── training/
│       ├── evaluation/
│       └── pipelines/
├── scripts/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── reference_results/
├── data/
└── outputs/
```

Each current directory contains its own README describing its responsibility,
files, inputs, outputs, and relevant commands. Start with
`configs/README.md`, `src/README.md`, `docs/README.md`, or `tests/README.md`
when navigating the repository.

### Configuration layout

Reusable configuration groups should describe datasets, separators, feature
extractors, fusion models, and training protocols. Experiment manifests should
map directly to the paper tables:

```text
configs/experiments/paper/
├── table_03_fma_htdemucs.yaml
├── table_04_fma_fusion.yaml
├── table_05_gtzan.yaml
├── table_06_ismir04.yaml
├── table_07_jamendo_musicfm.yaml
├── table_08_jamendo_mert.yaml
├── table_09_pmemo_valence.yaml
├── table_10_pmemo_arousal.yaml
├── table_11_emomusic_valence.yaml
├── table_12_emomusic_arousal.yaml
├── table_13_deam_valence.yaml
└── table_14_deam_arousal.yaml
```

Each table manifest may enumerate several source or fusion variants, but all
directly comparable variants must inherit the same training hyperparameters.

## Intended command-line workflow

Separation, feature extraction, training, and evaluation must remain separate
stages because the first two are expensive and their outputs should be cached.

```bash
stemfusion separate --config configs/experiments/paper/table_03_fma_htdemucs.yaml
stemfusion extract --config configs/experiments/paper/table_03_fma_htdemucs.yaml
stemfusion train --config configs/experiments/paper/table_03_fma_htdemucs.yaml
stemfusion evaluate --config configs/experiments/paper/table_03_fma_htdemucs.yaml
stemfusion reproduce --table 3
```

## Data and artifact policy

The following must not be mixed with source code or committed to the
repository:

- raw datasets;
- separated audio files;
- feature caches;
- model checkpoints;
- TensorBoard or other training logs;
- local virtual environments;
- machine-specific paths.

Local paths should be supplied through an ignored local configuration derived
from `configs/paths.example.yaml`.

Generated runs should follow a predictable layout:

```text
outputs/<experiment>/<run_id>/
├── resolved_config.yaml
├── environment.json
├── data_manifest_checksum.txt
├── metrics.json
├── predictions.npz
├── checkpoint.pt
└── logs/
```

The resolved configuration, seed, software versions, data-manifest checksum,
and source revision should be recorded for every run.

## Canonical data conventions

- Use the source names `mixture`, `guitar`, `drums`, `piano`, `bass`,
  `vocals`, and `other` throughout the cleaned code.
- Support old names such as `mix`, `drum`, and `vocal` only at the legacy
  import boundary.
- Store track IDs explicitly as strings rather than deriving them throughout
  the training code from arbitrary file paths.
- Build one aligned manifest per dataset containing the track ID, split,
  labels, mixture path, and available stem paths.
- Detect missing, corrupt, and silent sources while constructing or validating
  the manifest. Directly comparable methods must use the same track set.
- Keep the RMS value separate from the learned embedding. Legacy feature files
  that append RMS as an extra dimension should be converted by a migration
  adapter.
- Fit preprocessing statistics only on the training split.
- Select checkpoints and classification thresholds using validation data only.

## Task-specific behavior

The cleaned implementation must explicitly distinguish these task types:

- multiclass genre classification;
- multilabel mood/theme tagging;
- binary valence classification;
- binary arousal classification.

Loss functions, averaging modes, checkpoint criteria, threshold selection,
and reported metrics must be declared in configuration rather than hidden in
dataset-specific scripts.

## Testing priorities

Tests should cover the parts of the pipeline most likely to silently change an
experiment:

- dataset split integrity and leakage detection;
- track and source alignment;
- deterministic source ordering;
- embedding and RMS dimensions;
- mid-fusion concatenation order;
- unweighted and RMS-weighted late-fusion calculations;
- metric averaging conventions;
- validation-only threshold selection;
- checkpoint selection criteria;
- behavior for missing, corrupt, and silent audio;
- a small end-to-end run using synthetic cached features.

## Migration and validation plan

1. Inventory the original scripts, configurations, cached features, result
   files, and checkpoints associated with each paper table.
2. Record the mapping in `docs/legacy_map.md`, including which original file
   is authoritative and which files are abandoned experiments.
3. Create dataset manifests and validate their splits and source alignment.
4. Implement legacy feature readers before changing any stored feature format.
5. Consolidate common models, metrics, and training behavior into the package.
6. Reproduce one small genre experiment end to end before migrating the full
   experiment matrix.
7. Compare cleaned outputs with the paper values stored in
   `reference_results/paper_metrics.csv`.
8. Migrate the remaining paper tables and document expected deviations.
9. Archive or remove original code only after its cleaned replacement has been
   identified and validated.

## Questions that must be resolved from the original code

- Whether each reported HT-Demucs "All Stems" result uses the five named stems
  or all six HT-Demucs outputs including `other`.
- Whether legacy MusicFM caches were generated with command-line checkpoint or
  layer overrides. The cleaned reproduction consistently follows the paper's
  MusicFM-MSD layer-7 specification.
- The exact macro, micro, or weighted averaging convention for every reported
  F1, ROC-AUC, and PR-AUC value.
- The authoritative classifier dimensions, dropout settings, loss functions,
  and checkpoint-selection rule for each experiment family.
- The exact continuous-to-binary label conversion used for valence and
  arousal.
- Whether weighted late-fusion logits are summed directly or normalized by the
  total predicted weight.
- Which random seeds and number of repeated runs produced the reported values.
- Which original scripts and checkpoints correspond to the final manuscript,
  especially where filenames contain labels such as `old`, `fixed`, `newest`,
  or `wrong`.

## Current source-code situation

The existing work is distributed across multiple directories, including the
FMA HT-Demucs experiments, GTZAN, ISMIR04, MTG-Jamendo, and
Music2Emotion. Source files are currently mixed with large cached features,
checkpoints, logs, and local environments. Many scripts repeat data loading,
model definitions, metric calculation, and training loops with small changes.

Those directories are evidence for reconstructing the paper and should be
treated as read-only inputs during the first migration phase.
