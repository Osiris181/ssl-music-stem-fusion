# FMA-Medium HT-Demucs training

## Purpose

This stage provides three separate classifier workflows for the HT-Demucs
FMA-Medium genre-classification experiments:

| Paper table | Command | Classifier | Features |
| --- | --- | --- | --- |
| Table 3 | `fma-htdemucs-mid-fusion` | One MLP over concatenated source embeddings | MusicFM |
| Table 4 | `fma-htdemucs-late-fusion` | One MLP voter per source; equal logit sum | MusicFM |
| Table 4 | `fma-htdemucs-rms-weighted-late-fusion` | One MLP voter and one sigmoid RMS gate per source | MusicFM and separate RMS |

These workflows do not rerun HT-Demucs or MusicFM. They consume an aligned
FMA manifest and cached features created by the HT-Demucs extraction workflow.

## Legacy provenance

The retained scripts are:

    /home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/classifier.py
    /home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/late_fusion.py
    /home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/late_fusion_with_energy_classifier.py

`classifier.py` is the mid-fusion MLP used for Table 3. `late_fusion.py` is
the unweighted Table 4 late-fusion MLP. `late_fusion_with_energy_classifier.py`
is the RMS-weighted Table 4 late-fusion MLP. The legacy weighted script reads
1,025-dimensional caches that append energy to MusicFM; the cleaned workflow
uses separate 1,024-dimensional MusicFM caches and one-value RMS caches.

## Shared paper-aligned protocol

All three workflows use the same settings wherever the methods are
comparable:

- MusicFM-MSD, layer 7, temporal mean pooling, and 1,024 dimensions per
  source.
- Three-linear-layer source MLPs: 1,024 to 1,024 to 512 to 16, ReLU, and
  dropout 0.2.
- Cross-entropy loss, Adam, learning rate `1e-4`, batch size 128, 20 epochs,
  and seed 42.
- Select the validation checkpoint by micro accuracy plus macro ROC-AUC plus
  macro PR-AUC.
- Report micro accuracy plus macro precision, recall, F1, ROC-AUC, and PR-AUC
  on the test split.

The RMS-weighted method adds only the paper's source-specific gate: a
two-linear-layer `1 -> 16 -> 1` MLP with ReLU and sigmoid. Its RMS cache stays
separate and is never concatenated to the MusicFM vector.

Table 3 defines the 14 mid-fusion configurations in
`configs/experiments/paper/table_03_fma_htdemucs.yaml`. Table 4 defines its
three shared source combinations in
`configs/experiments/paper/table_04_fma_htdemucs.yaml`.

## Commands

Build the aligned HT-Demucs manifest and extract caches first, as described in
`docs/fma_htdemucs_features.md`.

### Table 3 mid fusion

    stemfusion train fma-htdemucs-mid-fusion \
      --manifest data/manifests/fma_htdemucs.jsonl \
      --feature-dir data/features/fma_htdemucs/musicfm \
      --source mixture \
      --source guitar \
      --source drums \
      --source piano \
      --output-dir outputs/table_03/mixture_guitar_drums_piano

The command preserves the legacy all-zero-vector exclusion. Use
`--allow-missing-features` only for a legacy-cache reproduction that requires
the old intersection behavior.

### Table 4 unweighted late fusion

    stemfusion train fma-htdemucs-late-fusion \
      --manifest data/manifests/fma_htdemucs.jsonl \
      --feature-dir data/features/fma_htdemucs/musicfm \
      --source mixture \
      --source guitar \
      --source drums \
      --source piano \
      --output-dir outputs/table_04/unweighted_mixture_guitar_drums_piano

Each selected source gets its own MLP. Their logits are summed with equal
weight before cross-entropy and metric computation.

### Table 4 RMS-weighted late fusion

    stemfusion train fma-htdemucs-rms-weighted-late-fusion \
      --manifest data/manifests/fma_htdemucs.jsonl \
      --feature-dir data/features/fma_htdemucs/musicfm \
      --energy-feature-dir data/features/fma_htdemucs/rms \
      --source mixture \
      --source guitar \
      --source drums \
      --source piano \
      --output-dir outputs/table_04/rms_weighted_mixture_guitar_drums_piano

Modern RMS cache names are `fma_rms_SOURCE_SPLIT.pkl`. For an audit using the
legacy appended-energy caches, point `--energy-feature-dir` at their directory
and add `--legacy-appended-energy`; this is an adapter only and is not the
format emitted by the cleaned extractor.

All commands support `--dry-run`, which validates the manifest and the exact
cache paths without loading PyTorch or the full feature caches.

## Outputs and validation

Every run writes a validation-selected `checkpoint.pt`, `metrics.json`,
`predictions.npz`, `training_history.json`, resolved configuration, software
versions, and the SHA-256 checksum of the input manifest. Existing artifacts
are protected unless `--overwrite` is supplied.

The scripts have synthetic unit coverage for source ordering, separate RMS
cache loading, equal and gated logit aggregation, dry runs, and output
artifacts. Full Table 3 and Table 4 legacy-cache reproductions remain to be
run and recorded before numerical equivalence is claimed.
