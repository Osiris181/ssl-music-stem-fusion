# Stem Fusion for Music Classification

Research code for studying whether source-separated instrument representations
add predictive value beyond strong pretrained representations of the original
music mixture. This repository supports the revision of *Enhancing Music
Classification via Source Separation and Feature Fusion*. The Python package
and command-line interface are named `stemfusion`.

## Current status

As of 2026-10-28, repository cleanup is in progress. The active package contains
HT-Demucs separation, FMA-Medium manifest construction, MusicFM extraction,
and historical genre-classification workflows.

Banquet code, configurations, dedicated documentation, and Table 2 experiment
outputs have been removed from the active project with a recovery archive
outside it. See [the cleanup record](docs/cleanup_20261028.md).
RMS extraction, weighted fusion, old table-based configurations, and
superseded entry points are now archived outside the active project as well;
see [the FMA organization record](docs/fma_medium_organization.md).

FMA-Medium genre classification is the only active dataset experiment.
Start with [its dataset guide](src/stemfusion/experiments/fma_medium/README.md).
Shared separation, MusicFM, model, and training code remains available to
support that experiment.

No previous run artifacts remain in the active project. The obsolete
`reference_results/` directory has been removed from this project; `outputs/`
is currently empty and reserved for future runs. Reusable feature caches and
split manifests are inputs, rather than previous experimental results.

The matched mixture-only controls, five-seed evaluation, and revised results
are **not yet implemented or completed**. Existing code and historical results
must not be presented as evidence from the revised protocol.

The project context is recorded in:

- [Research direction](RESEARCH_DIRECTION.md): scientific question, prior-work
  positioning, historical setup, and interpretation limits.
- [Phase 1: novelty revision](PHASE1_NOVELTY_REVISION.md): revised motivation,
  empirical objectives, and manuscript drafting notes.
- [Phase 2: experimental design](PHASE2_EXPERIMENTAL_DESIGN.md): the current,
  provisionally agreed comparison matrix and implementation checks.

Phase 2 takes precedence over earlier experimental proposals. The cleanup
record documents repository changes made after those design notes.

## Research question

When pretrained music models already provide informative representations of
the original mixture, does explicitly adding representations of separated
instrument stems improve downstream classification? Under which evaluated
conditions does it help, make little difference, or hurt?

The study evaluates incremental predictive value while controlling for the
extra classifier parameters and ensemble branches introduced by fusion.
Source separation may make source-related cues easier to exploit; it does
not create acoustic information absent from the original recording.

The three empirical objectives are to assess incremental stem value against
matched mixture-only controls, examine a fixed sequence of stem additions,
and characterize outcomes across genre, mood/theme, and binary emotion
settings. These are proposed empirical contributions. Separation, stem fusion,
and combining separation with pretrained representations have prior precedent;
the project makes no general first-use claim.

## Revised pipeline and scope

The original mixture remains an independent input. Frozen pretrained
HT-Demucs `htdemucs_6s` supplies separated audio, and frozen representation
extractors encode the mixture and selected stems separately. Only downstream
classifiers are trained.

HT-Demucs produces six sources natively. Revised comparisons use guitar,
drums, piano, bass, and vocals, excluding `other`. The mixture input must
come from the original recording, rather than a reconstruction from stems.

Two representation pipelines are retained:

- MusicFM-MSD: seventh Conformer-layer features with temporal mean pooling.
- Kang & Herremans-style features: frozen MERT-95M layers five/six plus
  reconstructed chord and global-key descriptors.

The inspected FMA piano training cache contains 1,024-dimensional
`torch.float32` vectors. This confirms the width for that cache; feature
dimensions, checkpoint/layer provenance, pooling, and track alignment still
need verification across the remaining inputs. MERT plus chord/key dimensions
must be measured separately.

## Planned comparisons

Keep the original single-MLP mixture-only baseline. Evaluate both
feature-concatenation mid-fusion and unweighted prediction-level late-fusion
for the following fixed nested configurations.

Here `M` is the original mixture, `G` guitar, `D` drums, `P` piano,
`B` bass, and `V` vocals.

| Fusion inputs | Input views | Mixture-only control for mid-fusion | Mixture-only control for late-fusion |
| --- | ---: | --- | --- |
| `M+G+D+P` | 4 | MLP matched to the fusion model's trainable parameter count | 4 independently trained mixture classifiers |
| `M+G+D+P+B` | 5 | MLP matched to the fusion model's trainable parameter count | 5 independently trained mixture classifiers |
| `M+G+D+P+B+V` (`M+All Stems`) | 6 | MLP matched to the fusion model's trainable parameter count | 6 independently trained mixture classifiers |

For mid-fusion, the planned control widens a mixture-only MLP while retaining
the original mixture representation. Compute matched widths from the actual
architecture and report full trainable parameter counts. Repeating the same
mixture vector is not the intended capacity control.

For late-fusion, all control branches receive the same mixture features but
have independent initialization and training randomness. Use the same branch
architectures, validation selection, aggregation rule, and training budget
for stem fusion and its corresponding mixture ensemble.

Parameter matching does not equalize optimization difficulty, inductive bias,
or computation. Fair comparisons also require consistent splits,
preprocessing, losses, regularization, metrics, and training settings within
each dataset, representation, and target.

These three configurations examine adding bass, then vocals, to one selected
starting group. They are representative nested sets, not an exhaustive search
or evidence of globally optimal stem combinations. Mid- and late-fusion are
evaluation strategies, not claimed new algorithms.

## Planned datasets and representations

| Dataset | Task / target | Representation pipeline |
| --- | --- | --- |
| FMA-Medium | 16-class genre classification | MusicFM |
| GTZAN | 10-class genre classification | MusicFM |
| ISMIR04 | 6-class genre classification | MusicFM |
| MTG-Jamendo mood/theme | Multilabel tagging; 56 tags in the original manuscript | MusicFM and MERT + chord/key, evaluated separately |
| PMEmo | Binary valence and binary arousal, evaluated separately | MERT + chord/key |
| EmoMusic | Binary valence and binary arousal, evaluated separately | MERT + chord/key |
| DEAM | Binary valence and binary arousal, evaluated separately | MERT + chord/key |

This gives 11 dataset × representation × target settings in the full proposed
matrix. The emotion tasks binarize annotations; they are not regression
experiments. Exact label conversion, sample IDs, and split provenance require
verification before rerunning.

## Planned training and reporting protocol

Retain existing train/validation/test partitions, subject to checks for
duplicate IDs, overlap, and feature alignment. Fix five outer training seeds
before new runs; the actual seed list is still open. Derive independent
late-fusion branch seeds deterministically from each outer seed.

The provisionally agreed rules retain the original task-specific training
budgets:

| Task | Optimizer and training budget | Validation checkpoint criterion | Primary reported metric |
| --- | --- | --- | --- |
| Genre | Adam, learning rate `1e-4`, batch size 128, up to 20 epochs | Maximum accuracy + PR-AUC + ROC-AUC, all on a 0–1 scale | Accuracy |
| Mood/theme | Adam, learning rate `1e-4`, batch size 8, up to 500 epochs; patience 10, minimum delta `1e-4` | Maximum PR-AUC | PR-AUC |
| Binary valence/arousal | Adam, learning rate `1e-4`, batch size 8, up to 500 epochs; patience 10, minimum delta `1e-4` | Maximum accuracy, provisionally pending label-balance/code checks | Accuracy |

Exact metric averaging, PR-AUC computation, threshold selection, tie handling,
and undefined-AUC handling must be fixed before evaluation. Choose thresholds
using validation data only. Apply the same selection criterion to every
compared model within a setting; historical checkpoints selected differently
are not automatically reusable.

Report mean ± sample standard deviation (`ddof=1`) over five runs, plus the
mean of paired per-seed differences between fusion and its matched control.
Retain appropriate secondary metrics. Formal significance tests and bootstrap
confidence intervals are optional; standard deviation alone does not
establish statistical significance.

Save per-track IDs, labels, scores/probabilities, predictions, validation-chosen
thresholds, configs, seeds, trainable parameter counts, checkpoint metadata,
and validation histories. Keep branch outputs so nested late-fusion ensembles
can be evaluated without retraining compatible branches.

## Interpretation and historical limitations

The selected separator and stem configurations were influenced by earlier
FMA-Medium test results. Fixing them for revised runs does not make that
previously examined test split an independent confirmatory holdout. Preserve
and disclose this exploratory history.

Task comparisons also differ in datasets, labels, and representation
pipelines. Treat explanations such as instrument-localized genre cues versus
holistic emotion cues as hypotheses. Report positive, negligible, and negative
outcomes without asserting a universal task-type rule.

The Kang & Herremans chord/key stages were reconstructed because the original
extraction scripts were unavailable. The original manuscript documents a
reproduction gap. Compare stem fusion against the local mixture-only baseline
under the same implementation, and distinguish it from published results;
the reasons for the gap remain possible explanations rather than proven causes.

## Current repository layout

The tree below shows the current main files and directories, rather than a
proposed future package structure.

```text
stem-fusion-music-classification/
├── README.md
├── RESEARCH_DIRECTION.md
├── PHASE1_NOVELTY_REVISION.md
├── PHASE2_EXPERIMENTAL_DESIGN.md
├── STEM_Separation.pdf
├── pyproject.toml
├── configs/
│   ├── separators/
│   ├── features/
│   ├── training/
│   └── experiments/fma_medium/
│       ├── extraction.yaml
│       ├── mid_fusion.yaml
│       └── late_fusion.yaml
├── src/stemfusion/
│   ├── cli.py
│   ├── configuration.py
│   ├── data/
│   ├── separation/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── experiments/fma_medium/
│       ├── extraction.py
│       ├── train_mid.py
│       └── train_late.py
├── tests/unit/
├── docs/
├── data/manifests/
└── outputs/
```

Directory guides explain their code, inputs, outputs, and usage. Start with
[configs](configs/README.md), [source code](src/README.md),
[documentation](docs/README.md), or [tests](tests/README.md).

Public tasks now load their standard YAML presets automatically. Use
`--config PATH` to load an experiment or alternate preset; explicit CLI
arguments override YAML values. For training, `--run NAME` selects one source
condition from the experiment file. See [configuration usage](docs/configuration.md).
The [FMA-Medium configuration directory](configs/experiments/fma_medium/README.md)
contains the selected input matrices and references retained training presets.
These matrices do not implement the pending matched controls or five-seed
evaluation. Old paper-table manifests are archived.

The retained late-fusion runner trains all source branches jointly through
summed logits and selects one whole-model checkpoint. Independent branch
training and checkpoint selection required by Phase 2 remain to be implemented.
Mid-fusion's retained zero-vector exclusion also requires a common-track-set
audit before new comparisons.

## Using the current code

From the repository root, inspect the supported commands without installing
the package:

```bash
PYTHONPATH=src python -m stemfusion --help
PYTHONPATH=src python -m stemfusion separate htdemucs --help
PYTHONPATH=src python -m stemfusion data fma-manifest --help
PYTHONPATH=src python -m stemfusion extract fma-medium --help
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion --help
PYTHONPATH=src python -m stemfusion train fma-medium late-fusion --help
```

Python 3.10 or newer is required. Editable installation is available with
`python -m pip install -e .`; optional dependency groups in
[pyproject.toml](pyproject.toml) are `htdemucs`, `musicfm`, and `training`.
The upstream MusicFM repository and checkpoints remain external dependencies.

See the [separation guide](docs/htdemucs_separation.md),
[feature-extraction guide](docs/fma_medium_extraction.md), and
[training guide](docs/fma_htdemucs_training.md) for complete commands and local
path arguments. These guides describe the retained runners in their new
dataset-specific organization.
There are no standalone `evaluate` or `reproduce` CLI commands.

Run the existing tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests use temporary synthetic inputs and mocked external inference.
They do not establish real-model feature equivalence or revised research
findings.

## Data, artifacts, and next implementation steps

Keep raw audio, separated stems, feature caches, checkpoints, training logs,
and local environments out of source control. Supply machine-specific paths
through CLI arguments or ignored local YAML overrides. Reuse existing features
after verifying their
provenance and alignment rather than automatically rerunning separation or
pretrained extraction.

Current FMA trainers write run artifacts under a chosen output directory,
including `resolved_config.json`, `environment.json`,
`data_manifest_checksum.txt`, `training_history.json`, `metrics.json`,
`predictions.npz`, and `checkpoint.pt`. Revised runs should use distinct,
versioned output directories under `outputs/`. Historical artifacts remain in
the external recovery archive described in the cleanup record.

The remaining work is:

1. Audit existing features, split manifests, labels, metrics, actual MLP
   parameters, and late-fusion aggregation. Verify ISMIR04 split provenance
   without inventing an official split or citation.
2. Lock the five seeds and precise per-setting protocol before examining new
   test outcomes.
3. Implement matched controls and consistent prediction/configuration saving,
   preserving the original mixture-only reference.
4. Complete migration of the remaining dataset and representation pipelines,
   validate a small pilot, then run the agreed repeated comparisons.
5. Analyze matched differences and limitations, and revise results-dependent
   manuscript claims after the evidence is available.

See [the legacy migration map](docs/legacy_map.md) for existing source mappings.
Further code changes, experiments, and manuscript edits proceed through
specific author requests.
