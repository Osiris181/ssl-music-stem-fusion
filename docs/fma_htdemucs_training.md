# FMA-Medium HT-Demucs training

The active public workflows are organized under
[src/stemfusion/experiments/fma_medium](../src/stemfusion/experiments/fma_medium/README.md).

| Command | Retained behavior |
| --- | --- |
| `stemfusion train fma-medium mid-fusion` | Train one MLP on a single source or concatenated sources. |
| `stemfusion train fma-medium late-fusion` | Jointly train source MLPs through their equal-sum logits. |

Both consume an aligned FMA manifest and cached MusicFM vectors; neither
reruns separation or extraction. See [the feature guide](fma_medium_extraction.md).

## Configuration and usage

Source matrices are in
[configs/experiments/fma_medium](../configs/experiments/fma_medium/README.md).
Select the original mixture alone for the reference, or the three agreed
mixture-plus-stem configurations. `other` is excluded from those matrices.

Inspect complete arguments with:

```bash
stemfusion train fma-medium mid-fusion --help
stemfusion train fma-medium late-fusion --help
```

The matching training preset is loaded automatically. Use `--config` with
an experiment YAML and `--run` for a named source condition, or supply repeated
`--source` arguments. Supply manifest/feature/output paths through CLI or
YAML `paths`. See [configuration usage](configuration.md).
Add `--dry-run` to validate paths without loading
the feature contents or training. Full examples and code explanations are in
[the dataset guide](../src/stemfusion/experiments/fma_medium/README.md).

## Preserved settings and pending work

The retained defaults are 1024-dimensional source features; hidden widths
1024/512; ReLU; dropout 0.2; cross-entropy; Adam at `1e-4`; batch size 128;
20 epochs; and seed 42. Checkpoint selection maximizes validation accuracy +
macro ROC-AUC + macro PR-AUC. Per-run outputs include the checkpoint, metrics,
predictions, resolved config, history, software versions, and manifest checksum.

Mid-fusion retains legacy all-zero-vector exclusion. The late runner selects
one whole-model checkpoint, rather than independently trained branch checkpoints.
These behaviors need a fairness/protocol audit before new comparisons.

Matched-capacity mixture MLPs, independent mixture ensembles, independent
branch selection, and five-seed reporting are not implemented yet. The old
weighted workflow and table-specific commands are archived; see
[the organization record](fma_medium_organization.md). No real-data experiment
was run during cleanup.
