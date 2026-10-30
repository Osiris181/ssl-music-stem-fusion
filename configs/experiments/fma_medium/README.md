# FMA-Medium experiment configurations

| File | Active usage |
| --- | --- |
| `extraction.yaml` | Shared feature preset, FMA manifest/output paths, and six selected inputs. |
| `mid_fusion.yaml` | Seven conditions: original mixture, three capacity-matched mixture controls, three source combinations. |
| `late_fusion.yaml` | Three stem ensembles and three matched mixture-only ensembles with independent branch selection. |

The corresponding entry script loads its experiment YAML automatically.
Both training matrices inherit common settings from
`configs/training/fma_medium_genre.yaml`, including outer seeds
32, 64, 128, 256, 512 and manifest/feature/output paths.

Without `--run`, training executes every condition. Use `--run NAME` to
select one condition, `--seed 32` for one outer seed, or repeated `--seed`
for a subset. CLI values override YAML. All conditions require the common
six-source track set, even when the classifier only uses the mixture.

Use `--dry-run` to inspect paths, conditions, seeds, and parameter counts.
Use `--validate-only` to check cached vectors and extraction reports.
Both avoid training. Compatible selected branches are shared between workflows
automatically, and completed runs are reused with verified checksums.
`--resume` remains accepted; `--overwrite` explicitly requests retraining.

Set the MusicFM checkout path in `configs/features/musicfm_msd_layer7.yaml`.
Extraction never starts training or separation. The revised matrices exclude
`other`, Banquet, and RMS weighting.

See [the training guide](../../../docs/fma_htdemucs_training.md) and
[the dataset scripts](../../../src/stemfusion/experiments/fma_medium/README.md).
