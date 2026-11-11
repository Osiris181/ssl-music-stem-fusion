# FMA-Medium experiment configurations

| File | Active usage |
| --- | --- |
| `manifest.yaml` | Official CSV, mixture/stem roots, and generated manifest output. |
| `extraction.yaml` | Shared feature preset, FMA manifest/output paths, and six selected inputs. |
| `separation.yaml` | Dataset input/output roots; inherits the shared HT-Demucs inference preset. |
| `mid_fusion.yaml` | Seven conditions: original mixture, three repeated-mixture input controls, three source combinations. |
| `late_fusion.yaml` | Three stem ensembles and three matched mixture-only ensembles with independent branch selection. |

The corresponding entry script loads its experiment YAML automatically.
`prepare_manifest.py` and `stemfusion data fma-manifest` both load
`manifest.yaml`. Its default CSV is the local copy in
`data/manifests/fma_medium/metadata/tracks.csv`; other users must supply their
official metadata and configure their own audio roots. `genres.csv` is
reference-only. Repeated CLI `--htdemucs-root` values replace the YAML root
list. Existing manifests require explicit
`--overwrite` before replacement; dry runs check paths without writing.

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

The five standalone stem evaluations reuse saved late-fusion branch results;
they are not additional training conditions in the YAML matrix. Full
late-fusion runs report them automatically. Run
`python train_late.py --report-stems-only` from the FMA experiment directory
to generate supporting summaries from existing branches without training.

Set the MusicFM checkout path in `configs/features/musicfm_msd_layer7.yaml`.
Extraction never starts training or separation. The revised matrices exclude
`other`, Banquet, and RMS weighting.

See [the training guide](../../../docs/fma_htdemucs_training.md) and
[the dataset scripts](../../../src/stemfusion/experiments/fma_medium/README.md).
