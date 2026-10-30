# YAML configuration

Public extraction, training, and HT-Demucs separation commands now load YAML.
Loading configuration never starts another task: extraction does not launch
separation, and selecting a run does not launch a batch of experiments.

## Defaults and precedence

| Command | Automatically loaded preset |
| --- | --- |
| `extract fma-medium` | `configs/experiments/fma_medium/extraction.yaml`, referencing the shared MusicFM preset |
| `train fma-medium mid-fusion` | `configs/experiments/fma_medium/mid_fusion.yaml`, inheriting the shared genre profile |
| `train fma-medium late-fusion` | `configs/experiments/fma_medium/late_fusion.yaml`, inheriting the shared genre profile |
| `separate htdemucs` | `configs/separators/htdemucs_6s.yaml` |

Use `--config PATH` to load another preset or an experiment file. Experiment
files can refer to `feature_profile` or `training_profile`; the current
task's referenced preset is loaded first, then experiment settings override
it. Explicit CLI arguments override those YAML values.

The FMA extraction entry point owns its argument parser and default task-YAML
selection. The shared `features/_musicfm.py` has no CLI parser. A future
dataset entry point can select its own preset and argument interface.

For example, a YAML layer value of 12 with an explicit `--layer-index 7`
uses layer 7. Repeated CLI values such as `--source` or `--split` replace
the YAML list; they are not appended to it.

Checkpoint/statistics basenames come from the feature preset and are resolved
under `<musicfm_root>/data/`. Explicit `--checkpoint-path` and
`--stats-path` take precedence. Direct Python dataclass construction remains
available for programmatic use; the public CLI resolves YAML before creating
those dataclasses.

## Extraction

Set `paths.musicfm_root` in `configs/features/musicfm_msd_layer7.yaml` and
`paths.manifest`/`paths.output_dir` in
`configs/experiments/fma_medium/extraction.yaml`. From the project root:

```bash
python src/stemfusion/experiments/fma_medium/extraction.py --dry-run
```

This experiment file references the MSD/layer-7 feature preset and provides
the fresh output directory `data/features/fma_medium/musicfm_msd_layer7/`.
Without `--source`, extraction processes the YAML `sources` list in order:
mixture, guitar, drums, piano, bass, and vocals. One encoder is reused across
pending sources. Use `--source guitar` for one source, or repeat it for a
subset, for example `--source guitar --source piano`. Explicit sources replace
the YAML list. Use `--output-dir` to override the destination.

Complete split caches are skipped when rerunning without `--overwrite`.
Each source retains its own cache/report files. Multi-source runs also write
`extraction_report_musicfm_batch.json`; a custom `--report` names that aggregate
report. Extraction stops after a source reports failures, preserving its
partial cache and listing the sources not processed yet.

A dry run validates configuration and local manifest/model-file paths. It
neither reads the audio nor loads the encoder or writes feature caches. Do not remove `--dry-run` until
the audio manifest and local paths have been checked.

## One named training condition

```bash
PYTHONPATH=src python -m stemfusion train fma-medium mid-fusion \
  --config configs/experiments/fma_medium/mid_fusion.yaml \
  --run mixture_guitar_drums_piano \
  --manifest /path/to/fma_manifest.jsonl \
  --feature-dir /path/to/musicfm_msd_layer7 \
  --output-dir outputs/fma_medium/mid_fusion/example \
  --dry-run
```

`--run` selects one named condition, including its control/fusion role.
Without `--run`, training runs the entire matrix in the default experiment
file. Both methods inherit `configs/training/fma_medium_genre.yaml` for common
paths, hyperparameters, metric definitions, and outer seeds
`[32, 64, 128, 256, 512]`. Repeated `--seed` selects a subset and replaces
the YAML seed list. Training uses the agreed named conditions rather than
arbitrary `--source` lists. Source-subset arguments remain available for extraction.

The FMA implementation includes capacity-matched mixture controls,
independently selected late-fusion branches, matched mixture ensembles,
and paired five-seed summaries. Use `--validate-only` to check all cached
features without training. See [the training guide](fma_htdemucs_training.md).

## Local paths and overrides

The shared MusicFM preset contains the checkout path:

```yaml
paths:
  musicfm_root: /home/osiris181/musicfm
```

Change this value for another machine. The FMA task YAML contains its manifest
and output paths; those stay separate from the shared encoder settings so
future datasets can reuse the MusicFM profile. A separate `paths.local.yaml`
is no longer needed. Explicit `--musicfm-root`, `--manifest`, and `--output-dir`
arguments still override the corresponding configured paths.

A legacy `source: guitar` field still selects one source; prefer
`sources: [guitar, piano]` for a subset.

Both profile references and values under `paths` are resolved relative to
the YAML file containing them. CLI paths remain relative to the working
directory. Do not put filenames in YAML expecting shell environment-variable
expansion; it is not performed.

Extraction paths: `manifest`, `musicfm_root`, `output_dir`,
`checkpoint_path`, `stats_path`, `report`.
Training paths: `manifest`, `feature_dir`, `output_dir`, `summary_dir`.
The summary directory can be separate from checkpoints and prediction arrays.
An explicit `--output-dir` redirects summaries as well unless an explicit
`--summary-dir` is given.
Separation paths: `input_root`, `existing_output_root`, `output_root`,
`report`; input roots can be lists.

The nested feature, model, optimization, reproducibility, and runtime fields
in the standard presets are active settings. Use the supported CLI options
for additional per-run controls. The extraction option `--no-is-flash`
overrides YAML. Training always retains valid zero vectors; missing-feature
intersections and method-specific silent-vector removal are not supported by
the revised protocol.

## Validation and provenance

Unknown keys, duplicate keys, missing required settings, invalid types,
cyclic profile references, and unsupported methods fail before processing.
Fixed conventions such as ReLU, Adam/cross-entropy, temporal mean pooling,
source order, and metric averaging are validated against the current
implementation. Editing YAML does not implement a new architecture, loss,
or metric. Those changes require corresponding code.

Ancillary profile references describe already-produced inputs: a training
file's `feature_profile` and an extraction file's `separator_profile`
are recorded as context, not executed as additional tasks. Fields such as
dataset labels, implementation status, selected-view lists, and historical
ISMIR04 annotations are descriptive metadata.

Training saves actual effective values and configuration provenance in
`resolved_config.json`; extraction/separation reports include
`resolved_config`. Provenance records configuration filenames, SHA-256
hashes, selected run, and explicit CLI overrides. This preserves the settings
used even if the YAML files change later. A snapshot of merged YAML settings
before CLI overrides is kept alongside the actual resolved configuration.

The canonical presets are also included in package installation data, so
the same loader works outside a source checkout.
