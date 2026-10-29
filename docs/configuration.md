# YAML configuration

Public extraction, training, and HT-Demucs separation commands now load YAML.
Loading configuration never starts another task: extraction does not launch
separation, and selecting a run does not launch a batch of experiments.

## Defaults and precedence

| Command | Automatically loaded preset |
| --- | --- |
| `extract fma-medium` | `configs/experiments/fma_medium/extraction.yaml`, referencing the shared MusicFM preset |
| `train fma-medium mid-fusion` | `configs/training/fma_medium_mid_fusion.yaml` |
| `train fma-medium late-fusion` | `configs/training/fma_medium_late_fusion.yaml` |
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

From the project root:

```bash
PYTHONPATH=src python -m stemfusion extract fma-medium \
  --config configs/experiments/fma_medium/extraction.yaml \
  --manifest /path/to/fma_manifest.jsonl \
  --musicfm-root /home/osiris181/musicfm \
  --dry-run
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

A dry run validates paths and reports effective settings. It neither loads
the encoder nor writes feature caches. Do not remove `--dry-run` until
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

`--run` selects one source list from `runs`. It does not run every
condition. Alternatively supply repeated `--source` arguments. With no
`--config`, training loads its default training preset and requires explicit
sources. The late-fusion command uses its own experiment file in the same way.

This adds configuration loading, not matched controls, independent branch
training, five-seed scheduling, or result aggregation. Those remain pending.

## Local paths and overrides

Keep machine-specific YAML files outside tracked presets, for example an
ignored `configs/paths.local.yaml`:

```yaml
feature_profile: experiments/fma_medium/extraction.yaml
paths:
  manifest: ../data/manifests/fma_manifest.jsonl
  musicfm_root: /home/osiris181/musicfm
  output_dir: ../data/features/fma_medium/musicfm_msd_layer7
features:
  conformer_layer_index: 7
runtime:
  device: cuda
execution:
  dry_run: true
```

This example inherits the dataset's full source list. Set `execution.dry_run`
to `false` when ready to extract, or use a local config without that setting
and control preview mode with the CLI `--dry-run` flag. A legacy `source: guitar`
field still selects one source; prefer `sources: [guitar, piano]` for a subset.

Both profile references and values under `paths` are resolved relative to
the YAML file containing them. CLI paths remain relative to the working
directory. Do not put filenames in YAML expecting shell environment-variable
expansion; it is not performed.

Extraction paths: `manifest`, `musicfm_root`, `output_dir`,
`checkpoint_path`, `stats_path`, `report`.
Training paths: `manifest`, `feature_dir`, `output_dir`.
Separation paths: `input_root`, `existing_output_root`, `output_root`,
`report`; input roots can be lists.

The nested feature, model, optimization, reproducibility, and runtime fields
in the standard presets are active settings. Use the supported CLI options
for additional per-run controls. For Boolean settings with negative options,
such as `--no-is-flash` and `--no-drop-silent-features`, the explicit
negative option also overrides YAML.

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
