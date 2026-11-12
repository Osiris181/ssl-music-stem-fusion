# YAML configuration

Public extraction, training, and HT-Demucs separation commands now load YAML.
Loading configuration never starts another task: extraction does not launch
separation, and selecting a run does not launch a batch of experiments.

## Defaults and precedence

| Command | Automatically loaded preset |
| --- | --- |
| `data fma-manifest` | `configs/experiments/fma_medium/manifest.yaml` |
| `extract fma-medium` | `configs/experiments/fma_medium/extraction.yaml`, referencing the shared MusicFM preset |
| `train fma-medium mid-fusion` | `configs/experiments/fma_medium/mid_fusion.yaml`, inheriting the shared genre profile |
| `train fma-medium late-fusion` | `configs/experiments/fma_medium/late_fusion.yaml`, inheriting the shared genre profile |
| `data gtzan-manifest` | `configs/experiments/gtzan/manifest.yaml` |
| `data ismir04-manifest` | `configs/experiments/ismir04/manifest.yaml` |
| `data mtg-jamendo-manifest` (default `--purpose features`) | `configs/experiments/mtg_jamendo/feature_manifest.yaml` |
| `data mtg-jamendo-manifest --purpose separation` | `configs/experiments/mtg_jamendo/separation_manifest.yaml` |
| `data pmemo-manifest` (default `--purpose features`) | `configs/experiments/pmemo/feature_manifest.yaml` |
| `data pmemo-manifest --purpose separation` | `configs/experiments/pmemo/separation_manifest.yaml` |
| `train pmemo mid-fusion` | `configs/experiments/pmemo/mid_fusion.yaml`, inheriting the shared emotion and Music2Emo profiles |
| `train pmemo late-fusion` | `configs/experiments/pmemo/late_fusion.yaml`, same training profiles plus independent branch conditions and the explicit mid-reference path |
| `separate htdemucs --config configs/experiments/pmemo/separation.yaml` | PMEmo paths inheriting `configs/separators/htdemucs_6s.yaml` |
| `data deam-manifest` (default `--purpose features`) | `configs/experiments/deam/feature_manifest.yaml` |
| `data deam-manifest --purpose separation` | `configs/experiments/deam/separation_manifest.yaml` |
| `train deam mid-fusion` | `configs/experiments/deam/mid_fusion.yaml`, same binary model/fitter with native 1-9 midpoint 5 and no score normalization |
| `train deam late-fusion` | `configs/experiments/deam/late_fusion.yaml`, same native labels/training profile, independent branch conditions and configured mid-reference root |
| `separate htdemucs --config configs/experiments/deam/separation.yaml` | DEAM paths inheriting the same shared HT-Demucs profile |
| `data emomusic-manifest` (default `--purpose features`) | `configs/experiments/emomusic/feature_manifest.yaml` |
| `data emomusic-manifest --purpose separation` | `configs/experiments/emomusic/separation_manifest.yaml` |
| `train emomusic mid-fusion` | `configs/experiments/emomusic/mid_fusion.yaml`, same shared binary recipe and native >= 5 labels; supplied columns unchanged, column provenance caveat unresolved |
| `train emomusic late-fusion` | `configs/experiments/emomusic/late_fusion.yaml`, same binary settings with independent matched branches, shared logit-sum evaluation and configured protected mid references |
| `separate htdemucs --config configs/experiments/emomusic/separation.yaml` | EmoMusic paths retaining the shared DEAM stem root, with the same HT-Demucs profile |
| `extract mtg-jamendo chords-keys` | `configs/experiments/mtg_jamendo/chords_keys.yaml`, paired with MERT |
| `extract mtg-jamendo mert` | `configs/experiments/mtg_jamendo/mert.yaml`, inheriting the pinned MERT profile |
| `train mtg-jamendo author-reproduction` | `configs/experiments/mtg_jamendo/reproduction.yaml`, isolated five-seed mixture-only baseline with a 200-epoch cap |
| `train mtg-jamendo mid-fusion` | `configs/experiments/mtg_jamendo/mid_fusion.yaml`, seven controlled conditions inheriting the same retained Music2Emo recipe |
| `train mtg-jamendo late-fusion` | `configs/experiments/mtg_jamendo/late_fusion.yaml`, twelve report conditions from independently selected mixture/stem branches, inheriting the same retained recipe |
| `extract gtzan` | `configs/experiments/gtzan/extraction.yaml`, referencing the same MusicFM preset |
| `extract ismir04` | `configs/experiments/ismir04/extraction.yaml`, with explicit long-track chunking |
| `train gtzan mid-fusion` | `configs/experiments/gtzan/mid_fusion.yaml`, inheriting the shared genre profile |
| `train gtzan late-fusion` | `configs/experiments/gtzan/late_fusion.yaml`, inheriting the shared genre profile |
| `train ismir04 mid-fusion` | `configs/experiments/ismir04/mid_fusion.yaml`, inheriting the shared genre profile |
| `train ismir04 late-fusion` | `configs/experiments/ismir04/late_fusion.yaml`, inheriting the shared genre profile |
| `separate htdemucs` | `configs/separators/htdemucs_6s.yaml` |

Use `--config PATH` to load another preset or an experiment file. Experiment
files can refer to `feature_profile` or `training_profile`; the current
task's referenced preset is loaded first, then experiment settings override
it. Explicit CLI arguments override those YAML values.

Jamendo uses one `prepare_manifest.py` entry point for both purposes. The
purpose is selected before YAML loading, so it determines the default preset
and its validation rules. The older installed commands
`data mtg-jamendo-feature-manifest` and `data mtg-jamendo-separation-manifest`
remain compatibility aliases with their original default purposes. Existing
manifest paths and extraction settings are unchanged.

PMEmo follows the same purpose-aware preparation flow. Its feature purpose
calls the unchanged original PMEmo builder; the separation purpose reads only
the supplied split IDs and binds exact stem destinations. Installed commands
`data pmemo-feature-manifest` and `data pmemo-separation-manifest` are aliases
with feature and separation defaults respectively. Existing features and split
assignments do not change.

DEAM uses the same flow with `data deam-manifest` and `--purpose separation`.
`data deam-feature-manifest` remains a feature-default compatibility alias,
and `data deam-separation-manifest` defaults to separation. The separation
purpose reads only the supplied split IDs; the original feature builder and
its 1-9 annotations stay unchanged. The direct script is `prepare_manifest.py`.

EmoMusic follows the same purpose-aware entry and aliases:
`data emomusic-manifest`, `data emomusic-feature-manifest` and
`data emomusic-separation-manifest`. Both purposes preserve the current
EmoMusic mixtures and same-ID DEAM stem destinations. Preparation does not
resolve the annotation-provenance question or enable training.

Each extraction entry point selects its dataset's default task YAML and uses
the shared parser in `experiments/_extraction.py`. The shared
`features/_musicfm.py` has no CLI parser.

For example, a YAML layer value of 12 with an explicit `--layer-index 7`
uses layer 7. Repeated CLI values such as `--source` or `--split` replace
the YAML list; they are not appended to it.

Checkpoint/statistics basenames come from the feature preset and are resolved
under `<musicfm_root>/data/`. Explicit `--checkpoint-path` and
`--stats-path` take precedence. Direct Python dataclass construction remains
available for programmatic use; the public CLI resolves YAML before creating
those dataclasses.

## Extraction

Jamendo's shared BTC chord/key command loads
`configs/experiments/mtg_jamendo/chords_keys.yaml`.
`execution.retry_failures: true` enables one post-extraction retry pass for
recognized tiny CQT tails; `--retry-failures` / `--no-retry-failures` override
it. The switch changes scheduling, not cache identity. The CQT adaptation is
recorded in output provenance, and verified successful pre-retry caches remain
reusable without being rewritten. It does not retry unavailable key estimates
or unrelated failures. See the
[shared guide](../src/stemfusion/experiments/mtg_jamendo/README.md#chordkey-extraction).

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

ISMIR04 uses this same runtime and explicitly sets `audio.chunk_seconds: 30`
while retaining `audio.max_seconds: null`. This covers the full recording
with balanced chunks and pools by latent-frame count. Encoder context resets
between chunks; reports record the policy. The shared default remains null
for FMA/GTZAN. Use `--chunk-seconds N` or `--no-chunking` for explicit overrides.

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
the YAML seed list. GTZAN uses `configs/training/gtzan_genre.yaml` for its
ten-class target and paths. ISMIR04 uses `configs/training/ismir04_genre.yaml`
for six classes and its local paths. All three dataset profiles inherit architecture,
optimization, evaluation, and seed settings from `configs/training/genre.yaml`.
Training uses the agreed named conditions rather than
arbitrary `--source` lists. Source-subset arguments remain available for extraction.

The FMA implementation includes repeated-mixture input/parameter-count controls,
independently selected late-fusion branches, matched mixture ensembles,
and paired five-seed summaries. Use `--validate-only` to check all cached
features without training. See [the training guide](fma_htdemucs_training.md).

## Local paths and overrides

Jamendo mid and late fusion support `runtime.verify_checksums: true`
in their respective dataset YAMLs. `--no-verify-checksums` trusts declared feature completion
markers instead of hashing feature payloads; `--verify-checksums` restores the
default. Both override YAML. Structural/alignment checks and trained-artifact
integrity checks remain mandatory. This option does not affect the retained
reproduction. The reviewed loading-only transition preserves completed
checkpoint compatibility; actual execution-source identity is recorded
separately in new fusion summaries. Both Jamendo fusion commands now use this
shared YAML loader and the task-level `_mood_theme_tagging.py` controller, with the
same thin-entry-point flow as genre experiments. See the
[Jamendo commands and integrity tradeoff](../src/stemfusion/experiments/mtg_jamendo/README.md#commands-and-checkpoint-reuse).

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

FMA's `configs/experiments/fma_medium/manifest.yaml` defines `tracks_csv`,
`mixture_root`, a list of `htdemucs_root` directories, and `output`.
The default CSV is the official local copy in
`data/manifests/fma_medium/metadata/tracks.csv`. The preset checks configured
audio-file availability to build a common track set. Use `--no-allow-missing`
for strict availability checks. Both FMA manifest entry points load this YAML;
CLI arguments override it. Existing manifests require `--overwrite`.

GTZAN's `configs/experiments/gtzan/manifest.yaml` defines `train_split`,
`validation_split`, `test_split`, `mixture_root`, `htdemucs_root`, and `output`.
It binds the provided split IDs to local paths and checks configured file
availability by default, retaining a common track set across all sources.
Use `--no-allow-missing` for strict checks, or combine `--no-check-audio` and
`--no-allow-missing` to bind paths without file checks. The generated manifest
and its sidecars remain ignored by Git. See the
[GTZAN guide](../src/stemfusion/experiments/gtzan/README.md).

ISMIR04 preparation uses `configs/experiments/ismir04/manifest.yaml` for its
metadata/audio roots, audit path, duplicate priority, verified filename
corrections, and stem-repair locations. It retains the provided partition
assignments and records exact-duplicate exclusions instead of generating a
new split. See the
[ISMIR04 preparation guide](../src/stemfusion/experiments/ismir04/README.md).

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
`manifest`, `report`; input roots can be lists. Each experiment folder's
`separation.py` loads `configs/experiments/<dataset>/separation.yaml`, inheriting
the common HT-Demucs inference profile. When `manifest` is supplied, its exact
stem destinations beneath `output_root` are used and directory discovery is
skipped. Omit `existing_output_root` in manifest mode; the manifest already
identifies each retained track's stem location.

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
