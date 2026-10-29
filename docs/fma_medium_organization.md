# FMA-Medium organization and legacy archive — 2026-10-28

The author approved organizing the active experiments by dataset and
archiving older code to avoid mixing it with the revised work.

## Active code

- `src/stemfusion/experiments/fma_medium/`: public mid-/late-fusion entry points.
- `configs/experiments/fma_medium/`: selected source matrices.
- `src/stemfusion/data/`, `separation/`, `features/`, `models/`, and
  `training/`: retained shared implementations required by FMA-Medium.

No other dataset implementations had been migrated into this package.
MusicFM extraction and HT-Demucs separation are shared preprocessing, not
separate non-FMA experiments.

## External archive

A snapshot of the project immediately before reorganization, including the
previous documentation edits, is available at:

```text
/home/osiris181/stem-fusion-fma-archive-20261028-Ul7bMH/project-before-fma-organization.tar.gz
```

Files removed from the active project are also available under
`/home/osiris181/stem-fusion-fma-archive-20261028-Ul7bMH/legacy/`, preserving
their former relative paths:

- RMS extraction implementation, its preset, and dedicated tests.
- Weighted late-fusion entry point and training preset.
- Superseded table-specific training entry points and combined late-fusion engine.
- Old table-based experiment matrices and duplicate generic genre-training preset.
- Old mixed weighted/unweighted tests, replaced by tests of the retained paths.

The weighted model definition and original guides remain in the snapshot.
The active late-fusion model and engine now contain only the unweighted path.
Superseded bytecode caches are removed from the active tree.

## Command changes

| Previous command | Active command |
| --- | --- |
| `train fma-genre` / `train fma-htdemucs-mid-fusion` | `train fma-medium mid-fusion` |
| `train fma-htdemucs-late-fusion` | `train fma-medium late-fusion` |
| `train fma-htdemucs-rms-weighted-late-fusion` | Archived; no active replacement |
| `extract htdemucs rms` | Archived; no active replacement |

## Scientific status

Unweighted model architecture, optimization settings, validation selection,
and existing feature handling are preserved by this organization step.
The historical late runner trains source branches jointly through summed
logits; implementing independent branch training and per-branch selection is
still required by Phase 2. Mid-fusion's retained zero-vector exclusion also
needs a common-track-set audit.

Matched controls and five-seed evaluation remain unimplemented. No real-data
training, separation, or feature extraction was launched. Input data,
existing feature caches, split manifests, manuscript, and scientific context
documents were preserved. The older Banquet/results archive is unchanged.

## Verification

- All 28 active unit tests passed, including synthetic mid-/late-fusion runs
  and checks that archived commands are unavailable.
- All Markdown links and 17 documented help-command variants were checked.
- The archived and retained unweighted runners produced identical metrics,
  predictions, and checkpoint tensors on the same seeded synthetic inputs.
- Eleven retained core/input/context files, both training preset contents
  (apart from title comments), the retained unweighted model architecture,
  and all 14 archived legacy files matched the pre-change snapshot.
- No real-data experiments were run; `outputs/` remains empty.

## Subsequent extraction-task naming

The extraction entry point is now grouped with the FMA-Medium training tasks:
`src/stemfusion/experiments/fma_medium/extraction.py`. Its public command is
`stemfusion extract fma-medium`; this replaces `extract htdemucs musicfm`.
The MusicFM implementation remains shared in `features/_musicfm.py`, and
extraction settings, cache format, and failure behavior are unchanged.

The corresponding guide is `docs/fma_medium_extraction.md`, and the dataset
configuration directory includes `extraction.yaml`. No extraction was started
as part of this naming change.

## Subsequent active YAML configuration

`src/stemfusion/configuration.py` now loads task presets and referenced
profiles, validates supported fields, applies explicit CLI overrides, and
records configuration provenance. The extraction, training, and separator
YAML files are active inputs to their public commands; earlier descriptions
of documentation-only YAML refer to the pre-loader state.

See [configuration usage](configuration.md). Adding this loader does not
implement new scientific comparisons or start real-data processing.

## Subsequent one-command extraction

Extraction now consumes the YAML `sources` list in one invocation, with
repeated `--source` options available for subsets. The shared extraction
helper reuses one encoder, retains per-source caches, and stops after a failed
source. Multi-source jobs also produce an aggregate report. Scientific
settings and track selection are unchanged; no real extraction was launched
during implementation.
