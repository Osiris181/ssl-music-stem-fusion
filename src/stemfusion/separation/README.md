# Source separation

## Purpose

This package contains wrappers around external music source-separation
systems. It does not vendor their model implementations or checkpoints.

## Code

| File | Command | Purpose |
| --- | --- | --- |
| `htdemucs.py` | `stemfusion separate htdemucs` | Extract all six `htdemucs_6s` sources and resume across storage locations. |
| `manifest.py` | Used by manifest-mode separation | Validate IDs, splits, and source paths independently of single- or multi-label targets. |

The wrapper makes effective inference settings explicit, validates outputs,
supports dry runs, and writes machine-readable reports. Original research
scripts remain outside this repository until output equivalence is checked.

## How to use it

Inspect the required model and path arguments, then begin with a dry run:

    stemfusion separate htdemucs --help

Full commands and output layouts are documented in
`docs/htdemucs_separation.md`.

FMA-Medium, GTZAN, ISMIR04, MTG-Jamendo, and PMEmo also expose `python separation.py` from their
experiment directories. These thin entry scripts load dataset-specific YAML
and call this same backend. Run `python separation.py --dry-run` first.

Directory mode discovers unique input basenames. Manifest mode (`--manifest`)
uses exact per-record stem paths beneath `--output-root`, skips complete native
six-stem sets, and stages temporary input symlinks for missing targets. This
preserves distinct recordings with identical filenames. It never edits the
manifest or source audio; dry runs create no staging/output/report files.

All entry points show shared tqdm progress on stderr: directory discovery
(visited paths/audio found), existing-stem checks (existing/pending tracks),
manifest validation/completeness checks, input staging, separation batches and
individual retries. Unknown directory totals are not estimated in advance.
The batch bar advances when each subprocess batch finishes; Demucs retains its
own within-track output. Dry runs stop after checking/planning, without a
separation bar. These displays do not change inference settings or reuse rules,
and the final readable summary is printed on stdout. It lists reused/newly
separated/failed/unprocessed tracks, retry counts and the saved report path.
Dry runs instead show pending separation and write no report. The detailed
JSON report remains saved on disk for normal runs.
