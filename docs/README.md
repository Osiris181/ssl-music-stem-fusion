
# Documentation

## Purpose

This directory contains component-level provenance and usage guides.

Previous result artifacts are no longer kept in the active project. Future
run metrics, predictions, and checkpoints belong under `outputs/`, accompanied
by their resolved configurations and validation-selection metadata.

## Contents

| File | Contents |
| --- | --- |
| `configuration.md` | Active YAML loading, defaults/overrides, named runs, local paths, validation, and provenance. |
| `htdemucs_separation.md` | Consolidated HT-Demucs configuration, storage handling, and command. |
| `fma_medium_extraction.md` | Multi-disk FMA HT-Demucs manifest and MusicFM extraction. |
| `fma_htdemucs_training.md` | Active FMA-Medium mid-/late-fusion commands, retained behavior, and pending protocol changes. |
| `fma_medium_organization.md` | Dataset organization, archived legacy code, command changes, and recovery paths. |
| `cleanup_20261028.md` | Retired separator cleanup, preserved shared components, and recovery archive. |
| `legacy_map.md` | Mapping from original research scripts to cleaned replacements and validation status. |

## How to use this directory

Read `legacy_map.md` first to identify the original and replacement files.
Then open the relevant component guide for commands, expected inputs and
outputs, failure behavior, and validation status.

Add a new document when a migrated component needs more explanation than its
directory README. Keep historical script paths in `legacy_map.md` so the
origin of each cleaned implementation remains traceable.
