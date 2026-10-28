
Completed numeric comparisons are stored separately in `reference_results/`.
# Documentation

## Purpose

This directory contains component-level provenance and usage guides.

## Contents

| File | Contents |
| --- | --- |
| `htdemucs_separation.md` | Consolidated HT-Demucs configuration, storage handling, and command. |
| `fma_htdemucs_features.md` | Multi-disk FMA HT-Demucs manifest plus vanilla MusicFM and separate RMS extraction. |
| `fma_htdemucs_training.md` | Table 3 mid fusion and Table 4 unweighted and RMS-weighted late-fusion training. |
| `cleanup_20261028.md` | Retired separator cleanup, preserved shared components, and recovery archive. |
| `legacy_map.md` | Mapping from original research scripts to cleaned replacements and validation status. |

## How to use this directory

Read `legacy_map.md` first to identify the original and replacement files.
Then open the relevant component guide for commands, expected inputs and
outputs, failure behavior, and validation status.

Add a new document when a migrated component needs more explanation than its
directory README. Keep historical script paths in `legacy_map.md` so the
origin of each cleaned implementation remains traceable.
