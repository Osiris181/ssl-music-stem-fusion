# ISMIR04 experiment settings

`manifest.yaml` configures the provided CSV/audio roots, legacy separated
stems, a separate repair location, and the generated manifest/audit paths.
It preserves existing assignments, applies the six CSV genre labels, and
removes exact decoded-audio duplicates using a fixed partition priority.

`filename_overrides` records verified CSV-to-filesystem corrections.
`verified_legacy_stems` identifies full source paths for which a colliding
stem-folder correspondence has been checked. Neither setting edits source
files. Adapt these values to your dataset copy and retain the resulting audit.

Run `python prepare_manifest.py` from the ISMIR04 experiment directory. See
[the preparation guide](../../../src/stemfusion/experiments/ismir04/README.md).

`separation.yaml` inherits the same `htdemucs_6s` inference settings as the
other datasets. Its manifest path selects retained tracks and exact output
folders, while `output_root` bounds the permitted destinations. It uses a
batch size of 32 and accepts MP3, WAV, and AU input. Reports go under
`outputs/ismir04/`; complete existing stem sets are skipped automatically.

`extraction.yaml` selects the shared MusicFM-MSD layer-7 profile and the new
`data/features/ismir04/musicfm_msd_layer7/` cache folder. It explicitly sets a
30 s chunk cap for long tracks while retaining full duration. All latent
frames are mean pooled across balanced chunks. Reports record this policy;
the FMA/GTZAN defaults remain single pass.

`mid_fusion.yaml` and `late_fusion.yaml` define the same conditions as
FMA-Medium and GTZAN. They load `training/ismir04_{mid,late}_fusion.yaml`,
which inherit the six-class/path profile `training/ismir04_genre.yaml`
and the common `training/genre.yaml`. These training paths are relative
to `configs/`. Architecture, optimization, selection, metrics, and seeds
remain shared across all three genre datasets.

Run `python train_mid.py` and `python train_late.py` from the ISMIR04
experiment directory. Use `--validate-only` to verify caches and reports
without training or writing outputs. Checkpoints go under
`outputs/ismir04/runs/`; TXT/JSON summaries go directly in `outputs/ismir04/`.
See the experiment guide for controls, checkpoint reuse, and stem-only reports.
