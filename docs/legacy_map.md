# Legacy migration map

Original research files remain unchanged until their replacements have been
validated against the paper experiments.

| Pipeline component | Authoritative legacy source | Clean replacement | Status |
| --- | --- | --- | --- |
| HT-Demucs batch separation | /home/osiris181/htdemucs_separation.py; SSD and new-HDD files are storage continuations; ISMIR04 changes only inputs and scheduling | src/stemfusion/separation/htdemucs.py | Migrated as one method; lightweight tests added; output equivalence still pending |
| FMA-Medium HT-Demucs MusicFM and RMS extraction | /home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/fixed_get_musicfm_htdemucs.py; /home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/get_musicfm_htdemucs_add_rms_features.py | src/stemfusion/data/fma.py; src/stemfusion/features/htdemucs.py | Migrated to the paper-aligned MSD/layer-7 profile; real-cache equivalence still pending |
| FMA-Medium HT-Demucs Table 3 and Table 4 genre training | `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/classifier.py`; `late_fusion.py`; `late_fusion_with_energy_classifier.py` | `src/stemfusion/training/htdemucs_mid_fusion.py`; `htdemucs_late_fusion.py`; `htdemucs_rms_weighted_late_fusion.py` | Migrated with modern separate RMS caches and a legacy appended-energy adapter; metric equivalence pending |

`src/stemfusion/training/genre.py` and `src/stemfusion/models/mlp.py` remain
shared HT-Demucs dependencies. The retired separator's migration history and
Table 2 reproduction artifacts are recoverable as recorded in
[cleanup_20261028.md](cleanup_20261028.md).
