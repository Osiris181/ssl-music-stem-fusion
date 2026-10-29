# Legacy migration map

Original scripts and cached inputs outside the project remain unchanged.
Only FMA-Medium genre-classification entry points are currently active.

| Component | Legacy source | Active implementation | Status |
| --- | --- | --- | --- |
| HT-Demucs separation | `/home/osiris181/htdemucs_separation.py`; storage continuations | `src/stemfusion/separation/htdemucs.py` | Shared wrapper; real output-equivalence check pending. |
| FMA HT-Demucs MusicFM | `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/fixed_get_musicfm_htdemucs.py` | `src/stemfusion/data/fma.py`; `experiments/fma_medium/extraction.py` | MSD/layer-7 configured; full feature-provenance audit pending. |
| FMA mid-fusion | `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/classifier.py` | `experiments/fma_medium/train_mid.py` over `training/genre.py` | Historical runner retained; revised matched controls pending. |
| FMA unweighted late-fusion | `/home/osiris181/htdemucs_experiment(FMA_HTDEMUCS)/late_fusion.py` | `experiments/fma_medium/train_late.py` over `training/late_fusion.py` | Historical joint runner retained; independent branch training pending. |

Module paths in the last two columns are relative to `src/stemfusion/`
unless written in full. Dataset entry points contain routing/configuration;
shared models and engines contain the implementation.

Old RMS/weighted modules, duplicate wrappers, and paper-table configs are
recoverable through [the FMA organization archive](fma_medium_organization.md).
Earlier Banquet/Table 2 assets are covered by
[the previous cleanup record](cleanup_20261028.md).
