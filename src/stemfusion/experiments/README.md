# Dataset experiments

Dataset-specific entry points live here. Shared data, separation, feature,
model, and training implementations stay in their corresponding packages.

[PMEmo](pmemo/README.md) provides `prepare_manifest.py` with feature/separation
purposes, `separation.py`, `extract_mert.py` and `extract_chords_keys.py`. It
preserves supplied splits, uses the same HT-Demucs backend as Jamendo and
reuses its MERT/BTC runtimes. Existing feature-reader identities and outputs
are unchanged. `train_mid.py` calls `_emotion_recognition.py` for the shared
seven conditions/five seeds and separate binary targets. Model fitting stays
in `training/music2emo.py`; binary metrics are supplied by a task adapter.
`train_late.py` calls `_emotion_late_fusion.py` for independent branches,
position-paired mixture ensembles and stem-only results. It shares the fitter
and binary metrics, reusing the completed mid-fusion mixture references without
changing the fingerprinted mid controller or its outputs.

[DEAM](deam/README.md) provides `prepare_manifest.py` with the same feature/
separation purposes, `separation.py`, `extract_mert.py`
and `extract_chords_keys.py`, preserving
the supplied song-level annotations and splits. It shares the same extraction
backends through `_emotion_features.py` and `features/emotion_dataset.py`;
the running PMEmo adapter remains unchanged. Separation uses the same shared
HT-Demucs backend and exact existing stem destinations; the fingerprinted DEAM
feature reader and extracted features remain unchanged.
`train_mid.py` uses `_emotion_mid_fusion.py` and the same PMEmo Music2Emo fitter,
binary metrics, seven conditions and five seeds, with DEAM's native >= 5 labels.

[EmoMusic](emomusic/README.md) uses these same shared implementations with
its supplied partitions, EmoMusic mixtures and matching DEAM stem folders.

[FMA-Medium](fma_medium/README.md), [GTZAN](gtzan/README.md), and
[ISMIR04](ismir04/README.md) provide dataset entry points. ISMIR04 also
audits exact duplicates and explicitly chunks long tracks during extraction.

All three datasets provide `separation.py`, `prepare_manifest.py`, `extraction.py`, `train_mid.py`,
and `train_late.py`, each supporting direct execution.

[MTG-Jamendo](mtg_jamendo/README.md) provides `prepare_manifest.py` and
`separation.py` for shared audio preparation. Manifest preparation defaults to
the complete feature mapping; `--purpose separation` selects separation jobs.
Its separation manifest preserves multi-label tags and fixed split assignments.
Its dataset directory provides `extract_mert.py` for all-layer MERT segments. The dataset-level
`extract_chords_keys.py` generates whole-track harmonic descriptors paired with
MERT. Both stages use the dataset-level feature manifest.

`mtg_jamendo/train_reproduction.py` separately implements the
five-seed original-mixture Music2Emo baseline with its own paper-budget YAML.
Its author-specific orchestration stays in `mtg_jamendo/reproduction.py`;
it does not change the completed genre trainer or its cached-run signatures.

`mtg_jamendo/train_mid.py` runs the seven controlled mid-fusion conditions,
using its YAML matrix and the same shared Music2Emo training recipe. It reuses
the unchanged mixture reproduction when compatible and saves separate
`summary_mid_fusion.{json,txt}` results. `mtg_jamendo/train_late.py` independently
trains/selects mixture and stem branches, assembles matched logit-sum/sigmoid
ensembles, and includes five standalone stem rows in
`summary_late_fusion.{json,txt}` without extra stem-only training.
Both are thin launchers, matching the genre entry-point pattern. Their common
`_mood_theme_tagging.py` controller handles arguments, scheduling, and reporting;
the shared `configuration.py` validates both YAML roles. Actual model fitting
remains in `training/music2emo.py`, not duplicated in the launchers.

The Jamendo public workflow uses the original BTC/music21 harmonic stage,
with frozen MERT embeddings and its mixture-only baseline classifier.

Shared orchestration is:

| File | Purpose |
| --- | --- |
| `_datasets.py` | Select dataset-specific manifest and cache adapters. |
| `_separation.py` | Common direct separation CLI and dataset YAML selection; uses `separation/htdemucs.py`. |
| `_extraction.py` | Common extraction CLI and handoff to `features/_musicfm.py`. |
| `_genre.py` | Common training CLI, seed/condition scheduling, reuse, and summaries. |
| `_mood_theme_tagging.py` | MERT/harmony mood/theme experiment CLI, mid/late scheduling, protected reuse, and reporting; currently used by MTG-Jamendo. |
| `_emotion_recognition.py` | MERT/harmony binary-emotion mid-fusion CLI, two-target scheduling and reporting; currently used by PMEmo with the same fitting loop. |
| `_emotion_mid_fusion.py` | Native-score emotion mid-fusion adapters and scheduling; DEAM uses its reviewed original-score boundary without modifying PMEmo's running controller. |
| `_emotion_late_fusion.py` | Binary-emotion late-fusion CLI, paired independent branch scheduling, mid-reference reuse and per-target ensemble/stem-only reporting. |
| `_mid_controls.py` | Construct exact repeated-mixture inputs and fit parameter-count/input-width matched controls through the unchanged trainer. |
| `_stem_results.py` | Common standalone-branch validation and reporting. |

All genre launchers call `_genre.py` directly; Jamendo launchers call
`_mood_theme_tagging.py` directly. No dataset-level `training.py` wrapper is
needed. FMA's `stem_results.py` retains compatibility imports for saved-branch
reporting. These shared modules are implementation details; run the
dataset entry scripts or unified CLI. Inspect their options with:

    stemfusion extract fma-medium --help
    stemfusion train fma-medium --help
    stemfusion extract gtzan --help
    stemfusion train gtzan --help
    stemfusion extract ismir04 --help
    stemfusion train ismir04 --help
