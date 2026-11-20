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
`train_late.py` calls `_emotion_native_late_fusion.py` for the same independent
branch/ensemble schedule as PMEmo, reusing DEAM mid references and reporting
all five standalone stems through the unchanged shared binary evaluator.

[EmoMusic](emomusic/README.md) uses these same shared implementations with
its supplied partitions, EmoMusic mixtures and matching DEAM stem folders.
`prepare_manifest.py` combines feature/separation purposes, while
`separation.py` calls the same shared HT-Demucs backend. The feature reader,
existing caches and intentionally shared stem paths remain unchanged.
`train_mid.py` calls `_emomusic_mid_fusion.py` with DEAM/PMEmo-equivalent
loading/scheduling and the same shared model/fitter/metrics. It retains native
scores, applies high >= 5 per supplied target column, and uses seven conditions
and five seeds. The annotation-column provenance caveat remains documented
in the dataset guide; no automatic column exchange is applied.
`train_late.py` calls `_emomusic_late_fusion.py` for the same DEAM/PMEmo
independent branch schedule and shared binary ensemble evaluator, reusing
mid mixture-only references and reporting all five standalone stems.

[FMA-Medium](fma_medium/README.md), [GTZAN](gtzan/README.md), and
[ISMIR04](ismir04/README.md) provide dataset entry points. ISMIR04 also
audits exact duplicates and explicitly chunks long tracks during extraction.

All three datasets provide `separation.py`, `prepare_manifest.py`, `extraction.py`, `train_mid.py`,
and `train_late.py`, each supporting direct execution.

FMA-Medium, GTZAN and ISMIR04 also provide `extraction_cnn.py`, `train_mid_cnn.py`, and `train_late_cnn.py` for
Phase 3. They share the official frozen MSD_musicnn extraction backend and
genre fitting protocol, using each dataset's existing manifest and class map.
Run extraction in the isolated musicnn environment, then training in the main
PyTorch environment. Dataset READMEs document cache and result paths.
All CNN datasets use the same [guarded audio loading and cache recovery](../../../docs/cnn_audio_decoding.md).
It preserves healthy cached vectors, retries invalid decodes with FFmpeg, and
reports genuinely damaged/mismatched audio without changing the cohort.
MTG-Jamendo provides `extraction_cnn.py` through the same backend and
`train_mid_cnn.py` for seven conditions with a plain 512/256 hidden-layer head.
Jamendo retains its 56 mood/theme tags; the CNN classifier uses the existing
Music2Emo fitter, frequency weights, stopping and evaluation without harmonic
features or a chord Transformer.
`train_late_cnn.py` uses `_cnn_mood_late.py` with that same CNN loader,
model and fitter, reuses the completed CNN mixture reference, and evaluates
independent matched-mixture/source ensembles plus five standalone stems.

[PMEmo](pmemo/README.md) also provides `extraction_cnn.py` through the same
shared extractor and guarded decoder. Its existing numeric IDs, fixed splits
and continuous valence/arousal annotations are preserved; both tasks reuse one
six-view CNN cache. `train_mid_cnn.py` calls `_cnn_emotion.py` to fit both
binary targets across seven conditions and five seeds with the same shared
emotion fitter and metric adapter. The plain 512/256/1 CNN head has no harmonic
features. This does not change its MERT/harmony training entries or outputs.

[DEAM](deam/README.md) provides `extraction_cnn.py` through that same shared
extractor and decoder, preserving its supplied splits and native 1-9 scores.
The separate six-view cache serves both targets without modifying MERT/harmony
features; extraction does not normalize or binarize annotations.
`train_mid_cnn.py` uses the same `_cnn_emotion.py` controller, plain 512/256/1
head and shared fitter as PMEmo, retaining DEAM's native high >= 5 rule and
separate CNN checkpoints and readable/detailed summaries.

[EmoMusic](emomusic/README.md) provides `extraction_cnn.py` using that same
shared emotion-extraction workflow. It preserves the supplied split/annotation
mapping and intentional DEAM stem root, with independent EmoMusic CNN outputs;
the existing annotation-column provenance caveat remains unresolved.
`train_mid_cnn.py` calls the shared `_cnn_emotion.py` controller and unchanged
PMEmo/DEAM fitter, using native >= 5 labels without swapping supplied columns.
Its seven conditions, five seeds and separate target reports/checkpoints use
the same plain CNN head and optimization/evaluation settings.

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
| `_cnn_runtime.py` | Automatic legacy Keras and local CUDA setup before official musicnn extraction. |
| `_cnn_extraction.py` | Shared CNN extraction CLI for genre, Jamendo multi-label and PMEmo/DEAM/EmoMusic continuous-score manifests, source alignment, caching, and reports. |
| `_cnn_genre.py` | Inherit genre training YAML for CNN caches and call the shared mid/late genre scheduler/trainer; FMA, GTZAN, and ISMIR04 provide `train_late_cnn.py` with read-only CNN-M reuse. |
| `_genre_reference.py` | Load and verify a completed CNN genre mixture reference without fitting or changing it. |
| `_cnn_mood.py` | Schedule plain CNN Jamendo tagging experiments with the existing Music2Emo fitter and evaluation protocol. |
| `_cnn_mood_late.py` | Schedule independent plain CNN Jamendo branches with the same loader/model/fitter, read the completed CNN-M reference, and use shared multilabel late-fusion evaluation. |
| `_cnn_emotion.py` | Schedule separate binary emotion targets for CNN inputs using the retained shared fitter, training-only prevalence and validation-only thresholds. |
| `_genre.py` | Common training CLI, seed/condition scheduling, reuse, and summaries. |
| `_mood_theme_tagging.py` | MERT/harmony mood/theme experiment CLI, mid/late scheduling, protected reuse, and reporting; currently used by MTG-Jamendo. |
| `_emotion_recognition.py` | MERT/harmony binary-emotion mid-fusion CLI, two-target scheduling and reporting; currently used by PMEmo with the same fitting loop. |
| `_emotion_mid_fusion.py` | Native-score emotion mid-fusion adapters and scheduling; DEAM uses its reviewed original-score boundary without modifying PMEmo's running controller. |
| `_emomusic_mid_fusion.py` | EmoMusic native-score mid fusion, parity-tested against DEAM/PMEmo with the same shared model/fitter; supplied target columns are preserved and existing training identities are untouched. |
| `_emomusic_late_fusion.py` | EmoMusic native-score late fusion; DEAM/PMEmo-equivalent scheduling/branches with shared fitting/evaluation, protected mid references and standalone stem reports. |
| `_emotion_late_fusion.py` | Binary-emotion late-fusion CLI, paired independent branch scheduling, mid-reference reuse and per-target ensemble/stem-only reporting. |
| `_emotion_native_late_fusion.py` | Native-score DEAM late-fusion adapter; same branch schedule/fitter/evaluator as PMEmo without modifying existing scientific identities. |
| `_mid_controls.py` | Construct exact repeated-mixture inputs and fit parameter-count/input-width matched controls through the unchanged trainer. |
| `_stem_results.py` | Common standalone-branch validation and reporting. |

MusicFM genre launchers call `_genre.py` directly; CNN training launchers call
it through `_cnn_genre.py`. Jamendo launchers call
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
