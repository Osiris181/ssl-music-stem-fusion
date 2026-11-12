# Unit tests

Dedicated SheetSage2 tests are preserved inside its private experiment archive.
Active regression tests ensure the archived commands are unavailable and the
original BTC/music21, MERT and dataset workflows remain usable.

## Purpose

These tests cover deterministic behavior that can be checked without large
datasets or model checkpoints.

## Test files

| File | Coverage |
| --- | --- |
| `test_pmemo_extraction.py` | Original split/annotation preservation, strict numeric IDs and source paths, settings and scheduling parity with Jamendo, direct/unified commands, retained tails, verified resume and bounded CQT retry through the same per-view writers. |
| `test_pmemo_preparation.py` | Unified feature/separation manifest purposes, unchanged feature bytes, fixed split IDs and exact audio destinations, protected outputs, targeted selection, strict YAML, direct commands, shared Jamendo inference settings and complete-stem reuse. |
| `test_pmemo_mid_fusion.py` | Midpoint/equality, separate targets and training-only prevalence, source alignment, optional checksums, binary macro metrics, actual architecture matching, synthetic 70-fit matrix, exact resume/reuse and direct/unified commands. |
| `test_pmemo_late_fusion.py` | Two-target/five-seed branch matrix, protected mid-reference reuse, position-paired controls, real binary stem fit, logit-sum metrics, stem-only summaries, strict artifacts, exact resume and direct/unified commands. |
| `test_music2emo_task_adapter.py` | Strict optional-task compatibility, unchanged Jamendo identities and rejection of optimizer or compatibility-policy edits. |
| `test_deam_extraction.py` | Supplied DEAM splits/scores, strict IDs and paths, explicit dataset adapters, PMEmo/Jamendo setting parity, scheduler parity, direct/unified commands, real per-view writer fixtures, corrupted-cache detection and targeted CQT retry. |
| `test_deam_preparation.py` | Combined feature/separation purposes, byte-identical feature manifests, PMEmo preparation-policy parity, fixed split/ID paths, selection/protection, CLI aliases, exact shared HT-Demucs settings and completed-stem reuse. |
| `test_deam_mid_fusion.py` | Native >= 5 labels without score changes, AST-identical cache algorithm, PMEmo representation/fit parity, target-specific prevalence, 70-fit synthetic matrix, strict caches/reuse, real architecture, exact resume and direct/unified commands. |
| `test_deam_late_fusion.py` | PMEmo-identical scheduling/branch ASTs and evaluator reuse, bit-identical synthetic branch/ensemble predictions, native-score reports, five-seed/two-target matrix, protected references, stem-only rows, strict artifacts, exact resume and direct/unified commands. |
| `test_emomusic_extraction.py` | Supplied EmoMusic splits and unchanged annotation columns, shared DEAM stem configuration, four-dataset settings parity, identical scheduler routing, direct commands, retained tails, strict cache reuse and targeted CQT retry. |
| `test_emomusic_preparation.py` | PMEmo-equivalent label-independent preparation, unchanged feature manifest bytes, fixed split/ID paths and shared DEAM stem roots, protected selection/output files, CLI aliases, exact separation settings and completed-stem reuse. |
| `test_emomusic_mid_fusion.py` | Unchanged supplied score columns, native >= 5 labels, AST-identical DEAM loading/scheduling, shared model/fitter/metrics, bit-identical synthetic PMEmo fits, 70-fit matrix, checksum choices, strict caches/resume and direct/unified commands. |
| `test_emomusic_late_fusion.py` | AST-identical DEAM late orchestration/branches and PMEmo branch parity, shared evaluator, bit-identical synthetic predictions, five-seed/two-target matrix, protected mid references, standalone stems, strict integrity/resume and direct/unified commands. |
| `test_archived_harmony_variants.py` | Rejection of archived S-KEY/ChordFormer/SheetSage2 commands and absence of their modules, with original MERT/BTC/music21 imports, presets and commands intact. |
| `test_jamendo_relocation.py` | Flattened entry points, strict path-only trainer identities, unchanged dependency hashing, and rejection of actual optimizer/adapter changes for legacy cache reuse. |
| `test_music2emo_configuration.py` | Shared/task preset inheritance, preserved reproduction settings and artifact paths, flat MERT presets, CLI overrides, strict stopping validation, and separation of future binary-emotion task settings. |
| `test_jamendo_mid_fusion.py` | Six-view cache alignment, source-specific harmony, shared temporal sampling, exact single-input model parity, trainable-parameter matching and identical repeated vectors, validation-only thresholds, five-seed/seven-condition smoke test, protected reference reuse, bit-exact resume, optional-checksum YAML/CLI precedence, trusted-cache structural rejection, bit-exact cross-mode fits, and historical-checkpoint reuse without rewriting per-run artifacts. |
| `test_jamendo_checksum_compatibility.py` | Exact reviewed loading/layout compatibility for completed mid/late runs, combined controller/thin entries and removed redundant modules, actual execution identity recording, and rejection of changed fusion/sampling/threshold policies or schema/loader/training dependencies. |
| `test_task_controller_layout.py` | All genre launchers route directly to `_genre.py`, Jamendo launchers route directly to `_mood_theme_tagging.py`, dataset training wrappers are absent, and YAML defaults remain unchanged for all eight mid/late entry points. |
| `test_jamendo_late_fusion.py` | Fixed twelve-report/eleven-branch plan, inherited settings, paired actual seeds, five-seed matrix with exactly 50 new fits when M1 is cached, stem-only reuse, logit-sum/sigmoid aggregation, per-ensemble validation thresholds, original-reference parity/protection, optional payload checksums with exact loader/training parity and recorded integrity policy, mandatory array/metadata/alignment rejection in trusted mode, rerun/subset reuse, and direct/unified commands. |
| `test_direct_extraction_script.py` | Direct Python-file execution without PYTHONPATH, YAML source selection, and non-writing dry runs. |
| `test_yaml_configuration.py` | YAML precedence, relative paths, run selection, strict validation, safe loading, and saved resolved settings. |
| `test_multi_source_extraction.py` | YAML source lists, CLI subsets, one-time encoder loading, cache skipping, partial failures, and aggregate reports. |
| `test_htdemucs.py` | Multi-root discovery, six-stem completion, commands, storage resume, and rescue behavior. |
| `test_separation_progress.py` | Shared discovery/checking/staging/batch/retry progress, unchanged commands and reuse counts, closed bars on failures, non-writing dry runs, and readable stdout summaries versus stderr progress across all seven dataset scripts. |
| `test_separation_reporting.py` | Readable success/reuse/failure/dry-run summaries, unprocessed tracks, bounded failure previews, and unchanged saved JSON reports and exit codes. |
| `test_dataset_separation.py` | Direct scripts for all three datasets, shared inference settings, manifest aliases, exact destination reuse, collision/root validation, and staging cleanup on failure. |
| `test_mtg_jamendo_separation.py` | Provided mood/theme TSV selection, split/tag preservation, label-independent manifests, output reuse, direct scripts, shared settings, full-waveform validation, lossless WAV round trips, and rejection of invalid durations. |
| `test_jamendo_segments.py` | Shared full feature manifests, numeric ordering, temporal sampling, retained tails, encoder-specific padding, bounded alignment, active MERT/harmony commands, and rejection of the archived Jamendo MusicFM command. |
| `test_jamendo_mert_segments.py` | Architecture-derived 400-sample minimum, minimal padding, author-compatible all-layer pooling including single-frame inputs, preserved mixture/aligned stems, verified caches, failed-publication backup safety, reports, strict YAML, and direct/unified commands without audio/model access. |
| `test_chord_cqt_retry.py` | One-sample-only CQT tail padding, unchanged valid blocks, true sample-boundary metadata, and rejection of empty audio or unvalidated frontends. |
| `test_jamendo_reproduction.py` | Verified mixture caches, author token/layer/key conventions, weighted loss, chord-model gradients, five-seed synthetic runs, sample SD, strict minimum-delta/patience behavior, independent best-checkpoint selection, top-five pruning/checksums, bit-exact epoch/patience resume, strict cache reuse, YAML defaults, and non-writing direct commands. |
| `test_fma_manifest.py` | FMA splits, distributed HT-Demucs roots, YAML defaults and CLI overrides, missing-file handling, direct scripts, and protection of existing manifests. |
| `test_musicfm_features.py` | HT-Demucs MusicFM cache keys, reports, dry runs, and partial failures. |
| `test_feature_entry_points.py` | MusicFM routing and rejection of archived RMS commands. |
| `test_training_protocol.py` | Complete synthetic five-seed matrix, identical repeated-mixture inputs and exact parameter matching, old-control cache rejection, unchanged fusion/branch reuse, deterministic fits, validation-only selection, artifact checksums, paired summaries, standalone stem reporting without training, and direct training scripts. |
| `test_genre_identity.py` | Historical genre identity preservation, task-scoped dependencies, and rejection of changed optimizers, relevant dependencies, fingerprint logic, or compatibility policy. |
| `test_gtzan_pipeline.py` | Provided split preservation, generic availability filtering and strict overrides, mixed NumPy/PyTorch vectors, provenance checks, shared FMA/GTZAN settings, five-seed 85-fit smoke test, checkpoint reuse, and direct GTZAN commands. |
| `test_ismir04_preparation.py` | CSV-authoritative labels, collision-safe IDs, fixed split preservation, duplicate priority, conflict rejection, stem-repair routing, PCM hashes, and audit reuse. |
| `test_ismir04_extraction.py` | Complete chunk coverage, latent-frame-weighted pooling, unchanged single-pass mean, explicit layer selection, YAML overrides, unique cache keys, reports, and direct dry runs. |
| `test_ismir04_training.py` | Shared three-dataset settings, NumPy/PyTorch alignment, chunk-policy provenance, strict cache validation, synthetic five-seed 85-fit matrices, checkpoint reuse, direct commands, and standalone-stem reporting. |

## How to run them

`test_jamendo_chords_keys.py` checks released interval/MIDI policies, key
normalization, per-source outputs, YAML/CLI paths, lazy loading, verified
resume, no-key reporting, failed-overwrite preservation, bounded two-pass
retries and opt-out, initial/final reports, unchanged legacy-cache reuse,
strict rejection of incompatible/corrupted caches, and planning without
audio/model access. Fixtures do not need BTC weights or optional MIDI packages.

Run the unit directory with:

    PYTHONPATH=src python -m unittest discover -s tests/unit -v

Run one module while developing with:

    PYTHONPATH=src python -m unittest tests.unit.test_musicfm_features -v

New tests should avoid relying on user-specific absolute paths. Use temporary
directories and dependency injection for expensive model calls.
