# Phase 3 — Frozen Pretrained CNN Across All Seven Datasets (Mid and Late Fusion)

**Project:** Source Separation Project — revised music classification study  
**Status:** Seven-condition CNN mid fusion is complete for all ten dataset/target settings; late fusion is the next experiment.

**Revision:** 2026-11-19 — retain the completed mid-fusion design and define late-fusion branches, matched mixture ensembles, and standalone-stem evaluations.

**Purpose:** Fixed experiment settings and implementation sequence for Phase 3 CNN late fusion.

## A. Research motivation and interpretation

**Research question:** Does adding separated-source features to an original-mixture representation improve music classification differently when the feature extractor is a **frozen supervised-pretrained CNN** rather than the frozen pretrained music SSL-based pipelines evaluated in Phase 2?

**Working hypothesis (to test, not assume):** Separated-stem representations might provide more incremental predictive benefit for a conventional CNN encoder, while MusicFM- and MERT-based representations might already make much of the useful source information accessible to the downstream classifier.

The primary quantity is **fusion improvement within a feature pipeline**, not the CNN's raw classification score versus MusicFM/MERT. If CNN stem fusion improves its matched control but the SSL-based system does not, this supports an **extractor-/pipeline-dependent association**. It does **not** prove that SSL pretraining caused the difference: architecture, supervised versus self-supervised objectives, training corpora, input context, embedding sizes, and (for the MERT-based experiments) the chord/key processing all differ.

**Phase 3 is supplementary and exploratory.** Keep the late-fusion branch schedule, matched controls, aggregation, metrics, and analysis fixed across all datasets; do not alter them in response to late-fusion test results or claim that the experiment prospectively isolated the effect of SSL pretraining.

## B. Fixed scope and dataset matrix

Reuse the original mixtures and the outputs of the existing pretrained HT-Demucs `htdemucs_6s` source separator. The named stems are **G**uitar, **D**rums, **P**iano, **B**ass, and **V**ocals. Exclude HT-Demucs's `other` output, exactly as in Phase 2. **M** denotes the *original unseparated mixture*, not a resynthesis from stems.

| Dataset | Task/target | Phase 2 representation pipeline | Phase 3 representation pipeline | Target-specific runs |
| --- | --- | --- | --- | ---: |
| FMA-Medium | 16-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| GTZAN | 10-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| ISMIR04 | 6-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| MTG-Jamendo | 56-label mood/theme tagging | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 1 |
| PMEmo | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| EmoMusic | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| DEAM | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| **Total** | **7 datasets; 10 target settings** | | | **10** |

**Interpretation by grouping:**

- **Primary cross-extractor contrast:** Three genre datasets, where Phase 2 uses MusicFM and Phase 3 substitutes a CNN embedding.
- **Extended cross-pipeline contrast:** MTG-Jamendo and the three binary-emotion datasets, where Phase 2 uses MERT **plus chord/key** and Phase 3 uses a plain frozen CNN. The removal of chord/key processing changes the representation pipeline, not merely the neural encoder. Treat these as pipeline-level comparisons and report the confound explicitly.

**Retain the existing splits and labels.** In particular, use the current source-aligned GTZAN IDs (499/250/250), ISMIR04 manifests (729/726/724), FMA-Medium and MTG-Jamendo mappings, and PMEmo/EmoMusic/DEAM binary targets. Do not regenerate splits or move tracks between train, validation, and test.

**FMA CNN loading:** The existing loader intentionally skips a track when any
required source embedding is unavailable. One FMA training track was skipped
because a separated-source embedding could not be extracted, leaving **19,906
CNN training tracks**. Use this same retained CNN cohort for CNN-M, all seven
mid conditions, every new late branch, and both late ensemble families; do not
impute a vector, hardcode the track ID, or regenerate the completed CNN runs.
The earlier MusicFM results remain unchanged.

**Existing binary-label decisions:** PMEmo high when native score `>= 0.5`; EmoMusic and DEAM high when native score `>= 5`. Keep separate valence/arousal targets. For EmoMusic, preserve the existing metadata columns but retain the **unresolved valence/arousal column-provenance caveat** documented in Phase 2; do not silently swap labels.

## C. Frozen CNN encoder

| Item | Existing fixed setting |
| --- | --- |
| Model/package | Official `musicnn` implementation, with source commit and checkpoint checksums recorded in cache metadata |
| Checkpoint | **Standard `MSD_musicnn`**, not `MSD_musicnn_big` |
| Pretraining | Supervised music audio tagging on a subset of the Million Song Dataset |
| Trainability | All CNN weights **frozen**; inference only |
| Intermediate representation | `features['penultimate']` |
| Window-embedding dimension | **200**, validated by the extractor and feature loaders |
| Audio input | Original mixture or one of five selected HT-Demucs stems; consistent mono 16 kHz processing and the model's official preprocessing |
| Input analysis window | Non-overlapping **187-frame** log-mel windows at 16 kHz (approximately 3 seconds), using a 187-frame hop |
| Final-window policy | Retain the final partial spectrogram window and right-pad its missing log-mel frames with zeros; verify recorded coverage |
| Track representation | Mean-pool the per-window penultimate vectors into one `d`-dimensional vector for each source and track |
| CNN feature extraction repetitions | **Once**, cached for reuse by both valence/arousal and all five MLP seeds |
| Random downstream seeds | `32`, `64`, `128`, `256`, `512` |

The active extractor checks the official source/checkpoint identity, uses the standard `MSD_musicnn` model, and rejects non-finite or non-200-dimensional window embeddings. The **187-frame** windowing and zero-padded final partial window are the implemented cache policy; do not substitute an assumed library-default overlap policy. The larger `MSD_musicnn_big` model is not part of this experiment.

**Implementation notes:**

1. Do **not** train, fine-tune, or replace the CNN based on downstream test scores.
2. Use the same sample and source coverage as the corresponding Phase 2 experiment wherever feasible; different encoder-specific sample rates/windowing are unavoidable and must be documented.
3. Use a **single fixed** crop/window policy within each dataset for the mixture and all five stems; align windows by the original track time. If very long tracks require chunked processing, ensure all audio is covered and pooling weights windows appropriately.
4. For nearly silent stems, check finite embeddings and preserve records without label-informed filtering. Assert train/validation/test ID uniqueness and exact source-to-mixture alignment.
5. Store each track-level feature vector using identifiers `(dataset, split, track_id, source, checkpoint, extraction_version)` plus waveform length and provenance metadata. Cache the embeddings once, not separately for five seeds.
6. Verify first on a small sample: source order, expected dimension, no NaNs/Infs, deterministic re-extraction, audio coverage and final-window handling.
7. If musicnn's older package dependencies need a compatible environment, document it. Do not silently substitute a reimplementation or checkpoint with potentially different weights/features.

### Pretraining-data overlap limitation

`MSD_musicnn` was pretrained on a large labeled music collection. It is not automatically free of overlap with downstream test recordings. Record any overlap evidence or unresolved risk, and avoid strong independent-generalization claims where overlap cannot be ruled out. ISMIR04's Magnatune origin makes the alternative `MTT_musicnn` checkpoint especially concerning; do not switch checkpoints based on test results.

## D. Experimental configurations — seven mid-fusion conditions

**The active Phase 2 control is repeated mixture, not widened mixture.** Phase 3 repeats the **same seven mid-fusion input conditions** using frozen CNN embeddings. No widened-mixture baseline is included in the active matrix; earlier widened controls are historical.

For an expected CNN embedding width `d = 200`, use the fixed order `M, G, D, P, B, V`:

| ID | Condition | Input feature construction | MLP input dimension | Role |
| --- | --- | --- | ---: | --- |
| `CNN-M` | Mixture Only | `[M]` | `d = 200` | Ordinary single-MLP reference |
| `CNN-R4` | Repeated Mixture (4) | `[M, M, M, M]` | `4d = 800` | Matched control for `CNN-F4` |
| `CNN-R5` | Repeated Mixture (5) | `[M, M, M, M, M]` | `5d = 1000` | Matched control for `CNN-F5` |
| `CNN-R6` | Repeated Mixture (6) | `[M, M, M, M, M, M]` | `6d = 1200` | Matched control for `CNN-F6` |
| `CNN-F4` | Mixture + G + D + P | `[M, G, D, P]` | `4d = 800` | Four-input stem fusion |
| `CNN-F5` | Mixture + G + D + P + B | `[M, G, D, P, B]` | `5d = 1000` | Five-input stem fusion |
| `CNN-F6` | Mixture + All Five Stems | `[M, G, D, P, B, V]` | `6d = 1200` | Six-input stem fusion |

Each repeated-mixture vector consists of **exact, identical copies** of the same cached original-mixture embedding. Do not compute multiple stochastic mixture embeddings, re-extract mixture copies, use feature noise, or mistakenly substitute the reconstruction of stems for the original mixture.

**Within-configuration comparisons:** `CNN-F4` vs `CNN-R4`, `CNN-F5` vs `CNN-R5`, and `CNN-F6` vs `CNN-R6`. The standard `CNN-M` reference is reported in addition to these matched controls. The three fusion conditions are one historically selected nested sequence (adding bass, then vocals); they are neither exhaustive nor newly optimized on Phase 3 test outcomes.

For **genre** datasets, retain the existing Phase 2 genre MLP hidden widths, `1024 -> 512`:

- `CNN-M`: `d -> 1024 -> 512 -> C`.
- `CNN-Rk` and `CNN-Fk`: `k*d -> 1024 -> 512 -> C` for `k in {4, 5, 6}` and genre `C in {16, 10, 6}`.

For **MTG-Jamendo and binary emotion**, the Phase 2 MERT+chord/key pipeline uses a separate **512 -> 256** classifier head and, in that system, a trainable chord-processing component. The plain CNN alternative retains **512 -> 256** hidden widths but **does not include the MERT-specific chord Transformer or harmonic features**:

- `CNN-M`: `d -> 512 -> 256 -> C`.
- `CNN-Rk` and `CNN-Fk`: `k*d -> 512 -> 256 -> C`, for `k in {4, 5, 6}`, with `C=56` on Jamendo and `C=1` for each binary target.

These are the implemented Phase 3 CNN classifier widths. The MERT+chord/key and CNN systems do **not** have identical total trainable capacity. Report this difference rather than claiming to isolate CNN versus SSL architecture.

Genre uses three biased linear layers with ReLU and **0.2 dropout after each hidden activation**. The Music2Emo and plain CNN mood/emotion heads use biased linear layers and ReLU with **no head dropout or normalization**; Phase 2's 0.1 Transformer dropout belongs only to its trainable chord component. Other classifier choices (loss weighting, optimizer, validation-based checkpoint/threshold selection) must match the dataset's active Phase 2 protocol **within each task family**, except for components inherently tied to the MERT+chord/key representations.

**Interpretation of repeated mixtures:** Repeated `M` matches each corresponding fusion model's input width, MLP architecture, and nominal number of trainable parameters. Identical copies add no new information; with a dense first layer their projections sum to a single equivalent mixture projection. This is an architecture/parameterization control, **not** an effective independent-feature-capacity control. A fusion gain over `CNN-M` alone does not establish a matched-control benefit.

**Illustrative FMA-Medium case** (assuming `d=200`, `C=16`, and only three biased linear layers):

| Condition | MLP | Trainable linear parameters |
| --- | --- | ---: |
| `CNN-M` | `200 -> 1024 -> 512 -> 16` | 738,832 |
| `CNN-R4` / `CNN-F4` | `800 -> 1024 -> 512 -> 16` | 1,353,232 |
| `CNN-R5` / `CNN-F5` | `1000 -> 1024 -> 512 -> 16` | 1,558,032 |
| `CNN-R6` / `CNN-F6` | `1200 -> 1024 -> 512 -> 16` | 1,762,832 |

For the implemented three biased linear layers, `P(D,H1,H2,C) = (D+1)*H1 + (H1+1)*H2 + (H2+1)*C`. Each repeated-mixture control matches its corresponding mid-fusion configuration **exactly** in architecture and nominal parameter count.

**Expanded scope:** Keep the seven completed mid-fusion conditions unchanged. Add the unweighted late-fusion ensembles, their independently trained matched mixture ensembles, and standalone evaluations of the already trained stem branches in Section E.1. Mid-fusion `M+G` or `M+G+D`, mixture-plus-individual-stem ablations, widened-mixture controls, RMS weighting, and CNN training from scratch remain outside this design. Adding late fusion does **not** require retraining the seven mid-fusion conditions.

## E. Training and validation protocols

Use the **same five actual outer seeds** as Phase 2: `[32, 64, 128, 256, 512]`. Each seed applies to the downstream CNN-feature classifier; the frozen CNN embedding cache is shared across seeds. Use the exact same split, target, class map, training-only weighting where applicable, and validation-only model-selection/threshold rules for comparable CNN configurations and branches on a given dataset/target.

| Setting | Genre | MTG-Jamendo mood/theme | PMEmo, EmoMusic, DEAM binary valence/arousal |
| --- | --- | --- | --- |
| CNN downstream head | `1024/512` | `512/256` | `512/256` |
| Optimizer | Adam, `1e-4` | Adam, `1e-4` | Adam, `1e-4` |
| Batch size | 128 | 8 | 8 |
| Epoch budget | 20 | Up to 200 | Up to 200 |
| Checkpoint score | Max validation `(accuracy + PR-AUC + ROC-AUC)` on 0–1 scale | **Minimum validation weighted BCE** | **Minimum validation weighted BCE** |
| Early stopping | Best checkpoint within 20 epochs; no newly invented patience | Validation loss patience 10, min delta `1e-4` | Validation loss patience 10, min delta `1e-4` |
| Labels/outputs | Fixed multiclass `C` logits | Multi-label 56 sigmoid logits | Single sigmoid logit per valence/arousal target |
| Class weighting | Unweighted cross-entropy | Preserve author frequency-weighted BCE using training-label prevalence | Preserve training-set-positive-prevalence weighting |
| Prediction threshold | Multiclass argmax | One validation-only global threshold maximizing macro F1; fixed grid `0.00`–`1.00` step `0.01`, lowest tie | Validation-only threshold maximizing macro F1; same grid and tie rule |
| Primary report metric | **Accuracy** | **Macro average precision / PR-AUC** | **Accuracy** |
| Additional metrics | Macro F1, macro ROC-AUC, macro average precision | Agreed multi-label F1, ROC-AUC, accuracy definitions | Two-class macro F1, ROC-AUC, average precision |

**Fixed task settings:** MTG-Jamendo and binary emotion use up to **200 epochs**, minimum validation weighted-BCE checkpoint selection, and the same loss and early-stopping rules in mid and late fusion. Genre uses macro **average precision** for the PR-AUC/AP term in its validation-selection score.

**Audio/representation mismatch caveat:** Phase 2's MERT+chord/key pipeline includes a segment-level strategy and chord processing not replicated by track-mean-pooled CNN features. CNN extraction is standardized *within the Phase 3 conditions*; direct comparisons with MERT are comparisons of whole pipelines. Keep the same test evaluation IDs and labels across pipelines; carefully label different feature construction and model-head capacity.

### E.1. Late fusion — independent CNN branches and matched ensembles

Use the existing frozen, six-view `MSD_musicnn` cache; **do not** run CNN training, source separation, or feature extraction again merely to add late fusion. For each dataset/target and outer seed, train one downstream MLP per new branch. Each branch takes exactly one source's 200-dimensional track embedding, never a concatenation. The source order is `M, G, D, P, B, V`. The original mixture-only CNN-M model is the intended **M1** reference; M2–M6 all receive the *same cached original-mixture embedding* but have independently initialized and fitted weights. G/D/P/B/V receive their respective source-specific embeddings and independent weights.

The branch head is `200 -> 1024 -> 512 -> C` for genre, `200 -> 512 -> 256 -> 56` for MTG-Jamendo, and `200 -> 512 -> 256 -> 1` for each binary valence/arousal target. Retain the task-specific activations, dropout, biases, loss, optimizer, epoch budget, checkpoint selection, and evaluation in Section E. Omit the Phase 2 MERT-specific chord Transformer and chord/key inputs from the CNN branches. The resulting plain CNN and MERT+chord/key systems match the downstream **method**, not total trainable parameter count.

| Reported condition | Stem-fusion branches | Matched mixture control | Control branches |
| --- | --- | --- | --- |
| `CNN-LF4` | M1 + G + D + P | `CNN-LE4` | M1 + M2 + M3 + M4 |
| `CNN-LF5` | M1 + G + D + P + B | `CNN-LE5` | M1 + M2 + M3 + M4 + M5 |
| `CNN-LF6` | M1 + G + D + P + B + V | `CNN-LE6` | M1 + M2 + M3 + M4 + M5 + M6 |

Each branch is fitted and checkpointed **once per target and outer seed** and reused across the nested ensembles. The existing mid-fusion `CNN-R4/R5/R6` are single classifiers on repeated concatenated mixture inputs; they are **not** members of `CNN-LE4/LE5/LE6`. Do not compare `CNN-LFk` with `CNN-M` or `CNN-Rk` as evidence of a matched late-fusion benefit; the matched comparison is `CNN-LFk - CNN-LEk`.

**Position-paired branch seeds:** Outer seeds are `[32, 64, 128, 256, 512]`. M1 uses the outer seed `s`. At positions 1–5, derive the same seed for each mixture/stem pair using the implemented Phase 2 rule: take the first four bytes of `SHA256(UTF-8("stemfusion-phase2:{s}:{index}"))` as an unsigned big-endian integer. Position 1 pairs M2/G; 2 pairs M3/D; 3 pairs M4/P; 4 pairs M5/B; 5 pairs M6/V. The two classifiers at a position are *separate fits* with the same random seed, not shared weights. Preserve the existing genre seeding and Music2Emo per-epoch seeding inside their respective fitters.

**Aggregation:** Select each branch's checkpoint independently on its validation split. For both fusion and matched mixture ensembles, require identical split, track IDs **in identical order**, labels, finite logit shapes, and cohort before aggregation; reject mismatches rather than silently reindexing. Sum selected branches' raw float32 logits without weights, averaging, normalization, probability averaging, calibration, or a learned fusion layer. Apply softmax and argmax for genre, or sigmoid for Jamendo and binary emotion. The ensemble itself has no trainable parameters or checkpoint-selection step. Test labels must not affect branch selection, thresholds, or model choice.

**Task-specific selection and decisions:** Genre branches use unweighted cross-entropy, Adam `1e-4`, batch 128, 20 full epochs, and the highest validation sum of micro accuracy + macro one-vs-rest ROC-AUC + macro average precision (`PR-AUC` in current reports), with earliest exact tie. No genre early stopping or tuned probability threshold is used. Jamendo and binary emotion branches use the existing training-prevalence-weighted BCE, Adam `1e-4`, batch 8, at most 200 epochs, and validation-loss early stopping with patience 10 and min delta `1e-4`. `best.pt` is selected on **any strictly lower** validation weighted-BCE (earliest exact tie); `min_delta` governs stopping, not best-checkpoint replacement. For each Jamendo or binary ensemble separately, select a **single** probability threshold on validation macro F1 over `0.00, 0.01, ..., 1.00`, taking the lowest tied value, and freeze it for test. Jamendo accuracy is label-wise; binary F1, ROC-AUC, and average precision average the low/high classes. Genre uses multiclass argmax.

**Standalone sources and M1 reuse:** Report G, D, P, B, and V from their already selected late-branch checkpoints with no additional fit. A Jamendo or binary standalone branch uses its **own validation-selected threshold**, not an ensemble threshold or test-tuned threshold. Reuse the completed Phase 3 CNN-M checkpoint and its saved validation/test predictions as M1 for the same dataset, target, outer seed, and CNN feature cohort. Load that reference without changing its model, protocol signature, or saved files. If a required reference artifact is missing or inconsistent with the late branches' IDs, labels, or feature cohort, stop that run and report it; do not silently retrain or substitute a Phase 2 MusicFM/MERT model.

**Implementation boundary:** Reuse the genre fitter/aggregator and Music2Emo shared fitter and task metrics with CNN-specific model and dataset factories. Do not send CNN features through the MERT/harmony-specific Phase 2 loaders or add chord/key inputs. Put new late branches, ensembles, and summaries under separate Phase 3 CNN late-fusion output directories; leave all Phase 2 and completed CNN mid-fusion artifacts unchanged. Save branch IDs, outer and actual training seeds, selected checkpoints, validation/test raw logits and targets, thresholds where applicable, and the M1 reference path. Logit-scale or error-overlap summaries may be reported as diagnostics but do not change fusion.

## F. Run matrix and workload

The completed mid-fusion runs cover all ten settings with seven conditions and five outer seeds. Reuse their CNN-M model as M1; train **ten additional branches per setting/seed**—M2–M6 and G/D/P/B/V. The three late source ensembles, three late matched mixture ensembles, and five standalone-stem evaluations use those branches without extra fitting.

| Dataset/target setting | Existing mid fits: 7 x 5 | Added late branch fits: 10 x 5 | Nominal total fits |
| --- | ---: | ---: | ---: |
| FMA-Medium genre | 35 | 50 | 85 |
| GTZAN genre | 35 | 50 | 85 |
| ISMIR04 genre | 35 | 50 | 85 |
| MTG-Jamendo mood/theme | 35 | 50 | 85 |
| PMEmo valence | 35 | 50 | 85 |
| PMEmo arousal | 35 | 50 | 85 |
| EmoMusic valence | 35 | 50 | 85 |
| EmoMusic arousal | 35 | 50 | 85 |
| DEAM valence | 35 | 50 | 85 |
| DEAM arousal | 35 | 50 | 85 |
| **Total** | **350** | **500** | **850** |

The run plan is `10 settings x 5 seeds x 10 new branches = 500` additional fits, for `350 + 500 = 850` distinct classifier fits including completed mid fusion. A missing or incompatible M1 stops the affected run rather than silently changing this plan. Each setting/seed has **18 reported conditions**: seven existing mid conditions, three matched late mixture ensembles, three late source ensembles, and five standalone stems; `18 x 10 x 5 = 900` reportable evaluations. Ensemble evaluations are not classifier fits.

The frozen CNN cache is computed **once per dataset/track/source**, shared by mid and late fusion, all five seeds, and both valence/arousal targets within an emotion dataset. Do not rerun separation or feature extraction without an identified missing or invalid cache. CNN parameters remain frozen throughout.

**Run sequence:** Add the late-fusion entry points/configurations for genre, Jamendo, and binary emotion using their existing CNN caches and shared training code; smoke-test one non-test branch and ensemble per task family; then execute the fixed branch and reporting matrix across all ten settings. Mid-fusion results remain untouched.

## G. Analysis, reporting, and scientific limits

**Three existing mid-fusion within-CNN contrasts**, for every dataset/target and reported metric:

- `Delta_CNN_4 = Metric(CNN-F4) - Metric(CNN-R4)`.
- `Delta_CNN_5 = Metric(CNN-F5) - Metric(CNN-R5)`.
- `Delta_CNN_6 = Metric(CNN-F6) - Metric(CNN-R6)`.

Report all three, alongside the ordinary `CNN-M` mixture-only reference. Examine the nested inclusion sequence (bass, then vocals) descriptively; do not infer an optimal stem subset from three chosen combinations, and do not select a single favorable `k` post hoc as the only headline result.

**Three new late-fusion within-CNN contrasts:** For `k in {4, 5, 6}`, report `Delta_CNN_late_k = Metric(CNN-LFk) - Metric(CNN-LEk)`. Keep mid and late effects separate, each against its own matched control. The `CNN-Rk` repeated-mixture MLP is not a matched control for a late ensemble, and `CNN-LEk` is not a matched control for a mid-fusion MLP. Standalone stems are diagnostic results, not evidence of incremental benefit over M.

**Corresponding Phase 2 effects:** For each `k in {4, 5, 6}`, calculate the Phase 2 mid effect `Metric(Phase2-Fk) - Metric(Phase2-Rk)` using the **active repeated-mixture** controls, never archived widened-mixture controls. Calculate the Phase 2 late effect `Metric(Phase2-LFk) - Metric(Phase2-LEk)` using its independently trained matched mixture ensembles. Contrast CNN and Phase 2 effects **within the same fusion strategy** and as descriptive cross-pipeline comparisons only. For genre the representation contrast is CNN versus MusicFM; for Jamendo and binary emotion it is plain CNN versus MERT+chord/key. Report the intentionally different FMA CNN training count when presenting cross-pipeline comparisons.

**Five-seed reporting:** Mean ± sample SD (`ddof=1`) of each metric for all 18 conditions, plus **per-outer-seed** fusion-minus-matched-control differences and their mean/sample SD for each mid and late `k`. Do not derive paired uncertainty by subtracting group means or SDs. Save configs, outer and branch seeds, trainable-parameter counts, source IDs, selected checkpoints and validation scores, thresholds, track IDs, labels, and raw logits/probabilities. Include a provenance record for every reused M1. Seed variability is not dataset-sampling uncertainty; do not claim statistical significance without appropriate tests. The late extension follows inspection of earlier results and must be labeled exploratory. If interpreting multiple dataset/metric/k comparisons inferentially, address multiplicity rather than highlighting only positive contrasts.

**Interpretation scenarios:**

| Observation | Supported conclusion |
| --- | --- |
| Some CNN fusion combinations exceed their own matched Rk controls, but the corresponding Phase 2 Fk-Rk comparisons do not | Stem usefulness differs across these evaluated feature pipelines at those configurations; the cause is not isolated to SSL pretraining |
| CNN benefits vary across k, tasks, or metrics | Conditional effects along the tested nested source sequence; no global stem ranking or universal benefit |
| Neither CNN nor Phase 2 mid fusion exceeds Rk, or late fusion exceeds its LEk matched controls | No demonstrated benefit for the evaluated representations, strategies, and configurations under these protocols |
| CNN mixture-only has lower absolute accuracy than MusicFM/MERT | This does **not** establish that CNN stem features provide more conditional predictive value |

Never claim that separated stems add waveform information absent from the mixture, that SSL representations are universally sufficient, or that extractor-specific patterns constitute controlled causal evidence about SSL pretraining. Document historical FMA configuration-selection bias and unresolved CNN pretraining-overlap risk.

## H. Implementation sequence

1. Add separate CNN late-fusion configurations and output roots for each dataset, preserving existing mid-fusion files and the fixed split/label policies.
2. Reuse each completed CNN-M model as M1; train M2–M6 and G/D/P/B/V independently with the fixed branch seeds and the same task-specific classifier settings.
3. Combine saved branch logits into LE4/5/6 and LF4/5/6; select Jamendo/binary ensemble thresholds on validation only, and report standalone branches from their existing checkpoints.
4. Save per-seed predictions and readable/detailed summaries. Report all late `LFk-LEk` and existing mid `Fk-Rk` contrasts with five-seed sample SD, keeping the intentional FMA CNN track skip and EmoMusic label-provenance caveat in their appropriate dataset context.
5. Treat CNN-from-scratch models or additional stem combinations as separate follow-up experiments, not changes to this run matrix.

## I. References and handoff context

- Pons et al. (2019), *musicnn: Pre-trained convolutional neural networks for music audio tagging*: https://arxiv.org/abs/1909.06654
- Official `musicnn` model and Python API documentation: https://github.com/jordipons/musicnn/blob/master/DOCUMENTATION.md
- Official `musicnn` FAQ: https://github.com/jordipons/musicnn/blob/master/FAQs.md
- Internal Phase 2 design (`PHASE2_EXPERIMENTAL_DESIGN.md`, author-provided revised version, November 2026): active repeated-mixture mid baselines, independently trained matched-mixture late ensembles, position-paired branch seeds, MusicFM on three genre datasets, MERT+chord/key on mood/theme and binary emotion, current task protocols, and ten task settings. Active code and resolved runtime configurations take precedence over historical prose.

---

**Expanded scope summary:** Seven datasets / ten targets; one frozen standard `MSD_musicnn` encoder; seven completed mid-fusion conditions (`CNN-M`, `CNN-R4/R5/R6`, `CNN-F4/F5/F6`), three late matched-mixture ensembles (`CNN-LE4/5/6`), three late source ensembles (`CNN-LF4/5/6`), and five standalone source-branch evaluations; five outer seeds. Reusing completed CNN-M as M1 gives **500** new branch fits, **850** total classifier fits including mid fusion, and **900** reportable setting/seed/condition evaluations. No widened-mixture model, RMS weighting, or CNN-from-scratch model is included. The goal remains an exploratory comparison of *incremental stem-fusion benefit across feature pipelines*, not proof of a causal SSL mechanism.
