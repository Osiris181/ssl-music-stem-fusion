# Phase 3 — Frozen Pretrained CNN Across All Seven Datasets

**Project:** Source Separation Project — revised music classification study  
**Status:** Agreed experimental design; no Phase 3 results are asserted here  
**Revision:** 2026-11-18 — seven mid-fusion configurations across all seven datasets; aligned with active Phase 2 repeated-mixture controls  
**Purpose:** Research design / implementation handoff, not authorization to run training, alter original results, or refactor Phase 2 code.

## A. Research motivation and interpretation

**Research question:** Does adding separated-source features to an original-mixture representation improve music classification differently when the feature extractor is a **frozen supervised-pretrained CNN** rather than the frozen pretrained music SSL-based pipelines evaluated in Phase 2?

**Working hypothesis (to test, not assume):** Separated-stem representations might provide more incremental predictive benefit for a conventional CNN encoder, while MusicFM- and MERT-based representations might already make much of the useful source information accessible to the downstream classifier.

The primary quantity is **fusion improvement within a feature pipeline**, not the CNN's raw classification score versus MusicFM/MERT. If CNN stem fusion improves its matched control but the SSL-based system does not, this supports an **extractor-/pipeline-dependent association**. It does **not** prove that SSL pretraining caused the difference: architecture, supervised versus self-supervised objectives, training corpora, input context, embedding sizes, and (for the MERT-based experiments) the chord/key processing all differ.

**Phase 3 is supplementary and exploratory.** It is being designed after Phase 2 results have been examined; do not claim independent prospective hypothesis selection. Fix the CNN checkpoint, feature layer, all seven fusion/control conditions, splits, and metrics before looking at Phase 3 test results. Do not optimize the model choice specifically to obtain positive fusion gains.

## B. Fixed scope and dataset matrix

Reuse the original mixtures and the outputs of the existing pretrained HT-Demucs `htdemucs_6s` source separator. The named stems are **G**uitar, **D**rums, **P**iano, **B**ass, and **V**ocals. Exclude HT-Demucs's `other` output, exactly as in Phase 2. **M** denotes the *original unseparated mixture*, not a resynthesis from stems.

| Dataset | Task/target | Phase 2 representation pipeline | Phase 3 representation pipeline | Target-specific runs |
| --- | --- | --- | --- | ---: |
| FMA-Medium | 16-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| GTZAN | 10-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| ISMIR04 | 6-class genre | Frozen MusicFM-MSD | Frozen `MSD_musicnn` | 1 |
| MTG-Jamendo | 56-label mood/theme tagging (verify live manifest) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 1 |
| PMEmo | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| EmoMusic | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| DEAM | Binary valence; binary arousal (independent classifiers) | Frozen MERT + chord/key pipeline | Frozen `MSD_musicnn` **without additional chord/key** | 2 |
| **Total** | **7 datasets; 10 target settings** | | | **10** |

**Interpretation by grouping:**

- **Primary cross-extractor contrast:** Three genre datasets, where Phase 2 uses MusicFM and Phase 3 substitutes a CNN embedding.
- **Extended cross-pipeline contrast:** MTG-Jamendo and the three binary-emotion datasets, where Phase 2 uses MERT **plus chord/key** and Phase 3 uses a plain frozen CNN. The removal of chord/key processing changes the representation pipeline, not merely the neural encoder. Treat these as pipeline-level comparisons and report the confound explicitly.

**Retain Phase 2 splits and labels.** In particular, preserve existing source-aligned GTZAN IDs (the Phase 2 source cache includes 499/250/250, not the original 500/250/250 nominal training count); the audited ISMIR04 inclusion/exclusion and split manifests (729/726/724 as documented in Phase 2); existing FMA-Medium and MTG-Jamendo split/target mappings; and the established PMEmo/EmoMusic/DEAM binary targets. Verify actual live files rather than regenerating splits from prose. No track may move between train, validation, and test due to a different extractor.

**Existing binary-label decisions:** PMEmo high when native score `>= 0.5`; EmoMusic and DEAM high when native score `>= 5`. Keep separate valence/arousal targets. For EmoMusic, preserve the existing metadata columns but retain the **unresolved valence/arousal column-provenance caveat** documented in Phase 2; do not silently swap labels.

## C. Frozen CNN encoder

| Item | Proposed fixed setting |
| --- | --- |
| Model/package | Official `musicnn` implementation; record exact version/commit |
| Checkpoint | **Standard `MSD_musicnn`**, not `MSD_musicnn_big` |
| Pretraining | Supervised music audio tagging on a subset of the Million Song Dataset |
| Trainability | All CNN weights **frozen**; inference only |
| Intermediate representation | `features['penultimate']` |
| Expected window-embedding dimension | **200** for the standard model; inspect output shape before processing the full dataset |
| Audio input | Original mixture or one of five selected HT-Demucs stems; consistent mono 16 kHz processing and the model's official preprocessing |
| Input analysis window | 3 seconds, with **zero overlap** proposed for efficient extraction; verify the library's `input_overlap` behavior instead of relying on defaults |
| Final-window policy | Must be explicitly fixed after verifying official implementation (pad/retain/coverage); never silently drop arbitrary endings |
| Track representation | Mean-pool the per-window penultimate vectors into one `d`-dimensional vector for each source and track |
| CNN feature extraction repetitions | **Once**, cached for reuse by both valence/arousal and all five MLP seeds |
| Random downstream seeds | `32`, `64`, `128`, `256`, `512` |

The official musicnn documentation lists `MSD_musicnn`, `MSD_musicnn_big`, the `penultimate` output, and 3-second training/input windows. The standard model has 200 units in its temporal-pooling back end, while the larger model has 500; confirm that the actual selected feature tensor has `d=200` before constructing MLPs.

**Implementation notes:**

1. Do **not** train, fine-tune, or replace the CNN based on downstream test scores.
2. Use the same sample and source coverage as the corresponding Phase 2 experiment wherever feasible; different encoder-specific sample rates/windowing are unavoidable and must be documented.
3. Use a **single fixed** crop/window policy within each dataset for the mixture and all five stems; align windows by the original track time. If very long tracks require chunked processing, ensure all audio is covered and pooling weights windows appropriately.
4. For nearly silent stems, check finite embeddings and preserve records without label-informed filtering. Assert train/validation/test ID uniqueness and exact source-to-mixture alignment.
5. Store each track-level feature vector using identifiers `(dataset, split, track_id, source, checkpoint, extraction_version)` plus waveform length and provenance metadata. Cache the embeddings once, not separately for five seeds.
6. Verify first on a small sample: source order, expected dimension, no NaNs/Infs, deterministic re-extraction, audio coverage and final-window handling.
7. If musicnn's older package dependencies need a compatible environment, document it. Do not silently substitute a reimplementation or checkpoint with potentially different weights/features.

### Pretraining-data overlap gate

`MSD_musicnn` was pretrained on a large labeled music collection. **Check feasible overlap** between its pretraining recordings and every downstream test collection using available identifiers, metadata, and audio matching where possible. An MSD checkpoint is **not automatically leak-free**. ISMIR04 was sourced from Magnatune, making the `MTT_musicnn` alternative especially concerning; nevertheless, the MSD checkpoint must also be audited as far as practicable. Record unresolved leakage risk and do not make strong independent-generalization claims where overlap cannot be ruled out. Do not choose MSD versus MTT based on test results.

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

These are proposed Phase 3 CNN classifier widths, to verify against the actual Phase 2 heads before implementation. The MERT+chord/key and CNN systems therefore do **not** have identical total trainable capacity. Report this difference rather than claiming to isolate CNN versus SSL architecture.

Other classifier choices (ReLU/dropout, biases, loss weighting, optimizer, validation-based checkpoint/threshold selection) must match the dataset's active Phase 2 protocol **within each task family**, except for components inherently tied to the MERT+chord/key representations.

**Interpretation of repeated mixtures:** Repeated `M` matches each corresponding fusion model's input width, MLP architecture, and nominal number of trainable parameters. Identical copies add no new information; with a dense first layer their projections sum to a single equivalent mixture projection. This is an architecture/parameterization control, **not** an effective independent-feature-capacity control. A fusion gain over `CNN-M` alone does not establish a matched-control benefit.

**Illustrative FMA-Medium case** (assuming `d=200`, `C=16`, and only three biased linear layers):

| Condition | MLP | Trainable linear parameters |
| --- | --- | ---: |
| `CNN-M` | `200 -> 1024 -> 512 -> 16` | 738,832 |
| `CNN-R4` / `CNN-F4` | `800 -> 1024 -> 512 -> 16` | 1,353,232 |
| `CNN-R5` / `CNN-F5` | `1000 -> 1024 -> 512 -> 16` | 1,558,032 |
| `CNN-R6` / `CNN-F6` | `1200 -> 1024 -> 512 -> 16` | 1,762,832 |

For three biased linear layers only, `P(D,H1,H2,C) = (D+1)*H1 + (H1+1)*H2 + (H2+1)*C`. Verify actual model counts, since extra trainable layers or normalization may alter them. Each repeated-mixture control must match its corresponding fusion configuration **exactly** in architecture and nominal parameter count.

**Out of the initial Phase 3 scope:** Mid-fusion `M+G` or `M+G+D`; mixture-plus-individual-stem ablations; unweighted late-fusion ensembles; standalone stem-only branches; widened-mixture controls; RMS weighting; and training a CNN from scratch. The seven selected mid-fusion conditions are the fixed initial scope, not a promise to reproduce all Phase 2 late-fusion experiments.

## E. Training and validation protocols

Use the **same five actual outer seeds** as Phase 2: `[32, 64, 128, 256, 512]`. Each seed applies to the downstream CNN-feature classifier; the frozen CNN embedding cache is shared across seeds. Use the exact same split, target, class map, training-only class weighting, and validation-only model-selection/threshold rules for all seven CNN configurations on a given dataset/target.

| Setting | Genre | MTG-Jamendo mood/theme | PMEmo, EmoMusic, DEAM binary valence/arousal |
| --- | --- | --- | --- |
| CNN downstream head | `1024/512` | `512/256` | `512/256` |
| Optimizer | Adam, `1e-4` | Adam, `1e-4` | Adam, `1e-4` |
| Batch size | 128 | 8 | 8 |
| Epoch budget | 20 | Up to 200 | Up to 200 |
| Checkpoint score | Max validation `(accuracy + PR-AUC + ROC-AUC)` on 0–1 scale | **Minimum validation weighted BCE** | **Minimum validation weighted BCE** |
| Early stopping | Best checkpoint within 20 epochs; no newly invented patience | Validation loss patience 10, min delta `1e-4` | Validation loss patience 10, min delta `1e-4` |
| Labels/outputs | Fixed multiclass `C` logits | Multi-label 56 sigmoid logits | Single sigmoid logit per valence/arousal target |
| Class weighting | Preserve active Phase 2 setting | Preserve author frequency-weighted BCE | Preserve training-set-positive-prevalence weighting |
| Prediction threshold | Multiclass argmax | One validation-only global threshold maximizing macro F1; fixed grid `0.00`–`1.00` step `0.01`, lowest tie | Validation-only threshold maximizing macro F1; same grid and tie rule |
| Primary report metric | **Accuracy** | **Macro average precision / PR-AUC** | **Accuracy** |
| Additional metrics | Macro F1, macro ROC-AUC, macro average precision | Agreed multi-label F1, ROC-AUC, accuracy definitions | Two-class macro F1, ROC-AUC, average precision |

**Phase 2 supersession notes:** The *current* MTG-Jamendo setup is **200 epochs with minimum validation weighted BCE**, not the older 500-epoch / validation PR-AUC scheme. The earlier Phase 3 document was outdated on this point. Follow the active Phase 2 training configuration and inspect the code before running. For genre, use the exact existing definition of PR-AUC/AP used in the selection sum; do not substitute a different metric silently.

**Audio/representation mismatch caveat:** Phase 2's MERT+chord/key pipeline includes a segment-level strategy and chord processing not replicated by track-mean-pooled CNN features. CNN extraction is standardized *within the Phase 3 conditions*; direct comparisons with MERT are comparisons of whole pipelines. Keep the same test evaluation IDs and labels across pipelines; carefully label different feature construction and model-head capacity.

## F. Run matrix and workload

| Dataset/target setting | Mid-fusion configurations | Classifier seeds | New classifier fits |
| --- | ---: | ---: | ---: |
| FMA-Medium genre | 7 | 5 | 35 |
| GTZAN genre | 7 | 5 | 35 |
| ISMIR04 genre | 7 | 5 | 35 |
| MTG-Jamendo mood/theme | 7 | 5 | 35 |
| PMEmo valence | 7 | 5 | 35 |
| PMEmo arousal | 7 | 5 | 35 |
| EmoMusic valence | 7 | 5 | 35 |
| EmoMusic arousal | 7 | 5 | 35 |
| DEAM valence | 7 | 5 | 35 |
| DEAM arousal | 7 | 5 | 35 |
| **Total** | **10 target settings** | | **350** |

`10 target settings x 7 configurations x 5 outer seeds = 350` independent downstream classifier fits. Of these, `200` are the four added conditions per setting (R4/R5/F4/F5) relative to the prior three-configuration draft. The count **does not** include frozen CNN feature extraction, any missing source separation, or optional reruns.

Compute track-level frozen CNN features **once per dataset/track/source**, for the six source views (M/G/D/P/B/V), independent of configuration and seed. The valence and arousal tasks within one emotion dataset reuse the same cached six-view embeddings. The 350 classifier fits are not 350 CNN training runs: **CNN parameters remain frozen throughout**.

**Suggested staged execution without changing the agreed full scope:** Smoke-test extraction and all seven input shapes on a small sample; then implement and validate genre training; then MTG-Jamendo; then binary-emotion settings. Staging is for debugging and compute management, **not** permission to stop or selectively promote configurations because early test results support the hypothesis. Final reporting covers all seven conditions and all ten settings.

## G. Analysis, reporting, and scientific limits

**Three prespecified within-CNN contrasts**, for every dataset/target and reported metric:

- `Delta_CNN_4 = Metric(CNN-F4) - Metric(CNN-R4)`.
- `Delta_CNN_5 = Metric(CNN-F5) - Metric(CNN-R5)`.
- `Delta_CNN_6 = Metric(CNN-F6) - Metric(CNN-R6)`.

Report all three, alongside the ordinary `CNN-M` mixture-only reference. Examine the nested inclusion sequence (bass, then vocals) descriptively; do not infer an optimal stem subset from three chosen combinations, and do not select a single favorable `k` post hoc as the only headline result.

**Corresponding Phase 2 effects:** For each `k in {4, 5, 6}`, calculate `Delta_Phase2_k = Metric(Phase2-Fk) - Metric(Phase2-Rk)` using the **active repeated-mixture** controls on exactly the same downstream dataset/target/test split. Never use superseded Phase 2 widened-mixture controls as if they were the active R4/R5/R6 controls. Report `Delta_CNN_k - Delta_Phase2_k` as a descriptive cross-pipeline contrast for each k, preserving the encoder/training-pipeline caveats. For genre, this is CNN versus MusicFM; for Jamendo and binary emotion, it is CNN versus MERT+chord/key.

**Five-seed reporting:** Mean ± sample SD (`ddof=1`) of each metric, plus the per-seed paired fusion-minus-matched-control difference and its mean (and SD when useful), separately for each `k`. Save configs, seeds, trainable-parameter counts, selected checkpoint and validation scores, thresholds, track IDs, labels, and logits/probabilities. Seed variability is not dataset-sampling uncertainty; do not claim statistical significance without appropriate tests. If interpreting multiple dataset/metric/k comparisons inferentially, address multiplicity rather than highlighting only positive contrasts.

**Interpretation scenarios:**

| Observation | Supported conclusion |
| --- | --- |
| Some CNN fusion combinations exceed their own matched Rk controls, but the corresponding Phase 2 Fk-Rk comparisons do not | Stem usefulness differs across these evaluated feature pipelines at those configurations; the cause is not isolated to SSL pretraining |
| CNN benefits vary across k, tasks, or metrics | Conditional effects along the tested nested source sequence; no global stem ranking or universal benefit |
| Neither CNN nor Phase 2 fusion exceeds the corresponding Rk matched controls | No demonstrated benefit for the evaluated representations and configurations under these protocols |
| CNN mixture-only has lower absolute accuracy than MusicFM/MERT | This does **not** establish that CNN stem features provide more conditional predictive value |

Never claim that separated stems add waveform information absent from the mixture, that SSL representations are universally sufficient, or that extractor-specific patterns constitute controlled causal evidence about SSL pretraining. Document historical FMA configuration-selection bias and unresolved CNN pretraining-overlap risk.

## H. Execution readiness / guardrails

- [ ] Check actual Phase 2 repo configuration and split manifests, including current **repeated-mixture** results rather than archived widened-mixture results.
- [ ] Confirm `MSD_musicnn` checkpoint version, output `penultimate` shape, audio preprocessing, segmentation/overlap/final-window policy, and a deterministic six-view smoke test.
- [ ] Audit available pretraining/test-track overlap evidence, recording limitations for every dataset.
- [ ] Verify exact original mixtures, stem files, track IDs, no split leakage, and six-view time alignment.
- [ ] Verify 3 genre target maps, Jamendo tag map, PMEmo/EmoMusic/DEAM binary labels and target-column provenance.
- [ ] Verify and implement 1024/512 **genre** and 512/256 **mood/emotion** CNN heads, as specified; preserve task-level validation/weighting/threshold conventions.
- [ ] Implement/cache one frozen-CNN embedding per `(track, source)` and reuse across seeds and valence/arousal.
- [ ] Run exactly `CNN-M`, `CNN-R4`, `CNN-R5`, `CNN-R6`, `CNN-F4`, `CNN-F5`, and `CNN-F6` under outer seeds 32/64/128/256/512 for all 10 target settings.
- [ ] Save per-seed predictions and compare all three `Fk-Rk` effects (k=4,5,6) against the **current** Phase 2 repeated-mixture comparisons using the same datasets/splits/metrics.
- [ ] Only after full Phase 3 reporting, evaluate whether CNN-from-scratch, additional stem combinations beyond the three existing nested sets, or late-fusion additions answer a genuinely new question.

## I. References and handoff context

- Pons et al. (2019), *musicnn: Pre-trained convolutional neural networks for music audio tagging*: https://arxiv.org/abs/1909.06654
- Official `musicnn` model and Python API documentation: https://github.com/jordipons/musicnn/blob/master/DOCUMENTATION.md
- Official `musicnn` FAQ: https://github.com/jordipons/musicnn/blob/master/FAQs.md
- Internal Phase 2 design (`PHASE2_EXPERIMENTAL_DESIGN.md`, author-provided revised version, November 2026): active repeated-mixture baselines, MusicFM on three genre datasets, MERT+chord/key on mood/theme and binary emotion, current task protocols, and ten task settings.

---

**Frozen scope summary:** Seven datasets / ten targets; a **single standard frozen `MSD_musicnn`** encoder; **seven mid-fusion classifier conditions** (`CNN-M`, `CNN-R4`, `CNN-R5`, `CNN-R6`, `CNN-F4`, `CNN-F5`, `CNN-F6`); **five actual seeds**; **350 classifier fits**; no widened-mixture model, late fusion, or CNN-from-scratch initially. The main claim is an exploratory comparison of *incremental stem-fusion benefit across different feature pipelines*, not proof of a causal SSL mechanism.
