# Research Direction — Music Source Separation and Pretrained Music Representations

> **Status (2026-10-28):** Phase 1 research framing revised by the author; **Phase 2 experimental design agreed provisionally** and documented separately. New controls and five-seed experiments have **not yet been implemented**. Scientific conclusions remain conditional on results.
>
> **Purpose of this file:** Provide Codex (and future collaborators) with the scientific objective, the *original* experimental setup, what the literature does and does not establish, and guardrails. The **current agreed Phase 2 design** is in [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md). **These are context/design notes, not permission to clean code, start experiments, or rewrite the paper automatically.**
>
> **Working manuscript:** *Enhancing Music Classification via Source Separation and Feature Fusion* (original submission: APSIPA Transactions on Signal and Information Processing, manuscript ATSIP-07-2026-0064, July 2026; rejected October 2026).

## Current agreed revision scope (design, not completed work)

- **HT-Demucs (`htdemucs_6s`) only** for revised source separation, using five named stems (guitar, drums, piano, bass, vocals) and excluding `other`; **Banquet is removed** from the revised comparison scope. Historical Banquet experiments/code remain part of the unrevised paper, and must not be deleted without authorization.
- **Keep both representation pipelines:** frozen MusicFM, and Kang & Herremans-style frozen MERT plus reconstructed chord/key features. They may be used for different datasets; avoid unqualified task-type causal comparisons.
- **Keep two fusion approaches:** feature-concatenation mid-fusion and **unweighted** prediction-level late-fusion. **Remove RMS-weighted late fusion** from the revised primary matrix. These methods are evaluation mechanisms, not claimed novel algorithms.
- **Fix three representative, nested mixture-plus-stem configurations** for revised comparisons: `M+G+D+P`, `M+G+D+P+B`, and `M+G+D+P+B+V` (`M+All Stems`). The sets were retained after earlier exploratory **FMA test** comparisons; they are **not historically prespecified** or proven optimal.
- **Add matched mixture-only controls:** width/parameter-matched MLPs for 4/5/6-input mid-fusion and 4/5/6-branch independently trained mixture ensembles for late-fusion. Keep the original mixture-only reference.
- **Five training seeds**, fixed dataset splits, task-appropriate validation checkpoint selection, and **mean ± SD** plus mean paired differences. Statistical significance tests and bootstrap CIs are **optional**, never implied by SD alone.
- **Current task:** document the design for Codex. The **original repository is not yet cleaned**. Do not implement anything based solely on these notes. See [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md) for concrete planned comparisons and checks.

## 1. Core motivation and research question

Modern pretrained self-supervised learning (SSL) models, including MusicFM and MERT, can extract rich musical representations directly from an **original, unseparated music mixture**. Prior research has shown that music source separation (MSS) can improve certain music-classification settings. The central question for our revised work is therefore **not** whether MSS has ever been used for classification, nor whether pretrained features have ever been combined with separation.

**Primary research question**

> **When powerful pretrained music representations of the original mixture are already available, does explicitly adding representations of source-separated musical stems provide complementary predictive value for downstream music classification? Under what conditions does it help, do little, or hurt?**

In this project, *complementary predictive value* means improvements attributable to stem-derived inputs **relative to an informative mixture-only representation**, after controlling as well as feasible for additional classifier capacity and ensemble effects. It does **not** mean MSS creates acoustic information absent from the input waveform; rather, separation may make existing source-related cues easier to exploit.

## 2. Candidate contributions / research objectives

These are **hypotheses and proposed empirical contributions**, not established claims of originality or proven findings.

1. **Explicit mixture-plus-stem fusion using pretrained representations.** Study embeddings extracted separately from the original mixture and selected stems using frozen, general-purpose pretrained music feature extractors. The MSS model is likewise pretrained/frozen; train the downstream classifier only. The scientifically important contrast is against strong *mixture-only* feature baselines.
2. **Incremental value of selected stem combinations.** Test whether incorporating selected instrument-stem subsets **in addition to the original mixture** adds predictive value, which combinations are beneficial, and whether including more stems necessarily improves results. Distinguish this from a *stems-only* experiment. Exhaustive enumeration is not required.
3. **Cross-task effectiveness.** Investigate whether explicit mixture-plus-stem fusion gives different results across **genre classification, mood/theme tagging, and binary valence/arousal emotion recognition**, including neutral or negative effects. Apparent task differences are exploratory unless dataset, label, feature-extractor, classifier, and other confounders are addressed.

### Main positioning / suggested motivation paragraph

> Recent pretrained music SSL models have demonstrated the ability to extract highly informative representations directly from music mixtures. Although previous research has explored MSS for music classification, we investigate whether explicit mixture-plus-stem fusion provides additional predictive value **beyond the original mixture's pretrained representation**. We further examine how the choice of instrument stems and the downstream task affect its usefulness, using controlled comparisons with strong mixture-only baselines.

**Avoid:** “This is the first study to use MSS for different music classification tasks”; “No one has combined MSS with pretrained feature extraction”; “Stem fusion proves instrument-specific complementary information” without controls; unqualified causal claims that genre benefits because it is instrument-localized or that emotion relies on holistic context.

## 3. Existing system (from original manuscript; preserve until independently inspected)

### Pipeline

1. Input: original music-mixture waveform.
2. MSS: pretrained **HT-Demucs (`htdemucs_6s`)** and/or query-based **Banquet** for instrument separation. The original experiments use the five named stems **vocals, drums, bass, piano, guitar** and generally exclude the `other` stem. Note that HT-Demucs has a six-stem model; do not infer that only five are produced natively.
3. Feature extraction: frozen pretrained representations of the original mixture and/or each stem. The original study uses **MusicFM-MSD** (features from the seventh Conformer layer, then temporal averaging) and, for other tasks, the **Kang & Herremans feature pipeline** (frozen **MERT-95M**, layers five/six, plus estimated chord/key features). RMS loudness is used in the original weighted late-fusion method.
4. Downstream classification: the original **mid-fusion MLP** (concatenate feature vectors), **late fusion of independently trained MLP outputs (unweighted sum)**, and **late fusion with RMS-based weights**. The MSS and pretrained feature extractors are frozen; downstream classifiers are trained.

### Datasets and task definitions

| Family | Dataset | Task / target in original manuscript |
| --- | --- | --- |
| Genre | FMA-Medium | 16-class genre classification |
| Genre | GTZAN | 10-class genre classification |
| Genre | ISMIR04 | 6-class genre classification |
| Mood/theme | MTG-Jamendo mood/theme subset | Multilabel tagging, 56 tags in original manuscript |
| Emotion | PMEmo | Binary valence and binary arousal classification |
| Emotion | EmoMusic | Binary valence and binary arousal classification |
| Emotion | DEAM | Binary valence and binary arousal classification |

The original emotion task **binarizes** continuous emotion annotations; this is not equivalent to the continuous regression targets in EmoMucs. Verify precise label conversions, averaging, metric definitions, and split provenance in the existing code before rerunning.

### Existing experiments versus proposals

- The original FMA-Medium stem analysis evaluated individual stems, **selected four-stem leave-one-out conditions**, all five stems, and **three mixture-plus-stem settings**: Mixture + Guitar + Drums + Piano; Mixture + Guitar + Drums + Piano + Bass; Mixture + All Five Stems. It did **not** exhaustively evaluate all two-/three-/four-stem combinations, and did **not** report mixture-plus-two-stem configurations.
- The original paper used the FMA-Medium **test-set results** in selecting the favored separator (HT-Demucs) and configurations. This is exploratory selection, **not clean validation-based model selection**. Even on a rerun, the already-examined FMA test set is not a pristine unseen holdout.
- Original **genre** results were reported from **one run**. The user reports **emotion** results were averages of **three seeds**; verify stored per-seed metrics/predictions and implementation. The mood/theme seed protocol must be checked rather than assumed.
- Existing extracted features/embeddings are **still available** according to the user. Do not rerun MSS or pretrained extractors unnecessarily; first audit sample IDs, source identity, feature dimensions, layer choice, dataset splits, transforms, labels, and feature provenance.
- The original results are useful **exploratory evidence**, not final conclusive evidence of complementarity, model superiority, or a task-type law.

## 4. Prior work: what is already known, and how our question differs

Focus on research questions and experimental contrasts rather than superficial changes of model names.

| Study | Established / evaluated | Important distinction for this project |
| --- | --- | --- |
| **Rosner & Kostek (2018)** | Genre classification with NMF/openBliSSART-separated components and handcrafted audio descriptors; **original mixture + separated-source feature combinations** already studied. | Not a modern frozen foundation-model embedding evaluation across the three task families. Do **not** claim that mixture-plus-stem fusion originated here. |
| **EmoMucs — De Berardinis et al. (2020)** | Demucs stems; task-trained source-specialized CNNs (C1D/C2D); early/mid/late fusion; PMEmo continuous valence/arousal regression; includes **two-, three-, and four-source subset combinations**. | Does not explicitly retain the original-mixture embedding as an extra fusion branch in its main source-combination study, nor use the same frozen general-purpose music-SSL embedding setup. **It does not rely solely on handcrafted features.** |
| **Garoufis et al. (2024)** | MSS as a pretraining objective; U-Net representations plus classifier backends; joint downstream fine-tuning; genre and music auto-tagging, including per-tag analyses. | Distinct from post-hoc fusion of separately extracted **frozen** mixture and stem embeddings. MSS use across *multiple classification tasks* is not new. |
| **audioLIME — Haunschmid et al. (2020)** | Separated-source perturbations to explain existing music tagging/classification predictions. | Explanation/source attribution is not the same as training a better predictor by adding stem representations. |
| **Li et al. (2026)** | Spleeter vocals/accompaniment, frozen pretrained audio encoders (including MERT), lyrics, cross-attention for emotion recognition; mixture-input versus separated-input ablation. | **Already combines MSS + frozen pretrained representations**, so do not claim that combination as first. Its ablation primarily compares *replacement of mixed inputs* with separated inputs in a lyric/audio system, not incremental **original-mixture + multiple fine-grained stems** under capacity-matched controls across three task families. |
| **Liu et al. (2024)** | Spleeter vocals/accompaniment, MFCC-based CNN/LSTM genre classification, classifier ensembling / late fusion. | Late fusion of separated sources is known; does not address the same strong frozen mixture-embedding baseline. |
| **Lampropoulos et al. (2010)** | Traditional separation + source-specific SVM ensembles for genre recognition. | Historical precedent for separated-source ensemble classification; full-text methodological details may need verification. |
| **Xu et al. (2014)** | Vocals/accompaniment separation with handcrafted features and late fusion for **categorical emotion classification**. | Emotion *classification* with source fusion was already studied, not only emotion regression. |
| **Kang & Herremans (2025)** | Pretrained **MERT** representations extracted from the original mixture, fused with estimated chord/key features for music emotion-related predictions. | No MSS/stem branches in their pipeline. **MERT is a music representation model, not a source-separation model.** This is a relevant strong mixture-based baseline, subject to faithful reproduction. |

### Primary references / starting links

1. Rosner & Kostek (2018): https://doi.org/10.1007/s10844-017-0464-5
2. EmoMucs (ISMIR 2020): https://archives.ismir.net/ismir2020/paper/000308.pdf
3. Garoufis et al. (EUSIPCO 2024): https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000411.pdf
4. audioLIME expanded ISMIR paper (2020): https://arxiv.org/pdf/2009.02051
5. Li et al. (2026): https://doi.org/10.1007/s11042-026-21719-3
6. Liu et al. (2024 preprint): https://arxiv.org/abs/2412.15602
7. Lampropoulos et al. (2010): https://doi.org/10.3233/IDT-2010-0083
8. Xu et al. (ICMR 2014): https://doi.org/10.1145/2578726.2578784
9. Kang & Herremans (2025): https://arxiv.org/abs/2502.03979

This is a **focused non-exhaustive survey**. Don't invent a first-in-literature claim or claim a previous paper omitted an analysis without checking its methods and supplemental material. The technical distinctiveness of our setup does not automatically guarantee sufficient scientific novelty for journal publication.

## 5. Primary experimental concerns to address in revision

These arise from reviewer feedback and subsequent discussion; they are not yet implemented fixes.

1. **Capacity confound — mid fusion:** concatenating mixture and stem embeddings increases input dimensionality and typically classifier parameters. Include a **capacity-matched mixture-only MLP**, reporting full **trainable** parameter counts and using the same training/validation/evaluation rules. Do not use simple copies of the same mixture vector as a supposedly independent information control.
2. **Ensemble confound — late fusion:** multiple branch classifiers introduce an ensemble effect. Compare to a **mixture-only ensemble** with the same branch count, branch architectures, fusion/aggregation, optimization budget, and training protocol. Ensemble branches can receive the same mixture embeddings but should be independently initialized/trained where the method requires independent models.
3. **Training variability:** the revised design uses **five predetermined training seeds**, fixed splits, and **mean ± sample SD** with paired mean differences. A formal significance test or bootstrap CI is **optional**, not automatically required or implied by five-seed reporting. One-run accuracy does not characterize training-seed variability; track-level and training-run uncertainty are different.
4. **Prediction saving:** save sample IDs, ground truth, predicted labels, and scores/probabilities (plus validation-selected thresholds) for every final model/seed. This enables appropriate paired tests (e.g., McNemar for paired correctness, paired bootstrap for score-based metrics). **Aggregate scalar metrics alone are insufficient for such paired prediction-level tests.**
5. **Model-selection leakage:** the chosen revised separator and fixed nested stem sets were influenced historically by FMA test outcomes; **this cannot be retroactively undone**. Keep selections fixed in future reruns, use validation only for new checkpoint/hyperparameter/threshold decisions, disclose the FMA exploration, and do not call its previously examined test set an independent holdout. Any new independent evaluation would be a separate future decision.
6. **Across-task interpretation:** distinguish dataset/task/label effects from feature-extractor, fusion, classifier, split, and sample-size effects. The "instrument-localized genre vs holistic emotion" explanation is **a hypothesis**, not a demonstrated general principle. Report per-class and per-tag analysis if making source-/task-dependent claims.
7. **Baseline reproduction:** the original MTG-Jamendo Kang & Herremans reproduction fell below the published score; the manuscript **already reports and discusses** unavailability of the original chord/key extraction scripts, reconstruction, and classifier-hyperparameter differences. Treat these as possible rather than proven explanations, preserve transparency, and compare stem fusion to the reproduced **within-implementation** mixture-only baseline. Do not require a fresh reproduction effort merely to restate this documented limitation.
8. **Multiple testing / selection bias:** predefine primary metrics, distinguish exploratory from independent confirmatory comparisons, and interpret many comparisons cautiously. Formal tests/multiplicity adjustments are **optional** under the currently agreed Phase 2 plan; **do not use "statistically significant"** unless actually supported by appropriate testing.

## 6. Intended implementation workflow (after advisor approval)

**Do not start training solely because this file exists.** When tasked to implement:

1. **Repository audit:** discover existing code, dataset manifests, feature directories, shapes, checkpoints, seeds, paper-table generation scripts, and current reproducibility issues. Avoid guessing local paths or changing original outputs.
2. **Feature audit:** verify existing mixture/stem representations are aligned on the same tracks, splits, normalizations, and label definitions. Note missing combinations. Reuse valid features instead of recomputing separator or SSL outputs.
3. **Lock a written protocol:** define selected datasets, fixed stems, fusion configurations, primary metrics, hyperparameters/search budget, seeds, validation selection rule, and test evaluation policy **before** looking at new test outcomes. Selected subsets are acceptable; exhaustive combinations are *not* a requirement.
4. **Baseline suite:** retain original mixture-only and original fusion architectures; **add**, rather than replace, the capacity-matched mixture-only MLP and the ensemble-matched mixture-only control.
5. **Pilot implementation:** validate shape handling, matched parameter counts, output-saving, and metric computation on a limited representative dataset **without using its test results for substantive model selection**.
6. **Repeated evaluation:** run matched comparisons across task families using predetermined seeds and existing fixed splits; save all per-seed predictions, provenance, split IDs, and summary tables. Do **not** characterize previously inspected test sets (especially FMA) as newly independent confirmatory evaluations.
7. **Analysis:** report paired effect sizes with uncertainty, comparison to matched controls, per-class/tag patterns where justified, and negative findings. Avoid sweeping task-type generalizations not established by the data.

### Proposed minimum model comparisons

| Model ID | Inputs | Why it is needed |
| --- | --- | --- |
| A — Original mixture-only baseline | Mixture embedding | Historical main reference |
| B — Capacity-matched mixture-only | Mixture embedding; wider/otherwise matched MLP | Controls mid-fusion parameter count |
| C — Mixture + stems, mid fusion | Concatenated original-mixture + selected-stem embeddings | Test incremental stem benefit against B |
| D — Ensemble-matched mixture-only | Multiple independently trained branches using mixture embeddings | Controls late-fusion ensembling |
| E — Mixture + stems, late fusion | Mixture and selected stems fed to comparable branches | Test source diversity against D |

The revised primary matrix **excludes RMS-weighted late fusion**. Parameter matching is a useful control **but does not by itself equalize optimization dynamics, inductive bias, or computational cost**. See the dedicated Phase 2 design file for the agreed 4/5/6-branch and 4/5/6-input comparisons.

### Reproducibility conventions for Codex

- Prioritize **fair paired comparisons**: equal splits, fixed feature provenance, same evaluation code, comparable training budgets, and appropriately matched seeds.
- Keep original data and previously reported results **immutable**. Put revised runs in versioned experiment directories.
- Record configuration, git commit, model/feature version, seed, input stems, output paths, preprocessing, and trainable parameter count in a machine-readable experiment manifest.
- Never silently change split definitions, label binarization, averaging mode, early stopping, or PR-AUC computation in only one method.
- When information is missing, inspect existing project code or ask for the specific path/configuration rather than assuming a conventional implementation.
- Prefer explicit, deterministic experimental settings wherever possible. Document remaining nondeterminism.
- If a requested code change would alter an established architecture, explain and flag the change before proceeding.

## 7. Current decision status and immediate next step

- **Phase 1:** research motivation, literature positioning, contributions and removal of broad "first-work" claims were revised by the author. The abstract and results/conclusion claims remain to be finalized *after* new evidence.
- **Phase 2 design:** main separator/feature/fusion decisions, fixed nested stems, fair controls, five-seed mean±SD reporting, and proposed validation criteria are documented in [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md). **Design decisions are not implemented experiments.**
- **Historical Concern 5:** wording overclaiming HT-Demucs superiority was corrected by the author; the revised scope no longer includes Banquet. This does **not** resolve the separate FMA test-selection issue.
- **Still open:** actual feature/model audits, implementation and clean-up, evaluation with matched controls, test-selection limitation, scientific significance, and results-dependent manuscript revisions.
- **Immediate Codex action:** only understand and preserve this design. The original repository is not fully cleaned; **wait for an explicit request before changing code, running classifiers, editing data, or replacing experimental results**.

---

*Updated 2026-10-28. Research-context and design handoff only; this is not a claim that any new control experiment has been implemented or evaluated.*
