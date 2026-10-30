# Phase 2 — Experimental Design and Fairness Controls (Codex Handoff)

> **Status (updated 2026-10-30): REVISED DESIGN + PARTIAL RESULTS.** The agreed matrix now includes five **standalone stem-branch evaluations** as supporting analyses across all dataset/extractor/target settings. These reuse the independently trained late-fusion branches; **no additional stem-only training jobs** are added. The author has supplied five-seed **FMA-Medium mid- and late-fusion summaries**; the remaining dataset results and standalone stem-branch results have not been provided here. The original repository's implementation/cleanup status has not been independently audited.
>
> **Purpose:** Tell Codex **why** the revision needs new experiments, **which comparisons have been agreed**, and **what still must be checked** before implementing anything. Do not infer existing filenames, implementations, feature sizes, or successful experiments from this plan.
>
> **Related context:** [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md) describes the research question and prior literature; [PHASE1_NOVELTY_REVISION.md](PHASE1_NOVELTY_REVISION.md) records the Phase 1 framing. The original manuscript is historical evidence, **not** the finalized revision protocol.

## A. Objective and interpretation

**Central question:** Given informative frozen pretrained mixture representations (MusicFM or MERT-based features), does adding separately extracted HT-Demucs instrument-stem representations provide **incremental predictive value** for music classification?

**Three linked objectives / contributions:**

1. Does mixture-plus-stem fusion improve performance **relative to strong mixture-only controls**, not just relative to a smaller single MLP?
2. **Effects of different stem combinations:** How do the **three selected nested mixture-plus-stem configurations** affect classification performance, and does adding bass, then vocals, consistently improve the evaluated metrics along this particular sequence?
3. How do positive, negligible, and negative effects differ across the evaluated genre, mood/theme, and binary-emotion **dataset/representation settings**?

**Supporting analysis (not a separate contribution):** Evaluate the individual G/D/P/B/V classifiers already trained as late-fusion branches. Report their standalone performance as context for the mixture–stem fusion results, including possible explanations for unweighted late-fusion degradation. Standalone results do **not** test whether a stem representation adds predictive value beyond the mixture. They may appear in the main Results tables without changing Contribution 2.

**Interpretation limits:** This is a comparative representation study, not a new separation algorithm, fusion mechanism, or an assertion that separation introduces information absent from the mixture waveform. Mid/late fusion are **evaluation strategies**, not claimed methodological novelties. Do not infer a causal task-type law from data collected with different feature extractors or datasets.

## B. Agreed experiment scope

| Dimension | Decision | Notes |
| --- | --- | --- |
| Separator | **Pretrained HT-Demucs only**, `htdemucs_6s` | The revised experimental scope **removes Banquet**; this is not evidence HT-Demucs is globally better. Preserve old Banquet code/results until repository cleanup is separately authorized. |
| Stems | **G**uitar, **D**rums, **P**iano, **B**ass, **V**ocals | The native HT-Demucs checkpoint also outputs `other`, which is excluded from these five selected stems. Here **All Stems means all five named stems, NOT `other`.** |
| Representation pipelines | Frozen **MusicFM**; frozen **Kang & Herremans-style MERT + chord/key** pipeline | Keep both existing feature-extraction paths. Check how chord/key descriptors are attached to mixture/stem inputs before assuming equal embedding sizes. |
| Fusion architectures | **Mid-fusion MLP** and **unweighted late-fusion MLP ensemble** | **RMS-weighted late-fusion is removed from the revised primary scope.** |
| Main fusion stem sets | **M+G+D+P**, **M+G+D+P+B**, **M+G+D+P+B+V** | `M` means original mixture; `M+All Stems` means the mixture plus five selected stems. The three nested sets were historically influenced by FMA-Medium test results; they are **not optimized, exhaustive, or independently prespecified**. |
| Supporting standalone inputs | **G Only, D Only, P Only, B Only, V Only** | Evaluate the five **existing late-fusion stem branches separately** across **all 11 dataset × extractor × target settings** and five outer seeds. No new mid-fusion or stem-only classifier training is required if the same branch checkpoints/predictions are reusable. |
| Excluded additions | **No `M+G` or `M+G+D` progression**, no `M+each individual stem` primary matrix | Additional progressively added inputs and separate mixture-plus-one-stem configurations are **not part of the frozen scope**. |
| Main repetitions | **Five actual outer seeds: `32`, `64`, `128`, `256`, `512`** | These replace the example seed identifiers in the earlier draft; apply the same fixed split and task protocol to each run. |
| Main reporting | **Mean ± sample standard deviation** over the five independent runs; mean paired differences | Formal significance tests and confidence intervals are **optional**, not mandatory, and must not be claimed if not performed. |

### Fixed dataset/feature-extractor plan

| Dataset | Task/target | Representation pipeline(s) |
| --- | --- | --- |
| FMA-Medium | Genre (16 classes) | MusicFM |
| GTZAN | Genre (10 classes) | MusicFM |
| ISMIR04 | Genre (6 classes) | MusicFM |
| MTG-Jamendo mood/theme | Multi-label tagging (56 tags in original manuscript) | **MusicFM and** Kang & Herremans-style MERT+chord/key |
| PMEmo | Binary valence and binary arousal (separate heads/runs) | Kang & Herremans-style MERT+chord/key |
| EmoMusic | Binary valence and binary arousal (separate heads/runs) | Kang & Herremans-style MERT+chord/key |
| DEAM | Binary valence and binary arousal (separate heads/runs) | Kang & Herremans-style MERT+chord/key |

The seven datasets are **not seven identical model settings**: treating valence/arousal as distinct targets and MTG-Jamendo's two feature pipelines separately gives **11 dataset × extractor × target settings** in the full proposed matrix. Verify actual label definitions, feature provenance, and usable dataset partitions in code. Different representations by task limit causal cross-task conclusions.

## B.1 Detailed original-pipeline settings (for Codex to audit, not change automatically)

**Source and status conventions.** The facts marked **[MANUSCRIPT]** below are documented in the original submitted manuscript *Enhancing Music Classification via Source Separation and Feature Fusion* (especially Sections **3.1–3.4 and 4.1–4.4**). They describe the **historical implementation as reported**, not proof that every current code path agrees. **[DECIDED]** identifies a revision decision made with the author. **[VERIFY]** is not specified precisely enough in the manuscript or conversation; inspect the existing implementation before writing a new config. **Do not silently replace an unknown with a common library default.** If the PDF is present alongside this handoff, it is named `ATSIP-07-2026-0064_Proof_hi.pdf`; otherwise locate it only if the user supplies access.

### B.1.1 Audio and source-separation path

| Item | Recorded setting / intended handling | Status / provenance |
| --- | --- | --- |
| Original input | Original, unseparated music mixture remains an independent feature input (`M`), **not** a sum reconstructed from the selected stems | **[MANUSCRIPT]** §3.1; **[DECIDED]** keep mixture branch |
| Music source separator | Pretrained **HT-Demucs** checkpoint **`htdemucs_6s`** | **[MANUSCRIPT]** §§3.2, 4.2; **[DECIDED]** sole separator in revision |
| Native separator outputs | `vocals`, `drums`, `bass`, `piano`, `guitar`, `other` | **[MANUSCRIPT]** §3.2 |
| Used source views | `guitar (G)`, `drums (D)`, `piano (P)`, `bass (B)`, `vocals (V)`; **exclude `other`** from proposed combinations | **[MANUSCRIPT]** selected source set; **[DECIDED]** scope |
| Model weights | Freeze pretrained MSS weights; run separator as inference-only preprocessing | **[MANUSCRIPT]** §1 system overview |
| Source ordering for concatenation | Use consistent named order `M, G, D, P, B, V` for the planned configurations | **[DECIDED]** convention; **[VERIFY]** existing feature-order mappings |
| Separation parameters | Input sample rate, channel handling, waveform loudness/normalization, chunking/overlap, resampling, silence handling, inference device, output length and file format | **[VERIFY]** not established precisely in manuscript |
| Feature–audio alignment | All views for a given sample must refer to the same track/crop and retained source IDs; no source should leak across split boundaries | **[DECIDED]** fairness requirement; **[VERIFY]** implementation |
| Historical alternatives | Banquet query audio and RMS-weighted late-fusion were present in the **original** work, but are **not** part of the revised primary experiments | **[MANUSCRIPT]** §§3.2–3.4; **[DECIDED]** exclude from new matrix, preserve old assets |

**Important:** `htdemucs_6s` separates **six** sources natively. `M+All Stems` in this project means the original mixture **plus five selected instrument/vocal stems**, so there are **six classifier inputs** but **not** the `other` output.

### B.1.2 MusicFM feature-extraction path

| Item | Setting | Status / provenance |
| --- | --- | --- |
| Model | **MusicFM**, pretrained **MusicFM-MSD** checkpoint | **[MANUSCRIPT]** §§3.3, 4.2 |
| Pretraining/model family | Conformer architecture, self-supervised learning using **BERT-RQ** | **[MANUSCRIPT]** §3.3; descriptive context, not a new training step |
| Feature tap | Frame-level representation **after the seventh Conformer layer** | **[MANUSCRIPT]** §§3.3, 4.2 |
| Temporal aggregation | **Average-pool framewise features** to obtain a track/audio-level vector | **[MANUSCRIPT]** §3.3 |
| Trainability | Keep pretrained MusicFM **frozen**; train downstream MLP(s) only | **[MANUSCRIPT]** §1 system overview; **[DECIDED]** retained |
| Audio views | Extract/use features for the **original mixture** and the **selected HT-Demucs stems** in each fusion condition | **[MANUSCRIPT]** §§3.1, 3.3; **[DECIDED]** retained |
| Embedding width | **1024 for MusicFM**, confirmed by the author for the FMA-Medium runs; verify each dataset's saved vectors and the exact first MLP layer | **[AUTHOR-CONFIRMED FMA]**; audit other settings |
| Technical extraction details | Exact MusicFM package/commit, sample rate and conversion, mono/stereo policy, audio length/cropping or windowing, frame mask/padding, batching, mean-pooling axes, optional normalization, saved tensor format and feature filenames | **[VERIFY]** from repository/features; **do not guess** |

**Expected data contract (after verification):** one deterministic track-level embedding for each `(dataset, track_id, source_view)` entry. If a saved tensor is `(N_tracks, d)`, the final axis is `d`; if it is `(N_tracks, T, d)`, verify that the original pooling step still needs to be applied rather than assuming it has been performed.

### B.1.3 Kang & Herremans-style MERT + chord/key path

This is a **representation pipeline**, **not** a source-separation model. The revised study retains this pipeline because its MERT component is a pretrained music representation extractor.

| Item | Setting | Status / provenance |
| --- | --- | --- |
| Core pretrained audio encoder | **MERT-95M**, frozen | **[MANUSCRIPT]** §4.2 |
| Encoder feature layers | **Fifth and sixth transformer layers** | **[MANUSCRIPT]** §4.2 |
| Additional harmonic descriptors | Estimated **chord** and **global key** information | **[MANUSCRIPT]** §3.3 |
| Chord inference | Pretrained **Bi-directional Transformer for Chord Recognition (BTC)**, producing **frame-level chord estimates** | **[MANUSCRIPT]** §4.2 |
| Chord-to-note processing | Use **`mir_eval`** and **`pretty_midi`** to convert detected chords to MIDI note events | **[MANUSCRIPT]** §4.2 |
| Key estimation | Use **`music21`** to estimate a **global key** from those MIDI events | **[MANUSCRIPT]** §4.2 |
| Reproduction provenance | Original Kang & Herremans key/chord extraction scripts were not publicly available; these stages were **re-implemented**, so exact feature equivalence is **not** established | **[MANUSCRIPT]** §§4.2, 5.2 |
| MERT feature combination | Whether layer 5 and 6 outputs are averaged, concatenated, pooled independently, or transformed, and whether any task-specific dimensionality reduction occurs | **[VERIFY]** manuscript gives layers, **not** complete tensor construction |
| Chord/key encoding and scope | Encoding dimensions, one-hot/statistical representation, normalization, whether descriptors come from the mixture only or separately from each stem, and how appended features affect fusion inputs | **[VERIFY]** original method description does not fully specify |
| Encoder/extraction runtime | Exact model revision, sampling rate, cropping, length handling, pooling, normalization, tensor shape, inference determinism and output file paths | **[VERIFY]** original code/features |

**Capacity-control implication:** Do **not** assume that a MERT+chord/key source vector has the same width as MusicFM or even that all MERT-based branches share the same final dimension. Measure **actual concatenated dimensions**. If the input-source widths differ, replace the shorthand `k*d` with the **sum of the actual widths** when matching trainable parameter counts.

**Reproduction-result boundary:** For the original MTG-Jamendo MERT+chord/key experiment the published Kang & Herremans mixture-only PR-AUC was **0.1521**, while the reported local reproduction achieved **0.1279**, and mixture+all-stems achieved **0.1421**. These are **historical single-setting results, not new five-seed measurements**. Do not portray the local fusion result as better than the published baseline. The absent original chord/key scripts and different classifier hyperparameters are **possible**, not proven, reasons for the discrepancy.

### B.1.4 Dataset manifests, labels, and target shapes

The following are dataset characteristics and split descriptions **reported in the manuscript**, not independent validation of every CSV file. Keep exact IDs and class maps from the repository rather than regenerating labels from these summaries.

| Dataset | Original manuscript description | Split handling / checks |
| --- | --- | --- |
| **FMA-Medium** | 25,000 clips, **30 s** each; **16** unbalanced genre classes | **[MANUSCRIPT]** official train/validation/test partitions; **[VERIFY]** actual track-ID sets and counts |
| **GTZAN** | 1,000 clips, **30 s**, ten genres (100 each) | **[MANUSCRIPT]** artist-filtered **500/250/250** train/validation/test partition, attributed there to **Kereliuk et al. (2015)**; **[VERIFY]** manifests and artist overlap |
| **ISMIR04 genre** | 1,458 full tracks, **6** genre labels; originally **729 training + 729 test** in the contest description | **[USER-REPORTED]** existing `ismir04_genre` **train/validation/test CSVs**; **[VERIFY]** provenance and whether validation is drawn only from the original training portion, no duplicate tracks; author prefers to retain existing split |
| **MTG-Jamendo mood/theme** | **17,982** tracks, **56** simultaneous mood/theme labels; multi-label, imbalanced | **[MANUSCRIPT]** official train/validation/test; **[VERIFY]** exact tag list and manifest correspondence |
| **PMEmo** | **794** clips with continuously annotated valence/arousal | **[MANUSCRIPT]** use splits from Kang & Herremans; **[VERIFY]** precise binary-label construction/thresholds and IDs |
| **EmoMusic** | **744** clips, **45 s**, emotion annotations | **[MANUSCRIPT]** use splits from Kang & Herremans; **[VERIFY]** binary-label construction/thresholds and IDs |
| **DEAM** | **1,802** clips, second-by-second valence/arousal; manuscript uses **per-clip mean** label | **[MANUSCRIPT]** use splits from Kang & Herremans; **[VERIFY]** mean interval, threshold(s), IDs |

- **Genre targets:** one mutually exclusive class per track; use saved class-index mapping, not an arbitrary new alphabetical ordering. **[MANUSCRIPT / VERIFY mapping]**
- **Mood/theme targets:** a multi-hot vector of 56 tags per track; metric and threshold handling must support multi-label evaluation. **[MANUSCRIPT / VERIFY encoding]**
- **Emotion targets:** *separate binary valence and binary arousal classifiers* for each of three datasets; do not turn these into regression tasks or a new four-way label unless specifically authorized. **[MANUSCRIPT / DECIDED]**
- **Overlap checks:** train/validation/test IDs must be pairwise disjoint; also check for known artist/recording duplicates where metadata permits. **[VERIFY]**
- **IS MIR 2004 citation caution:** Teacher recalled a 2015 source, but the exact source for the current `ismir04_genre` CSVs has **not** been established. Do not cite the GTZAN 2015 split paper as the ISMIR04 split source without evidence.

### B.1.5 Models, loss, optimizer, and original training settings

**Original classifier description:** Three-layer MLP for **mid-fusion** (concatenate features before classifier) and separate three-layer MLP branches for **late-fusion** (unweighted sum of predictions). The author reports original hidden widths **`1024 -> 512`**. The original manuscript does **not** fully specify activation functions, dropout, normalization, loss functions, regularization, scheduler, or whether late fusion aggregates logits or probabilities; these are **[VERIFY]** items, not design defaults.

| Setting | Genre classification | Mood/theme tagging | Binary valence/arousal |
| --- | --- | --- | --- |
| Optimizer | **Adam** **[MANUSCRIPT]** | **Adam** **[MANUSCRIPT]** | **Adam** **[MANUSCRIPT]** |
| Learning rate | **1e-4** **[MANUSCRIPT]** | **1e-4** **[MANUSCRIPT]** | **1e-4** **[MANUSCRIPT]** |
| Batch size | **128** **[MANUSCRIPT]** | **8** **[MANUSCRIPT]** | **8** **[MANUSCRIPT]** |
| Maximum training epochs | **20** **[MANUSCRIPT]** | **500** **[MANUSCRIPT]** | **500** **[MANUSCRIPT]** |
| Early stopping | Not described beyond best checkpoint within 20 epochs; **do not invent patience** | Patience **10**, minimum `delta=1e-4` **[MANUSCRIPT]** | Patience **10**, minimum `delta=1e-4` **[MANUSCRIPT]** |
| Checkpoint score | Maximize **validation Accuracy + PR-AUC + ROC-AUC**, each scaled 0–1 **[MANUSCRIPT / DECIDED]** | Maximize **validation PR-AUC** **[DECIDED / VERIFY original code]** | Maximize **validation accuracy** **[DECIDED, provisional pending class-balance/code review]** |
| Prediction threshold | Class prediction rule from code; no fixed threshold described for multiclass | Manuscript: choose threshold maximizing **validation F1** **[MANUSCRIPT / VERIFY global vs per-tag]** | Determine original binary threshold rule/positive-label encoding from code **[VERIFY]** |

**Fairness rules:** Apply the **same** data partitions, loss type, optimizer, learning-rate setting, regularization, epoch budget, checkpoint-scoring code, evaluation metrics, and validation-only threshold tuning to each original baseline, matched baseline, and stem-fusion model **within the same dataset/target/representation experiment**. When checkpoint rules change from the historical implementation, **retrain all compared models** under the revised rule. For unweighted late fusion, select each branch checkpoint using the agreed validation protocol, then combine branch predictions by one fixed aggregation rule for both fusion and matched mixture ensembles. Do not allow ensemble-specific test-time tuning.

### B.1.6 Metric definitions and results artifacts

| Item | Documented status | Action for Codex when later authorized |
| --- | --- | --- |
| Metrics | **[MANUSCRIPT]** Accuracy, F1-score, ROC-AUC, PR-AUC | Retain task-appropriate reported metrics; verify exact implementation and which are undefined for particular labels |
| Main metric | **[DECIDED]** Genre: accuracy; mood/theme: PR-AUC; binary valence/arousal: accuracy | Fix exact per-task averaging for F1/ROC/PR and consistency of micro/macro/weighted variants |
| Accuracy scale in genre checkpoint sum | **[DECIDED]** all terms on `[0,1]` scale | Verify source code computes scaled values correctly; do not sum percentage accuracy with unit-scale AUCs |
| Multi-label threshold | **[MANUSCRIPT]** validation-F1-maximizing threshold | **[VERIFY]** one global vs separate tag thresholds; grid/tie handling; never choose threshold using test data |
| PR-AUC computation | **[VERIFY]** average precision vs trapezoidal precision–recall area, class-wise handling, averaging | Preserve original method if possible; report explicitly and compare only identically computed metrics |
| AUC undefined cases | **[VERIFY]** class absent in subset/bootstraps or validation folds | Decide/report stable handling; do not silently substitute arbitrary 0.0 scores |
| Five-seed results | **[DECIDED]** mean ± sample SD, per-seed differences vs matched controls | Save per-track IDs, labels, probabilities/logits, per-epoch validation scores, checkpoint epoch/path, seed and trainable parameter count |
| Extra testing | **[DECIDED]** formal significance tests/CI optional | Do not assert statistical significance from mean±SD alone |

### B.1.7 Required feature/configuration audit outputs (read-only until implementation is authorized)

When the user **later requests an audit**, Codex should **report** — not silently patch — the following configuration inventory separately for **MusicFM** and **MERT+chord/key**, and separately for genre, mood/theme, valence, and arousal where they differ:

1. Actual checkpoint identifier/commit, input audio sampling/cropping/normalization policy, which encoder layers are tapped, pooling axes, feature shapes for mixture and every selected stem, and saved-file paths plus track-ID alignment.
2. The exact feature construction used for chord/key (source audio, model and library versions, representation shape, layer-combination rule, concatenation order) and any differences from the published Kang & Herremans procedure.
3. Dataset CSV locations, partition sizes, ID overlap and label-map checks, train/eval transformations, exact binary emotion threshold(s), and how validation split provenance is recorded.
4. Full MLP architecture including activation/dropout/norm, biases, loss function, optimizer kwargs, weight decay, scheduler, input and output sizes, trainable parameter count, and any task-specific classifier-head differences.
5. Training randomness (worker seeds, shuffle, determinism settings), stopping and best-checkpoint implementation, metric definitions, validation thresholds, probability vs logit aggregation for late fusion, and output-artifact directory conventions.
6. Which historic checkpoints/features/results are valid to **reuse** in the proposed new matrix and which need rerunning to ensure comparable conditions. **No training, extraction, deletion, or refactoring** during a documentation-only audit.

---

## C. Model-comparison matrix (same dataset/extractor/target and fixed splits)

| Model / input condition | Mid-fusion | Unweighted late-fusion | Intended control or comparison |
| --- | :---: | :---: | --- |
| Original **Mixture Only** single MLP | Yes | Shared reference | Historical baseline; **do not replace** with a wider model. |
| **Capacity-matched Mixture Only (4)** | Yes | — | Approximately match M+G+D+P mid-fusion trainable parameters. |
| **Capacity-matched Mixture Only (5)** | Yes | — | Approximately match M+G+D+P+B mid-fusion trainable parameters. |
| **Capacity-matched Mixture Only (6)** | Yes | — | Approximately match M+All Stems mid-fusion trainable parameters. |
| **Mixture-only ensemble (4)** | — | Yes | Four independent classifiers, each fed the same mixture features. |
| **Mixture-only ensemble (5)** | — | Yes | Five independent mixture classifiers. |
| **Mixture-only ensemble (6)** | — | Yes | Six independent mixture classifiers. |
| **M+G+D+P** | Yes | Yes | Compare mid against matched (4), late against mixture ensemble (4). |
| **M+G+D+P+B** | Yes | Yes | Compare mid against matched (5), late against mixture ensemble (5). |
| **M+All Stems** | Yes | Yes | Compare mid against matched (6), late against mixture ensemble (6). |
| **Guitar Only (`G`)** | — | **Evaluate branch** | Standalone accuracy/F1/AUC/AP of the existing guitar late-fusion branch; supporting, not evidence of added information beyond M. |
| **Drums Only (`D`)** | — | **Evaluate branch** | Same standalone analysis for drums. |
| **Piano Only (`P`)** | — | **Evaluate branch** | Same standalone analysis for piano. |
| **Bass Only (`B`)** | — | **Evaluate branch** | Same standalone analysis for bass. |
| **Vocals Only (`V`)** | — | **Evaluate branch** | Same standalone analysis for vocals. |

**Standalone-stem analysis policy:** Report all five stem-branch metrics **for each of the 11 dataset/extractor/target settings and five outer seeds**, subject to checkpoint/prediction availability. These are **evaluations of the already trained late-fusion branches**, not a new fusion architecture or a second checkpoint-tuning procedure. Use the exact fixed split and the branch checkpoints selected using the predetermined validation rules; do not select new checkpoints based on stem-only test scores. For the MERT+chord/key path, **verify the source of chord/key descriptors**. If any stem branch receives descriptors extracted from the original mixture, call the condition "stem audio representation + mixture-derived chord/key descriptors," **not strictly stem-only**. Differences in standalone accuracy indicate different *predictive effectiveness* of representations, **not** new waveform information and **not** incremental benefit beyond M. Weaker individual branches may help explain late-fusion failure but are not sufficient: also examine complementary errors and logit scales if making mechanistic claims.

Report each proposed fusion model **against its corresponding matched control**. Also retain the ordinary single-MLP mixture-only reference for interpretability. Do not present fusion versus a single mixture MLP as proof that stem features outperform a capacity/ensemble-controlled baseline.

## C.1 Complete seeded evaluation matrix — fixed scope, partial completion

**Fixed outer-seed IDs:** `32`, `64`, `128`, `256`, `512`, matching the author's five-seed FMA-Medium experiments. Earlier example IDs (`42`, `123`, `2024`, `2025`, `2026`) were **not** the actual runs and have been removed from the planning manifests. Keep the same dataset partitions across seeds. Checkmarks below mean **required by the final matrix**, not that every dataset's run is complete.

| Reportable model / setting | 32 | 64 | 128 | 256 | 512 | Independent trainings per outer seed |
| --- | :---: | :---: | :---: | :---: | :---: | ---: |
| Mixture Only (`M1`) | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: matched Mixture Only (4 inputs) | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: matched Mixture Only (5 inputs) | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: matched Mixture Only (6 inputs) | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: M+G+D+P | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: M+G+D+P+B | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Mid: M+All Stems (=M+G+D+P+B+V) | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| Late: Mixture Ensemble (4) | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse mixture branches) |
| Late: Mixture Ensemble (5) | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse mixture branches) |
| Late: Mixture Ensemble (6) | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse mixture branches) |
| Late: M+G+D+P | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse M1+G+D+P) |
| Late: M+G+D+P+B | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse M1+G+D+P+B) |
| Late: M+All Stems | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (reuse M1+G+D+P+B+V) |
| **Supporting: Guitar Only** | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (evaluate late `G` branch) |
| **Supporting: Drums Only** | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (evaluate late `D` branch) |
| **Supporting: Piano Only** | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (evaluate late `P` branch) |
| **Supporting: Bass Only** | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (evaluate late `B` branch) |
| **Supporting: Vocals Only** | ✓ | ✓ | ✓ | ✓ | ✓ | **0 new** (evaluate late `V` branch) |

Late-fusion ensembles still require **five additional mixture branches (`M2..M6`) and five stem branches (`G,D,P,B,V`)** in addition to the seven ordinary mid/single MLP jobs. Reusing compatible `M1` yields **17 independent classifier trainings** but now **18 reportable evaluations** (13 former main comparisons + 5 standalone stems) per dataset/extractor/target per outer seed. No extra training comes from reporting stem-only branch performance. If the stems' checkpoints/predictions are missing, or the task's pipeline differs, audit/recover them before asserting reuse.

**Dataset / feature-pipeline / target coverage (five seeds and all 18 evaluations for every setting):**

| Dataset | Feature pipeline | Target(s) | Outer seeds | Nominal distinct MLP trainings (with reuse) |
| --- | --- | --- | --- | ---: |
| FMA-Medium | MusicFM | Genre | 32, 64, 128, 256, 512 | 85 |
| GTZAN | MusicFM | Genre | 32, 64, 128, 256, 512 | 85 |
| ISMIR04 | MusicFM | Genre | 32, 64, 128, 256, 512 | 85 |
| MTG-Jamendo | MusicFM | Mood/theme | 32, 64, 128, 256, 512 | 85 |
| MTG-Jamendo | MERT+chord/key | Mood/theme | 32, 64, 128, 256, 512 | 85 |
| PMEmo | MERT+chord/key | Valence | 32, 64, 128, 256, 512 | 85 |
| PMEmo | MERT+chord/key | Arousal | 32, 64, 128, 256, 512 | 85 |
| EmoMusic | MERT+chord/key | Valence | 32, 64, 128, 256, 512 | 85 |
| EmoMusic | MERT+chord/key | Arousal | 32, 64, 128, 256, 512 | 85 |
| DEAM | MERT+chord/key | Valence | 32, 64, 128, 256, 512 | 85 |
| DEAM | MERT+chord/key | Arousal | 32, 64, 128, 256, 512 | 85 |
| **Total** | **11 settings** | | **5 outer seeds each** | **935 training jobs; 990 reportable evaluations** |

**Late-fusion deterministic branch-seed convention — retain the implemented FMA-Medium approach:** Use the existing [`branch_seed()` implementation](../src/stemfusion/training/late_fusion.py) for all dataset/extractor/target settings. The outer seeds remain `[32, 64, 128, 256, 512]`. For outer seed `s`, branch index `0` uses `s`; indices `1..5` use the first four bytes of the SHA256 digest of the UTF-8 string `stemfusion-phase2:{s}:{index}`, interpreted as an unsigned big-endian integer:

```python
import hashlib

def branch_seed(outer_seed: int, index: int) -> int:
    if index == 0:
        return outer_seed
    message = f"stemfusion-phase2:{outer_seed}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:4], "big")
```

| Branch index | Mixture-only branch | Stem branch at the same position |
| --- | --- | --- |
| 0 | M1; also the ordinary single-MLP mixture reference | Original mixture branch in stem ensembles |
| 1 | M2 | Guitar |
| 2 | M3 | Drums |
| 3 | M4 | Piano |
| 4 | M5 | Bass |
| 5 | M6 | Vocals |

For outer seed `32`, the six position seeds are `32, 2682760092, 2560740051, 3596998195, 2746825346, 2361236136`. Mixture-only and stem branches at matching positions receive the same derived seed, but are trained separately on their respective features. Branch seeds within each ensemble are distinct. M1 is shared with the original mixture-only reference when settings match.

All **55 saved FMA-Medium branch records** (11 branches × 5 outer seeds) were checked against this function and agree. **Preserve these seeds, checkpoints, and results; no FMA retraining or seed reassignment is required.** Any planning-manifest field named `actual_training_seed` must follow this implemented convention. For other datasets, the derived schedule specifies intended seeds; completion must still be established from the saved run records.

**Machine-readable planning manifests:** [`PHASE2_SEEDED_EXPERIMENT_MATRIX.csv`](PHASE2_SEEDED_EXPERIMENT_MATRIX.csv) now has **990 reportable dataset × outer-seed × model rows** (`11 × 5 × 18`). [`PHASE2_SEEDED_TRAINING_JOBS.csv`](PHASE2_SEEDED_TRAINING_JOBS.csv) retains **935 nominal MLP training jobs** (`11 × 5 × 17`). The manifests are **planning inventories**, not proof that runs have been executed. The newly added stem-only report rows **reuse** existing `late_stem_*` training jobs.

## C.2 FMA-Medium results received from author (not yet independently audited)

For the five outer seeds above, the author supplied these aggregate metrics (accuracy in %, macro ROC-AUC unit scale). These results are **specific to FMA-Medium / MusicFM** and should not be generalized before other dataset results are available.

| Condition (matched comparison) | Fusion accuracy | Matched control accuracy | Fusion macro ROC-AUC | Matched control macro ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Mid M+G+D+P vs matched (4) | 70.60 | 70.31 | 0.9007 | 0.8897 |
| Mid M+G+D+P+B vs matched (5) | 70.53 | 70.89 | 0.9073 | 0.8909 |
| Mid M+G+D+P+B+V vs matched (6) | 70.49 | 70.43 | 0.9044 | 0.8906 |
| Late M+G+D+P vs mixture ensemble (4) | 69.05 | 70.74 | 0.8933 | 0.8941 |
| Late M+G+D+P+B vs mixture ensemble (5) | 68.98 | 70.93 | 0.8939 | 0.8947 |
| Late M+G+D+P+B+V vs mixture ensemble (6) | 68.91 | 70.92 | 0.8981 | 0.8948 |

The ordinary Mixture Only reference was **70.52 ± 0.27% accuracy** and **0.8944 ± 0.0032 macro ROC-AUC**. Mid-fusion improves mean macro ROC-AUC in all three matched comparisons but has mixed accuracy/F1/AP; unweighted late-fusion underperforms matched mixture ensembles on accuracy, F1 and AP in all three configurations. **No individual-stem-only metrics have yet been provided in this conversation.** Do not insert invented values. These comparisons do **not** establish statistical significance; the FMA test set was historically used in configuration selection.

## D. Mid-fusion parameter-matching design — provisionally specified

**Existing architecture (author-reported):** a three-linear-layer MLP with hidden widths **1024 and 512**, i.e. `input -> 1024 -> 512 -> output`. The exact code may have dropout, normalization, additional trainable layers, or other details; **audit it rather than assuming the prose describes every parameter**.

**Author-confirmed for FMA-Medium:** MusicFM per-source embedding dimension `d = 1024`. Audit actual saved feature widths for remaining settings. MERT+chord/key final input dimension must be measured separately.

For a source embedding width `d`, output width `C`, and `k` inputs (the mixture plus `k-1` stems):

- Stem-fusion MLP: `k*d -> 1024 -> 512 -> C`.
- Capacity-matched mixture-only MLP: `d -> h_k -> 512 -> C`.
- For **three biased linear layers only**, trainable parameters of `D -> h1 -> h2 -> C` are `(D+1)*h1 + (h1+1)*h2 + (h2+1)*C`.
- Under the same restricted assumptions, a first-hidden-layer width approximately matching a `k`-input fusion model is `h_k ≈ 1024*(k*d + 513)/(d + 513)`, rounded to a positive integer with minimum actual parameter-count difference. **Compute from the real model, not just this expression**, if it includes other trainable components.

**Illustrative FMA-Medium calculation** (`d=1024`, `C=16`, three biased linear layers and nothing else; not a substitute for auditing trainable model counts):

| Model | Architecture | Trainable parameters |
| --- | --- | ---: |
| Original mixture only | `1024 -> 1024 -> 512 -> 16` | 1,582,608 |
| Fusion M+G+D+P | `4096 -> 1024 -> 512 -> 16` | 4,728,336 |
| Matched mixture only (4) | `1024 -> 3071 -> 512 -> 16` | 4,728,847 |
| Fusion M+G+D+P+B | `5120 -> 1024 -> 512 -> 16` | 5,776,912 |
| Matched mixture only (5) | `1024 -> 3753 -> 512 -> 16` | 5,777,081 |
| Fusion M+All Stems | `6144 -> 1024 -> 512 -> 16` | 6,825,488 |
| Matched mixture only (6) | `1024 -> 4435 -> 512 -> 16` | 6,825,315 |

These example counts are **not confirmed project measurements**. Parameter matching controls one confound but does not equalize input information, architecture inductive biases, compute cost, optimization difficulty, or generalization. Do **not** create controls by merely concatenating the same mixture feature vector multiple times.

## E. Unweighted late-fusion ensemble-matching design — agreed principle

Use the same original branch MLP architecture (`input -> 1024 -> 512 -> C`, subject to code audit), same input width per source, same training/selection rules and **same aggregation rule** in fusion and its matched control.

| Stem-fusion ensemble | Matched mixture-only ensemble | Number of branches |
| --- | --- | ---: |
| `M + G + D + P` | `M1 + M2 + M3 + M4` | 4 |
| `M + G + D + P + B` | `M1 + M2 + M3 + M4 + M5` | 5 |
| `M + G + D + P + B + V` | `M1 + M2 + M3 + M4 + M5 + M6` | 6 |

`M1`–`M6` **all receive the same original mixture embedding**, but must have **independent initializations/training stochasticity**. Do not copy one trained model six times. For each outer run, the five stem-specific branches and the six mixture-only branches can be **reused** when assembling nested configurations, if their training/selection procedures truly match. The single mixture-only MLP may be shared with `M1` and the mid-fusion mixture-only reference if implementations and settings are identical.

**Author-confirmed implementation rule:** late-fusion **sums branch logits without per-branch weighting, then applies softmax** for multiclass prediction. For multilabel/binary heads, verify the equivalent head-specific output transformation in code; do not assume softmax is valid for multilabel tasks. Use the **same logit-sum rule** for stem-fusion and matched mixture-only ensembles. Preserve per-branch raw logits for standalone evaluation and diagnosis; check any thresholding/calibration using validation data only. Audit code to confirm that all task-specific pathways implement the reported rule.

**Selection principle:** Prefer selecting **each branch checkpoint** under the task's predefined validation criterion, then aggregating the selected branches without additional test-driven ensemble selection. Apply the same branch procedure in both types of ensemble. This is the planned procedure, not an observed fact about original code.

## F. Dataset splits, validation, and checkpoints

**Retain existing dataset splits** for the revised comparisons, subject to a non-destructive audit of track IDs and overlap. The author reports that the current `ismir04_genre` dataset uses **three CSV manifests (train/validation/test)**. The teacher recalls that the split came from a 2015 paper, but the exact provenance/citation has **not** been established; **do not invent a citation or silently claim the split is official**. Preserve the existing CSVs; verify uniqueness/overlap and mapping to original partitions before evaluation.

**Decided task-specific checkpoint rules:**

| Task | Checkpoint selection on validation | Training budget / early stopping |
| --- | --- | --- |
| Genre (FMA-Medium, GTZAN, ISMIR04) | Maximize `validation_accuracy + validation_PR_AUC + validation_ROC_AUC`, each on **0–1** scale | Original regime: Adam `1e-4`, batch size 128, max 20 epochs; choose best checkpoint. |
| MTG-Jamendo mood/theme | Maximize **validation PR-AUC** (averaging definition must be verified/fixed) | Original regime: Adam `1e-4`, batch size 8, max 500 epochs, patience 10, min delta `1e-4`. |
| Binary emotion (valence and arousal) | Maximize **validation accuracy**, provisionally (audit label balance and existing implementation) | Same original mood/emotion regime: Adam `1e-4`, batch size 8, max 500 epochs, patience 10, min delta `1e-4`. |

These are **provisionally chosen rules**, not necessarily the exact rules already in original mood/emotion training code. A metric change must be implemented **for every model within the same dataset/target**, including the original mixture-only baseline; never mix checkpoints selected under different criteria in one comparison. Fix conventions for macro/micro/weighted AUCs and F1, ties, invalid one-class AUC cases, and validation-selected prediction thresholds. Do not tune/checkpoint on test data.

## G. Five-seed evaluation and results reporting

1. Use the **fixed outer seed list `[32, 64, 128, 256, 512]`** and the **existing FMA-Medium SHA256 branch-seed convention** in Section **C.1** for every dataset/extractor/target setting. Keep dataset splits fixed across runs; vary training randomness. Do not treat different branches within one ensemble as separate outer runs.
2. Run the full comparison consistently within each dataset/extractor/target setting. Do not reuse a checkpoint trained under incompatible features, labels, split, metric definition, or architecture.
3. Report **mean ± sample SD (`ddof=1`)** across five runs for each reported metric. Also report the **mean of per-seed paired differences** between each fusion system and the corresponding matched control; optionally its SD. Match the outer run/seed conditions for each difference.
4. **Formal tests / bootstrap confidence intervals are optional.** Five seeds and SD characterize *training-run variability*, **not statistical significance**. Never write "statistically significant" without a justified performed test. Do not interpret overlapping/nonoverlapping SD ranges as significance testing.
5. Save for every run: model/training config and seed; checkpoint path/epoch/validation metric; per-track IDs and ground truth; per-track output probabilities or logits; final metrics; trainable parameter counts; any validation-chosen thresholds. Preserve source-level branch predictions for late fusion so ensembles can be analyzed without retraining.
6. Report mixed, null, and negative findings, and separate exploratory diagnostic metrics from primary task metrics: **genre accuracy**, **mood/theme PR-AUC**, and **binary emotion accuracy** (valence and arousal separately) are proposed primary headline measures. Still report agreed secondary metrics where defined.
7. For the **five standalone stem branches**, collect metrics for **every** dataset/extractor/target setting using the existing late-fusion branch checkpoints and saved logits. Compare descriptively against the matching mixture-only single-branch classifier; do **not** treat lower standalone accuracy as proof that the stem contains no complementary information. Where feasible, report branch error agreement, positive/negative decision changes after fusion, and logit-scale summaries to examine possible explanations for late-fusion degradation. Never claim such an explanation is established from standalone scores alone.

**Optional future analyses:** paired example-level uncertainty intervals, formal tests with multiple-comparison handling, per-class genre/per-tag mood analyses. These are not preconditions for starting the agreed design, but keeping predictions now makes them feasible later.

## H. Important scientific/reporting constraints

- **Historical FMA test selection cannot be undone:** The three nested stem configurations and original decision to favor HT-Demucs were influenced by prior FMA-Medium **test** scores. They may be fixed *now* for the revised matrix, but do **not** label them "predetermined before any test results" or claim a rerun on the same test split is newly independent/confirmatory. Report the historical exploratory selection transparently. Removing Banquet from the revised scope does not erase the history.
- **The three stem combinations are intentionally limited:** They test one nested addition sequence (bass, then vocals), not global optimality, exhaustive combinations, source attribution, or the effect of all possible inclusion orders. **Neither `M+G` nor `M+G+D` is included.** Extra mixture-plus-one-stem/two-stem/leave-one-out experiments are **not** required by the current agreed design.
- **Stem-only results are supporting diagnostics:** The five late-fusion branches are evaluated alone across all 11 settings. Their classification scores measure standalone predictive performance, **not** whether a stem supplies incremental predictive value *conditional on M*. Strong/weak stems do not automatically make a late ensemble improve/degrade. Preserve the MERT chord/key provenance caveat. No new training is assumed under valid checkpoint reuse.
- **Kang & Herremans reproduction discrepancy is already addressed at the reporting level:** The original manuscript reports published MTG-Jamendo PR-AUC `0.1521`, our reproduced mixture-only `0.1279`, and mixture+all-stems `0.1421`. It explains that key/chord extraction scripts were unavailable and had to be reconstructed; classifier hyperparameters also differ. These are **plausible** sources of the gap, not proven causal explanations. Preserve transparent comparison; do not claim our fusion exceeds the published paper's result or require a new reproduction experiment just to restate this limitation.
- **Do not overclaim task effects:** Performance contrasts between MusicFM on genre and MERT+chord/key on emotion are confounded by representation and datasets; use a conditional, descriptive interpretation. MTG-Jamendo contains both feature setups and may provide a within-dataset comparison.
- **Fairness isn't only a parameter count:** Preserve matching splits, preprocessing, losses, optimizer/scheduler, batch construction, regularization, validation selection, metric definitions, and comparable training budgets. Report mismatches honestly.
- **No general "first" claim:** The revised contribution is an empirical question about the *incremental predictive value* of retaining mixture embeddings and adding instrument stems from frozen pretrained representations, not first use of MSS, first stem fusion, or first SSL+MSS.

## I. Implementation-readiness audit / Codex boundaries

This section describes checks to perform **when implementation is explicitly requested later**. It is **not permission to refactor or run** now.

- [ ] Locate source modules, configs, scripts, original result files, saved features, and split CSVs. Record provenance; do not rename/delete them.
- [ ] Reconfirm author-reported MusicFM 1024-dimensional FMA embeddings against feature files; verify remaining MusicFM and MERT/chord/key dimensions, temporal pooling/normalization, correct mixture/stem alignment, `other`-stem omission, data types and sample IDs.
- [ ] Inspect the **actual** three-layer MLP and count trainable parameters including optional normalization. Produce exact width matching from real counts for every dataset/extractor/target.
- [ ] Verify author-reported unweighted **logit sum → softmax** for multiclass genre; audit multilabel/binary heads and branch checkpoint selection. Ensure independent mixture ensemble branch training, and preserve each stem branch's per-example logits and standalone scores across all settings.
- [ ] Check all dataset split overlaps and especially the `ismir04_genre` train/validation/test CSV manifests. Identify 2015 split provenance only if available; do not fabricate.
- [ ] Audit existing mood/theme and emotion early-stopping metric; implement the agreed criterion consistently for **all** baselines if it differs.
- [ ] Freeze metrics (averaging, label binarization, thresholds, positive-class orientation) and the fixed five outer seeds in a manifest before any new test-based exploration.
- [ ] Preserve original code/outputs. Place any future revised experiment implementation in clearly versioned configs or modules, with an explicit review/diff before execution.
- [ ] Preserve complete predictions and trained checkpoint metadata so results can be recomputed without new training.

### Training-count planning note (not a run request)

Under the **full** matrix, for **one dataset × extractor × target setting and one outer seed**, the independent model trainings could be approximately:

- Mid-fusion: **1** standard mixture-only + **3** wider matched mixture-only + **3** stem-fusion classifiers = **7**.
- Late-fusion: **6** independently trained mixture-only branches + **5** independent stem branches = **11**, reused to form 4/5/6-member ensembles.
- If the mid-fusion single mixture-only model is identical to late-fusion's `M1`, **one training can be shared**, giving **17 distinct trainings per outer seed** and **18 reportable model evaluations**, five of which are standalone evaluations of already-trained stem branches.

At 5 seeds × 11 settings, this is a *nominal upper planning figure* of **935 distinct classifier trainings** before accounting for existing valid checkpoints or changes to scope. This figure is conditional on the assumptions above; **do not schedule experiments from it**, and do not confuse it with separator or feature-extractor inference costs. The number of ensemble evaluations is not the number of independently trained branches.

## J. Current handoff state

**Decided in conversation:** separator and fusion scope, **three existing nested mixture-plus-stem sets** (no incremental 1/2-stem additions), **five standalone stem-only diagnostic evaluations across all settings**, matched-control principles, both feature pipelines, actual outer seeds `32,64,128,256,512`, task-specific validation metric proposal, five-seed mean±SD reporting, optional significance tests, and retaining current split manifests.

**Provisional / must verify from implementation:** MusicFM feature-file width beyond author-confirmed FMA `d=1024`, sample rate/cropping/normalization and feature pooling; exact MLP activations/dropout/norm/loss and optimizer kwargs; MERT layers 5/6 combination plus chord/key encoding and stem-specific handling; author-reported logit-sum aggregation in each task head; exact mood/emotion early-stopping code; class imbalance/metric definitions and emotion-label binarization; consistent application of the verified FMA-Medium SHA256 branch-seed convention to remaining datasets; ISMIR04 CSV overlap and bibliographic provenance; availability of all stem-branch checkpoints/predictions. Consult detailed inventory in **B.1**.

**Not completed / not independently audited:** code cleanup, end-to-end matched comparisons across the remaining datasets, standalone-stem metrics, paper-result table updates, abstract/conclusion finalization, independent-holdout resolution of historical FMA test selection. **Five-seed FMA-Medium mid- and late-fusion summaries have been supplied by the author**, as captured in §C.2; do not imply that no new experiments exist.

**Codex instruction:** Treat this as a **design and purpose document**. **Do not edit application code, clean the repository, train models, delete previous experiments, or revise the manuscript automatically** merely because this file is present. Wait for a specific author request, then audit before implementing.
