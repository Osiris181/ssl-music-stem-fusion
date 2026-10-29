# Phase 1 — Research Direction, Contributions, and Related Work

> **Status (updated 2026-10-28):** Phase 1 novelty/research-direction revisions have been discussed and revised by the author. The paragraphs below preserve earlier **drafting/reference language**, not necessarily the author's exact latest LaTeX and not evidence that the new controls have been run. The abstract remains deferred until results are available.
>
> **Phase 2 agreed design:** [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md). Use that file, not preliminary proposals in this Phase 1 draft, for current separator, fusion, stem and statistical decisions. **Do not rewrite the manuscript or change code automatically.**
>
> References use author–year names to avoid inventing BibTeX keys. Replace with the manuscript's actual `\cite{...}` commands when integrating into LaTeX.

## A. Central research direction (agreed framing)

**Research question:** Given that modern pretrained music self-supervised learning (SSL) models can already extract informative representations directly from mixture audio, to what extent does explicitly adding separated-instrument representations provide incremental predictive value for downstream music classification? Under what conditions does adding stems help, provide little benefit, or hurt?

This concerns *additional predictive value beyond a strong mixture-only pretrained representation*, not whether source separation has ever been used for MIR classification, or whether MSS has ever been combined with pretrained feature extractors. This question is empirical and depends on appropriate controls.

## B. Proposed replacement: Introduction motivation / research gap

Advances in music source separation (MSS) make it possible to isolate different musical sources from polyphonic recordings, potentially exposing source-specific cues that are difficult for downstream models to exploit directly from a mixture. Previous studies have shown that representations or predictions derived from separated sources can benefit music classification and emotion recognition (e.g., Rosner and Kostek, 2018; De Berardinis et al., 2020; Garoufis et al., 2024). Meanwhile, pretrained self-supervised learning (SSL) models such as MusicFM and MERT have demonstrated the ability to extract informative music representations directly from the original, unseparated audio mixture.

These advances raise a more specific question: when an informative pretrained representation of the mixture is already available, does incorporating additional representations from separated musical sources still improve downstream classification? Source separation does not introduce new information into the original recording, but it may reorganize acoustic information in a way that makes source-related patterns more accessible to a downstream classifier. Whether such benefits persist with pretrained mixture embeddings—and whether they justify the increased complexity of processing and fusing multiple stems—requires careful empirical evaluation.

Prior studies have examined related but not identical settings. For example, Rosner and Kostek (2018) combined features of the original recording with features from separated components, but relied on handcrafted descriptors; EmoMucs (De Berardinis et al., 2020) investigated different separated-source combinations using task-specific learned CNN features; and Li et al. (2026) combined pretrained audio encoders with vocal/accompaniment separation and lyrical features for music emotion recognition. The present study focuses on the incremental predictive contribution of explicitly adding selected fine-grained instrument-stem embeddings to a **retained, pretrained mixture embedding**. It evaluates this question across genre classification, mood/theme tagging, and binary emotion recognition, while distinguishing potential stem-specific benefits from differences in classifier capacity and ensembling.

## C. Proposed replacement: Contributions (Introduction)

The revised study is organized around three empirical objectives:

1. **Incremental predictive value of separated stems given pretrained mixture representations.** We evaluate explicit mixture-plus-stem feature fusion using a fixed pretrained HT-Demucs MSS model and frozen music representation extractors, comparing it against mixture-only baselines. The revised experimental design will include appropriate controls for the increased trainable classifier capacity and ensemble effects introduced by fusion.

2. **Complementarity of selected stem combinations.** We investigate whether adding selected groups of instrument stems to the mixture representation changes downstream predictive performance, and whether adding more stems consistently provides additional benefit. This analysis concerns **mixture-plus-stem combinations**, which should be distinguished from combining separated stems without an original-mixture input. The selection of stem subsets will be specified and not described as exhaustive.

3. **Variation across downstream classification tasks.** We characterize the benefits, negligible effects, and possible performance degradations of mixture-plus-stem fusion across genre classification, mood/theme tagging, and binary valence/arousal classification. Any explanation in terms of instrument-localized versus holistic musical cues is treated as a hypothesis to be examined, not an established universal principle.

> Writing caution: The phrase “will include” in Contribution 1 is deliberate: capacity-/ensemble-matched controls are planned but not yet completed. Reword to completed-work tense only once corresponding experiments are finished. The **current agreed** stem analysis uses three nested combinations (`M+G+D+P`, `M+G+D+P+B`, `M+All Stems`); mixture-plus-two-stem experiments are **not required** under that selected-combination objective. Do not claim this sequence is exhaustive or proves globally optimal subsets.

## D. Proposed replacement / expansion: Related Work (music classification using separated sources)

Music source separation has previously been explored as a way to improve music-information-retrieval classification models. Rosner and Kostek (2018) used NMF-based separation to obtain harmonic, percussion, and instrument-related components, and extended feature vectors from the original recording with handcrafted descriptors derived from separated signals for genre classification. Consequently, combining mixture and separated-source features is **not** itself a new idea. Earlier studies also investigated separated-source ensemble models for genre classification (Lampropoulos et al., 2010) and vocals/accompaniment fusion for categorical music emotion recognition (Xu et al., 2014).

More recent work incorporates neural source separation and learned features. EmoMucs (De Berardinis et al., 2020) applied Demucs to obtain vocals, drums, bass, and other accompaniment, processed log-mel inputs with source-specific CNNs, and evaluated early, mid, and late fusion for continuous valence/arousal regression on PMEmo. Importantly, EmoMucs also evaluated two-, three-, and four-source combinations; therefore, stem-combination analysis cannot be claimed as new. Its source-combination study, however, did not explicitly preserve the **original mixture representation as an additional input** alongside those combinations, and it did not use frozen, general-purpose pretrained music SSL embeddings in the manner studied here. Garoufis et al. (2024) proposed MSS-based pretraining for genre classification and music tagging, with separation-network representations integrated into downstream classifiers and joint fine-tuning. audioLIME (Haunschmid et al., 2020) used separated-source perturbations to explain predictions rather than studying the accuracy of trained mixture-plus-stem feature fusion.

The use of pretrained music representations is also relevant. Kang and Herremans (2025) combined MERT embeddings with musical key and chord information without first applying MSS. Li et al. (2026) used pretrained Data2Vec and MERT to represent separated vocals and accompaniment and fused these audio sources with lyrics through cross-attention for music emotion recognition. Their separation ablation also compared separated-source inputs with mixture inputs, providing direct precedent for studying MSS alongside pretrained encoders. Unlike these approaches, the present work centers on a retained pretrained **mixture representation** as the reference input and tests the incremental value of explicitly adding selected fine-grained instrument-stem embeddings, including controls for confounding model capacity and ensemble effects.

## E. Claims to remove or revise in the original manuscript

- **Delete:** “To our knowledge, this is the first work that studies the effectiveness of using MSS for different music classification tasks.”
- **Delete:** “the first systematic study of modern MSS models ... across three tasks” unless an independently justified, carefully bounded claim can be substantiated; preferable to avoid “first”.
- **Replace:** “MSS paradigm comparison” as a headline contribution, and any categorical assertion that HT-Demucs is generally superior to Banquet. **Updated decision:** remove Banquet from the revised main experiments entirely; retain historical code/data until cleanup is explicitly authorized.
- **Replace:** “Task-type principle” and any categorical explanation that genre necessarily benefits because of instrument-localized cues while emotion does not. State an empirical pattern and hypothesis, subject to controlled validation.
- **Correct:** EmoMucs did **not** merely use handcrafted features; it used source-specialized learned CNN features and tested stem subsets.
- **Correct:** MERT is a pretrained music representation model, **not** a source-separation model.
- **Acknowledge:** Rosner and Kostek had mixture-plus-separated-component feature combinations; Li et al. had source separation plus frozen pretrained representations; Garoufis et al. addressed more than one music classification task.
- **Avoid:** claiming improvements are significant or confidently caused by complementary stem information before matched controls and uncertainty analyses are completed.

## F. Phase 1 tracking and next decisions

| Item | Status | Note |
| --- | --- | --- |
| Literature survey | Done as a focused survey | Check and insert BibTeX entries for included works |
| Research question | Agreed framing | Incremental stem value beyond pretrained mixture embeddings |
| Introduction motivation and novelty | Draft prepared | Needs paper integration and advisor approval |
| Contributions | Draft prepared | Final language depends on completed controls and analyses |
| Related Work / EmoMucs distinction | Draft prepared | Accurately acknowledge stem subsets and source CNNs |
| Editor concern about scientific significance | Still open | Writing alone is insufficient; depends on resulting experiments |
| Reviewer concern No. 5: HT-Demucs vs. Banquet overclaim | **Corrected by author** | Banquet removed from revised primary scope; historical code/results still present |
| Reviewer concern No. 8: test-set selection | **Open / exploratory limitation** | Existing three stem sets were influenced by FMA test results; cannot retroactively make that split independent |
| Phase 2 experimental design | **Agreed provisionally** | See [PHASE2_EXPERIMENTAL_DESIGN.md](PHASE2_EXPERIMENTAL_DESIGN.md): HT-Demucs, 2 fusion methods, matched controls, five-seed mean ± SD, task-specific validation; code not yet cleaned or modified |
| Abstract | **Deferred** | Revise after final experimental evidence |

## G. Source starting points

- Rosner and Kostek (2018): https://doi.org/10.1007/s10844-017-0464-5
- De Berardinis et al. / EmoMucs (2020): https://archives.ismir.net/ismir2020/paper/000308.pdf
- Garoufis et al. (2024): https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000411.pdf
- Haunschmid et al. / audioLIME (2020): https://arxiv.org/pdf/2009.02051
- Kang and Herremans (2025): https://arxiv.org/abs/2502.03979
- Li et al. (2026): https://doi.org/10.1007/s11042-026-21719-3
- Xu et al. (2014): https://doi.org/10.1145/2578726.2578784
- Lampropoulos et al. (2010): https://doi.org/10.3233/IDT-2010-0083

**Codex caution:** Do not automatically rewrite LaTeX, rename citation keys, alter research code, or run experiments from this draft alone. Compare each section with the source manuscript and obtain author approval before integrating edits.
