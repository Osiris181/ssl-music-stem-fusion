# Models

## Purpose

This directory contains reusable trainable architectures. Frozen MusicFM and
source-separation models belong to their own pipeline stages.

## Contents

| File | Purpose |
| --- | --- |
| `mlp.py` | Three-linear-layer MLP used by the paper's mid-fusion genre experiments. |
| `late_fusion.py` | Separately parameterized source MLPs with equal logit aggregation. |

For FMA genre classification, every source MLP uses hidden dimensions 1024
and 512 with dropout 0.2. Mid fusion receives 1024 times the number of selected
sources. Late fusion gives each source MLP one 1024-dimensional vector, then
aggregates its class logits. The output dimension is the number of genres.

Late fusion trains and selects each branch independently through the shared
trainer. Ensembles sum selected branch logits equally, then apply softmax for
probabilities. Both mid fusion and individual ensemble branches use the same
`DeepMLPClassifier`. Repeated-mixture controls concatenate identical copies
of the mixture vector and keep both hidden widths unchanged, exactly matching
the corresponding fusion input width and parameter count. This does not add
independent input information or equalize effective representational capacity.
The RMS-gated model is archived outside the active project.

## Music2Emo author baseline

`music2emo.py` implements the single-task author's chord Transformer and
feature-level harmonic fusion: frozen MERT (1536) + learned chords (8) + scalar
key mode (1), then `1545 -> 512 -> 256 -> 56`. Its two-layer/eight-head
Transformer is trainable. This is separate from the genre fusion MLP.
The upstream MIT attribution is in `THIRD_PARTY_NOTICES.md`; see the
[baseline guide](../../../docs/music2emotion_reproduction.md).

`music2emo_mid_fusion.py` shares the same chord Transformer across source
views and concatenates complete 1,545-dimensional representations before the
512/256 classifier. Four/five/six-input repeated-mixture controls duplicate
one encoded mixture, matching the corresponding fusion model's full trainable
parameter count. The single-input case preserves the original model's state
dictionary and forward calculation exactly.

`music2emo_emotion.py` subclasses that model and replaces only its final output
layer with one logit per independently trained binary target. The shared chord
Transformer, 1,545 encoded values per source, 512/256 hidden layers and
encode-once repeated-mixture construction are unchanged. PMEmo uses separate
valence/arousal classifiers, not a joint regression head.
