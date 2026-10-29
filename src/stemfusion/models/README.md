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

The current late runner trains those source MLPs jointly. Independent branch
training and per-branch checkpoint selection remain pending. The RMS-gated
model is archived outside the active project.
