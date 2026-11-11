"""Binary output adapter for the exact shared harmonic mid-fusion architecture."""

from __future__ import annotations

import torch
from torch import nn

from stemfusion.models.music2emo_mid_fusion import Music2EmoMidFusion


class Music2EmoEmotionMidFusion(Music2EmoMidFusion):
    """Retain the shared chord Transformer, source fusion, and 512/256 MLP."""

    def __init__(self, inputs: int, *, repeated_mixture: bool = False):
        super().__init__(inputs, repeated_mixture=repeated_mixture)
        self.output_proj[-1] = nn.Linear(256, 1)


def parameter_count(inputs: int) -> int:
    with torch.random.fork_rng(devices=[]):
        return sum(parameter.numel() for parameter in Music2EmoEmotionMidFusion(inputs).parameters())
