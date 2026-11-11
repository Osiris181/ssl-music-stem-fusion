"""Parameter-matched multi-view extension of the retained Music2Emo model."""

from __future__ import annotations

import torch
from torch import nn

from stemfusion.models.music2emo import Music2EmoClassifier


class Music2EmoMidFusion(Music2EmoClassifier):
    """Share one chord encoder and concatenate source-specific representations.

    Repeated-mixture controls encode the mixture once and repeat the complete
    1,545-dimensional representation, including the same dropout realization.
    They have exactly the same trainable parameters as the matched stem model.
    """

    def __init__(self, inputs: int, *, repeated_mixture: bool = False):
        if inputs not in (1, 4, 5, 6) or (repeated_mixture and inputs == 1):
            raise ValueError("Use one reference input or four/five/six matched inputs.")
        super().__init__()
        self.number_of_inputs = inputs
        self.repeated_mixture = repeated_mixture
        if inputs != 1:
            self.input_proj[0] = nn.Linear(inputs * 1545, 512)

    def encode_view(self, values):
        roots = self.chord_root_embedding(values["x_chord_root"])
        attributes = self.chord_attr_embedding(values["x_chord_attr"])
        chords = self.positional_encoding(torch.cat((roots, attributes), dim=-1))
        cls = torch.zeros_like(chords[:, :1, :])
        harmonic = self.chord_transformer(torch.cat((cls, chords), dim=1))[:, 0, :]
        return torch.cat((values["x_mert"], harmonic, values["x_key"].float()), dim=1)

    def forward(self, values):
        expected = 1 if self.repeated_mixture else self.number_of_inputs
        if values["x_mert"].ndim != 3 or values["x_mert"].shape[1:] != (expected, 1536):
            raise ValueError("Expected a batch of source-ordered 1,536-dimensional MERT vectors.")
        views = [{name: tensor[:, index] for name, tensor in values.items()} for index in range(expected)]
        if self.number_of_inputs == 1:
            return super().forward(views[0])
        encoded = [self.encode_view(view) for view in views]
        if self.repeated_mixture:
            encoded *= self.number_of_inputs
        return self.output_proj(self.input_proj(torch.cat(encoded, dim=1)))


def parameter_count(inputs: int) -> int:
    """Actual count without modifying the caller's random-number state."""
    with torch.random.fork_rng(devices=[]):
        return sum(parameter.numel() for parameter in Music2EmoMidFusion(inputs).parameters())
