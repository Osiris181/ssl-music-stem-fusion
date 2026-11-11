"""Music2Emo's single-task harmonic-fusion classifier.

Adapted from AMAAI-Lab/Music2Emotion model/linear_attn_ck.py at
c5f7984f78d9928b841f25bd05a47377c12c58ca.
Copyright (c) 2025 The Audio, Music, and AI (AMAAI) Lab.
MIT-licensed; see THIRD_PARTY_NOTICES.md for the complete notice.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int = 8, max_length: int = 100):
        super().__init__()
        encoding = torch.zeros(max_length, dimension)
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dimension, 2).float() * (-np.log(10000.0) / dimension))
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        # The author's tensor is not a state-dict entry. A nonpersistent buffer
        # preserves that format while supporting device movement automatically.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values):
        return values + self.encoding[:, :values.size(1), :].to(values.device)


class Music2EmoClassifier(nn.Module):
    """Frozen audio vector + learned chord sequence + scalar key mode -> logits."""

    def __init__(self, input_size: int = 1536, output_size: int = 56):
        super().__init__()
        self.chord_root_embedding = nn.Embedding(14, 4)
        self.chord_attr_embedding = nn.Embedding(14, 4)
        nn.init.xavier_uniform_(self.chord_root_embedding.weight)
        nn.init.xavier_uniform_(self.chord_attr_embedding.weight)
        self.positional_encoding = PositionalEncoding()
        self.chord_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=8, nhead=8, dim_feedforward=64,
                                       dropout=0.1, batch_first=True), num_layers=2)
        self.input_proj = nn.Sequential(nn.Linear(input_size + 8 + 1, 512), nn.ReLU())
        self.output_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, output_size))

    def forward(self, values):
        roots = self.chord_root_embedding(values["x_chord_root"])
        attributes = self.chord_attr_embedding(values["x_chord_attr"])
        chords = self.positional_encoding(torch.cat((roots, attributes), dim=-1))
        # Preserve the released zero CLS token and unmasked zero-token padding.
        cls = torch.zeros_like(chords[:, :1, :])
        harmonic = self.chord_transformer(torch.cat((cls, chords), dim=1))[:, 0, :]
        combined = torch.cat((values["x_mert"], harmonic, values["x_key"].float()), dim=1)
        return self.output_proj(self.input_proj(combined))
