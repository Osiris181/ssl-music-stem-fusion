"""One-logit CNN-feature head with the retained emotion hidden widths."""
from __future__ import annotations

import torch
from torch import nn


class CNNEmotionClassifier(nn.Module):
    def __init__(self, inputs=1):
        super().__init__()
        if type(inputs) is not int or inputs not in (1, 4, 5, 6):
            raise ValueError("Choose one, four, five or six CNN inputs")
        self.head = nn.Sequential(
            nn.Linear(inputs * 200, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, values):
        return self.head(values["x_mert"])


def parameter_count(inputs):
    with torch.random.fork_rng(devices=[]):
        return sum(parameter.numel() for parameter in CNNEmotionClassifier(inputs).parameters())
