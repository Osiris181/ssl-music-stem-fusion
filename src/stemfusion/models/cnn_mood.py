"""Plain frozen-CNN feature head with the retained Music2Emo hidden widths."""
from torch import nn

from stemfusion.models.mlp import DeepMLPClassifier


class CNNMoodClassifier(nn.Module):
    def __init__(self, inputs=1):
        super().__init__()
        if inputs not in (1, 4, 5, 6):
            raise ValueError("Choose one, four, five or six CNN inputs")
        # Music2Emo's classifier head has no dropout; its dropout belongs to
        # the removed chord Transformer. Keep the two hidden ReLUs only.
        self.head = DeepMLPClassifier(inputs * 200, 56, (512, 256), dropout=0.)

    def forward(self, values):
        return self.head(values["x_mert"])
