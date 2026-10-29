"""FMA-Medium unweighted late-fusion training entry point.

The retained engine uses joint training through summed logits. Independent
branch training for the revised evaluation is not implemented yet.
"""

from __future__ import annotations

import argparse

from stemfusion.configuration import configure_yaml

from stemfusion.training.late_fusion import (
    LateFusionTrainingConfig,
    config_from_args,
    configure_parser as _configure_parser,
    plan_late_fusion_training,
    run_from_args,
    train_late_fusion_classifier,
)

__all__ = [
    "LateFusionTrainingConfig",
    "config_from_args",
    "configure_parser",
    "plan_late_fusion_training",
    "run_from_args",
    "train_late_fusion_classifier",
]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the retained FMA-Medium unweighted late-fusion runner."""

    parser.description = (
        "Train the FMA-Medium unweighted late-fusion MLP. The retained "
        "runner jointly trains source branches through summed logits; "
        "the revised independent-branch protocol is not implemented yet."
    )
    _configure_parser(parser, command_handler=run_from_args)
    configure_yaml(parser, "late_fusion", "training/fma_medium_late_fusion.yaml")
