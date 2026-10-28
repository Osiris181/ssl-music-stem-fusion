"""Public Table 4 unweighted late-fusion workflow for HT-Demucs features."""

from __future__ import annotations

import argparse

from stemfusion.training._htdemucs_late_fusion import (
    LateFusionTrainingConfig,
    config_from_args as _config_from_args,
    configure_parser as _configure_parser,
    plan_late_fusion_training,
    run_from_args as _run_from_args,
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
    """Configure Table 4 unweighted late-fusion training."""

    parser.description = (
        "Train the Table 4 HT-Demucs late-fusion MLP with an equal logit "
        "sum across independently trained source classifiers."
    )
    _configure_parser(
        parser,
        weighted=False,
        command_handler=run_from_args,
    )


def config_from_args(args: argparse.Namespace) -> LateFusionTrainingConfig:
    """Build an unweighted late-fusion configuration from CLI arguments."""

    return _config_from_args(args, weighted=False)


def run_from_args(args: argparse.Namespace) -> int:
    """Run Table 4 unweighted late-fusion training from CLI arguments."""

    return _run_from_args(args, weighted=False)
