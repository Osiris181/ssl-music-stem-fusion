"""Genre-only dependencies and narrowly verified fingerprint-edit compatibility.

The original genre identity included every training module in the directory.
Adding unrelated task implementations consequently invalidated completed runs.
Keep its original dependency set, and recognize only the reviewed identity-only
trainer edit. Changes to fitting, metrics, models, or this policy still invalidate
cached runs; saved run metadata and artifact checks are never bypassed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from stemfusion.training.reporting import file_checksum, fingerprint

GENRE_DEPENDENCIES = (
    "training/__init__.py",
    "training/late_fusion.py",
    "training/metrics.py",
    "training/mid_fusion.py",
    "training/reporting.py",
    "training/trainer.py",
    "models/mlp.py",
    "data/features.py",
    "data/fma_features.py",
)
LEGACY_TRAINER_SHA256 = "0299f441ef6fa80e4be2c4996a6e3681b058a5012f2808109a72527a882621dc"
REVIEWED_TRAINER_AST = "c74efc7730bf24b678263ea8bea885910df9fcf2da1ba2a1fad507a173af2287"
REVIEWED_POLICY_AST = "c3eb38032bd764ea1d827e47bc672cccf06b8d7cf703e07c32558151c2417f38"


def _ast_checksum(path: Path, *, policy: bool = False) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if policy:
        # Exclude only the self-check value, not the policy or dependency list.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "REVIEWED_POLICY_AST"
        )]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def canonical_trainer_checksum(path: Path) -> str:
    """Preserve identity only for the exact reviewed fingerprint-only edit.

    Full trainer AST matching includes its identity function. Any substantive
    trainer edit uses its real checksum. Policy edits also receive a new identity.
    """
    policy = Path(__file__)
    if _ast_checksum(policy, policy=True) != REVIEWED_POLICY_AST:
        return fingerprint({"trainer": file_checksum(path), "identity_policy": file_checksum(policy)})
    if _ast_checksum(path) == REVIEWED_TRAINER_AST:
        return LEGACY_TRAINER_SHA256
    return file_checksum(path)


def genre_implementation_fingerprint() -> str:
    package = Path(__file__).resolve().parents[1]
    identities = {
        name: canonical_trainer_checksum(package / name)
        if name == "training/trainer.py" else file_checksum(package / name)
        for name in GENRE_DEPENDENCIES
    }
    return fingerprint(identities)
