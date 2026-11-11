"""Strict compatibility for adding optional binary metrics to the shared loop.

The reviewed task=None path retains Jamendo's fitting, metrics and signatures.
Unknown trainer/policy changes receive new identities; run metadata is not edited.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from stemfusion.training.reporting import file_checksum, fingerprint

LEGACY_TRAINER_SHA256 = "be82d26b60b65f8d596abe09e37e2df9e6e14a20c0084bbb4a794adc3c7e5c64"
REVIEWED_TRAINER_AST = "fab91862b4acd783a5fc69643a725ec6262695ada3d540cdfa5093480aed9547"
REVIEWED_POLICY_AST = "012a68239f912452b1f375799385e7cb795110e8d21b797d962f03521f14782e"


def _ast_checksum(path, *, policy=False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if policy:
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REVIEWED_POLICY_AST")]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def canonical_trainer_checksum(path):
    policy = Path(__file__)
    if _ast_checksum(policy, policy=True) != REVIEWED_POLICY_AST:
        return fingerprint({"trainer": file_checksum(path), "identity_policy": file_checksum(policy)})
    return LEGACY_TRAINER_SHA256 if _ast_checksum(path) == REVIEWED_TRAINER_AST else file_checksum(path)
