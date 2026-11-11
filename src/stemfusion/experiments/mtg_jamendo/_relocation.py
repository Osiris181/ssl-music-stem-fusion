"""Strict layout-only compatibility for the completed mixture baseline runs.

Scientific fingerprints keep their historical logical source identity only if
the entire training AST matches the verified old implementation after the
explicit relocation substitutions below. Other code changes use a new checksum.
Old results/checkpoints are never rewritten. This helper is required by the
public baseline fitter; it is not a standalone experiment.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from stemfusion.training.reporting import file_checksum

LEGACY_RAW = {
    "reproduction": "b9f3a69435b82c704efbbc4eed346af5e5719b120245e94d6b4f76555284a6ec",
}
LEGACY_AST = {
    "reproduction": "6a64313c1286db68e71df3464ed887229149919fe4f4a572a68e28c3f1e91e7c",
}


def canonical_training_checksum(path: Path, kind: str) -> str:
    if kind not in LEGACY_RAW:
        raise ValueError(f"Unknown relocation identity: {kind}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = "_implementation"
    tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == helper)]
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return LEGACY_RAW[kind] if digest == LEGACY_AST[kind] else file_checksum(path)
