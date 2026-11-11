"""Strict scientific compatibility for reviewed Jamendo loading/layout changes.

The combined controller and shared YAML schema must match the reviewed ASTs,
and the optional cache loader must retain its parity-tested bytes. Otherwise
new scientific identities are used. Historical checkpoints/results are never
rewritten; current execution sources are fingerprinted separately.
Historical logical source paths remain in scientific identities; the controller's
actual task-level location is recorded by execution_identity.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from stemfusion.training.reporting import file_checksum, fingerprint

LEGACY_RAW = {
    "mid": "1cb7ec9f09e2fd6f47ee0107d34a6e8dda34d85ee8d8aab172d701867a007bba",
    "late": "abb90c19105cf9a7b17259b194dd132ca1bd5645e296d035b52e95c4cdbc5119",
}
LEGACY_LATE_CONFIGURATION = "c1d10734896bdcd501df3e02c0e18d068f2b2d4f09fa21ace301122aefc96c14"
LEGACY_LATE_ENTRY = "70a8251f2863feefec29d0c2cdd9227d14d6b15673cfc499abf4863821938f1c"
OPTIONAL_LOADER_SHA256 = "0768f1422a9183a6b62b76ac0828817a7a142f66f424aab41cdb63c7535d8513"
REVIEWED_TRAINING_AST = "e89630ea3c6b36c4e798393793118b957b163e0b9c4c77317074f56901e9ab09"
REVIEWED_CONFIGURATION_AST = "8c7e8bf31eca63677cb2999c0c86d54289185a79ee9ac20cbc3cdc7191b9fb17"
REVIEWED_ENTRY_AST = {
    "mid": "d907a7fdc2c0749a758fd1877359de4b45d6953da8b8936806c964d14a4dbf64",
    "late": "f88dd931a8754752bb2dcd140bc7f57848f3646507f340372e61fc5f3417c56a",
}


def _ast_checksum(path):
    return hashlib.sha256(ast.dump(ast.parse(path.read_text(encoding="utf-8")),
                                   include_attributes=False).encode()).hexdigest()


def _configuration_checksum():
    package = Path(__file__).resolve().parents[2]
    tree = ast.parse((package / "configuration.py").read_text(encoding="utf-8"))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {
                "JAMENDO_MID_CONDITIONS", "JAMENDO_LATE_CONDITIONS"} for target in node.targets):
            definitions.append(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.FunctionDef) and node.name == "load_configuration":
            for statement in node.body:
                if isinstance(statement, ast.If) and any(isinstance(value, ast.Constant) and
                        value.value == "jamendo_reproduction" for value in ast.walk(statement.test)):
                    definitions.append(ast.dump(statement, include_attributes=False))
    return fingerprint(definitions)


def _reviewed_layout(path):
    package = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    return (_ast_checksum(path) == REVIEWED_TRAINING_AST
            and _configuration_checksum() == REVIEWED_CONFIGURATION_AST
            and file_checksum(package / "data/jamendo_late_fusion.py") == OPTIONAL_LOADER_SHA256
            and all(_ast_checksum(experiment / f"train_{kind}.py") == expected
                    for kind, expected in REVIEWED_ENTRY_AST.items()))


def compatible_source_checksum(path: Path, kind: str) -> str:
    if kind not in LEGACY_RAW:
        raise ValueError(f"Unknown checksum compatibility kind: {kind}")
    if _reviewed_layout(path):
        return LEGACY_RAW[kind]
    package = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    return fingerprint({"controller": file_checksum(path), "configuration": file_checksum(package / "configuration.py"),
                        "optional_cache_loader": file_checksum(package / "data/jamendo_late_fusion.py"),
                        "entry_points": {name: file_checksum(experiment / f"train_{name}.py") for name in LEGACY_RAW}})


def implementation_identity(kind="mid"):
    if kind not in LEGACY_RAW:
        raise ValueError(f"Unknown checksum compatibility kind: {kind}")
    from stemfusion.experiments.mtg_jamendo import reproduction
    from stemfusion.training._music2emo_identity import canonical_trainer_checksum

    package = Path(__file__).resolve().parents[2]
    controller = package / "experiments/_mood_theme_tagging.py"
    names = ["data/jamendo_mid_fusion.py", "models/music2emo_mid_fusion.py", "training/music2emo.py"]
    identities = {name: canonical_trainer_checksum(package / name) if name == "training/music2emo.py"
                  else file_checksum(package / name) for name in names}
    identities["experiments/mtg_jamendo/mid_fusion.py"] = compatible_source_checksum(controller, "mid")
    if kind == "late":
        reviewed = _reviewed_layout(controller)
        identities.update({
            "experiments/mtg_jamendo/late_fusion.py": compatible_source_checksum(controller, "late"),
            "experiments/mtg_jamendo/late_configuration.py": LEGACY_LATE_CONFIGURATION if reviewed else file_checksum(package / "configuration.py"),
            "experiments/mtg_jamendo/train_late.py": LEGACY_LATE_ENTRY if reviewed else file_checksum(Path(__file__).resolve().parent / "train_late.py"),
            **{name: file_checksum(package / name) for name in (
                "data/jamendo_late_fusion.py", "training/music2emo_late_fusion.py", "training/late_fusion.py")},
        })
    return fingerprint({"retained_baseline": reproduction._implementation(), **identities})


def execution_identity(kind):
    if kind not in LEGACY_RAW:
        raise ValueError(f"Unknown checksum compatibility kind: {kind}")
    package = Path(__file__).resolve().parents[2]
    names = ("experiments/_mood_theme_tagging.py",
             "experiments/mtg_jamendo/train_mid.py", "experiments/mtg_jamendo/train_late.py",
             "experiments/mtg_jamendo/_checksum_compatibility.py",
             "data/jamendo_late_fusion.py", "configuration.py")
    return fingerprint({"family": kind, **{name: file_checksum(package / name) for name in names}})
