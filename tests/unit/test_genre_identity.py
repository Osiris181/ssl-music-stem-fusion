"""Unrelated task modules must not invalidate verified genre runs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stemfusion.training import _identity, trainer
from stemfusion.training.reporting import file_checksum, fingerprint

LEGACY_GENRE_IDENTITY = "7b39adcd615291b602a399342bf3d9f0d12d377eaff631a26772f7fe4aee6f7b"


class GenreIdentityTests(unittest.TestCase):
    def test_reviewed_policy_and_trainer_match_exactly(self):
        self.assertEqual(_identity._ast_checksum(Path(_identity.__file__), policy=True),
                         _identity.REVIEWED_POLICY_AST)
        self.assertEqual(_identity._ast_checksum(Path(trainer.__file__)), _identity.REVIEWED_TRAINER_AST)
        self.assertEqual(_identity.canonical_trainer_checksum(Path(trainer.__file__)),
                         _identity.LEGACY_TRAINER_SHA256)

    def test_saved_genre_identity_is_preserved(self):
        self.assertEqual(trainer.implementation_fingerprint(), LEGACY_GENRE_IDENTITY)

    def test_only_historical_genre_dependencies_are_hashed(self):
        package = Path(trainer.__file__).resolve().parents[1]
        expected = {name: file_checksum(package / name) for name in _identity.GENRE_DEPENDENCIES}
        expected["training/trainer.py"] = _identity.LEGACY_TRAINER_SHA256
        self.assertEqual(trainer.implementation_fingerprint(), fingerprint(expected))
        with patch.object(_identity, "file_checksum", wraps=file_checksum) as checksum:
            self.assertEqual(trainer.implementation_fingerprint(), LEGACY_GENRE_IDENTITY)
        hashed = {str(call.args[0].relative_to(package)) for call in checksum.call_args_list}
        self.assertNotIn("training/music2emo.py", hashed)
        self.assertNotIn("training/music2emo_late_fusion.py", hashed)
        self.assertNotIn("training/_identity.py", hashed)

    def test_changes_to_every_genre_dependency_invalidate_identity(self):
        original = _identity.file_checksum
        for name in _identity.GENRE_DEPENDENCIES:
            if name == "training/trainer.py":
                continue  # Its full AST guard is checked separately below.
            with self.subTest(name=name):
                def changed_checksum(path):
                    return "changed-dependency" if str(path).endswith(name) else original(path)
                with patch.object(_identity, "file_checksum", side_effect=changed_checksum):
                    self.assertNotEqual(trainer.implementation_fingerprint(), LEGACY_GENRE_IDENTITY)

    def test_optimizer_edits_are_not_treated_as_identity_only(self):
        original = Path(trainer.__file__).read_text()
        changed = original.replace("betas=(0.9, 0.999)", "betas=(0.8, 0.999)")
        self.assertNotEqual(original, changed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trainer.py"
            path.write_text(changed)
            self.assertEqual(_identity.canonical_trainer_checksum(path), file_checksum(path))
            self.assertNotEqual(_identity.canonical_trainer_checksum(path), _identity.LEGACY_TRAINER_SHA256)

    def test_unknown_identity_function_edits_are_not_accepted(self):
        changed = Path(trainer.__file__).read_text().replace(
            "return genre_implementation_fingerprint()", 'return "unsafe-fixed-identity"')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trainer.py"
            path.write_text(changed)
            self.assertEqual(_identity.canonical_trainer_checksum(path), file_checksum(path))

    def test_policy_edits_invalidate_identity(self):
        original = Path(_identity.__file__).read_text()
        changed = original.replace('"training/metrics.py",', '')
        self.assertNotEqual(original, changed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "_identity.py"
            path.write_text(changed)
            with patch.object(_identity, "__file__", str(path)):
                self.assertNotEqual(_identity.canonical_trainer_checksum(Path(trainer.__file__)),
                                    _identity.LEGACY_TRAINER_SHA256)


if __name__ == "__main__":
    unittest.main()
