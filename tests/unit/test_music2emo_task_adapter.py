"""Adding binary metrics must preserve Jamendo's reviewed default protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stemfusion.experiments.mtg_jamendo._checksum_compatibility import implementation_identity
from stemfusion.training import _music2emo_identity as identity, music2emo
from stemfusion.training.reporting import file_checksum


class Music2EmoTaskAdapterTests(unittest.TestCase):
    def test_exact_reviewed_trainer_and_policy_preserve_legacy_identity(self):
        self.assertEqual(identity._ast_checksum(Path(music2emo.__file__)), identity.REVIEWED_TRAINER_AST)
        self.assertEqual(identity._ast_checksum(Path(identity.__file__), policy=True), identity.REVIEWED_POLICY_AST)
        self.assertEqual(identity.canonical_trainer_checksum(Path(music2emo.__file__)), identity.LEGACY_TRAINER_SHA256)
        self.assertEqual(implementation_identity('mid'), 'ba940df5a28bb5ed138adfa3365e35db670a9b3c1719a47a30630c241515a544')
        self.assertEqual(implementation_identity('late'), '57d13078c227db3e22002f1d5ae85ac230f8482d869c6b11c65da6b3da441678')

    def test_unreviewed_optimizer_edits_are_never_treated_as_task_adaptation(self):
        original = Path(music2emo.__file__).read_text()
        changed = original.replace('betas=(0.9, 0.999)', 'betas=(0.8, 0.999)')
        self.assertNotEqual(changed, original)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'changed.py'
            path.write_text(changed)
            self.assertEqual(identity.canonical_trainer_checksum(path), file_checksum(path))
            self.assertNotEqual(identity.canonical_trainer_checksum(path), identity.LEGACY_TRAINER_SHA256)

    def test_policy_edits_invalidate_old_training_identity(self):
        original = Path(identity.__file__).read_text()
        changed = original.replace('else file_checksum(path)', 'else LEGACY_TRAINER_SHA256')
        self.assertNotEqual(original, changed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.py'
            path.write_text(changed)
            with patch.object(identity, '__file__', str(path)):
                self.assertNotEqual(identity.canonical_trainer_checksum(Path(music2emo.__file__)), identity.LEGACY_TRAINER_SHA256)


if __name__ == '__main__':
    unittest.main()
