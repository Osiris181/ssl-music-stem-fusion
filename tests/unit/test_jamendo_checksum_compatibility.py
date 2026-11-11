"""Reviewed controller consolidation must preserve runs, not changed science."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stemfusion.experiments.mtg_jamendo import _checksum_compatibility as compatibility
from stemfusion.experiments import _mood_theme_tagging as training
from stemfusion.experiments.mtg_jamendo import reproduction
from stemfusion.training.reporting import file_checksum


class JamendoChecksumCompatibilityTests(unittest.TestCase):
    def test_completed_scientific_identities_are_preserved_but_actual_source_is_recorded(self):
        expected = {"mid": "ba940df5a28bb5ed138adfa3365e35db670a9b3c1719a47a30630c241515a544",
                    "late": "57d13078c227db3e22002f1d5ae85ac230f8482d869c6b11c65da6b3da441678"}
        for kind in ("mid", "late"):
            with self.subTest(kind=kind):
                path = Path(training.__file__)
                self.assertEqual(compatibility.implementation_identity(kind), expected[kind])
                self.assertEqual(compatibility.compatible_source_checksum(path, kind), compatibility.LEGACY_RAW[kind])
                self.assertNotEqual(file_checksum(path), compatibility.LEGACY_RAW[kind])
                self.assertNotEqual(compatibility.execution_identity(kind), expected[kind])
        self.assertEqual(reproduction._implementation(), "d67a02a28eb3da94524277edc430ccd29679e29c7c70e54728f77a0e0a9ac406")

    def test_changed_fusion_protocol_is_not_accepted_as_layout_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.py"
            for kind, old, new in (
                ("mid", '"harmony_encoder": "shared"', '"harmony_encoder": "independent"'),
                ("mid", '"sampling": "shared_epoch_seeded_conditional_uniform_consecutive_segment_mean"', '"sampling": "different"'),
                ("late", '"harmony_encoder": "independent_per_branch"', '"harmony_encoder": "shared"'),
                ("late", '"threshold": "validation_macro_f1_global_0_to_1_step_0.01_ties_lowest"', '"threshold": "test_optimized"'),
            ):
                with self.subTest(kind=kind, old=old):
                    source = Path(training.__file__).read_text()
                    changed = source.replace(old, new)
                    self.assertNotEqual(source, changed)
                    path.write_text(changed)
                    self.assertNotEqual(compatibility.compatible_source_checksum(path, kind), compatibility.LEGACY_RAW[kind])

    def test_optional_loader_changes_invalidate_both_identities(self):
        original = compatibility.file_checksum
        def changed(path):
            return "different-loader" if path.name == "jamendo_late_fusion.py" else original(path)
        before = [compatibility.implementation_identity(kind) for kind in ("mid", "late")]
        with patch.object(compatibility, "file_checksum", side_effect=changed):
            for kind, identity in zip(("mid", "late"), before, strict=True):
                self.assertNotEqual(compatibility.implementation_identity(kind), identity)

    def test_other_training_dependencies_still_invalidate_identities(self):
        original = compatibility.file_checksum
        def changed(path):
            return "changed-training-dependency" if path.name == "music2emo.py" else original(path)
        before = [compatibility.implementation_identity(kind) for kind in ("mid", "late")]
        with patch.object(compatibility, "file_checksum", side_effect=changed):
            for kind, identity in zip(("mid", "late"), before, strict=True):
                self.assertNotEqual(compatibility.implementation_identity(kind), identity)

    def test_unknown_configuration_or_entry_edits_are_not_accepted_as_relocation(self):
        path = Path(training.__file__)
        for name in ("_configuration_checksum", "_ast_checksum"):
            with self.subTest(name=name), patch.object(compatibility, name, return_value="unreviewed-code"):
                for kind in ("mid", "late"):
                    self.assertNotEqual(compatibility.compatible_source_checksum(path, kind), compatibility.LEGACY_RAW[kind])

    def test_public_controller_replaces_old_modules_and_both_entries_are_thin(self):
        root = Path(reproduction.__file__).parent
        self.assertEqual(Path(training.__file__), root.parent / '_mood_theme_tagging.py')
        for name in ("training.py", "mid_fusion.py", "late_fusion.py", "late_configuration.py"):
            self.assertFalse((root / name).exists())
        for kind in ("mid", "late"):
            source = (root / f"train_{kind}.py").read_text()
            self.assertIn("experiments._mood_theme_tagging import", source)
            self.assertNotIn("add_argument", source)
            self.assertNotIn("load_configuration", source)

    def test_unknown_compatibility_kinds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown checksum compatibility"):
            compatibility.compatible_source_checksum(Path(training.__file__), "unknown")
        with self.assertRaisesRegex(ValueError, "Unknown checksum compatibility"):
            compatibility.execution_identity("unknown")


if __name__ == "__main__":
    unittest.main()
