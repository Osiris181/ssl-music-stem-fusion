"""Path-only flattening must preserve runs without accepting changed training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stemfusion.experiments.mtg_jamendo import reproduction
from stemfusion.experiments.mtg_jamendo._relocation import LEGACY_RAW, canonical_training_checksum
from stemfusion.training.reporting import file_checksum, fingerprint
from stemfusion.training._identity import canonical_trainer_checksum


class JamendoRelocationTests(unittest.TestCase):
    def test_flat_source_layout_has_no_old_package_directory(self):
        root = Path(reproduction.__file__).parent
        self.assertFalse((root / "mert_chord_key").exists())
        for name in ("extract_mert.py", "extract_chords_keys.py", "train_reproduction.py", "reproduction.py", "README.md"):
            self.assertTrue((root / name).is_file(), name)
        self.assertFalse((root / "mert.md").exists())
        self.assertFalse((root / "chords_keys.md").exists())

    def test_verified_relocation_preserves_canonical_source_identities(self):
        self.assertEqual(canonical_training_checksum(Path(reproduction.__file__), "reproduction"), LEGACY_RAW["reproduction"])
        self.assertNotEqual(file_checksum(Path(reproduction.__file__)), LEGACY_RAW["reproduction"])

    def test_public_helper_supports_only_the_active_baseline(self):
        self.assertEqual(set(LEGACY_RAW), {"reproduction"})
        for kind in ("skey_training", "sheetsage2", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unknown relocation identity"):
                canonical_training_checksum(Path(reproduction.__file__), kind)

    def test_public_files_do_not_include_private_archive_notes(self):
        self.assertFalse((Path(reproduction.__file__).parent / "archived_variants.md").exists())

    def test_actual_baseline_optimizer_change_is_not_treated_as_relocation(self):
        source = Path(reproduction.__file__).read_text()
        changed = source.replace("betas=(0.9, 0.999)", "betas=(0.8, 0.999)")
        self.assertNotEqual(changed, source)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "changed.py"
            path.write_text(changed)
            self.assertEqual(canonical_training_checksum(path, "reproduction"), file_checksum(path))
            self.assertNotEqual(canonical_training_checksum(path, "reproduction"), LEGACY_RAW["reproduction"])

    def test_dependency_changes_still_invalidate_baseline_fingerprint(self):
        identity = reproduction._implementation()
        with patch.object(reproduction, "file_checksum", return_value="changed-dependency"):
            self.assertNotEqual(reproduction._implementation(), identity)

    def test_baseline_keeps_legacy_logical_path_but_actual_dependency_hashes(self):
        package = Path(reproduction.__file__).resolve().parents[2]
        paths = ("models/music2emo.py", "data/jamendo_reproduction.py", "data/segments.py",
                 "training/trainer.py", "training/reporting.py")
        identities = {name: file_checksum(package / name) for name in paths}
        identities["training/trainer.py"] = canonical_trainer_checksum(package / "training/trainer.py")
        identities["experiments/mtg_jamendo/mert_chord_key/reproduction.py"] = LEGACY_RAW["reproduction"]
        self.assertEqual(reproduction._implementation(), fingerprint(identities))


if __name__ == "__main__":
    unittest.main()
