"""Archived harmony variants must not break the original Jamendo workflow."""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from stemfusion.cli import build_parser
from stemfusion.configuration import default_config_path, load_configuration


class ArchivedHarmonyVariantsTests(unittest.TestCase):
    def test_archived_public_commands_are_rejected(self):
        parser = build_parser()
        for stage in ("extract", "train"):
            for method in ("skey", "chordformer-skey", "sheetsage2"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parser.parse_args([stage, "mtg-jamendo", method, "--dry-run"])
                self.assertEqual(error.exception.code, 2)

    def test_active_commands_and_presets_remain_available(self):
        parser = build_parser()
        for method in ("mert", "chords-keys"):
            self.assertTrue(parser.parse_args(["extract", "mtg-jamendo", method, "--dry-run"]).dry_run)
        for method in ("author-reproduction",):
            self.assertTrue(parser.parse_args(["train", "mtg-jamendo", method, "--dry-run"]).dry_run)
        values, _ = load_configuration(default_config_path("experiments/mtg_jamendo/reproduction.yaml"),
                                       "jamendo_reproduction")
        self.assertEqual(values["seeds"], [32, 64, 128, 256, 512])

    def test_original_backends_do_not_need_archived_modules(self):
        from stemfusion.features.chords_keys import extract_chords_keys
        from stemfusion.features.mert_segments import extract_mert_segments
        from stemfusion.experiments.mtg_jamendo.reproduction import ReproductionConfig, run_reproduction
        self.assertTrue(callable(extract_chords_keys))
        self.assertTrue(callable(extract_mert_segments))
        self.assertTrue(callable(run_reproduction))
        self.assertIsNotNone(ReproductionConfig)

    def test_archived_entrypoints_are_not_left_in_active_source(self):
        root = Path(__file__).resolve().parents[2] / "src/stemfusion/experiments/mtg_jamendo"
        for name in ("extract_skey.py", "train_skey.py", "extract_chordformer.py", "train_chordformer.py",
                     "chordformer_training.py", "extract_sheetsage2.py", "train_sheetsage2.py", "sheetsage2_training.py",
                     "skey_training.py"):
            self.assertFalse((root / name).exists(), name)
        package = root.parents[1]
        for name in ("features/sheetsage2.py", "features/sheetsage2_assets.json", "features/skey.py",
                     "data/jamendo_sheetsage2.py", "data/jamendo_skey.py"):
            self.assertFalse((package / name).exists(), name)
