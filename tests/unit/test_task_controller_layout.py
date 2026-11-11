"""Dataset launchers delegate directly to the matching task controller."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from stemfusion.cli import build_parser
from stemfusion.experiments import _genre, _mood_theme_tagging


class TaskControllerLayoutTests(unittest.TestCase):
    def test_genre_launchers_use_one_controller_without_dataset_training_wrappers(self):
        for dataset in ("fma_medium", "gtzan", "ismir04"):
            for family in ("mid", "late"):
                with self.subTest(dataset=dataset, family=family):
                    module = importlib.import_module(f"stemfusion.experiments.{dataset}.train_{family}")
                    self.assertIs(module.direct_main, _genre.direct_main)
                    self.assertIs(module._configure_parser, _genre.configure_parser)
                    self.assertFalse(Path(module.__file__).with_name("training.py").exists())

    def test_mood_launchers_use_the_task_controller_without_a_dataset_training_wrapper(self):
        for family in ("mid", "late"):
            with self.subTest(family=family):
                module = importlib.import_module(f"stemfusion.experiments.mtg_jamendo.train_{family}")
                self.assertIs(module.direct_main, _mood_theme_tagging.direct_main)
                self.assertIs(module._configure_parser, _mood_theme_tagging.configure_parser)
                self.assertFalse(Path(module.__file__).with_name("training.py").exists())

    def test_both_task_routes_preserve_the_agreed_default_training_settings(self):
        parser = build_parser()
        for dataset in ("fma-medium", "gtzan", "ismir04", "mtg-jamendo"):
            for family in ("mid-fusion", "late-fusion"):
                with self.subTest(dataset=dataset, family=family):
                    args = parser.parse_args(["train", dataset, family, "--dry-run"])
                    self.assertEqual(args.seeds, [32, 64, 128, 256, 512])
                    self.assertEqual(args.learning_rate, 1e-4)
                    self.assertTrue(args.dry_run)
                    if dataset == "mtg-jamendo":
                        self.assertEqual((args.epochs, args.batch_size), (200, 8))
                        self.assertTrue(args.verify_checksums)
                        self.assertEqual((args.early_stopping_patience, args.early_stopping_min_delta), (10, 1e-4))
                    else:
                        self.assertEqual((args.epochs, args.batch_size), (20, 128))
                        self.assertEqual(args.hidden_dimensions, [1024, 512])
                        self.assertEqual(args.dropout, .2)


if __name__ == "__main__":
    unittest.main()
