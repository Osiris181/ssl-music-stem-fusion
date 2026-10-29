from __future__ import annotations

import contextlib
import io
import unittest

from stemfusion.cli import build_parser


class FMAMediumEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    @staticmethod
    def shared_arguments() -> list[str]:
        return [
            "--manifest", "manifest.jsonl",
            "--feature-dir", "musicfm",
            "--output-dir", "output",
            "--source", "mixture",
            "--source", "drums",
            "--dry-run",
        ]

    def test_mid_fusion_preserves_silent_vector_exclusion(self) -> None:
        args = self.parser.parse_args(
            ["train", "fma-medium", "mid-fusion", *self.shared_arguments()]
        )

        self.assertTrue(args.drop_silent_features)
        self.assertEqual(args.training_dataset, "fma-medium")
        self.assertEqual(args.training_method, "mid-fusion")

    def test_comparable_training_defaults_match(self) -> None:
        mid = self.parser.parse_args(
            ["train", "fma-medium", "mid-fusion", *self.shared_arguments()]
        )
        late = self.parser.parse_args(
            ["train", "fma-medium", "late-fusion", *self.shared_arguments()]
        )

        for name in (
            "expected_feature_dimension", "hidden_dimensions", "dropout",
            "batch_size", "learning_rate", "epochs", "seed",
        ):
            self.assertEqual(getattr(mid, name), getattr(late, name), name)
        self.assertFalse(hasattr(late, "energy_feature_dir"))
        self.assertEqual(late.training_method, "late-fusion")

    def test_archived_training_commands_are_not_available(self) -> None:
        for command in (
            ["train", "fma-genre"],
            ["train", "fma-htdemucs-mid-fusion"],
            ["train", "fma-htdemucs-rms-weighted-late-fusion"],
        ):
            with self.subTest(command=command):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        self.parser.parse_args(command)
                self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
