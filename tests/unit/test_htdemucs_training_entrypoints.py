from __future__ import annotations

import unittest

from stemfusion.cli import build_parser


class HTDemucsTrainingEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    @staticmethod
    def shared_arguments() -> list[str]:
        return [
            "--manifest",
            "manifest.jsonl",
            "--feature-dir",
            "musicfm",
            "--output-dir",
            "output",
            "--source",
            "mixture",
            "--source",
            "drums",
            "--dry-run",
        ]

    def test_mid_fusion_enables_legacy_silent_vector_exclusion(self) -> None:
        args = self.parser.parse_args(
            ["train", "fma-htdemucs-mid-fusion", *self.shared_arguments()]
        )

        self.assertTrue(args.drop_silent_features)
        self.assertEqual(args.training_task, "fma-htdemucs-mid-fusion")

    def test_unweighted_late_fusion_uses_musicfm_only(self) -> None:
        args = self.parser.parse_args(
            ["train", "fma-htdemucs-late-fusion", *self.shared_arguments()]
        )

        self.assertFalse(hasattr(args, "energy_feature_dir"))
        self.assertEqual(args.training_task, "fma-htdemucs-late-fusion")

    def test_rms_weighted_late_fusion_requires_an_rms_directory(self) -> None:
        args = self.parser.parse_args(
            [
                "train",
                "fma-htdemucs-rms-weighted-late-fusion",
                *self.shared_arguments()[:-1],
                "--energy-feature-dir",
                "rms",
                "--dry-run",
            ]
        )

        self.assertEqual(args.energy_feature_dir.name, "rms")
        self.assertEqual(
            args.training_task,
            "fma-htdemucs-rms-weighted-late-fusion",
        )


if __name__ == "__main__":
    unittest.main()
