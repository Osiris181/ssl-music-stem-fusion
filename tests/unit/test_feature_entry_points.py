from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from stemfusion.cli import build_parser


class DatasetFeatureEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_fma_medium_accepts_mixture_source(self) -> None:
        args = self.parser.parse_args(
            [
                "extract",
                "fma-medium",
                "--manifest",
                "manifest.jsonl",
                "--output-dir",
                "features",
                "--source",
                "mixture",
                "--musicfm-root",
                "musicfm",
                "--dry-run",
            ]
        )

        self.assertEqual(args.extraction_dataset, "fma-medium")
        self.assertEqual(args.source, ["mixture"])

    def test_fma_medium_preserves_extraction_defaults(self) -> None:
        musicfm_args = self.parser.parse_args(
            [
                "extract",
                "fma-medium",
                "--manifest",
                "manifest.jsonl",
                "--output-dir",
                "features/musicfm",
                "--source",
                "other",
                "--musicfm-root",
                "musicfm",
                "--dry-run",
            ]
        )
        self.assertEqual(musicfm_args.extraction_dataset, "fma-medium")
        self.assertEqual(musicfm_args.layer_index, 7)
        self.assertEqual(musicfm_args.sample_rate, 24000)
        self.assertIsNone(musicfm_args.max_seconds)
        self.assertIsNone(musicfm_args.checkpoint_path)
        self.assertIsNone(musicfm_args.stats_path)

    def test_fma_dataset_yaml_supplies_its_output_default(self) -> None:
        args = self.parser.parse_args([
            "extract", "fma-medium", "--manifest", "manifest.jsonl",
            "--musicfm-root", "musicfm", "--source", "guitar", "--dry-run",
        ])
        project = Path(__file__).resolve().parents[2]
        self.assertEqual(
            args.output_dir,
            project / "data/features/fma_medium/musicfm_msd_layer7",
        )
        self.assertEqual(args.expected_dimension, 1024)
        self.assertEqual(args.splits, ["train", "validation", "test"])
        self.assertEqual(
            args.yaml_provenance["config_files"][0]["path"],
            str(project / "configs/experiments/fma_medium/extraction.yaml"),
        )

    def test_superseded_extraction_commands_are_not_available(self) -> None:
        for command in (
            ["extract", "htdemucs", "musicfm"],
            ["extract", "htdemucs", "rms"],
        ):
            with self.subTest(command=command):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        self.parser.parse_args(command)
                self.assertEqual(result.exception.code, 2)



if __name__ == "__main__":
    unittest.main()
