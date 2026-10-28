from __future__ import annotations

import unittest

from stemfusion.cli import build_parser


class ModelSpecificFeatureEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_htdemucs_accepts_mixture_source(self) -> None:
        args = self.parser.parse_args(
            [
                "extract",
                "htdemucs",
                "musicfm",
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

        self.assertEqual(args.extractor, "htdemucs")
        self.assertEqual(args.source, "mixture")

    def test_htdemucs_exposes_musicfm_and_rms_modes(self) -> None:
        musicfm_args = self.parser.parse_args(
            [
                "extract",
                "htdemucs",
                "musicfm",
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
        rms_args = self.parser.parse_args(
            [
                "extract",
                "htdemucs",
                "rms",
                "--manifest",
                "manifest.jsonl",
                "--output-dir",
                "features/rms",
                "--source",
                "other",
                "--dry-run",
            ]
        )

        self.assertEqual(musicfm_args.htdemucs_feature, "musicfm")
        self.assertEqual(rms_args.htdemucs_feature, "rms")


if __name__ == "__main__":
    unittest.main()
