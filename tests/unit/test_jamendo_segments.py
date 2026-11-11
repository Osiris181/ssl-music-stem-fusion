"""Shared Jamendo manifest/segment checks retained after archiving MusicFM."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stemfusion.cli import build_parser
from stemfusion.data.mtg_jamendo import build_feature_manifest
from stemfusion.data.segments import pool_segment_views, segment_files
from stemfusion.features.musicfm_segments import align_stem, prepare_encoder_segment, segment_ranges


class JamendoSharedSegmentTests(unittest.TestCase):
    def test_partition_keeps_short_tails_without_padding_or_balancing(self):
        self.assertEqual(segment_ranges(305, 10, 30), [(0, 300), (300, 305)])
        self.assertEqual(segment_ranges(300, 10, 30), [(0, 300)])
        self.assertEqual(segment_ranges(1, 10, 30), [(0, 1)])
        self.assertEqual(segment_ranges(601, 10, 30), [(0, 300), (300, 600), (600, 601)])
        with self.assertRaises(ValueError):
            segment_ranges(0, 10, 30)

    def test_numeric_order_and_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(12):
                np.save(root / f"segment_{index}.npy", np.zeros(3))
            self.assertEqual([p.stem for p in segment_files(root)], [f"segment_{i}" for i in range(12)])
            (root / "segment_4.npy").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous"):
                segment_files(root)

    def test_shared_temporal_window_and_evaluation_mean(self):
        values = np.arange(36, dtype=np.float32).reshape(12, 3)
        reference = np.random.default_rng(32)
        start = int(reference.integers(0, 12))
        end = int(reference.integers(start + 1, 13))
        train = pool_segment_views({"mixture": values, "guitar": values + 100},
                                   training=True, rng=np.random.default_rng(32))
        np.testing.assert_array_equal(train["mixture"], values[start:end].mean(0))
        np.testing.assert_array_equal(train["guitar"], (values + 100)[start:end].mean(0))
        evaluation = pool_segment_views({"mixture": values}, training=False)
        np.testing.assert_array_equal(evaluation["mixture"], values.mean(0))
        with self.assertRaisesRegex(ValueError, "counts differ"):
            pool_segment_views({"mixture": values, "guitar": values[:-1]}, training=False)
        with self.assertRaisesRegex(ValueError, "seeded"):
            pool_segment_views({"mixture": values}, training=True)

    def test_minimum_padding_is_encoder_specific_and_preserves_supported_inputs(self):
        for minimum in (400, 1025):
            original = torch.arange(minimum - 1, dtype=torch.float32)
            result, padding = prepare_encoder_segment(original, minimum, "pad_to_minimum")
            self.assertEqual(result.numel(), minimum)
            self.assertEqual(padding, 1)
            torch.testing.assert_close(result[:-1], original, rtol=0, atol=0)
            self.assertEqual(result[-1], 0)
            supported = torch.ones(minimum)
            self.assertIs(prepare_encoder_segment(supported, minimum, "pad_to_minimum")[0], supported)
            with self.assertRaises(ValueError):
                prepare_encoder_segment(original, minimum, "error")
        with self.assertRaises(ValueError):
            prepare_encoder_segment(torch.empty(0), 400, "pad_to_minimum")

    def test_alignment_changes_only_the_end_with_a_bounded_tolerance(self):
        original = torch.arange(1100, dtype=torch.float32)
        aligned, metadata = align_stem(original, 1000, 100)
        torch.testing.assert_close(aligned, original[:1000], rtol=0, atol=0)
        self.assertEqual(original.numel(), 1100)
        self.assertEqual(metadata["trimmed_end_samples"], 100)
        short = torch.arange(900, dtype=torch.float32)
        aligned, metadata = align_stem(short, 1000, 100)
        torch.testing.assert_close(aligned[:900], short, rtol=0, atol=0)
        self.assertTrue(torch.equal(aligned[900:], torch.zeros(100)))
        self.assertEqual(metadata["padded_end_samples"], 100)
        self.assertIs(align_stem(original, 1100, 100)[0], original)
        with self.assertRaisesRegex(ValueError, "allowed absolute difference"):
            align_stem(original, 999, 100)

    def test_public_jamendo_extractor_choices_exclude_archived_musicfm(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["extract", "mtg-jamendo", "musicfm", "--dry-run"])
        self.assertEqual(error.exception.code, 2)
        for representation in ("mert", "chords-keys"):
            args = parser.parse_args(["extract", "mtg-jamendo", representation, "--dry-run"])
            self.assertTrue(args.dry_run)
        for dataset in ("fma-medium", "gtzan", "ismir04"):
            args = parser.parse_args(["extract", dataset, "--dry-run"])
            self.assertTrue(args.dry_run)


class JamendoFeatureManifestTests(unittest.TestCase):
    def test_full_split_binding_and_repairs_preserve_metadata_without_audio_io(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tags = [f"mood/theme---tag_{index}" for index in range(56)]
            splits = {}
            for index, split in enumerate(("train", "validation", "test"), 1):
                splits[split] = root / f"{split}.tsv"
                with splits[split].open("w") as stream:
                    writer = csv.writer(stream, delimiter="\t")
                    writer.writerow(["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"])
                    writer.writerow([f"track_{index:07d}", "artist", "album", f"{index:02d}/{index}.mp3", "60", *tags])
            paths = {source: str(root / "repaired" / f"{source}.wav")
                     for source in ("mixture", "guitar", "drums", "piano", "bass", "vocals", "other")}
            override = root / "overrides.jsonl"
            row = {"track_id": "track_0000001", "split": "train", "relative_path": "01/1.mp3", "tags": tags, "paths": paths}
            override.write_text(json.dumps(row) + "\n")
            args = dict(split_files=splits, mixture_root=root / "original", htdemucs_root=root / "stems",
                        output=root / "manifest.jsonl", audio_overrides=override)
            self.assertEqual(build_feature_manifest(**args, dry_run=True)["records"], 3)
            self.assertFalse(args["output"].exists())
            build_feature_manifest(**args)
            rows = [json.loads(line) for line in args["output"].read_text().splitlines()]
            self.assertEqual(rows[0]["paths"], paths)
            self.assertEqual(rows[1]["paths"]["guitar"], str(root / "stems/2/guitar.mp3"))
            self.assertEqual(rows[0]["tags"], tags)
            self.assertFalse((root / "original").exists())
            with self.assertRaisesRegex(ValueError, "exists"):
                build_feature_manifest(**args)
            row["split"] = "test"
            override.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "changes split"):
                build_feature_manifest(**args, dry_run=True)


if __name__ == "__main__":
    unittest.main()
