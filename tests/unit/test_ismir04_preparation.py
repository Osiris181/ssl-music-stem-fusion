from __future__ import annotations

import csv
import json
import math
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from stemfusion.cli import build_parser
from stemfusion.data.audio_duplicates import audit_audio_files
from stemfusion.data.fma import read_manifest
from stemfusion.data.ismir04 import LABEL_MAPPING, SPLITS, read_ismir04_metadata, select_unique_tracks, prepare_ismir04_manifest


class ISMIR04PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.metadata = self.root / "metadata"
        self.audio = self.root / "audio"
        for source_split in SPLITS:
            path = self.metadata / source_split / "tracklist.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                for genre in LABEL_MAPPING:
                    for index in range(2):
                        stem = "shared_name" if genre == "classical" and index == 0 and source_split != "development" else f"{source_split}_{genre}_{index}"
                        folder = "jazz" if genre == "jazz_blues" else genre
                        relative = f"{folder}/{source_split}_artist/{stem}.wav"
                        row = [genre, "artist", "album, with comma", "title", str(index), relative]
                        if source_split == "evaluation":
                            row.append("extra-evaluation-field")
                        writer.writerow(row)
        self.tracks = read_ismir04_metadata(self.metadata, self.audio)
        self.audit_path = self.root / "audit.json"
        self.audit = {"complete": True, "files": {
            str(track.original_path): {"decoded_sha256": track.track_id} for track in self.tracks}}

    def tearDown(self):
        self.temporary.cleanup()

    def test_csv_labels_full_path_ids_and_evaluation_filename_column(self):
        self.assertEqual(len(self.tracks), 36)
        self.assertEqual(len({track.track_id for track in self.tracks}), 36)
        self.assertEqual({track.genre for track in self.tracks}, set(LABEL_MAPPING))
        jazz = next(track for track in self.tracks if track.genre == "jazz_blues")
        self.assertTrue(jazz.relative_path.startswith("jazz/"))
        shared = [track for track in self.tracks if track.legacy_stem_name == "shared_name"]
        self.assertEqual(len(shared), 2)
        self.assertNotEqual(shared[0].track_id, shared[1].track_id)
        self.assertTrue(all(track.original_path.suffix == ".wav" for track in self.tracks))

    def test_duplicate_priority_preserves_all_retained_assignments(self):
        members = [track for track in self.tracks if track.genre == "world" and track.relative_path.endswith("_0.wav")]
        for track in members:
            self.audit["files"][str(track.original_path)]["decoded_sha256"] = "same-world-audio"
        retained, excluded = select_unique_tracks(self.tracks, self.audit)
        self.assertEqual(len(excluded), 2)
        self.assertEqual({item["split"] for item in excluded}, {"validation", "test"})
        self.assertTrue(all(item["retained_split"] == "train" for item in excluded))
        original = {track.track_id: track.split for track in self.tracks}
        self.assertTrue(all(track.split == original[track.track_id] for track in retained))

    def test_filename_corrections_preserve_csv_identity_and_partition(self):
        original = self.tracks[0]
        key = f"{original.source_split}/{original.relative_path}"
        corrected = "classical/corrected_artist/corrected_filename.wav"
        tracks = read_ismir04_metadata(self.metadata, self.audio, {key: corrected})
        updated = next(track for track in tracks if track.track_id == original.track_id)
        self.assertEqual(updated.split, original.split)
        self.assertEqual(updated.genre, original.genre)
        self.assertEqual(updated.relative_path, original.relative_path)
        self.assertEqual(updated.original_path, self.audio / original.source_split / corrected)
        self.assertEqual(updated.legacy_stem_name, "corrected_filename")
        with self.assertRaisesRegex(ValueError, "present in the CSV"):
            read_ismir04_metadata(self.metadata, self.audio, {"unknown.wav": corrected})

    def test_conflicting_labels_and_incomplete_audits_fail(self):
        first, second = self.tracks[0], self.tracks[2]
        self.audit["files"][str(first.original_path)]["decoded_sha256"] = "same"
        self.audit["files"][str(second.original_path)]["decoded_sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "conflicting genre"):
            select_unique_tracks(self.tracks, self.audit)
        with self.assertRaisesRegex(ValueError, "complete"):
            select_unique_tracks(self.tracks, {"complete": False})

    def test_manifest_records_exclusions_and_unambiguous_stem_repair(self):
        training = next(track for track in self.tracks if track.genre == "world" and track.split == "train")
        evaluation = next(track for track in self.tracks if track.genre == "world" and track.split == "test")
        self.audit["files"][str(evaluation.original_path)]["decoded_sha256"] = training.track_id
        self.audit_path.write_text(json.dumps(self.audit))
        verified = next(track for track in self.tracks if track.legacy_stem_name == "shared_name" and track.split == "train")
        output = self.root / "manifests/ismir04.jsonl"
        kwargs = dict(metadata_root=self.metadata, audio_root=self.audio, htdemucs_root=self.root / "stems",
            repair_root=self.root / "repairs/htdemucs_6s", output=output, audit_path=self.audit_path,
            verified_legacy_stems=(f"{verified.source_split}/{verified.relative_path}",))
        with patch("stemfusion.data.ismir04.audit_audio_files", return_value=self.audit):
            report = prepare_ismir04_manifest(**kwargs)
            with self.assertRaisesRegex(ValueError, "already exist"):
                prepare_ismir04_manifest(**kwargs)
        records = read_manifest(output)
        self.assertEqual(report["split_counts"], {"train": 12, "validation": 12, "test": 11})
        self.assertEqual(len(report["required_stem_repairs"]), 1)
        self.assertEqual(len(report["excluded_duplicates"]), 1)
        self.assertEqual(len(records), 35)
        repair = report["required_stem_repairs"][0]
        target = next(record for record in records if record.track_id == repair["track_id"])
        self.assertIn("/repairs/htdemucs_6s/", target.paths["guitar"])
        self.assertTrue(output.with_suffix(".metadata.json").is_file())

    def test_yaml_manifest_paths_and_fixed_policy_are_loaded(self):
        args = build_parser().parse_args(["data", "ismir04-manifest", "--audit-only"])
        self.assertTrue(args.audit_only)
        self.assertEqual(args.audit_workers, 2)
        self.assertEqual(args.output.parent.name, "ismir04")
        self.assertEqual(len(args.verified_legacy_stem), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for decoded-audio hashing")
    def test_file_and_decoded_audio_hashes_and_resume(self):
        first, second = self.root / "first.wav", self.root / "second.wav"
        with wave.open(str(first), "wb") as stream:
            stream.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
            stream.writeframes(b"".join(struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * i / 8000))) for i in range(8000)))
        shutil.copyfile(first, second)
        with second.open("ab") as stream:
            stream.write(b"different trailing file metadata")
        report = audit_audio_files([first, second], self.audit_path, workers=1)
        a, b = report["files"][str(first)], report["files"][str(second)]
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["decoded_sha256"], b["decoded_sha256"])
        with patch("stemfusion.data.audio_duplicates.subprocess.run", side_effect=AssertionError("Unchanged audit must be reused")):
            reused = audit_audio_files([first, second], self.audit_path, workers=1)
        self.assertTrue(reused["complete"])


if __name__ == "__main__":
    unittest.main()
