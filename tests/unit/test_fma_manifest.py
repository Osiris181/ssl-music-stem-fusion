from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from stemfusion.data.fma import (
    FMAManifestConfig,
    FMAManifestError,
    HTDEMUCS_SOURCES,
    build_fma_manifest,
    read_manifest,
)


class FMAManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tracks_csv = self.root / "tracks.csv"
        self.mixture_root = self.root / "mixtures"
        self.source_root = self.root / "demucs"
        self.mixture_root.mkdir()
        self.source_root.mkdir()
        self.output_path = self.root / "fma_htdemucs.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_metadata(self) -> None:
        with self.tracks_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "set", "set", "track"])
            writer.writerow(["", "split", "subset", "genre_top"])
            writer.writerow(["track_id", "", "", ""])
            writer.writerow([2, "training", "small", "Rock"])
            writer.writerow([3, "validation", "medium", "Classical"])
            writer.writerow([4, "test", "large", "Jazz"])

    def write_audio(self, track_id: str) -> None:
        folder = track_id[:3]
        mixture = self.mixture_root / folder / f"{track_id}.mp3"
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_bytes(b"mixture")
        for source in HTDEMUCS_SOURCES:
            stem = self.source_root / "htdemucs_6s" / track_id / f"{source}.mp3"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"stem")

    def make_config(self, **overrides: object) -> FMAManifestConfig:
        values: dict[str, object] = {
            "tracks_csv": self.tracks_csv,
            "output_path": self.output_path,
            "mixture_root": self.mixture_root,
            "htdemucs_roots": (self.source_root,),
        }
        values.update(overrides)
        return FMAManifestConfig(**values)

    def test_builds_aligned_htdemucs_manifest(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        self.write_audio("000003")

        records, summary = build_fma_manifest(self.make_config())

        self.assertEqual([record.track_id for record in records], ["000002", "000003"])
        self.assertEqual([record.split for record in records], ["train", "validation"])
        self.assertEqual(summary.label_mapping, {"Classical": 0, "Rock": 1})
        self.assertEqual(summary.split_counts["train"], 1)
        self.assertEqual(summary.split_counts["validation"], 1)
        self.assertEqual(summary.split_counts["test"], 0)
        self.assertTrue(records[0].paths["drums"].endswith("htdemucs_6s/000002/drums.mp3"))
        self.assertEqual(read_manifest(self.output_path), records)

        metadata = json.loads(
            (self.root / "fma_htdemucs.metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metadata["sources"], sorted({"mixture", *HTDEMUCS_SOURCES}))

    def test_missing_audio_requires_explicit_aligned_drop(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        mixture = self.mixture_root / "000" / "000003.mp3"
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_bytes(b"mixture")

        with self.assertRaises(FMAManifestError):
            build_fma_manifest(self.make_config())

        records, summary = build_fma_manifest(
            self.make_config(allow_missing=True)
        )
        self.assertEqual([record.track_id for record in records], ["000002"])
        self.assertEqual(summary.dropped_tracks, 1)
        self.assertEqual(summary.missing_by_source, {source: 1 for source in HTDEMUCS_SOURCES})

    def test_supports_distributed_htdemucs_roots(self) -> None:
        self.write_metadata()
        first_root = self.root / "demucs_first"
        second_root = self.root / "demucs_second"
        first_root.mkdir()
        second_root.mkdir()

        for track_id, demucs_root in (
            ("000002", first_root),
            ("000003", second_root),
        ):
            folder = track_id[:3]
            mixture = self.mixture_root / folder / f"{track_id}.mp3"
            mixture.parent.mkdir(parents=True, exist_ok=True)
            mixture.write_bytes(b"mixture")
            for source in HTDEMUCS_SOURCES:
                stem = (
                    demucs_root
                    / "htdemucs_6s"
                    / track_id
                    / f"{source}.mp3"
                )
                stem.parent.mkdir(parents=True, exist_ok=True)
                stem.write_bytes(b"stem")

        partial = (
            first_root / "htdemucs_6s" / "000003" / "drums.mp3"
        )
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"incomplete duplicate")

        records, _ = build_fma_manifest(
            self.make_config(
                htdemucs_roots=(first_root, second_root),
            )
        )

        by_id = {record.track_id: record for record in records}
        self.assertEqual(
            set(by_id["000002"].paths),
            {"mixture", *HTDEMUCS_SOURCES},
        )
        self.assertTrue(
            by_id["000003"].paths["other"].startswith(
                str(second_root.resolve())
            )
        )
    def test_can_restrict_manifest_to_selected_track_ids(self) -> None:
        self.write_metadata()
        self.write_audio("000002")
        self.write_audio("000003")

        records, summary = build_fma_manifest(
            self.make_config(track_ids=("000003",))
        )

        self.assertEqual([record.track_id for record in records], ["000003"])
        self.assertEqual(records[0].label, 0)
        self.assertEqual(summary.metadata_tracks, 1)
        self.assertEqual(summary.written_tracks, 1)

if __name__ == "__main__":
    unittest.main()
