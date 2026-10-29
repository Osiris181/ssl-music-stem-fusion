from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from stemfusion.experiments.fma_medium.extraction import (
    MusicFMExtractionConfig,
    extract_musicfm,
)


class FMAMediumMusicFMExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.musicfm_root = self.root / "musicfm"
        data = self.musicfm_root / "data"
        data.mkdir(parents=True)
        (data / "msd_stats.json").write_text("{}", encoding="utf-8")
        (data / "pretrained_msd.pt").write_bytes(b"checkpoint")
        self.manifest_path = self.root / "manifest.jsonl"
        self.output_dir = self.root / "features"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(
        self,
        track_ids: tuple[str, ...],
        source: str = "drums",
    ) -> None:
        lines = []
        for track_id in track_ids:
            audio = self.root / "audio" / track_id / f"{source}.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"audio")
            lines.append(
                json.dumps(
                    {
                        "track_id": track_id,
                        "split": "train",
                        "genre": "Rock",
                        "label": 0,
                        "paths": {source: str(audio)},
                    }
                )
            )
        self.manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_config(self, **overrides: object) -> MusicFMExtractionConfig:
        values: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "output_dir": self.output_dir,
            "source": "drums",
            "musicfm_root": self.musicfm_root,
            "expected_dimension": 3,
            "splits": ("train",),
        }
        values.update(overrides)
        return MusicFMExtractionConfig(**values)

    def test_writes_track_id_keyed_legacy_compatible_cache(self) -> None:
        self.write_manifest(("000002", "000003"))

        summary = extract_musicfm(
            self.make_config(),
            feature_extractor=lambda _: [1.0, 2.0, 3.0],
        )

        output = self.output_dir / "fma_musicfm_drums_train.pkl"
        with output.open("rb") as handle:
            cache = pickle.load(handle)
        self.assertEqual(cache, {
            "000002": [1.0, 2.0, 3.0],
            "000003": [1.0, 2.0, 3.0],
        })
        self.assertEqual(summary.succeeded, 2)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.exit_code, 0)
        report = json.loads(
            (self.output_dir / "extraction_report_musicfm_drums.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["layer_index"], 7)
        self.assertTrue(report["checkpoint_path"].endswith("pretrained_msd.pt"))

    def test_failure_creates_partial_cache_and_nonzero_summary(self) -> None:
        self.write_manifest(("000002", "000003"))

        def extractor(path: Path) -> list[float]:
            if "000003" in str(path):
                raise RuntimeError("test extraction failure")
            return [1.0, 2.0, 3.0]

        summary = extract_musicfm(
            self.make_config(),
            feature_extractor=extractor,
        )

        final_path = self.output_dir / "fma_musicfm_drums_train.pkl"
        partial_path = self.output_dir / "fma_musicfm_drums_train.partial.pkl"
        self.assertFalse(final_path.exists())
        self.assertTrue(partial_path.is_file())
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.exit_code, 1)

    def test_dry_run_does_not_initialize_extractor_or_write(self) -> None:
        self.write_manifest(("000002",))

        def unexpected(_: Path) -> list[float]:
            raise AssertionError("Extractor should not run during a dry run.")

        summary = extract_musicfm(
            self.make_config(dry_run=True),
            feature_extractor=unexpected,
        )

        self.assertEqual(summary.planned, 1)
        self.assertEqual(summary.succeeded, 0)
        self.assertFalse(self.output_dir.exists())

    def test_accepts_mixture_source(self) -> None:
        self.write_manifest(("000002",), source="mixture")

        summary = extract_musicfm(
            self.make_config(source="mixture", dry_run=True),
        )

        self.assertEqual(summary.source, "mixture")
        self.assertEqual(summary.planned, 1)

if __name__ == "__main__":
    unittest.main()
