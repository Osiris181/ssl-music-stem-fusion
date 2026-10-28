from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from stemfusion.features.htdemucs import RMSExtractionConfig, extract_rms


class HTDemucsRMSExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest_path = self.root / "manifest.jsonl"
        self.output_dir = self.root / "features"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self, track_ids: tuple[str, ...]) -> None:
        lines = []
        for track_id in track_ids:
            audio = self.root / "audio" / track_id / "other.mp3"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"audio")
            lines.append(
                json.dumps(
                    {
                        "track_id": track_id,
                        "split": "train",
                        "genre": "Rock",
                        "label": 0,
                        "paths": {"other": str(audio)},
                    }
                )
            )
        self.manifest_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def make_config(self, **overrides: object) -> RMSExtractionConfig:
        values: dict[str, object] = {
            "manifest_path": self.manifest_path,
            "output_dir": self.output_dir,
            "source": "other",
            "splits": ("train",),
        }
        values.update(overrides)
        return RMSExtractionConfig(**values)

    def test_writes_separate_one_value_cache_and_report(self) -> None:
        self.write_manifest(("000002", "000003"))

        summary = extract_rms(
            self.make_config(),
            feature_extractor=lambda _: [0.25],
        )

        output = self.output_dir / "fma_rms_other_train.pkl"
        with output.open("rb") as handle:
            cache = pickle.load(handle)
        self.assertEqual(
            cache,
            {"000002": [0.25], "000003": [0.25]},
        )
        self.assertEqual(summary.succeeded, 2)
        self.assertEqual(summary.failed, 0)

        report = json.loads(
            (self.output_dir / "extraction_report_rms_other.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["sample_rate"], 24_000)
        self.assertEqual(report["frame_length"], 2_048)
        self.assertEqual(report["hop_length"], 512)
        self.assertTrue(report["center"])
        self.assertEqual(report["pad_mode"], "constant")
        self.assertEqual(report["scale"], 20.0)

    def test_failure_creates_partial_cache(self) -> None:
        self.write_manifest(("000002", "000003"))

        def extractor(path: Path) -> list[float]:
            if "000003" in str(path):
                raise RuntimeError("test extraction failure")
            return [0.25]

        summary = extract_rms(
            self.make_config(),
            feature_extractor=extractor,
        )

        final_path = self.output_dir / "fma_rms_other_train.pkl"
        partial_path = self.output_dir / "fma_rms_other_train.partial.pkl"
        self.assertFalse(final_path.exists())
        self.assertTrue(partial_path.is_file())
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.exit_code, 1)

    def test_dry_run_does_not_initialize_extractor_or_write(self) -> None:
        self.write_manifest(("000002",))

        def unexpected(_: Path) -> list[float]:
            raise AssertionError("Extractor should not run during a dry run.")

        summary = extract_rms(
            self.make_config(dry_run=True),
            feature_extractor=unexpected,
        )

        self.assertEqual(summary.planned, 1)
        self.assertEqual(summary.succeeded, 0)
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
