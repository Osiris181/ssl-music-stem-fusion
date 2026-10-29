from __future__ import annotations

import contextlib
import io
import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from stemfusion.cli import build_parser, main
from stemfusion.configuration import default_config_path
from stemfusion.features._musicfm import (
    MusicFMExtractionConfig,
    MusicFMExtractionError,
    extract_musicfm,
    extract_musicfm_sources,
)


class MultiSourceExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = self.root / "manifest.jsonl"
        self.output = self.root / "features"
        self.model_root = self.root / "musicfm"
        data = self.model_root / "data"
        data.mkdir(parents=True)
        (data / "msd_stats.json").write_text("{}")
        (data / "pretrained_msd.pt").write_bytes(b"synthetic")
        records = []
        for track in ("000002", "000003"):
            paths = {}
            for source in ("mixture", "guitar", "piano"):
                audio = self.root / "audio" / track / f"{source}.mp3"
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_bytes(b"synthetic")
                paths[source] = str(audio)
            records.append({
                "track_id": track, "split": "train",
                "genre": "Rock", "label": 0, "paths": paths,
            })
        self.manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def configs(self, sources=("mixture", "guitar"), dry_run=False):
        return [
            MusicFMExtractionConfig(
                manifest_path=self.manifest, output_dir=self.output,
                source=source, musicfm_root=self.model_root,
                expected_dimension=3, splits=("train",), dry_run=dry_run,
            )
            for source in sources
        ]

    @staticmethod
    def feature(_: Path):
        return [1.0, 2.0, 3.0]

    def test_sources_share_one_model_and_keep_individual_caches(self) -> None:
        with patch(
            "stemfusion.features._musicfm._MusicFMRuntime",
            return_value=self.feature,
        ) as factory:
            summaries = extract_musicfm_sources(self.configs())
        factory.assert_called_once()
        self.assertEqual([summary.source for summary in summaries], ["mixture", "guitar"])
        self.assertTrue(all(summary.succeeded == 2 for summary in summaries))
        for source in ("mixture", "guitar"):
            with (self.output / f"fma_musicfm_{source}_train.pkl").open("rb") as handle:
                cache = pickle.load(handle)
            self.assertEqual(set(cache), {"000002", "000003"})
            self.assertTrue((self.output / f"extraction_report_musicfm_{source}.json").is_file())

    def test_dry_run_never_loads_model_or_writes(self) -> None:
        with patch("stemfusion.features._musicfm._MusicFMRuntime") as factory:
            summaries = extract_musicfm_sources(self.configs(dry_run=True))
        factory.assert_not_called()
        self.assertTrue(all(summary.planned == 2 for summary in summaries))
        self.assertFalse(self.output.exists())

    def test_completed_sources_do_not_load_model(self) -> None:
        extract_musicfm_sources(self.configs(), feature_extractor=self.feature)
        with patch("stemfusion.features._musicfm._MusicFMRuntime") as factory:
            summaries = extract_musicfm_sources(self.configs())
        factory.assert_not_called()
        self.assertTrue(all(summary.split_summaries[0].skipped_existing_cache for summary in summaries))

    def test_only_pending_sources_use_the_shared_model(self) -> None:
        extract_musicfm(self.configs()[0], feature_extractor=self.feature)
        with patch(
            "stemfusion.features._musicfm._MusicFMRuntime",
            return_value=self.feature,
        ) as factory:
            summaries = extract_musicfm_sources(self.configs())
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0].source, "guitar")
        self.assertTrue(summaries[0].split_summaries[0].skipped_existing_cache)
        self.assertEqual(summaries[1].succeeded, 2)

    def test_failed_source_retains_partial_cache_and_stops_later_sources(self) -> None:
        def feature(path):
            if path.name == "guitar.mp3" and path.parent.name == "000003":
                raise RuntimeError("synthetic failure")
            return self.feature(path)
        summaries = extract_musicfm_sources(
            self.configs(("mixture", "guitar", "piano")),
            feature_extractor=feature,
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[1].failed, 1)
        self.assertTrue((self.output / "fma_musicfm_guitar_train.partial.pkl").is_file())
        self.assertFalse((self.output / "fma_musicfm_piano_train.pkl").exists())

    def test_duplicate_sources_or_inconsistent_encoder_settings_fail(self) -> None:
        configs = self.configs()
        with self.assertRaises(MusicFMExtractionError):
            extract_musicfm_sources([configs[0], configs[0]])
        with self.assertRaises(MusicFMExtractionError):
            extract_musicfm_sources([configs[0], replace(configs[1], layer_index=12)])
        self.assertFalse(self.output.exists())

    def cli_config(self, sources=("mixture", "guitar")) -> Path:
        config = self.root / "extraction.yaml"
        config.write_text(yaml.safe_dump({
            "feature_profile": str(default_config_path("features/musicfm_msd_layer7.yaml")),
            "sources": list(sources),
            "paths": {
                "manifest": str(self.manifest),
                "musicfm_root": str(self.model_root),
                "output_dir": str(self.output),
            },
            "features": {"dimension": 3},
            "output": {"files_per_source": ["train"]},
        }))
        return config

    def test_cli_uses_yaml_sources_and_writes_aggregate_report(self) -> None:
        config = self.cli_config()
        report = self.root / "batch.json"
        output = io.StringIO()
        with patch(
            "stemfusion.features._musicfm._MusicFMRuntime",
            return_value=self.feature,
        ) as factory, patch("stemfusion.cli.logging.basicConfig"), contextlib.redirect_stdout(output):
            result = main([
                "extract", "fma-medium", "--config", str(config),
                "--report", str(report),
            ])
        self.assertEqual(result, 0)
        factory.assert_called_once()
        saved = json.loads(report.read_text())
        self.assertEqual(saved["requested_sources"], ["mixture", "guitar"])
        self.assertEqual(saved["processed_sources"], ["mixture", "guitar"])
        self.assertEqual(saved["unprocessed_sources"], [])
        self.assertEqual(saved["succeeded"], 4)
        self.assertEqual(output.getvalue(), "")
        self.assertTrue((self.output / "extraction_report_musicfm_mixture.json").exists())
        self.assertTrue((self.output / "extraction_report_musicfm_guitar.json").exists())

    def test_cli_reports_failure_and_sources_not_processed(self) -> None:
        config = self.cli_config(("mixture", "guitar", "piano"))

        def feature(path):
            if path.name == "guitar.mp3" and path.parent.name == "000003":
                raise RuntimeError("synthetic failure")
            return self.feature(path)

        with patch(
            "stemfusion.features._musicfm._MusicFMRuntime", return_value=feature,
        ), patch("stemfusion.cli.logging.basicConfig"), contextlib.redirect_stdout(io.StringIO()):
            result = main(["extract", "fma-medium", "--config", str(config)])
        self.assertEqual(result, 1)
        report = json.loads((self.output / "extraction_report_musicfm_batch.json").read_text())
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["processed_sources"], ["mixture", "guitar"])
        self.assertEqual(report["unprocessed_sources"], ["piano"])
        self.assertEqual(report["failed"], 1)
        self.assertTrue((self.output / "fma_musicfm_guitar_train.partial.pkl").exists())

    def test_default_yaml_sources_and_explicit_subset(self) -> None:
        parser = build_parser()
        arguments = [
            "extract", "fma-medium", "--manifest", str(self.manifest),
            "--musicfm-root", str(self.model_root), "--dry-run",
        ]
        args = parser.parse_args(arguments)
        self.assertEqual(args.source, ["mixture", "guitar", "drums", "piano", "bass", "vocals"])
        subset = parser.parse_args([*arguments, "--source", "piano", "--source", "guitar"])
        self.assertEqual(subset.source, ["piano", "guitar"])
        again = parser.parse_args(arguments)
        self.assertEqual(again.source, args.source)


if __name__ == "__main__":
    unittest.main()
