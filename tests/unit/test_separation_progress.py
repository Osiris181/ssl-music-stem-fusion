"""Visible preparation/separation progress without changing jobs or artifacts."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from stemfusion.separation import htdemucs


class RecordingProgress:
    """Track progress lifecycle without terminal rendering or timing assumptions."""

    instances = []

    def __init__(self, iterable=None, *, total=None, desc="", **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable) if hasattr(iterable, "__len__") else None
        self.desc = desc
        self.n = 0
        self.closed = False
        self.postfix = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update()

    def update(self, count=1):
        self.n += count

    def set_postfix(self, *, refresh=True, **values):
        self.postfix = values


class SeparationProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.output = self.root / "stems"
        self.config = htdemucs.HTDemucsSeparationConfig((self.audio,), self.output, batch_size=2)
        RecordingProgress.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def tracks(self, count=3):
        paths = []
        for index in range(count):
            path = self.audio / f"{index}.mp3"
            path.write_bytes(b"synthetic mixture")
            paths.append(path)
        return paths

    def complete(self, path):
        target = self.output / "htdemucs_6s" / path.stem
        target.mkdir(parents=True, exist_ok=True)
        for stem in htdemucs.HTDEMUCS_6S_STEMS:
            (target / f"{stem}.mp3").write_bytes(b"synthetic stem")

    def manifest(self, count=3):
        records = [{"track_id": path.stem, "split": "train", "paths": {
            "mixture": str(path),
            **{stem: str(self.output / "htdemucs_6s" / path.stem / f"{stem}.mp3")
               for stem in htdemucs.HTDEMUCS_6S_STEMS},
        }} for path in self.tracks(count)]
        path = self.root / "manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        return path, records

    def runner(self, command, **kwargs):
        for argument in command:
            if argument.endswith(".mp3"):
                self.complete(Path(argument))
        return subprocess.CompletedProcess(command, 0)

    def bar(self, desc):
        matches = [bar for bar in RecordingProgress.instances if bar.desc == desc]
        self.assertEqual(len(matches), 1, desc)
        return matches[0]

    def assert_closed(self):
        self.assertTrue(RecordingProgress.instances)
        self.assertTrue(all(bar.closed for bar in RecordingProgress.instances))

    def test_discovery_reports_visited_paths_and_audio_without_a_fake_total(self):
        self.tracks(2)
        (self.audio / "notes.txt").write_text("not audio")
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            paths = htdemucs.discover_audio_files(self.config)
        bar = self.bar("Discover audio files")
        self.assertEqual([path.name for path in paths], ["0.mp3", "1.mp3"])
        self.assertEqual((bar.n, bar.total, bar.postfix["audio"]), (3, None, 2))
        self.assert_closed()

    def test_completion_bar_reports_existing_and_pending_without_running_dry_jobs(self):
        paths = self.tracks(3)
        self.complete(paths[0])
        runner = Mock(side_effect=AssertionError("Dry runs must not separate audio"))
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(replace(self.config, dry_run=True), runner=runner)
        bar = self.bar("Check existing stems")
        self.assertEqual((bar.n, bar.total), (3, 3))
        self.assertEqual(bar.postfix, {"existing": 1, "pending": 2})
        self.assertEqual((summary.skipped_existing, summary.planned), (1, 2))
        self.assertNotIn("HT-Demucs batches", [bar.desc for bar in RecordingProgress.instances])
        runner.assert_not_called()
        self.assert_closed()

    def test_successful_batches_keep_commands_and_track_counts(self):
        self.tracks(3)
        runner = Mock(side_effect=self.runner)
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(self.config, runner=runner)
        bar = self.bar("HT-Demucs batches")
        self.assertEqual((bar.n, bar.total, summary.succeeded, summary.failed), (2, 2, 3, 0))
        self.assertEqual(runner.call_count, 2)
        for call in runner.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--shifts") + 1], "1")
            self.assertEqual(command[command.index("--overlap") + 1], "0.25")
            self.assertFalse(call.kwargs["capture_output"])
        self.assert_closed()

    def test_failed_batch_retries_only_incomplete_tracks_with_its_own_bar(self):
        paths = self.tracks(2)
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                self.complete(paths[0])
                return subprocess.CompletedProcess(command, 1)
            self.assertEqual(command[-1], str(paths[1]))
            self.assertTrue(kwargs["capture_output"])
            return self.runner(command, **kwargs)
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(self.config, runner=runner)
        bar = self.bar("Retry incomplete tracks")
        self.assertEqual((bar.n, bar.total), (1, 1))
        self.assertEqual((summary.succeeded, summary.rescue_attempts, summary.failed_batches), (2, 1, 1))
        self.assertEqual(len(calls), 2)
        self.assert_closed()

    def test_process_start_failure_closes_progress_and_preserves_failures(self):
        self.tracks(3)
        runner = Mock(side_effect=OSError("synthetic missing executable"))
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(self.config, runner=runner)
        self.assertEqual((summary.failed, summary.failed_batches, summary.succeeded), (2, 1, 0))
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(summary.failures[0].reason, "synthetic missing executable")
        self.assertLess(self.bar("HT-Demucs batches").n, self.bar("HT-Demucs batches").total)
        self.assert_closed()

    def test_fail_fast_closes_main_and_retry_bars(self):
        self.tracks(3)
        runner = Mock(return_value=subprocess.CompletedProcess([], 1))
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(replace(self.config, continue_on_error=False), runner=runner)
        self.assertEqual((summary.failed, summary.rescue_attempts, runner.call_count), (1, 1, 2))
        self.assertEqual(self.bar("Retry incomplete tracks").n, 0)
        self.assert_closed()

    def test_single_track_failure_closes_bar_without_a_redundant_retry(self):
        self.tracks(1)
        runner = Mock(return_value=subprocess.CompletedProcess([], 1))
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_dataset(replace(self.config, continue_on_error=False), runner=runner)
        self.assertEqual((summary.failed, summary.rescue_attempts, runner.call_count), (1, 0, 1))
        self.assert_closed()

    def test_manifest_checks_existing_stems_and_stages_only_missing_jobs(self):
        manifest, rows = self.manifest()
        self.complete(Path(rows[0]["paths"]["mixture"]))
        original = manifest.read_bytes()
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_manifest(self.config, manifest, runner=self.runner)
        checks, staging = self.bar("Check manifest and stems"), self.bar("Prepare separation inputs")
        self.assertEqual((checks.n, checks.total), (3, 3))
        self.assertEqual(checks.postfix, {"existing": 1, "pending": 2})
        self.assertEqual((staging.n, staging.total), (2, 2))
        self.assertEqual((summary.skipped_existing, summary.succeeded, summary.planned), (1, 2, 2))
        self.assertEqual(manifest.read_bytes(), original)
        self.assert_closed()

    def test_manifest_dry_run_validates_all_destinations_even_with_limit(self):
        manifest, _ = self.manifest()
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_manifest(replace(self.config, dry_run=True, limit=1), manifest)
        checks = self.bar("Check manifest and stems")
        self.assertEqual((checks.n, checks.total, summary.selected, summary.planned), (3, 3, 1, 1))
        self.assertFalse(self.output.exists())
        self.assertNotIn("Prepare separation inputs", [bar.desc for bar in RecordingProgress.instances])
        self.assert_closed()

    def test_invalid_manifest_closes_progress_without_creating_stems(self):
        manifest, rows = self.manifest()
        rows[-1]["paths"]["guitar"] = str(self.root / "outside/guitar.mp3")
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with patch.object(htdemucs, "tqdm", RecordingProgress), self.assertRaises(ValueError):
            htdemucs.separate_manifest(replace(self.config, dry_run=True), manifest)
        self.assertFalse(self.output.exists())
        self.assert_closed()

    def test_all_completed_manifest_jobs_skip_staging_and_separation(self):
        manifest, rows = self.manifest()
        for row in rows:
            self.complete(Path(row["paths"]["mixture"]))
        runner = Mock(side_effect=AssertionError("Complete stems must be reused"))
        with patch.object(htdemucs, "tqdm", RecordingProgress):
            summary = htdemucs.separate_manifest(self.config, manifest, runner=runner)
        self.assertEqual((summary.skipped_existing, summary.planned), (3, 0))
        self.assertEqual([bar.desc for bar in RecordingProgress.instances], ["Check manifest and stems"])
        runner.assert_not_called()
        self.assert_closed()

    def test_all_dataset_scripts_keep_readable_summary_on_stdout_and_progress_on_stderr(self):
        manifest, _ = self.manifest(1)
        for dataset in ("fma_medium", "gtzan", "ismir04", "mtg_jamendo", "pmemo", "deam", "emomusic"):
            with self.subTest(dataset=dataset):
                entry = importlib.import_module(f"stemfusion.experiments.{dataset}.separation")
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = entry.main(["--manifest", str(manifest), "--output-root", str(self.output),
                                         "--extension", ".mp3", "--dry-run"])
                self.assertEqual(result, 0)
                self.assertIn("HT-Demucs separation summary", stdout.getvalue())
                self.assertIn("Pending separation: 1", stdout.getvalue())
                self.assertIn("Check manifest and stems", stderr.getvalue())
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
