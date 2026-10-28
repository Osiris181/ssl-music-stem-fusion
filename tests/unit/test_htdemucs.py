from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from stemfusion.separation.htdemucs import (
    HTDEMUCS_6S_STEMS,
    HTDemucsConfigurationError,
    HTDemucsSeparationConfig,
    build_command,
    discover_audio_files,
    is_complete_track,
    separate_dataset,
    track_output_directory,
)


class HTDemucsSeparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_config(self, **overrides: object) -> HTDemucsSeparationConfig:
        input_a = self.root / "input-a"
        input_b = self.root / "input-b"
        input_a.mkdir()
        input_b.mkdir()
        values: dict[str, object] = {
            "input_roots": (input_a, input_b),
            "output_root": self.root / "outputs",
        }
        values.update(overrides)
        return HTDemucsSeparationConfig(**values)

    def write_stems(
        self,
        config: HTDemucsSeparationConfig,
        input_path: Path,
        output_root: Path | None = None,
    ) -> None:
        track_directory = track_output_directory(
            output_root or config.output_root,
            config.model_name,
            input_path,
        )
        track_directory.mkdir(parents=True, exist_ok=True)
        for stem in HTDEMUCS_6S_STEMS:
            (track_directory / f"{stem}.mp3").write_bytes(b"stem")

    def test_discovery_supports_multiple_roots_and_extensions(self) -> None:
        config = self.make_config(extensions=(".mp3", ".wav", ".au"))
        input_a, input_b = config.input_roots
        (input_a / "genre").mkdir()
        (input_a / "genre" / "001.wav").write_bytes(b"audio")
        (input_b / "003.au").write_bytes(b"audio")
        (input_b / "002.mp3").write_bytes(b"audio")
        (input_b / "ignore.txt").write_text("ignore", encoding="utf-8")

        discovered = discover_audio_files(config)

        self.assertEqual(
            [path.name for path in discovered],
            ["001.wav", "002.mp3", "003.au"],
        )

    def test_duplicate_track_stems_are_rejected(self) -> None:
        config = self.make_config(extensions=(".mp3", ".wav"))
        input_a, input_b = config.input_roots
        (input_a / "same.mp3").write_bytes(b"first")
        (input_b / "same.wav").write_bytes(b"second")

        with self.assertRaises(HTDemucsConfigurationError):
            discover_audio_files(config)

    def test_command_makes_legacy_defaults_explicit(self) -> None:
        config = self.make_config()
        input_path = config.input_roots[0] / "track.mp3"

        command = build_command(config, [input_path])

        self.assertEqual(command[:3], [config.python_executable, "-m", "demucs.separate"])
        self.assertIn("htdemucs_6s", command)
        self.assertIn("--mp3", command)
        self.assertIn("320", command)
        self.assertIn("--shifts", command)
        self.assertIn("0.25", command)
        self.assertEqual(command[-1], str(input_path))

    def test_completion_requires_all_six_nonempty_stems(self) -> None:
        config = self.make_config()
        input_path = config.input_roots[0] / "track.mp3"
        input_path.write_bytes(b"audio")
        track_directory = track_output_directory(
            config.output_root,
            config.model_name,
            input_path,
        )
        track_directory.mkdir(parents=True)
        for stem in HTDEMUCS_6S_STEMS[:-1]:
            (track_directory / f"{stem}.mp3").write_bytes(b"stem")

        self.assertFalse(
            is_complete_track(config.output_root, config, input_path)
        )
        (track_directory / "vocals.mp3").write_bytes(b"stem")
        self.assertTrue(
            is_complete_track(config.output_root, config, input_path)
        )

    def test_completed_track_in_read_only_root_is_skipped(self) -> None:
        archive = self.root / "archive"
        archive.mkdir()
        config = self.make_config(existing_output_roots=(archive,))
        input_path = config.input_roots[0] / "track.mp3"
        input_path.write_bytes(b"audio")
        self.write_stems(config, input_path, output_root=archive)

        def unexpected_runner(*_: object, **__: object) -> subprocess.CompletedProcess:
            raise AssertionError("Demucs should not be invoked.")

        summary = separate_dataset(config, runner=unexpected_runner)

        self.assertEqual(summary.skipped_existing, 1)
        self.assertEqual(summary.succeeded, 0)

    def test_failed_batch_retries_only_incomplete_tracks(self) -> None:
        config = self.make_config(batch_size=2)
        input_a = config.input_roots[0] / "a.mp3"
        input_b = config.input_roots[1] / "b.mp3"
        input_a.write_bytes(b"audio")
        input_b.write_bytes(b"audio")
        calls: list[list[str]] = []

        def runner(command: list[str], **_: object) -> subprocess.CompletedProcess:
            calls.append(command)
            tracks = [
                Path(argument)
                for argument in command
                if argument in {str(input_a), str(input_b)}
            ]
            if len(tracks) == 2:
                self.write_stems(config, input_a)
                return subprocess.CompletedProcess(command, returncode=1)
            self.write_stems(config, tracks[0])
            return subprocess.CompletedProcess(command, returncode=0)

        summary = separate_dataset(config, runner=runner)

        self.assertEqual(summary.succeeded, 2)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.failed_batches, 1)
        self.assertEqual(summary.rescue_attempts, 1)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
