from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stemfusion.separation import htdemucs


class SeparationReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = htdemucs.HTDemucsSeparationConfig(
            input_roots=(), output_root=self.root / "stems"
        )

    def arguments(self, *extra):
        parser = argparse.ArgumentParser()
        htdemucs.configure_parser(parser)
        return parser.parse_args(["--output-root", str(self.config.output_root), *extra])

    def test_completed_summary_distinguishes_reuse_from_new_outputs_without_mutation(self):
        summary = htdemucs.HTDemucsSeparationSummary(
            discovered=25000, selected=24982, skipped_existing=24980, succeeded=2,
            rescue_attempts=1, failed_batches=1,
        )
        original = summary.to_dict()
        text = htdemucs.format_summary(summary, self.config)
        for line in (
            "Status: Completed", "Discovered tracks: 25,000", "Selected tracks: 24,982",
            "Reused existing tracks: 24,980", "Newly separated tracks: 2",
            "Failed tracks: 0", "Unprocessed tracks: 0", "Individual retry attempts: 1",
            "Failed batches: 1",
        ):
            self.assertIn(line, text)
        self.assertEqual(summary.to_dict(), original)

    def test_failure_summary_includes_unprocessed_tracks_and_bounds_failure_list(self):
        summary = htdemucs.HTDemucsSeparationSummary(
            discovered=20, selected=20, skipped_existing=2, succeeded=3,
            failures=[htdemucs.HTDemucsSeparationFailure(f"track_{index}.mp3", 1, "Decode failed")
                      for index in range(12)],
        )
        text = htdemucs.format_summary(summary, self.config, self.root / "report.json")
        self.assertIn("Status: Finished with failures or unprocessed tracks", text)
        self.assertIn("Failed tracks: 12", text)
        self.assertIn("Unprocessed tracks: 3", text)
        self.assertIn("track_0.mp3: Decode failed", text)
        self.assertNotIn("track_10.mp3", text)
        self.assertIn("2 additional failures", text)

    def test_all_reused_tracks_are_completed_not_newly_separated(self):
        summary = htdemucs.HTDemucsSeparationSummary(
            discovered=2, selected=2, skipped_existing=2,
        )
        text = htdemucs.format_summary(summary, self.config)
        self.assertIn("Status: Completed", text)
        self.assertIn("Reused existing tracks: 2", text)
        self.assertIn("Newly separated tracks: 0", text)
        self.assertIn("Unprocessed tracks: 0", text)

    def test_dry_run_is_readable_and_does_not_write_even_with_explicit_report(self):
        summary = htdemucs.HTDemucsSeparationSummary(
            discovered=3, selected=3, skipped_existing=1, planned=2,
        )
        report = self.root / "dry.json"
        stdout = io.StringIO()
        with patch.object(htdemucs, "separate_dataset", return_value=summary), \
                patch.object(htdemucs, "write_report") as write_report, \
                contextlib.redirect_stdout(stdout):
            code = htdemucs.run_from_args(self.arguments("--dry-run", "--report", str(report)))
        self.assertEqual(code, 0)
        self.assertIn("Dry run (no separation performed)", stdout.getvalue())
        self.assertIn("Pending separation: 2", stdout.getvalue())
        self.assertIn("Detailed report: not written (dry run)", stdout.getvalue())
        self.assertNotIn("Unprocessed tracks:", stdout.getvalue())
        write_report.assert_not_called()
        self.assertFalse(report.exists())
        self.assertFalse(self.config.output_root.exists())

    def test_cli_keeps_json_schema_saved_path_and_exit_code_for_success_and_failure(self):
        for failed in (False, True):
            for custom_report in (False, True):
                with self.subTest(failed=failed, custom_report=custom_report):
                    report = (self.root / "custom" / "report.json" if custom_report else
                              self.config.output_root / "separation_report_htdemucs.json")
                    extra = ["--report", str(report)] if custom_report else []
                    summary = htdemucs.HTDemucsSeparationSummary(
                        discovered=1, selected=1, succeeded=0 if failed else 1,
                        failures=[htdemucs.HTDemucsSeparationFailure("track.mp3", 1, "Decode failed")]
                        if failed else [],
                    )
                    stdout = io.StringIO()
                    with patch.object(htdemucs, "separate_dataset", return_value=summary), \
                            contextlib.redirect_stdout(stdout):
                        code = htdemucs.run_from_args(self.arguments(*extra))
                    self.assertEqual(code, int(failed))
                    self.assertEqual(json.loads(report.read_text()), summary.to_dict())
                    self.assertEqual(summary.resolved_config["model_name"], "htdemucs_6s")
                    self.assertIn(f"Detailed JSON report: {report}", stdout.getvalue())
                    self.assertFalse(stdout.getvalue().lstrip().startswith("{"))

    def test_manifest_route_preserves_paths_and_shows_manifest_in_summary(self):
        manifest = self.root / "tracks.jsonl"
        summary = htdemucs.HTDemucsSeparationSummary(
            discovered=1, selected=1, planned=1, manifest_path=str(manifest),
        )
        stdout = io.StringIO()
        with patch.object(htdemucs, "separate_manifest", return_value=summary) as separate, \
                contextlib.redirect_stdout(stdout):
            code = htdemucs.run_from_args(self.arguments("--manifest", str(manifest), "--dry-run"))
        self.assertEqual(code, 0)
        self.assertEqual(separate.call_args.args[1], manifest)
        self.assertTrue(separate.call_args.args[0].dry_run)
        self.assertIn(f"Input manifest: {manifest}", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
