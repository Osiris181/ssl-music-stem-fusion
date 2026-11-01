"""Read-only file and decoded-waveform duplicate checks for fixed partitions."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from stemfusion.training.reporting import file_checksum, write_json


def audit_audio_files(paths: list[Path], output: Path, *, workers: int = 2,
                      sample_rate: int = 24000) -> dict:
    """Hash listed files and mono float32 PCM decoded by ffmpeg.

    No directory discovery or audio modification is performed. Byte-identical
    files and identical decoded model inputs can be detected; re-encodings,
    offsets, near duplicates, and artist overlap are not comprehensively tested.
    Progress is saved so an interrupted audit can reuse unchanged local files.
    """
    if workers < 1 or sample_rate < 1:
        raise ValueError("Audit workers and sample rate must be positive.")
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg is required for decoded-audio duplicate auditing.")
    settings = {"sample_rate": sample_rate, "channels": 1, "sample_format": "pcm_f32le",
                "decoder": "ffmpeg", "duplicate_rule": "identical_decoded_model_input"}
    previous = {}
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("settings") == settings:
            previous = saved.get("files", {})
    report = {"settings": settings, "files": {}, "errors": {}, "complete": False,
              "audio_modified": False, "scope": "Only audio paths explicitly provided by dataset metadata."}

    def inspect(path):
        stat = path.stat()
        cached = previous.get(str(path), {})
        if cached.get("bytes") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            return cached
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_f32le", "-f", "hash", "-hash", "sha256", "pipe:1"],
            capture_output=True, text=True, check=True)
        value = result.stdout.strip()
        if not value.startswith("SHA256=") or len(value.removeprefix("SHA256=")) != 64:
            raise ValueError(f"No decoded audio hash returned for {path}")
        if value.removeprefix("SHA256=") == hashlib.sha256(b"").hexdigest():
            raise ValueError(f"Decoded audio is empty: {path}")
        return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": file_checksum(path), "decoded_sha256": value.removeprefix("SHA256="),
                "decoder_messages": result.stderr.strip()}

    unique = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inspect, path): path for path in unique}
        for index, completed in enumerate(tqdm(as_completed(pending), total=len(pending),
                                              desc="Audio duplicate audit", unit="track"), 1):
            path = pending[completed]
            try:
                report["files"][str(path)] = completed.result()
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                report["errors"][str(path)] = str(error)
            if index % 25 == 0:
                write_json(output, report)
    report["complete"] = not report["errors"] and len(report["files"]) == len(unique)
    write_json(output, report)
    if not report["complete"]:
        raise ValueError(f"Audio audit incomplete; inspect {output} for errors.")
    return report
