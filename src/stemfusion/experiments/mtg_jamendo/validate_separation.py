"""Validate separated audio and prepare lossless copies for decoder compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.configuration import default_config_path, load_configuration
from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS
from stemfusion.separation.manifest import read_manifest
from stemfusion.training.reporting import file_checksum, write_json


def validate_audio(manifest: Path, output_root: Path, report_path: Path,
                   validated_manifest: Path, *, write_wav_fallbacks: bool = False,
                   overwrite: bool = False) -> dict:
    """Fully decode each native stem; never replace MP3s or original recordings.

    WAV fallbacks preserve the float32 samples returned by libsndfile. They
    do not claim to reconstruct the pre-MP3 model waveform. Publication of a
    downstream manifest requires every source to pass validation.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    torch.set_num_threads(1)
    read_manifest(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    root = output_root.resolve()
    jobs = []
    for row in rows:
        for source in HTDEMUCS_6S_STEMS:
            path = Path(row["paths"][source]).resolve()
            if not path.is_relative_to(root) or path.suffix != ".mp3":
                raise ValueError(f"Expected native MP3 output under the configured repair root: {path}")
            jobs.append((row, source, path))

    def check(job):
        row, source, path = job
        result = {"track_id": row["track_id"], "source": source, "mp3_path": str(path), "status": "failed"}
        try:
            try:
                waveform, sample_rate = torchaudio.load(str(path))
                samples = waveform.numpy().T
                decoder = "torchaudio"
            except Exception as error:
                result["torchaudio_error"] = str(error)
                samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
                decoder = "soundfile"
            result["decoder"] = decoder
            if not samples.size or samples.ndim != 2 or not np.isfinite(samples).all():
                raise ValueError("Decoded waveform is empty or non-finite.")
            seconds = len(samples) / sample_rate
            if abs(seconds - row["duration_seconds"]) > 0.25:
                raise ValueError(f"Decoded duration {seconds:.6f}s differs from metadata {row['duration_seconds']}s.")
            result.update(sample_rate=int(sample_rate), frames=len(samples), channels=samples.shape[1],
                          duration_seconds=seconds, finite=True, peak=float(np.abs(samples).max()))
            usable = path
            if decoder == "soundfile":
                if not write_wav_fallbacks:
                    raise ValueError("Alternative decoder succeeded; use --write-wav-fallbacks for a torchaudio-readable copy.")
                usable = path.with_suffix(".wav")
                temporary = usable.with_name(f".{usable.name}.tmp")
                if usable.exists() and not overwrite:
                    existing, existing_sr = sf.read(str(usable), dtype="float32", always_2d=True)
                    if existing_sr != sample_rate or not np.array_equal(existing, samples):
                        raise ValueError(f"Existing WAV differs; use --overwrite explicitly: {usable}")
                else:
                    try:
                        sf.write(str(temporary), samples, sample_rate, format="WAV", subtype="FLOAT")
                        temporary.replace(usable)
                    finally:
                        temporary.unlink(missing_ok=True)
                roundtrip, actual_sr = torchaudio.load(str(usable))
                if actual_sr != sample_rate or not np.array_equal(roundtrip.numpy().T, samples):
                    raise ValueError("WAV does not exactly preserve the alternative decoder's float32 samples.")
                result["wav_sample_equality_verified"] = True
            result.update(status="ok", usable_path=str(usable), usable_sha256=file_checksum(usable))
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check, job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(jobs)} | {result['track_id']} | {result['source']} | {result['status']} | {result.get('decoder', 'none')}", flush=True)
    failed = sum(result["status"] != "ok" for result in results)
    report = {"manifest": str(manifest), "manifest_sha256": file_checksum(manifest),
              "tracks": len(rows), "audio_files": len(results), "failed": failed,
              "soundfile_fallbacks": sum(result.get("decoder") == "soundfile" for result in results),
              "write_wav_fallbacks": write_wav_fallbacks,
              "validated_manifest": str(validated_manifest) if not failed else None,
              "runtime_versions": {name: version(name) for name in ("torch", "torchaudio", "soundfile")},
              "wav_policy": "Float32 PCM copies of libsndfile-decoded MP3 samples; verified exact torchaudio round trip. MP3 files preserved.",
              "results": sorted(results, key=lambda result: (result["track_id"], result["source"]))}
    if not failed:
        lookup = {(result["track_id"], result["source"]): result["usable_path"] for result in results}
        for row in rows:
            row["paths"].update({source: lookup[row["track_id"], source] for source in HTDEMUCS_6S_STEMS})
        validated_manifest.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if validated_manifest.exists() and not overwrite and validated_manifest.read_text() != text:
            raise ValueError(f"Different validated manifest exists; use --overwrite: {validated_manifest}")
        validated_manifest.write_text(text)
    write_json(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path("experiments/mtg_jamendo/separation.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validated-manifest", type=Path)
    parser.add_argument("--write-wav-fallbacks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        defaults, provenance = load_configuration(args.config, "separation")
        if provenance["experiment_metadata"].get("dataset") != "mtg_jamendo":
            raise ValueError("Use an MTG-Jamendo separation configuration.")
        manifest = (args.manifest or defaults["manifest"]).expanduser().resolve()
        report = validate_audio(manifest, (args.output_root or defaults["output_root"]).expanduser().resolve(),
            (args.report or defaults["report"].with_name("separation_audio_validation.json")).expanduser().resolve(),
            (args.validated_manifest or manifest.with_name("mtg_jamendo_repaired_audio.jsonl")).expanduser().resolve(),
            write_wav_fallbacks=args.write_wav_fallbacks, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        print(f"Audio validation error: {error}")
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
