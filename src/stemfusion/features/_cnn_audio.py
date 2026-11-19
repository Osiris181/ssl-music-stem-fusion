"""Duration-checked CNN decoding; retain librosa's output for healthy inputs.

FFmpeg is a recovery decoder, not an alignment repair. A recovered waveform
must agree with its stream duration and the mixture within the same tolerance.
No track identities, silence thresholds, or dataset-specific exceptions apply.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from stemfusion.features.musicnn import SAMPLE_RATE

POLICY = "librosa_duration_guard_ffmpeg_fallback_v1"


def _validate(audio, reference_samples, tolerance_seconds):
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Decoded audio must be nonempty, finite and mono")
    if reference_samples is not None and abs(len(audio) - reference_samples) > round(tolerance_seconds * SAMPLE_RATE):
        raise ValueError(
            f"Source duration differs from mixture beyond configured tolerance: "
            f"decoded={len(audio) / SAMPLE_RATE:.6f}s, mixture={reference_samples / SAMPLE_RATE:.6f}s, "
            f"tolerance={tolerance_seconds:.6f}s"
        )


@lru_cache(maxsize=2)
def _version(executable):
    result = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.splitlines()[0]


def _decode_ffmpeg(path, tolerance_seconds, reference_samples):
    import librosa

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("Audio recovery requires the system ffmpeg and ffprobe executables")
    probe = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=60, check=True)
    if probe.stderr.strip():
        raise ValueError(f"FFprobe reported damaged input: {probe.stderr.strip()[:1000]}")
    details = json.loads(probe.stdout)
    stream = details["streams"][0]
    native_rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    duration = float(stream.get("duration", details.get("format", {}).get("duration", "nan")))
    if native_rate <= 0 or channels <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("FFprobe must provide positive stream rate, channels and duration")
    # Decode float32 at the native rate/channels. Keep librosa's channel mean
    # and resampler instead of introducing FFmpeg downmix/resampling defaults.
    decoded = subprocess.run([
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ], capture_output=True, timeout=180, check=True)
    errors = decoded.stderr.decode("utf-8", errors="replace").strip()
    if errors:
        raise ValueError(f"FFmpeg reported damaged input; replace or regenerate the audio: {errors[:1000]}")
    if not decoded.stdout or len(decoded.stdout) % (4 * channels):
        raise ValueError("FFmpeg returned empty or incomplete float32 audio frames")
    native = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, channels).T
    native_duration = native.shape[1] / native_rate
    if abs(native_duration - duration) > tolerance_seconds:
        raise ValueError(
            f"FFmpeg decoded duration disagrees with its stream: decoded={native_duration:.6f}s, "
            f"stream={duration:.6f}s, tolerance={tolerance_seconds:.6f}s"
        )
    audio = librosa.resample(librosa.to_mono(native), orig_sr=native_rate, target_sr=SAMPLE_RATE)
    _validate(audio, reference_samples, tolerance_seconds)
    return audio, {
        "decoder": "ffmpeg", "ffmpeg_version": _version(ffmpeg),
        "ffprobe_version": _version(ffprobe), "native_sample_rate": native_rate,
        "native_channels": channels, "stream_duration_seconds": duration,
        "decoded_duration_seconds": len(audio) / SAMPLE_RATE,
    }


def load_audio(path: Path, tolerance_seconds: float, *, reference_samples=None):
    """Use the unchanged primary decoder unless its output fails validation."""
    import librosa
    import soundfile as sf

    if not math.isfinite(tolerance_seconds) or not 0 <= tolerance_seconds <= 0.25:
        raise ValueError("CNN decoding requires a duration tolerance between zero and 0.25 seconds")
    try:
        header = sf.info(path)
        if not math.isfinite(header.duration) or header.duration <= 0:
            raise ValueError("Audio header has no positive finite duration")
        audio, rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if rate != SAMPLE_RATE:
            raise ValueError("Primary decoder returned an unexpected sample rate")
        _validate(audio, reference_samples, tolerance_seconds)
        if abs(len(audio) / SAMPLE_RATE - header.duration) > tolerance_seconds:
            raise ValueError(
                f"Primary decoded duration disagrees with its header: decoded={len(audio) / SAMPLE_RATE:.6f}s, "
                f"header={header.duration:.6f}s, tolerance={tolerance_seconds:.6f}s"
            )
        return audio, {
            "policy": POLICY, "decoder": "librosa", "header_duration_seconds": header.duration,
            "decoded_duration_seconds": len(audio) / SAMPLE_RATE,
        }
    except Exception as primary_error:
        try:
            audio, details = _decode_ffmpeg(path, tolerance_seconds, reference_samples)
        except Exception as fallback_error:
            raise ValueError(
                f"Audio decoding could not be validated. Primary: {primary_error}. "
                f"Fallback: {fallback_error}"
            ) from fallback_error
        return audio, {"policy": POLICY, "primary_error": str(primary_error), **details}
