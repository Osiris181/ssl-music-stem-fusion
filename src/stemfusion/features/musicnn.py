"""Frozen official MSD_musicnn inference, isolated from Phase 2 extraction.

The original log-mel calculation and model remain unchanged. Adaptations are
mixture-referenced length alignment and retaining/padding the final window.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")
SAMPLE_RATE = 16000
WINDOW_FRAMES = 187
VERSION = "msd_musicnn_aligned_tail_v1"
SOURCE_COMMIT = "516acb2a0ff5ef73f64547898e018e793152c506"
SOURCE_HASHES = {
    "models.py": "ca0a4e64b33fa79ef716d56188202cfdb31c123fa5721addb1e295405b671cf8",
    "extractor.py": "a634059ade166c3b7379c4243bc0728c060d564ca498676af755a1e37523288e",
    "configuration.py": "00fd595eb16bad3a605e10bb6eaf0152cb8125bed6e2c884a636f679ae100419",
}
CHECKPOINT_HASHES = {
    ".index": "87f3e0f3550c8d3bafc840374003e5c8af5eeb5ecf6add61af09a7822d3282c8",
    ".data-00000-of-00001": "8579734ee6388f799ed0def4bbd0d1fdcbeaf213ef2a9dc860917bd76a303de6",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def align_audio(audio, reference_samples: int, tolerance_seconds: float):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty, finite and mono")
    if abs(len(audio) - reference_samples) > round(tolerance_seconds * SAMPLE_RATE):
        raise ValueError("Source duration differs from mixture beyond configured tolerance")
    return np.pad(audio[:reference_samples], (0, max(0, reference_samples - len(audio))))


def log_mel(audio):
    """Match official batch_data preprocessing, including float16 conversion."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, hop_length=256, n_fft=512, n_mels=96,
    ).T.astype(np.float16)
    result = np.log10(10000 * mel + 1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite official log-mel representation")
    return result


def iter_windows(spectrogram, batch_size: int):
    """Cover every official frame once; only the final patch is right-padded."""
    if batch_size < 1 or spectrogram.ndim != 2 or spectrogram.shape[1] != 96 or not len(spectrogram):
        raise ValueError("Expected nonempty [frames, 96] spectrogram and positive batch size")
    for first in range(0, len(spectrogram), WINDOW_FRAMES * batch_size):
        patches = []
        for start in range(first, min(first + WINDOW_FRAMES * batch_size, len(spectrogram)), WINDOW_FRAMES):
            patch = spectrogram[start:start + WINDOW_FRAMES]
            patches.append(np.pad(patch, ((0, WINDOW_FRAMES - len(patch)), (0, 0))))
        yield np.stack(patches).astype(np.float32)


class OfficialMusicNN:
    """Strict original-checkpoint restore; no optimizer or training updates."""

    def __init__(self, device="/GPU:0", gpu_memory_mb=768):
        if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
            raise ValueError("Use musicnn/bin/python-musicnn (legacy Keras launcher)")
        import tensorflow as tf
        from musicnn import models
        if importlib.metadata.version("numpy") != "1.26.4":
            raise ValueError("Official float16 preprocessing requires the verified NumPy 1.26.4 runtime")
        package = Path(models.__file__).parent
        for name, expected in SOURCE_HASHES.items():
            if digest(package / name) != expected:
                raise ValueError(f"Official source checksum mismatch: {name}")
        checkpoint = package / "MSD_musicnn"
        for name, expected in CHECKPOINT_HASHES.items():
            if digest(checkpoint / name) != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        if device not in ("/GPU:0", "/CPU:0"):
            raise ValueError("device must be /GPU:0 or /CPU:0")
        if device == "/GPU:0":
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("GPU requested but unavailable; no silent CPU fallback")
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_mb)],
            )
        tf.config.experimental.enable_tensor_float_32_execution(False)
        graph = tf.Graph()
        prefix = str(checkpoint) + "/"
        with graph.as_default():
            with tf.device(device), tf.name_scope("model"):
                self.inputs = tf.compat.v1.placeholder(tf.float32, [None, WINDOW_FRAMES, 96])
                self.training = tf.compat.v1.placeholder(tf.bool)
                self.features = models.define_model(self.inputs, self.training, "MSD_musicnn", 50)[-1]
            shapes = tf.compat.v1.train.NewCheckpointReader(prefix).get_variable_to_shape_map()
            variables = tf.compat.v1.global_variables()
            for variable in variables:
                if shapes.get(variable.name.removesuffix(":0")) != variable.shape.as_list():
                    raise ValueError(f"Checkpoint variable mismatch: {variable.name}")
            saver = tf.compat.v1.train.Saver(var_list=variables)
        self.session = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(
            allow_soft_placement=False, intra_op_parallelism_threads=2, inter_op_parallelism_threads=1,
        ))
        try:
            with graph.as_default():
                saver.restore(self.session, prefix)
        except Exception:
            self.session.close()
            raise
        self.provenance = {
            "checkpoint": "MSD_musicnn", "source_commit": SOURCE_COMMIT,
            "checkpoint_sha256": CHECKPOINT_HASHES, "source_sha256": SOURCE_HASHES,
            "restored_variables": len(variables), "device": device,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("tensorflow", "tf-keras", "numpy", "librosa", "musicnn")},
        }

    def predict(self, batch):
        result = self.session.run(self.features, {self.inputs: batch, self.training: False})
        if result.shape != (len(batch), 200) or not np.isfinite(result).all():
            raise ValueError("Expected finite [windows, 200] penultimate embeddings")
        return result

    def close(self):
        self.session.close()


def pool_track(model, spectrogram, batch_size: int):
    total = np.zeros(200, dtype=np.float64)
    count = 0
    for batch in iter_windows(spectrogram, batch_size):
        vectors = model.predict(batch)
        total += vectors.sum(axis=0, dtype=np.float64)
        count += len(vectors)
    return (total / count).astype(np.float32), count


def save_cache(path: Path, vector, metadata):
    """Commit vector and provenance together, using atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, embedding=vector, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(path)
