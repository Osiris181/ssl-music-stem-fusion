"""BTC's released CQT frontend with a minimal, explicit tiny-tail fallback."""

from __future__ import annotations

import numpy as np

# Verified for the pinned 22,050 Hz / 144-bin / 24-bin-per-octave frontend:
# librosa 0.10.2.post1 rejects one sample; two samples produce finite features.
CQT_MINIMUM_INPUT_SAMPLES = 2


def cqt_features_with_tail_padding(audio, parameters):
    """Keep author-compatible blocks; right-pad only unsupported one-sample tails.

    The source file is never modified. Duration and block boundaries describe
    the original decoded waveform, not the minimally padded encoder input.
    """
    import librosa

    rate = parameters.mp3["song_hz"]
    seconds = parameters.mp3["inst_len"]
    feature_parameters = parameters.feature
    if (rate, seconds, feature_parameters["n_bins"], feature_parameters["bins_per_octave"],
            feature_parameters["hop_length"], parameters.model["timestep"]) != (22050, 10.0, 144, 24, 2048, 108):
        raise ValueError("CQT tiny-tail padding is validated only for the pinned BTC frontend.")
    waveform, rate = librosa.load(str(audio), sr=rate, mono=True)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("BTC requires finite, nonempty mono audio.")
    width = int(rate * seconds)
    blocks, padding_records = [], []
    for start in range(0, len(waveform), width):
        segment = waveform[start:start + width]
        samples = len(segment)
        padding = max(0, CQT_MINIMUM_INPUT_SAMPLES - samples)
        if padding:
            segment = np.pad(segment, (0, padding), mode="constant")
            padding_records.append({"start_sample": start, "end_sample": start + samples,
                                    "audio_samples": samples, "cqt_input_samples": len(segment),
                                    "right_zero_padding_samples": padding})
        blocks.append(librosa.cqt(segment, sr=rate, n_bins=feature_parameters["n_bins"],
                      bins_per_octave=feature_parameters["bins_per_octave"],
                      hop_length=feature_parameters["hop_length"]))
    feature = np.log(np.abs(np.concatenate(blocks, axis=1)) + 1e-6)
    metadata = {"policy": "right_zero_pad_to_minimum", "minimum_input_samples": CQT_MINIMUM_INPUT_SAMPLES,
                "sample_rate": rate, "decoded_samples": len(waveform), "padded_blocks": padding_records,
                "padded_samples": sum(row["right_zero_padding_samples"] for row in padding_records)}
    return feature, seconds / parameters.model["timestep"], len(waveform) / rate, metadata
