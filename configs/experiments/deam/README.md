# DEAM extraction presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Copied Music2Emotion splits/song-level annotations and existing DEAM mixture/stem roots. |
| `mert.yaml` | Shared pinned MERT profile, retained 30-second tails and mixture-duration stem alignment. |
| `chords_keys.yaml` | Shared BTC/music21 descriptors, independent per source, with bounded CQT retries. |

See the [dataset guide](../../../src/stemfusion/experiments/deam/README.md)
for commands and output locations. Paths are relative to the YAML file, not
the working directory or Music2Emotion checkout. Local metadata is ignored
input that other users must supply. No training or binary label settings are
implemented in these extraction presets.
