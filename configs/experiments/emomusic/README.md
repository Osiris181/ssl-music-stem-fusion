# EmoMusic extraction presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Copied Music2Emotion splits/annotations, EmoMusic mixtures and same-ID DEAM stem paths. |
| `mert.yaml` | Shared pinned MERT profile, retained 30-second tails and mixture-duration alignment. |
| `chords_keys.yaml` | Shared independent BTC/music21 features and bounded CQT retry. |

See the [dataset guide](../../../src/stemfusion/experiments/emomusic/README.md)
for commands, outputs and the annotation-provenance caveat to resolve before
training. The DEAM stem root is intentional. Paths resolve relative to the
YAML file, and CLI arguments override them. Local metadata is not bundled
public data. These presets do not define or train binary classifiers.
