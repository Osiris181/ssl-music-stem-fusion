# Dataset experiment configurations

[PMEmo](pmemo/README.md) defines fixed-split manifest preparation and MERT plus
BTC/music21 extraction with the same scientific settings as Jamendo.

[DEAM](deam/README.md) uses the same extraction rules with its own copied
Music2Emotion splits, supplied song-level scores and original/stem paths.

[EmoMusic](emomusic/README.md) uses the same rules and shared DEAM scheduler,
with EmoMusic mixtures, its copied splits and matching existing DEAM stems.

[FMA-Medium](fma_medium/README.md), [GTZAN](gtzan/README.md), and
[ISMIR04](ismir04/README.md) task files reference shared feature and
training presets. Dataset-specific profiles supply paths and class counts.

[MTG-Jamendo](mtg_jamendo/README.md) configures selected-track preparation
and the shared HT-Demucs separator for the MERT + chord/key pipeline.
Its shared `feature_manifest.yaml` binds audio for MERT and harmonic extraction;
`mert.yaml` configures MERT segments. The dataset-level `chords_keys.yaml`
configures independent harmonic extraction paired with those embeddings.
Its `mid_fusion.yaml` inherits the retained shared Music2Emo training recipe
and defines the seven mixture/reference/control/fusion conditions. Completed
mixture reproduction checkpoints can be reused without changing their protocol.
`late_fusion.yaml` inherits the same recipe and specifies matched independent
ensembles plus five stem-only report rows. It has a separate output root;
existing mid-fusion configs and checkpoint identities are unchanged.

Pass an experiment file with `--config`. For training, `--run` selects
one named source condition. Experiment values override their referenced
preset, and explicit CLI arguments override YAML.

The configuration loader only resolves settings. The training entry point
runs the full condition matrix when `--run` is omitted, using all five seeds
unless a seed subset is supplied. Mid fusion includes repeated-mixture
controls with equal fusion input widths and parameter counts; late fusion uses independently selected branches and
matched mixture ensembles. See [configuration usage](../../docs/configuration.md).
