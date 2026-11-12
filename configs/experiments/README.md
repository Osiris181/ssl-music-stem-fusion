# Dataset experiment configurations

[PMEmo](pmemo/README.md) defines fixed-split feature/separation manifest
preparation, shared HT-Demucs separation and MERT plus BTC/music21 extraction
with the same scientific settings as Jamendo. Existing extracted features can
be reused without regenerating their manifest or audio.
Its mid- and late-fusion presets share the same binary emotion training profile;
late fusion reuses the mid mixture reference and reports independent ensembles
and five stem-only rows per target, with outputs in separate directories.

[DEAM](deam/README.md) uses the same extraction rules with its own copied
Music2Emotion splits, supplied song-level scores and original/stem paths.
It also has feature/separation manifest presets selected by `prepare_manifest.py`
and the same shared HT-Demucs separation preset/backend as PMEmo. Existing
feature caches and their reader identity remain unchanged.
Its `mid_fusion.yaml` inherits the same PMEmo binary model/optimization recipe,
with native 1-9 annotations thresholded directly at 5 (no score normalization),
seven conditions, two separate targets and five seeds.
Its `late_fusion.yaml` uses the same recipe with independent matched ensembles
and stem-only rows, reusing configured mid mixture references without changing
their code/checkpoint identities.

[EmoMusic](emomusic/README.md) uses the same rules and shared DEAM scheduler,
with EmoMusic mixtures, its copied splits and matching existing DEAM stems.
It now uses the same `prepare_manifest.py` feature/separation purposes and
shared separation backend, preserving existing stem destinations and feature
reader identities. Binary training is not enabled by these preparation presets.
Its `mid_fusion.yaml` inherits the same binary-emotion training profile and
seven-condition/five-seed recipe as DEAM/PMEmo. Native scores use high >= 5
without normalization or column swapping; the annotation-column provenance
caveat remains unresolved and documented in its guide.
Its `late_fusion.yaml` uses the same DEAM/PMEmo branch/ensemble settings,
including position-paired seeds, configured protected mid references and all
five standalone stems without extra fits.

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
