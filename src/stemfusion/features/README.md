# Shared feature implementations

`emotion_dataset.py` provides explicit dataset adapters for DEAM and EmoMusic,
with the same frozen MERT/BTC runtimes and
per-view writers. Settings and scheduling are checked against the existing
PMEmo adapter without modifying its currently running extraction. See the
[DEAM guide](../experiments/deam/README.md).
The [EmoMusic guide](../experiments/emomusic/README.md) explains its supplied
splits, shared DEAM stem roots and identical extraction implementation.

`emotion_extraction.py` routes PMEmo manifests through the exact existing
MERT/BTC runtimes and atomic per-view writers, with dataset-specific reports
and added provenance. Scheduling parity is tested against Jamendo while its
fingerprinted modules remain unchanged. See the
[PMEmo extraction guide](../experiments/pmemo/README.md).

`mert_segments.py` implements frozen MERT-v1-95M extraction for Jamendo:
30-second retained-tail segments, mixture-duration stem matching, and padding
only below the architecture's 400-sample minimum. Each array preserves all
12 transformer layers as `(1, 12, 768)`, with mean pooling over time. This is
separate from MusicFM and from chord/key descriptors. Use the public
[MERT command](../experiments/mtg_jamendo/extract_mert.py), not this
implementation module directly; see its directory README for settings/usage.

`chords_keys.py` implements BTC whole-track chords, generated pitch-class MIDI,
music21 key, and key-normalized chords, using verified Music2Emotion utilities
without importing its UI. This harmonic stage processes each source's own
audio independently and saves symbolic descriptors separately from MERT arrays.
The dataset-level command serves the MERT pipeline; see the
[Jamendo chord/key guide](../experiments/mtg_jamendo/README.md#chordkey-extraction).
Normal extraction retains the released frontend; an optional, default-on
second pass uses `_chord_cqt.py` to right-pad unsupported one-sample CQT tails
to two samples. Only recognized failures are retried once. Valid blocks,
saved `None` keys, and verified successful legacy outputs remain unchanged.

This package contains reusable frozen feature extraction. `_musicfm.py`
handles MusicFM loading, layer extraction, temporal mean pooling, and
cache/report writing. It contains no CLI parser. Public dataset tasks in
`experiments/` select their YAML configurations and use the shared CLI in
`experiments/_extraction.py`.

`extract_musicfm_sources()` processes a job's sources sequentially with a
shared encoder. Dry runs and fully cached sources do not initialize the model.

The FMA-Medium task is
[experiments/fma_medium/extraction.py](../experiments/fma_medium/extraction.py):

```bash
stemfusion extract fma-medium --help
```

It reads an aligned JSONL audio manifest and writes source/split caches keyed
by six-digit FMA track IDs. Complete caches are written atomically; failures
produce partial caches and reports.

[GTZAN extraction](../experiments/gtzan/extraction.py) uses the exact same
runtime and pooling functions, with GTZAN track IDs and filenames
`gtzan_musicfm_{source}_{split}.pkl` (`mix` for the mixture). The shared runtime
saves CPU float32 tensors; the GTZAN loader also accepts original NumPy caches.

[ISMIR04 extraction](../experiments/ismir04/extraction.py) uses this same
implementation with its audited six-class manifest and unique IDs. Its YAML
selects a 30 s encoder chunk cap for long tracks. Balanced chunks cover all
samples; pooling weights chunks by their latent-frame counts. Chunking resets
encoder context and is recorded in reports. The shared default leaves
chunking disabled for the existing FMA/GTZAN workflows.

Defaults remain MusicFM-MSD, layer 7, mean pooling, 1024 dimensions, mono
24 kHz audio, and full duration, loaded from the feature YAML preset.
Explicit CLI settings override YAML. The revised FMA matrix uses mixture, guitar,
drums, piano, bass, and vocals. The shared implementation still supports
`other`, although it is excluded from those comparisons.

RMS extraction is archived. MusicFM code/checkpoints remain external
dependencies. See [the FMA extraction guide](../../../docs/fma_medium_extraction.md)
for inputs, commands, settings, and output names.

`musicfm_segments.py` retains shared manifest reading, temporal ranges, and
stem-alignment helpers imported by MERT and BTC chord/key workflows.
Its historical module name and implementation are kept unchanged because its
hash is part of existing MERT cache provenance. The Jamendo-specific MusicFM
experiment entry point and preset are no longer active. MusicFM extraction
for FMA-Medium, GTZAN, and ISMIR04 remains available through `_musicfm.py`.
`data/segments.py` provides numeric ordering and downstream consecutive-range
training pooling/all-segment evaluation pooling. This differs deliberately
from ISMIR04's balanced chunks and permanent track-level pooled vector.
