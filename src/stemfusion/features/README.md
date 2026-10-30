# Shared feature implementations

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

Defaults remain MusicFM-MSD, layer 7, mean pooling, 1024 dimensions, mono
24 kHz audio, and full duration, loaded from the feature YAML preset.
Explicit CLI settings override YAML. The revised FMA matrix uses mixture, guitar,
drums, piano, bass, and vocals. The shared implementation still supports
`other`, although it is excluded from those comparisons.

RMS extraction is archived. MusicFM code/checkpoints remain external
dependencies. See [the FMA extraction guide](../../../docs/fma_medium_extraction.md)
for inputs, commands, settings, and output names.
