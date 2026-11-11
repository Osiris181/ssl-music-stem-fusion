# Feature configurations

`mert_v1_95m.yaml` supplies the Jamendo MERT model's pinned revision,
24 kHz mono waveform settings, frozen float32 all-layer representation, device,
and cache-only model policy. Set its optional `paths.model_path` for a local
snapshot; otherwise the pinned Hugging Face snapshot is used. The dataset
`mert.yaml` defines segment/padding/alignment rules and output
paths. See the [MERT usage guide](../../src/stemfusion/experiments/mtg_jamendo/README.md#mert-segment-extraction).

`musicfm_msd_layer7.yaml` is the shared preset referenced by the FMA-Medium
and GTZAN/ISMIR04 task YAMLs in `configs/experiments/`. Each dataset entry point
selects its task YAML. This feature preset supplies checkpoint/statistics names,
sample rate, duration handling, encoder layer, pooling, dimension, runtime,
and split defaults.

`audio.chunk_seconds` defaults to null (one forward pass). ISMIR04 overrides
it to 30 s for long recordings without truncating them. The shared runtime
pools all latent frames from balanced chunks. Encoder context resets at chunk
boundaries, so this policy is documented separately from a full-track forward
pass. `--chunk-seconds N` or `--no-chunking` overrides YAML.

Set `paths.musicfm_root` in this preset to your MusicFM checkout. FMA manifest
and output paths are configured in `configs/experiments/fma_medium/extraction.yaml`.
GTZAN paths are in `configs/experiments/gtzan/extraction.yaml`.
No separate local-path YAML is required. Cache keys are track IDs: zero-padded
six-digit strings for FMA and genre-prefixed strings for GTZAN.

```bash
stemfusion extract fma-medium --help
```

Use `--config PATH` for an alternate preset or an experiment referencing
this file. Explicit CLI values override configured ones. Checkpoint/statistics
basenames are resolved under the configured MusicFM checkout's `data/`
directory unless an explicit file path overrides them.

See [configuration usage](../../docs/configuration.md). The RMS preset remains
archived outside the project.
