# PMEmo extraction presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Supplied split IDs, original 0-1 annotations, metadata and mixture/stem audio roots. |
| `mert.yaml` | Shared pinned MERT profile, 30-second retained-tail segments and aligned source views. |
| `chords_keys.yaml` | Shared BTC/music21 descriptors with default-on CQT-tail retries. |

See the [dataset guide](../../../src/stemfusion/experiments/pmemo/README.md)
for commands, required inputs, output locations and cache behavior. Training
presets are not provided yet. No binary emotion thresholds are applied here.
Edit audio/model paths for your installation; relative paths resolve against
the YAML file. Copied metadata is local ignored input, not bundled public data.
