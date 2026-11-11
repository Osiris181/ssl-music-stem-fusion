# PMEmo experiment presets

| Preset | Purpose |
| --- | --- |
| `feature_manifest.yaml` | Supplied split IDs, original 0-1 annotations, metadata and mixture/stem audio roots. |
| `separation_manifest.yaml` | Label-independent separation jobs from the same split IDs; all tracks by default, or an explicit track list. |
| `separation.yaml` | Shared HT-Demucs inference profile, separation job manifest and exact output/report locations. |
| `mert.yaml` | Shared pinned MERT profile, 30-second retained-tail segments and aligned source views. |
| `chords_keys.yaml` | Shared BTC/music21 descriptors with default-on CQT-tail retries. |
| `mid_fusion.yaml` | Separate binary valence/arousal, seven matched conditions and five seeds through the shared emotion/Music2Emo profiles. |
| `late_fusion.yaml` | Same binary task/training profile, independent mixture/stem branches, twelve reports including stem-only, and a configured mid-fusion reference root. |

See the [dataset guide](../../../src/stemfusion/experiments/pmemo/README.md)
for commands, required inputs, output locations and cache behavior. Training
uses the fixed original-score midpoint: high >= 0.5, low < 0.5. Preparation and
extraction do not create labels; prediction thresholds use validation only.
Edit audio/model paths for your installation; relative paths resolve against
the YAML file. Copied metadata is local ignored input, not bundled public data.

Like Jamendo, one `prepare_manifest.py` selects the feature preset by default
or the separation preset with `--purpose separation`. `separation.py` calls the
same shared backend and separator profile as Jamendo and the genre datasets.
Set its `output_root` to the parent of `htdemucs_6s`; this must agree with the
existing `feature_manifest.yaml` stem root. Existing feature presets and outputs
are unchanged; no re-separation or re-extraction is required to adopt this layout.
