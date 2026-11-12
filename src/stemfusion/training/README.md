# Shared training workflows

Dataset entry scripts call these modules; none of the training modules reads
FMA audio paths or hardcodes dataset cache filenames.

| File | Responsibility |
| --- | --- |
| `trainer.py` | Seed initialization, batch loading, model fitting, validation checkpoint selection, test evaluation, artifacts, and verified reuse. |
| `_identity.py` | Genre-scoped implementation fingerprints and strictly verified compatibility for the fingerprint-only trainer edit. |
| `mid_fusion.py` | Standard concatenation fits and parameter counting; the old width-matching helper remains only for historical compatibility, not the active matrix. |
| `late_fusion.py` | Train branches independently and assemble equal-logit-sum ensembles from their selected predictions. |
| `metrics.py` | Task loss, validation score, and metric definitions. Genre is implemented first; other task adapters remain future work. |
| `reporting.py` | Atomic artifacts, checksums, mean/sample SD, and paired differences. |
| `music2emo.py` | Shared multi-view weighted-BCE fitting, retained stopping/checkpoints and exact optimizer resume; Jamendo metrics by default, optional binary task adapter. |
| `emotion.py` | Binary low/high macro metrics and validation-only prediction-threshold selection for separate emotion targets. |
| `emotion_late_fusion.py` | One-output binary adaptation of the Jamendo logit-sum ensemble contract, using the same binary metrics and verified branch/checkpoint artifacts. |
| `_music2emo_identity.py` | Strict reviewed compatibility for optional binary hooks; preserves completed Jamendo identities while rejecting actual fitting changes. |
| `music2emo_late_fusion.py` | Verify selected branch predictions, sum logits then sigmoid, choose validation-only ensemble thresholds, and save multilabel ensemble results without additional fitting. |

FMA settings live in `configs/training/fma_medium_genre.yaml`. Both fusion
profiles inherit it. The default outer seeds are 32, 64, 128, 256, 512.
Every condition uses the same tracks, Adam at 1e-4, batch size 128, 20 epochs,
ReLU/dropout 0.2, and validation accuracy + macro ROC-AUC + macro average
precision checkpoint selection. Ties retain the earliest epoch.

The active repeated-mixture controls are constructed in
`experiments/_mid_controls.py` and fitted by the same `fit_classifier` backend.
They use exact copies of one mixture vector, matching the fusion input width,
1024/512 hidden architecture, and trainable parameter count. Their input
construction checksum is part of the checkpoint signature. The unchanged
fit backend keeps verified standard/fusion checkpoints reusable; old widened
controls cannot satisfy a repeated-control signature. See
[the control protocol](../../../docs/repeated_mixture_controls.md).

Run [the FMA entry scripts](../experiments/fma_medium/README.md). Dataset
loading belongs in `data/`; dataset conditions/scheduling belong in
`experiments/`. Additional task adapters can supply different losses,
selection scores, and metrics to the common trainer. The separate Jamendo
mood/theme implementation retains the Music2Emo protocol; PMEmo binary mid-
and late-fusion branches use the same fitter with a task adapter. Late fusion
reuses completed mid mixture-only models; each target has separate labels,
training prevalence, selected checkpoints and validation thresholds.
DEAM mid fusion and late branches use this same fitter/metric adapter and settings. Its dataset
adapter preserves the original 1-9 scores and labels high >= 5, low < 5; it
does not normalize scores or modify the running PMEmo fitting modules.
DEAM late ensembles also use the exact `emotion_late_fusion.py` evaluator,
including aligned logit summation, validation-only thresholds and artifact
verification. No task-specific training math is duplicated in the adapter.
EmoMusic mid fusion and late branches also use this exact fitter, model and binary metric adapter,
with the same native >= 5 midpoint, seven conditions and five seeds as DEAM.
Its supplied annotation columns are not exchanged; their provenance caveat
is documented in the dataset guide. Adding its adapter does not modify
fingerprinted DEAM/PMEmo fitting modules or invalidate their checkpoints.
Its late ensembles also use the same `emotion_late_fusion.py` evaluator and
position-paired schedule, preserving EmoMusic mid identities and checkpoints.

The isolated Music2Emo mixture-only reproduction is implemented in
`experiments/mtg_jamendo/reproduction.py`, with its model and
input adapter in `models/music2emo.py` and `data/jamendo_reproduction.py`.
It follows the author's weighted multilabel loss and paper training budget
without changing this genre backend. See the
[reproduction guide](../../../docs/music2emotion_reproduction.md).

`experiments/mtg_jamendo/train_mid.py` schedules seven conditions using
`data/jamendo_mid_fusion.py`, `models/music2emo_mid_fusion.py`, and this
directory's `music2emo.py`. The single-mixture condition reuses the original
reproduction trainer without changing its implementation identity. See the
[Jamendo training guide](../experiments/mtg_jamendo/README.md#mid-fusion-training).

`experiments/mtg_jamendo/train_late.py` uses the same retained fitter for each
independent single-source branch. It keeps the FMA SHA256 branch-seed convention,
reuses the original M1 reference when compatible, and reports five standalone
stems from the very same branch predictions. The new assembly backend does not
modify the existing mid-fusion fitter or its implementation identity.
Both public launchers use `experiments/_mood_theme_tagging.py` as their common
controller and `configuration.py` for YAML handling, matching the genre flow.
This directory remains responsible for model fitting and reusable ensemble
evaluation; the dataset controller only schedules those operations.

FMA experiment commands automatically reuse only complete runs with matching data, settings,
implementation/software metadata, and artifact checksums. It does not
continue an interrupted optimizer state. `--resume` remains an optional alias
for the default behavior. An incomplete run requires a new
output directory or explicit `--overwrite`. The original trainer is preserved
in `/home/osiris181/stem-fusion-training-archive-20261029/`.

Genre implementation identities hash their explicit historical dependency set,
not every module in this directory. Adding an unrelated task module therefore
does not invalidate completed genre fits. The identity-only trainer correction
preserves existing genre and Jamendo identities only when the complete reviewed
trainer and compatibility-policy ASTs match. Actual fitting/model/metric changes,
settings or input changes, and artifact corruption still prevent reuse; saved
run metadata is not rewritten to force compatibility.
