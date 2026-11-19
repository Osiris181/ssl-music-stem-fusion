# CNN audio decoding and recovery

All public `extraction_cnn.py` scripts share `experiments/_cnn_extraction.py`
and `features/_cnn_audio.py`. The guard applies equally to GTZAN, ISMIR04,
FMA-Medium, MTG-Jamendo, PMEmo, DEAM and EmoMusic. It contains no track IDs or dataset-specific audio
exceptions. It does not change the MusicFM/MERT extractors or training code.

## New audio

The primary path remains `librosa.load(..., sr=16000, mono=True)`. Its waveform
is used unchanged when it is nonempty, finite, and agrees with its audio-header
duration within the configured tolerance. Stem duration must also agree with
the decoded original mixture. Valid silence and genuinely short recordings are
retained; there is no minimum song length or amplitude threshold.

If loading or these checks fail, the shared loader tries system FFprobe and
FFmpeg. FFmpeg decodes float32 at the native stream sample rate and channel
count; librosa performs the channel mean and resampling as before. Recovery
requires a clean decoder exit without error-level diagnostics, agreement with
FFprobe's stream duration, and the same mixture-alignment tolerance. The
configured maximum remains 0.25 seconds. Several seconds are never silently
trimmed or padded to force a match.

Fallback failures remain in `extraction_report.json` and return nonzero. A
damaged or genuinely mismatched stem may require replacement or source
separation again; changing the decoder cannot guarantee recovery. The extractor
does not launch Demucs or overwrite source audio automatically.

Install the system `ffmpeg` package if `ffmpeg` and `ffprobe` are unavailable.
These executables are separate from the uv Python environment.

## Resume and existing features

Run the usual `python extraction_cnn.py` command with the original runtime
settings. Valid completed feature caches remain reusable and are not rewritten.
Existing settings/source/model identity checks remain strict.

For a track with any missing view, the original mixture is decoded and
validated before cached views are reused. Very short cached mixture references
(no longer than the alignment tolerance) are also revalidated, even if all
views exist. This is a check, not an exclusion rule. If an existing view's
stored aligned length differs from the validated reference, its embedding is
recomputed. This also corrects a stale mixture embedding that was published
before the missing stems failed.

Before this automatic wrong-length repair, the existing feature is copied to
`<track_id>.npz.before_audio_repair` in the same directory. This suffix is not
included in the training loader's `*.npz` file set. The new metadata records
the backup path and old checksum. A conflicting pre-existing backup is never
overwritten. Healthy cached views retain their original bytes and timestamps.

Full-length completed caches are not exhaustively re-decoded on every resume.
The guard does not certify that every previously accepted MP3 frame was clean.
New decodes use the validation above; incomplete tracks and very short cached
references receive the additional resume checks.

New cache metadata records the audio-loading policy and helper checksum,
selected decoder, duration checks, and FFmpeg versions when applicable. The
frozen checkpoint, original log-mel/windowing/pooling implementation and its
identity remain unchanged. The report lists `decoder_fallbacks`, `repaired`
and `failed` separately; the final readable line prints recovered/repaired
counts. No feature cohort is silently reduced.

Changing a repaired feature changes its dataset fingerprint. Already-trained
classifiers using that changed cache cannot be represented as having consumed
the repaired features; normal training compatibility checks remain active.
Training is never restarted automatically.

FMA CNN training can proceed when extraction has left missing views: its loader
automatically skips each incomplete track from all seven conditions, without
a policy option or hardcoded IDs. It logs the skipped count and records the
actual cohort in the data audit. This does not repair audio, change the manifest,
or remove existing feature files. Extraction still reports the underlying
failures; MusicFM's cached features and completed results remain unchanged.
