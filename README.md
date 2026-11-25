# AutoCut

**AutoCut** is an automatic video-editing pipeline for mountain-bike footage. It analyzes riding video to identify the moments that are most worth keeping, using motion analysis and a learned model of riding interest, then turns those moments into a coherent edited video.

The project is designed around a simple idea: **the goal is not to classify every frame or recognize specific MTB tricks, but to estimate how interesting each moment of a ride is and use that estimate to select the best parts of the footage.**

## Project status

AutoCut is currently being refactored from the original MTB-Video-Tools implementation into a cleaner, modular architecture.

The existing implementation has already established the core data pipeline and an initial interest-regression dataset. The current refactor aims to preserve that proven behavior while separating the system into well-defined components.

The architectural specification is maintained in:

`docs/AUTOCUT_SPEC.md`

That document is the authoritative reference for the intended architecture and behavior.

## Architecture

The project uses a `src/` layout:

```text
MTB-Video-Tools/
├── pyproject.toml
├── README.md
├── src/
│   └── autocut/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── config.py
│       ├── errors.py
│       ├── logging_utils.py
│       ├── annotations.py
│       ├── models.py
│       ├── manifests.py
│       ├── video/
│       ├── motion/
│       ├── dataset/
│       ├── ml/
│       ├── selection/
│       ├── audio/
│       └── cache.py
├── tests/
├── docs/
├── models/
├── data/
└── output/
```

### Directory responsibilities

| Directory | Purpose |
|---|---|
| `src/autocut/` | Application source code |
| `tests/` | Automated tests |
| `docs/` | Specifications and design documentation |
| `models/` | Trained ML models and associated artifacts |
| `data/` | Local datasets and generated analysis data |
| `output/` | Generated videos and other runtime output |

`data/` and `output/` are local working directories and are not committed to Git.

`models/` contains the models required by the application and is kept under version control.

## Core concept: riding interest

AutoCut does not attempt to assign independent labels such as `drop`, `rock garden`, `switchback`, or `stairs`.

Instead, the model estimates a continuous **interest score** on a 0–3 scale:

| Score | Meaning |
|---:|---|
| `0` | Irrelevant / non-riding |
| `1` | Ordinary or boring riding |
| `2` | Middling / borderline |
| `3` | Clearly interesting / worth keeping |

Training windows may overlap adjacent annotation intervals. A window is retained only when every annotation interval overlapping it has the same score; a window crossing a class boundary is discarded, even if one score covers most of its duration.

Training uses **ordinal learning-to-rank** with LightGBM LambdaRank. Every video is an independent ranking query, and its windows retain their human relevance grades (0–3). The grouped validation split keeps entire videos together; Optuna HPO evaluates leave-one-group-out folds. The production ranker is refit on the annotated corpus and emits raw ranking scores rather than class probabilities.

Windows with no annotation overlap are omitted. For partially overlapping annotations, each window receives the duration-weighted majority grade; ties are assigned to the higher grade.

The previous PU/BiLSTM implementation is no longer part of this development branch. Its history remains available on the `legacy` branch.

## Processing pipeline

At a high level, AutoCut is organized into these stages:

```text
Video
  │
  ▼
Video analysis
  │
  ├── motion / optical flow
  └── derived motion features
  │
  ▼
Interest estimation
  │
  ▼
Candidate generation
  │
  ▼
Candidate ranking
  │
  ▼
Candidate selection and duplicate suppression
  │
  ▼
Timeline
  │
  ├── audio / music
  └── rendering
  │
  ▼
Final video
```

The individual stages are deliberately separated so that the analysis, ML, selection and rendering components can evolve independently.

## Development

Create the development environment and install the project in editable mode:

```powershell
python -m pip install -e ".[dev,ml,hpo]"
```

The editable installation means changes made under `src/autocut/` are immediately available without reinstalling the package.

The application can then be invoked with:

```powershell
python -m autocut
```

## Testing

Run the test suite with:

```powershell
pytest
```

During the refactor, existing behavior should be preserved wherever it is part of the established pipeline. Tests are therefore an important part of the migration from the legacy implementation.

## Single-model LightGBM training

Train a LightGBM LambdaRank model that orders annotated windows by their
ordinal relevance grades (0–3). Each video is one ranking query: LightGBM gets
a group-size array describing the consecutive rows for each video. The
configured `data.train_groups` and `data.validation_groups` are kept separate
for group validation. Each video is one query. The 74-column representation
contains six statistics for each of eleven optical-flow signals, plus eight DJI
IMU statistics (roll/pitch/yaw angular-rate RMS and peaks, and dynamic
acceleration RMS and peak). Before annotation filtering, each video's features
are normalized across all its windows using the configured method:
`per_video_zscore_all_windows_ddof0` or `per_video_median_iqr_all_windows`.
The same transformation is applied during inference. Median/IQR is the current
first experiment and the default for new training runs; compare it with z-score
using the same nested group-CV procedure. The production ranker is then refit on all annotated manifest videos for the
selected number of iterations, including validation videos. It uses the
canonical feature cache and writes a LightGBM JSON checkpoint containing raw
ranking scores, not class probabilities.

```powershell
python -m autocut train
```

Override the manifest, configuration, or output checkpoint when needed:

```powershell
python -m autocut train data/manifest/annotations_manifest.yaml --config config/train.yaml --checkpoint models/lightgbm/ranker.json
```

The LightGBM hyperparameters are configured in the `training` YAML section. The former PU/BiLSTM `training` options (epochs, batch size, hidden dimensions, dropout, positive prior and threshold, and related fields) are no longer used and should be removed. The former `lightgbm` section has been renamed to `training`.

The IMU features are computed from DJI WXYZ quaternion telemetry and gravity-compensated accelerometer samples stored in each source MP4. Angular rates are derived from relative quaternion rotations; acceleration features are expressed in g after the telemetry parser identifies the source units. Set `data.pitch_time_offset` only when the telemetry and video elapsed-time origins differ (default: `0.0`; the key retains its legacy name).

The default checkpoint is `models/lightgbm/ranker.json`, overridable with
`--checkpoint` or `experiment.lightgbm_model_path` in the configuration.
Ranking does not require every grade in every split, but at least one
training video must contain multiple relevance grades so that preference pairs
exist. The summary includes the configured fixed iteration count, validation metric history,
configured group/video IDs, query sizes, checkpoint path, and row counts.
Older YAML settings `objective: multiclass` and `num_class: 4` are migrated to
the ranking defaults by the training configuration loader.

## LightGBM ranking HPO

Tune LightGBM hyperparameters with Optuna. The objective is NMS-suppressed,
pooled event Segment Recall@5: greedy temporal NMS (IoU 0.3) removes redundant
overlapping windows before selecting five predictions per video, and the score
is total distinct events hit divided by total class-3 events. It is not an
unweighted mean of per-video recalls.

The standard HPO command runs trials over grouped LOVO folds. Its best trial
is still a tuning result, not an unbiased performance estimate. Use nested
leave-one-group-out evaluation for the estimate:

```powershell
python -m autocut nested-cv --config config/train.yaml
```

Nested CV prepares features once, tunes only within the outer-training groups,
refits using the selected fixed iteration count, and evaluates each untouched
outer group once. Median/IQR is the default. Reports are written separately for each normalization method:
`output/hpo/lightgbm_nested_cv_segment_recall_at_5_zscore_all_windows_ddof0.json`
and
`output/hpo/lightgbm_nested_cv_segment_recall_at_5_median_iqr_all_windows.json`.

To compare normalization, run the same nested procedure twice, changing only:

```yaml
data:
  feature_normalization: per_video_zscore_all_windows_ddof0
```

or:

```yaml
data:
  feature_normalization: per_video_median_iqr_all_windows
```

The full-data HPO command remains available:

```powershell
python -m autocut hpo --config config/train.yaml
```

Its study name is versioned by metric and normalization so trials from the
previous early-stopped objective or a different normalizer cannot contaminate
the current search. To apply the exported parameters to production training,
set the results JSON as `hpo.best_params_path` and run `train`; the selected
fixed `num_iterations` is used unchanged by the production refit.

## Feature-cache cleanup

The SQLite feature-cache catalog is a rebuildable index; NPZ files are the stored
feature data. Rebuilding the catalog does not delete or rewrite NPZ files.
Deleting NPZs requires an explicit keep manifest produced after a successful
cache-regeneration run.

Create a UTF-8 JSON array containing the NPZ filenames (relative to the canonical
cache directory) that the completed run confirmed should be retained, for
example:

```json
[
  "a1b2...cache-key.npz",
  "c3d4...cache-key.npz"
]
```

Review the proposed deletion list first (dry-run is the default):

```powershell
python -m autocut.dataset.cache_cleanup --cache-dir data/cache/features --keep-manifest data/cache/feature-cache-keep.json
```

Only after reviewing the report, apply cleanup explicitly:

```powershell
python -m autocut.dataset.cache_cleanup --cache-dir data/cache/features --keep-manifest data/cache/feature-cache-keep.json --apply
```

The command validates every keep-manifest file before deleting anything, refuses
paths outside the cache directory and invalid/missing keep entries, deletes only
NPZs excluded from the manifest, and then rebuilds/reconciles the SQLite catalog.
It does not rewrite surviving NPZs or manually delete SQLite/WAL/SHM files.
Do not run cleanup while extraction/regeneration is in progress.

Run the focused tests with:

```powershell
pytest tests/test_cache_cleanup.py tests/test_cache_catalog.py
```

## Legacy implementation

The `legacy` Git branch contains the pre-refactor implementation.

It is retained as a reference and safety baseline while the `main` branch is migrated to the AutoCut architecture.

The legacy implementation should not be treated as the target architecture. The target is defined by `docs/AUTOCUT_SPEC.md`.

## Development philosophy

The refactor follows a few principles:

- **Preserve proven behavior before improving it.**
- **Keep responsibilities separated.**
- **Keep `pipeline.py` as orchestration rather than business logic.**
- **Keep domain models separate from ML model implementations.**
- **Make analysis and dataset generation independently testable.**
- **Avoid coupling the application unnecessarily to a particular video or rendering engine.**
- **Use explicit configuration rather than scattered constants.**
- **Treat dataset auditing and leakage prevention as first-class concerns.**

The objective is not simply to reorganize the existing files. It is to establish a maintainable foundation on which the complete AutoCut pipeline can be developed without losing the useful work already accomplished.

## License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for the full license text.