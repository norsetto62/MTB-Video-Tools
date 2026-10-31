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

Training uses positive-unlabeled (PU) learning:

- **P (positive):** only pure score-3 windows.
- **U (unlabeled):** pure score-0, score-1, and score-2 windows. These are not treated as confirmed negatives.
- **Conflicting windows:** discarded.

The model has a single raw logit for P versus U. Score 3 maps to the positive set P; scores 0, 1, and 2 map to the unlabeled set U. Training uses the non-negative PU risk only, with no auxiliary classification objective. U examples are not treated as confirmed negatives. Configure `training.positive_prior` for the assumed positive prevalence used by the PU risk; the code default is 0.1 and should be checked against the dataset. The model's sigmoid output is the estimated P probability. Candidate-generation integration is intentionally outside the scope of this experiment.

The purpose of the model is therefore to prioritize clearly interesting moments without flooding training with presumed negatives.

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
python -m pip install -e .
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
for iteration selection. Before annotation filtering, each video's 48-column
window features are independently z-score normalized across all windows in that
video (population standard deviation, ddof=0; constant columns become zero).
The same transformation is applied to all windows of a video during ranking
inference. This makes features relative to each video's own distribution; it
does not guarantee immunity to camera or terrain changes. Validation NDCG is
recorded at the configured cutoffs (default `ndcg@10`) and drives early stopping.
The production ranker is then refit on all annotated manifest videos for the
selected number of iterations, including validation videos. It uses the
canonical feature cache and writes a LightGBM JSON checkpoint containing raw
ranking scores, not class probabilities.

```powershell
python -m autocut lightgbm-train
```

Override the manifest, configuration, or output checkpoint when needed:

```powershell
python -m autocut lightgbm-train data/manifest/annotations_manifest.yaml --config config/train.yaml --checkpoint models/lightgbm/ranker.json
```

The default checkpoint is `models/lightgbm/ranker.json`, overridable with
`--checkpoint` or `experiment.lightgbm_model_path` in the configuration.
Ranking does not require every grade in every split, but at least one
training video must contain multiple relevance grades so that preference pairs
exist. The summary includes the selected iteration, validation NDCG history,
configured group/video IDs, query sizes, checkpoint path, and row counts.
Older YAML settings `objective: multiclass` and `num_class: 4` are migrated to
the ranking defaults by the training configuration loader.

## LightGBM ranking HPO

Tune the LightGBM ranking hyperparameters with Optuna. The default search
space explicitly compares `lambdarank` and `rank_xendcg` while holding the
data preparation and per-video query grouping fixed. Trials are scored by the
mean held-out-group NDCG at the configured `lightgbm_hpo.ndcg_cutoff` (default
10); use the same manifest and fold set when comparing objectives. The study
is persisted in `data/db/lightgbm_hpo.db` and the trial table plus best
parameters are exported to `output/hpo/lightgbm_ranking_hpo_results.json`.

```powershell
python -m autocut lightgbm-hpo --config config/train.yaml
```

Optional configuration belongs in a `lightgbm_hpo` YAML section:

```yaml
lightgbm_hpo:
  study_name: lightgbm_ranking_hpo
  n_trials: 30
  ndcg_cutoff: 10
  storage_path: data/db/lightgbm_hpo.db
  output_dir: output/hpo
  search_space:
    objective:
      choices: [lambdarank, rank_xendcg]
```

The default search space is used for parameters not explicitly listed. The
objective choices must include both ranking objectives so the comparison is
not accidentally disabled. HPO exports the best parameter set; it does not
silently launch a production refit. Review the results, set the selected values
under the `lightgbm` configuration section, then run `lightgbm-train` to
create the production checkpoint.
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