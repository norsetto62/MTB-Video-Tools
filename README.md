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

The dataset uses overlapping temporal windows. Each window receives a duration-weighted target derived from the annotations that overlap it. Consequently, intermediate values between 0 and 3 are valid model targets.

The purpose of the model is therefore to answer:

> **How interesting is this moment of the ride?**

rather than:

> **What particular MTB feature is visible here?**

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
Selection under duration / editing constraints
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