# AUTOCUT — Top Level Specification / Architectural Requirement Document

**Version:** 0.06 — Draft  
**Document:** `AUTOCUT_SPEC.md`  
**Status:** Architectural foundation for the AutoCut refactoring

---

## 1. Purpose and Scope

AutoCut is a self-contained software package for turning one or more MTB POV videos, together with a user-supplied `Annotations.txt` manifest, into a final highlight video.

The system selects segments that are interesting to a human observer, including fast or technically demanding riding, drops and jumps, rock gardens, stairs, roots, rough terrain, steep or tight switchbacks, rocky terrain and tight vegetation passages. The user can also force specific passages into the result, including atmospheric material or any other explicitly mandatory segment.

**AutoCut is the system.** Analysis, feature extraction, dataset generation, model inference, candidate selection, timeline construction, audio handling and rendering are internal components of that system.

The principal user workflow is:

```
autocut annotations.txt
```

Development/maintenance commands such as `autocut audit`, `autocut train` and `autocut validate` are interfaces to the same package, not separate products.

---

## 2. Architectural Principles

1. **Single product boundary** — users interact with AutoCut, not a collection of unrelated scripts.
2. **Separation of responsibilities** — each module owns one coherent responsibility.
3. **Explicit contracts** — data passed between stages has defined schemas and versions.
4. **No god scripts** — the coordinator orchestrates but does not contain all business logic.
5. **No global mutable runtime state** — configuration and execution state are passed explicitly.
6. **Deterministic logical behavior** — identical inputs, configuration, model and relevant artifacts produce the same logical timeline.
7. **Cache expensive work** — expensive media analysis is reusable without changing results.
8. **Traceability** — final results are explainable through source videos, annotations, configuration and model.
9. **Testability** — core logic can be tested without full video processing or GPU execution.
10. **Preserve proven behavior** — refactoring must not change working behavior unless deliberately documented.
11. **Version compatibility** — datasets, checkpoints and artifacts carry enough metadata to reject incompatible combinations.
12. **Media-engine isolation** — domain logic must not depend on MoviePy-specific objects.
13. **User simplicity** — the normal workflow remains simple despite internal modularity.

---

## 3. Target Package Structure

The refactored project shall use a `src/` layout. This is not strictly required by Python, but it prevents accidental imports from the repository root and provides a clean separation between application source code and project data/artifacts.

The target structure is:

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
│       │   ├── __init__.py
│       │   ├── probe.py
│       │   ├── reader.py
│       │   └── renderer.py
│       ├── motion/
│       │   ├── __init__.py
│       │   ├── flow.py
│       │   └── features.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── windows.py
│       │   ├── targets.py
│       │   ├── builder.py
│       │   └── audit.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── model.py
│       │   ├── scaling.py
│       │   ├── checkpoint.py
│       │   ├── training.py
│       │   └── inference.py
│       ├── selection/
│       │   ├── __init__.py
│       │   ├── candidates.py
│       │   ├── ranking.py
│       │   ├── budget.py
│       │   └── timeline.py
│       ├── audio/
│       │   ├── __init__.py
│       │   └── music.py
│       └── cache.py
├── tests/
├── docs/
├── models/
├── output/
└── data/
```

The exact file layout may evolve; the architectural responsibilities must remain stable.

All directories that are intended to be Python packages, including `video`, `motion`, `dataset`, `ml`, `selection` and `audio`, should contain `__init__.py`. The top-level `models/`, `output/` and `data/` directories are data/artifact directories and are not Python packages.

`__main__.py` shall support direct module execution: `python -m autocut`.

The project must not define a local module named `logging.py`, which could shadow Python's standard-library `logging` module. `logging_utils.py` (or an equivalent unambiguous name) shall be used instead.

The `data/` tree is intentionally separate from source code and final products. A recommended organization is:

```text
data/
├── raw/
├── annotations/
├── flow/
├── datasets/
└── audit/
```

Generated datasets, flow artifacts, audit reports and temporary/intermediate development artifacts belong under `data/` rather than `output/`. `models/` contains trained model/checkpoint artifacts, while `output/` contains final generated videos and other user-facing products.

---

## 4. Public Interface

The primary public interface is:

```powershell
autocut annotations.txt
```

The normal command parses and validates the manifest, inspects videos, processes requested ranges, computes features, runs the model, generates candidates, protects mandatory clips, constructs the timeline, applies audio policy, renders the result and writes a result manifest.

Development commands may expose audit, training and validation capabilities, but remain part of the same package.

---

## 5. Configuration

Configuration shall be explicit and preferably immutable during a run. It should cover input/output paths, target duration, clip limits, padding, merge gap, sampling rate, window duration/stride, feature and model schemas, scaling, selection parameters, audio policy, logging, cache policy and deterministic seed.

Any configuration value that materially affects an artifact must participate in its identity or manifest.

---

## 6. Annotation Manifest

`Annotations.txt` is the user-facing input manifest. It defines source videos, optional processing ranges, mandatory ranges and optional background music.

The parser shall immediately convert time expressions into absolute floating-point seconds and represent ranges as typed intervals using `[start, end)` semantics.

Validation shall cover file existence, timestamp syntax, non-negative times, `end > start`, bounds against source duration and directive syntax. Malformed input must produce a clear error rather than being silently altered.

---

## 7. Internal Data Model

Use explicit typed models for concepts including `VideoSource`, `VideoRange`, `Annotation`, `MusicDirective`, `FeatureSample`, `TemporalWindow`, `InterestTarget`, `Prediction`, `CandidateInterval`, `TimelineClip`, `Timeline`, `PipelineConfig` and `ResultManifest`.

`autocut.models` is the home for domain/data models. It must not become a generic container for neural-network implementations. ML architectures and their forward/inference behavior belong in `autocut.ml.model`; checkpoint handling belongs in `autocut.ml.checkpoint`.

Serialization formats such as CSV, NPZ and JSON are adapters around these domain objects, not the definition of the domain model. Domain models should not expose third-party media-engine objects or otherwise couple the business model to MoviePy, OpenCV or a particular storage format.

---

## 8. Pipeline Orchestration

`autocut.pipeline` is the orchestration layer for the application. It coordinates the major stages but does not contain their detailed algorithms.

Conceptually, the pipeline is:

```text
CLI
 │
 ▼
pipeline
 │
 ├── annotations / manifests
 ├── video
 ├── motion
 ├── dataset
 ├── ml
 ├── selection
 ├── audio
 └── rendering
```

The coordinator may invoke operations such as probing media, computing features, building datasets, running inference, generating candidates, constructing the timeline and rendering the result. The implementation of those operations remains in the corresponding package.

The pipeline must remain thin enough that individual stages can be unit-tested and replaced without rewriting unrelated stages. It is the application's workflow coordinator, not a replacement for the domain modules.

---

## 9. Video Inspection

Before processing, AutoCut shall inspect every referenced video and establish duration, frame rate, frame count where available, dimensions, pixel format where relevant, audio presence and relevant audio properties.

Video probing is centralized. Downstream modules receive typed video information rather than independently probing the same source. Invalid or inaccessible media must produce actionable errors.

---

## 10. Motion and Feature Analysis

The current proven baseline uses optical-flow-derived motion features at **2 FPS**, approximately one sample every 0.5 seconds.

The current temporal baseline is a **4.0-second window** with **2.0-second stride**. Feature extraction is modular so the feature representation can evolve without changing annotation parsing, selection or rendering.

Feature artifacts must identify the source, sampling configuration, feature schema/version, feature order and extraction configuration. Dimensions alone do not establish compatibility.

---

## 11. Temporal Windowing

The model operates on temporal windows rather than isolated frames.

Current baseline:

```text
sampling rate = 2 FPS
window       = 4.0 s
samples      = 8
stride       = 2.0 s
```

Adjacent windows therefore overlap by approximately 50%. Window generation is independent of the model architecture and each window retains source identity, start/end time, sample information and feature schema/version.

---

## 12. Training Data and Annotation Model

AutoCut uses **continuous MTB interestingness**, not five independent feature labels.

| Target | Meaning |
|---:|---|
| 0 | Irrelevant/non-riding: stopped, mounting/dismounting, walking the bike, camera manipulation, waiting, etc. |
| 1 | Boring/ordinary riding |
| 2 | Middling/borderline interest |
| 3 | Clearly interesting/keep: good drops, rock gardens, technical sections, etc. |

The scale is a scalar interest measure. Annotation descriptions are audit metadata, not independent prediction targets.

---

## 13. Window Target Generation

A window may overlap several annotation intervals. Its target is therefore derived from temporal overlap rather than from a single point or arbitrary annotation.

The accepted formulation is duration-weighted interest:

```text
target = sum(overlap_duration_i * annotation_value_i) / window_duration
```

Accordingly, generated targets may be intermediate values between 0 and 3. This represents continuous temporal interest even though the human annotation scale uses integer levels.

---

## 14. Dataset Generation

Dataset generation shall load validated annotations, inspect source videos, load or generate motion/features, create windows, calculate duration-weighted targets, preserve provenance, validate schemas and write versioned dataset artifacts.

The conceptual dataset shape is:

```text
X: [N, T, F]
y: [N]
```

where `N` is the number of windows, `T` the samples per window and `F` the feature dimension. The current baseline has `T = 8`; feature dimensions remain schema-defined rather than permanently hard-coded.

---

## 15. Dataset Audit

Audit is a first-class development capability. It shall expose annotation count/duration, target distributions, generated window counts, source-video distribution, exact/intermediate targets, temporal coverage, annotation transitions and malformed/missing source data.

It must be possible to inspect the actual generated examples around transitions, including source video, window timestamps, target and annotation overlap. Audit output should be deterministic and regression-testable.

---

## 16. ML Model

The model estimates continuous MTB interest over time:

```text
temporal feature sequence
        ↓
interest model
        ↓
scalar interest estimate
```

The current sequence-model/BiLSTM prototype is a valid baseline to preserve during refactoring. The architecture remains replaceable through a defined model interface for construction, loading, inference, serialization and compatibility validation.

---

## 17. Data Scaling

Normalization/scaling parameters must be learned from training data only. They are part of the model/data contract.

A checkpoint/model package must identify feature schema and order, feature mask where applicable, scaling method and parameters, and model architecture/configuration. Inference must reject incompatible feature data rather than silently applying an incorrect transformation.

---

## 18. Training and Validation

Because overlapping windows cause substantial temporal leakage, random window-level splitting is not acceptable as the primary validation strategy.

All windows from one source video must remain in one split. Preferred development evaluation includes grouped video-level splits and **Leave-One-Video-Out (LOVO)** validation.

After model configuration is frozen, a production model may be trained on the approved complete corpus. Training must record dataset identity, configuration, scaling, seed, validation results and source revision where available.

---

## 19. Inference

Inference converts feature windows into continuous interest predictions. Each prediction must retain enough information to map back to source video, temporal window, model/checkpoint and feature schema.

Inference produces evidence; it does not decide the final highlight timeline. This separation allows predictions to be inspected independently of editing policy.

---

## 20. Candidate Generation

Candidate generation converts interest estimates into candidate intervals. Candidates should contain source video, start, end, score, provenance, mandatory status and optional reason/selection metadata.

Candidate generation may use thresholds, local maxima, contiguous high-interest regions, temporal smoothing and padding. It remains separate from model inference.

---

## 21. Highlight Selection

Selection transforms candidates into a final timeline subject to target duration and user constraints.

It must account for mandatory clips, target duration, minimum duration, padding, maximum merge gap, overlap/deduplication, score, diversity, temporal distribution and multiple source videos.

Mandatory duration is accounted for first:

```text
automatic_budget = max(0, target_duration - mandatory_duration)
```

The selector must never create duplicate or overlapping mandatory material.

---

## 22. Human-Interest Ranking

Model score is evidence of interest, not a complete definition of a good highlight. Selection should distinguish model interest from temporal coherence, clip usability, redundancy, diversity and mandatory status.

A slightly lower-scoring clip can therefore be preferable to a near-duplicate of a higher-scoring clip. Selection should optimize the final collection rather than independently taking the highest-scoring windows.

---

## 23. Diversity and Temporal Distribution

The selector should avoid spending the entire budget on one small region unless explicitly configured to do so.

Selection should consider temporal spacing, overlap, similarity/redundancy, source-video balance and preservation of strong regions later in long recordings. This supersedes simplistic sequential accumulation from the start of a video.

---

## 24. Multi-Video Assembly

AutoCut may process multiple source videos in one invocation. Every timeline clip retains its source identity.

Selection must not assume that different videos share a time axis. Budgeting may be global or configurable, but it must be explicit and deterministic. Mandatory clips from every source are protected.

---

## 25. Timeline Data Model

The timeline is the authoritative description of what will be rendered.

A timeline clip contains at least:

```text
source_video
start
end
duration
mandatory
score
reason
```

Optional provenance may include clip ID, annotation reference, model/checkpoint, selection stage, ranking and padding information.

The renderer consumes the authoritative timeline and does not recalculate selection.

---

## 26. Rendering

Rendering converts the final timeline into the output video. It validates boundaries, extracts requested intervals, concatenates them in timeline order, applies audio policy and reports the actual duration.

**FFmpeg is the underlying media engine.** Business logic must not depend on MoviePy-specific objects. A Python media wrapper may be used where useful, but the domain model remains media-engine independent.

---

## 27. Audio

The manifest may specify background music. The audio stage supports original audio only, original audio mixed with background music, or background music replacing original audio.

Audio handling belongs at the audio/rendering boundary, not in ML selection logic. The selected policy must be recorded in the result manifest.

---

## 28. Caching

Expensive stages should be cacheable, including video probing, optical flow, feature generation, dataset generation and model inference.

Cache identity must incorporate all relevant source identity, configuration, feature schema, dataset schema, model/checkpoint and software/schema information. Deleting the cache must not change logical results.

Cache entries must be invalidated when their defining contract changes.

---

## 29. Intermediate Artifacts

Useful intermediate artifacts include feature data, datasets, audit reports, checkpoints, predictions, candidate intervals, timelines and result manifests.

Artifacts should be inspectable and versioned, while avoiding unnecessary permanent files in normal use. Debug/development modes may preserve additional intermediates.

---

## 30. Result Manifest

Every completed run should produce a machine-readable result manifest containing AutoCut version, configuration, source videos and identities, annotation manifest, model/checkpoint, feature schema, dataset identity where relevant, selected clips, mandatory clips, final duration, audio configuration, output path, artifact identities and Git revision where available.

The result manifest is the primary traceability record for a generated result.

---

## 31. Logging and Diagnostics

Logging must communicate progress through major stages, warnings, final selections, output location and duration during normal use.

Debug logging should allow diagnosis of annotation parsing, probing, feature extraction, model loading, schema mismatches, candidate generation, budget allocation, timeline construction and rendering failures. Errors must not be silently hidden.

---

## 32. Error Handling

Errors shall be explicit, actionable and associated with the relevant stage.

Examples include missing sources, invalid annotations, invalid time ranges, duration mismatches, unsupported schemas, incompatible checkpoints, missing scaling metadata, corrupted datasets, permissions failures and FFmpeg failures.

Fail early when continuation is unsafe. Do not silently substitute defaults for required information.

---

## 33. Testing Strategy

Testing is layered:

```text
Unit tests
    ↓
Dataset / audit tests
    ↓
Model validation
    ↓
Inference tests
    ↓
Selection / timeline tests
    ↓
Rendering tests
    ↓
End-to-end tests
    ↓
Human visual evaluation
```

Unit tests should avoid unnecessary video/GPU dependencies. Integration tests verify module contracts. End-to-end tests exercise the real pipeline on controlled media. Regression tests preserve known-good behavior from the current implementation.

---

## 34. Golden Reference Videos

The current reference set includes **Mentorella, Rosara, Cascata and Ascoli**.

These are useful for selection, timeline, rendering and qualitative regression evaluation. Ascoli must be interpreted carefully for generalization if it is part of the training corpus.

Golden tests should preserve expected behavior rather than relying exclusively on pixel-identical encoded output.

---

## 35. Known Failure Modes

Known failure modes include optical-flow false positives caused by rider struggles or camera motion, a rider stuck in a rut, disentangling from vegetation, short interesting events requiring padding, repeated high-motion patterns, temporal leakage from overlapping windows, early concentration of the selection budget and duplicate candidates.

Hard negatives are important because large motion is not necessarily interesting riding.

---

## 36. Current Baseline and Refactoring Strategy

The current repository is the baseline/reference implementation. The refactor is not a redesign detached from the working system.

The refactor must preserve useful behavior already achieved while moving it into a coherent package.

Current proven characteristics include:

- optical-flow/motion analysis at approximately 2 FPS;
- 4-second temporal windows;
- 2-second stride;
- overlapping windows;
- duration-weighted scalar interest targets;
- a sequence-based ML model;
- grouped/leakage-safe validation as the intended evaluation strategy;
- candidate selection followed by timeline construction;
- FFmpeg-based media processing;
- mandatory user-selected intervals.

Refactoring should be incremental and behavior should be compared with the previous implementation at each meaningful stage.

---

## 37. Refactoring Rules

1. Preserve proven behavior unless a change is deliberate.
2. Document deliberate behavioral changes.
3. Keep compatible datasets usable.
4. Version incompatible dataset changes.
5. Version incompatible checkpoint changes.
6. Add tests when moving or rewriting logic.
7. Keep domain logic independent of CLI parsing.
8. Keep model code independent of rendering.
9. Keep rendering independent of model implementation.
10. Avoid global mutable state.
11. Avoid duplicated parsing, timing and probing logic.
12. Prefer small explicit interfaces.
13. Keep the pipeline coordinator thin.
14. Do not preserve obsolete architectural concepts merely because they exist in old scripts.
15. Keep the `src/` package boundary intact; do not rely on imports from the repository root.
16. Keep generated datasets and intermediate artifacts under `data/`, trained models under `models/` and final products under `output/`.
17. Keep domain models independent of ML implementations and media-engine objects.
18. Keep the pipeline coordinator thin and stage-specific logic inside its owning package.

---

## 38. Public vs Internal API

The public API should remain deliberately small. The primary public entry point is the AutoCut CLI.

Python APIs may be exposed later, but only intentionally and with documentation. Internal modules may change during refactoring without becoming compatibility commitments.

Versioned artifacts are stronger compatibility boundaries than incidental Python function signatures.

---

## 39. Versioning

The following should be versioned independently where appropriate:

- annotation format;
- dataset schema;
- feature schema;
- model/checkpoint schema;
- result manifest schema;
- timeline schema.

Schema versions describe contracts, not merely software releases.

---

## 40. Compatibility

Compatibility must be checked before processing, including:

```text
dataset feature schema ↔ model feature schema
dataset scaling       ↔ model scaling
checkpoint architecture ↔ inference implementation
timeline schema       ↔ renderer
result manifest schema ↔ reporting tools
```

When compatibility cannot be established, the system must fail with a clear diagnostic. Matching array dimensions alone is not sufficient.

---

## 41. Reproducibility

A run should be reproducible from source media, annotation manifest, configuration, model checkpoint, feature/schema versions, scaling parameters, software revision and random seed where applicable.

Training must explicitly control randomness. Production inference and selection should be deterministic. Where media encoding has unavoidable nondeterminism, the logical timeline is the reproducibility target.

---

## 42. Platform Requirements

The initial target platform is **Windows**, with reliable PowerShell use.

Paths must be handled in a platform-safe manner. Developer-specific hard-coded paths should be avoided. External tools such as FFmpeg must be detected and reported clearly when unavailable.

---

## 43. Python and Dependencies

The project is Python-based. Dependencies should be declared centrally.

The implementation should prefer standard library facilities where sufficient, NumPy for numerical arrays, OpenCV for video/motion processing where appropriate, PyTorch for ML and FFmpeg for media processing.

Specific third-party dependencies should not leak into domain models. The final dependency set must reflect the actual implementation rather than historical prototype requirements.

---

## 44. Security and File Handling

AutoCut operates on user-supplied local files. File handling must validate paths, avoid accidental overwrites unless explicitly requested, use safe temporary directories, clean temporary artifacts appropriately, avoid unsafe shell construction and report permission failures clearly.

Media paths must not be interpolated unsafely into shell commands.

---

## 45. Performance

The architecture must preserve performance improvements already achieved in the prototype.

Expensive work should be streamed or batched where appropriate, performed at the required sampling rate rather than unnecessarily decoding every frame, and cached when reuse is possible.

The current optimized optical-flow sampling approach must not regress to full-rate decoding without justification. Performance-sensitive modules should expose measurable timings in diagnostics.

---

## 46. Acceptance Criteria

The refactored system is acceptable when:

1. A user can run `autocut annotations.txt` and obtain the requested final video.
2. `Annotations.txt` is the authoritative user-facing input manifest.
3. Mandatory clips are preserved.
4. Automatic selection uses the configured interest model.
5. Continuous interest predictions are separated from selection policy.
6. The final timeline is explicit and inspectable.
7. Rendering consumes the timeline rather than recomputing selection.
8. Model/checkpoint/data compatibility is validated.
9. Dataset splitting prevents temporal leakage.
10. Expensive stages can be cached.
11. The result is traceable through a result manifest.
12. Core logic can be unit tested without full video processing.
13. The system remains usable from Windows/PowerShell.
14. Existing proven behavior is preserved unless deliberately changed.
15. The package is importable through the `src/` layout without accidental repository-root imports.
16. `python -m autocut` works through `__main__.py`.
17. Generated datasets/intermediates are separated from final output and trained models.
18. Domain models are independent of ML architecture and media-engine implementations.
19. The pipeline coordinator orchestrates stages without absorbing their implementation logic.

---

## 47. Architectural Definition of Done

The architecture is complete when:

- AutoCut is the single coherent product boundary;
- internal modules have clear responsibilities;
- public and internal interfaces are distinguishable;
- domain models are explicit and typed;
- configuration is explicit;
- annotation parsing is isolated;
- video probing, reading and rendering are isolated from business logic;
- feature extraction is isolated from dataset generation;
- dataset generation is isolated from model training;
- scaling is part of the model/data contract;
- inference is isolated from candidate selection;
- candidate selection is isolated from timeline construction;
- timeline construction is isolated from rendering;
- audio handling is isolated from ML logic;
- caching is deterministic and invalidation-aware;
- artifacts are versioned;
- tests cover important boundaries;
- golden-video regression evaluation exists;
- the normal user workflow remains simple;
- the `src/` package boundary is respected;
- package directories have explicit `__init__.py` files where appropriate;
- `python -m autocut` is supported;
- `data/`, `models/` and `output/` have distinct responsibilities;
- domain models remain separate from ML implementations;
- the pipeline remains a thin orchestration layer.

---

## 48. Guiding Principle

> **AutoCut is the system. The individual analysis, dataset, inference, selection and rendering components are internal parts of that system.**

The architecture should make the whole pipeline coherent without making the internals opaque.

A user should be able to provide an annotation manifest and receive a finished video.

A developer should be able to inspect every major intermediate stage, test it independently, replace an implementation without rewriting unrelated components, and understand why a particular clip was selected.

The refactor succeeds when those two properties coexist:

**simple use for the user, explicit structure for the developer.**

---

## 49. Phase 11 — Audio, Synchronization and Rendering Contracts

Phase 11 implements music analysis, optional boundary synchronization, music caching and soundtrack rendering. It consumes the timeline produced by selection/timeline construction; it must not rerun inference or independently choose highlights. The pipeline coordinator orchestrates these operations but does not implement their algorithms.

### 49.1 Responsibilities and module boundaries

The intended responsibilities are:

- `autocut.audio.analysis`: analyze the music file and produce a validated, media-engine-independent `AudioAnalysis` result.
- `autocut.audio.cache`: load, validate, identify, and atomically save cached analysis.
- Timeline/editing synchronization logic: adjust eligible clip boundaries using analysis timestamps. Keep this operation separately testable from audio decoding and rendering.
- `autocut.editing.audio` (or the equivalent established rendering-layer module): apply the selected soundtrack policy, fades, and FFmpeg audio/video mapping.
- `autocut.pipeline`: pass explicit configuration and results between these stages.

The exact module placement may follow the existing package layout, but the responsibilities above must remain separate. Audio analysis must not depend on selection internals, and rendering must consume the final timeline rather than recalculate it.

### 49.2 Audio analysis result

The analysis result shall expose at least:

- source path for diagnostics;
- source duration in seconds;
- estimated tempo in beats per minute;
- ordered beat timestamps;
- ordered estimated measure timestamps;
- ordered onset-peak timestamps;
- ordered combined synchronization timestamps.

All timestamps are seconds from the beginning of the music file. Timestamp arrays must contain finite values, be monotonically non-decreasing (deduplicated where appropriate), and fall within the source duration, allowing only a documented small floating-point tolerance at the upper boundary. Invalid or empty event arrays must not be treated as usable synchronization data.

Measure timestamps are estimates. The initial implementation may derive them from every fourth detected beat, but must not claim that this always identifies true musical measures or assumes every track is in 4/4.

The analysis result should be a typed domain object, not a raw third-party library object or decoded waveform. The initial cache stores compact analysis metadata and event timestamps, not the full waveform.

### 49.3 Music-analysis cache

The default cache directory is `data/audio_cache/`, resolved relative to the configured `data_dir`. The initial implementation uses this default; a user-selected cache directory is deferred until needed.

The JSON cache retains the established fields:

- `cache_version`
- `source_hash`
- `source`
- `duration`
- `tempo`
- `beats`

It also stores `measures`, `onsets`, and `combined` synchronization timestamps. The source hash is SHA-256 over the music file contents. A cache entry is reusable only when its cache version and source content hash match and all cached values pass validation. The stored source path is diagnostic metadata, not a substitute for content identity.

Cache writes must use a temporary file followed by an atomic replacement where supported, so an interrupted write does not leave a partially written cache entry. Cache failures should be logged and handled without treating corrupt or incompatible data as valid; analysis may be recomputed when the source file remains readable.

### 49.4 Synchronization modes and boundary policy

Synchronization is configurable and supports these modes:

- `off`: leave both clip boundaries unchanged;
- `beat`: snap both eligible start and end boundaries to the nearest detected beat;
- `measure`: snap both eligible boundaries to estimated measure timestamps;
- `forward-beat`: move an eligible boundary forward to a suitable beat at or after its unsynchronized timestamp, rather than choosing the nearest beat;
- `onset`: snap both eligible boundaries to detected onset peaks;
- `combined`: snap both eligible boundaries using the combined beat-and-onset timestamp set.

These are the intended semantics for the new implementation. Before claiming exact compatibility with the legacy script, inspect its original implementation of `forward-beat` and `combined` and preserve any established nuance that does not conflict with the contracts here.

Synchronization adjusts both starts and ends where a valid adjustment is possible. It must not produce negative starts, empty/reversed intervals, clips shorter than the configured minimum, optional clips longer than the configured maximum, or a timeline that silently exceeds its applicable duration budget. If a proposed adjustment violates a constraint, retain the unsynchronized boundary or interval. Mandatory clips are exempt from synchronization and must remain unchanged.

If audio analysis fails or produces unusable synchronization data, issue a warning and continue with unsynchronized editing. Music playback/mixing may still proceed if the music file itself is usable. Do not fabricate beat positions or silently represent failed analysis as successful analysis.

### 49.5 Maximum optional clip duration

The configured maximum clip duration, referred to as `mc`, has these semantics:

- `mc = 0`: no per-clip maximum;
- `mc > 0`: no optional clip may exceed `mc` seconds in the final timeline.

Mandatory clips are cast in stone and are exempt from `mc`; their specified boundaries must not be shortened, discarded, or changed to satisfy this limit.

When an optional clip exceeds `mc`, attempt to retain the most interesting portion using its available inference-window evidence, centred around the strongest evidence where feasible. Do not rerun inference to trim a clip. If evidence-based trimming is unavailable, use a deterministic fallback that respects `mc` and the configured minimum clip duration; if no valid interval can be produced, reject that optional clip. The chosen interval must be finalized before rendering.

This is a cross-stage selection constraint: the component responsible for optional clip selection/trimming must enforce it, and the timeline must validate it. Synchronization must not subsequently lengthen an optional clip beyond `mc`. Mandatory clips may cause the overall timeline to exceed the requested target duration; in that case, preserve them and issue a warning rather than modifying them.

### 49.6 Music fade and soundtrack rendering

Music fade duration is configurable from the outset. The initial default is **2.0 seconds**. Validate that the value is non-negative and constrain the effective fade to the available music/output duration, including very short tracks. A zero duration means no fade.

Rendering must consume the final timeline and its explicit audio configuration. It must validate the requested output duration and audio stream mapping, preserve the intended video stream when stream-copying is appropriate, encode/mix the soundtrack according to the configured policy, apply the ending fade and report the actual output duration. FFmpeg failures must produce actionable diagnostics. Do not depend on MoviePy-specific objects in the domain model.

### 49.7 Failure handling and observability

- If the music file cannot be read and is required by the manifest, report a clear error.
- If only beat/onset analysis fails, warn and fall back to unsynchronized editing; do not fail the whole run solely because optional synchronization is unavailable.
- If the cache is missing, stale, corrupt, or incompatible, do not trust it; recompute when possible.
- Record the selected synchronization mode, effective fade duration, maximum clip duration, music identity, cache/analysis outcome and soundtrack policy in configuration/result metadata where applicable.
- Keep logging useful but avoid printing large event arrays in normal operation.

### 49.8 Phase 11 acceptance tests

Tests must cover at least:

1. Valid analysis output and timestamp-array validation.
2. Cache hit for matching version/hash and cache miss for changed content/version.
3. Rejection/recomputation of malformed cache entries.
4. Atomic cache-write behavior.
5. Each synchronization mode, including the directional behavior of `forward-beat`.
6. Both-boundary adjustment and invalid-adjustment fallback.
7. Analysis failure warning with unsynchronized fallback.
8. `mc = 0`, positive `mc`, evidence-centred trimming, deterministic fallback, and rejection when no valid trimmed interval exists.
9. Mandatory clips remaining unchanged and exempt from `mc` and synchronization.
10. Fade disabled, default fade, custom fade, and very short output duration.
11. FFmpeg command construction, stream mapping, failure reporting and actual-duration verification.
12. Integration confirming that rendering uses the supplied final timeline and does not rerun inference or selection.
