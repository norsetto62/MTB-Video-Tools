# AUTOCUT — Top Level Specification / Architectural Requirement Document

**Version:** 0.05 — Draft  
**Document:** `AUTOCUT_SPEC.md`  
**Status:** Architectural foundation for the AutoCut refactoring

---

## 1. Purpose and Scope

AutoCut is a self-contained software package for turning one or more MTB POV videos, together with a user-supplied `Annotations.txt` manifest, into a final highlight video.

The system selects segments that are interesting to a human observer, including fast or technically demanding riding, drops and jumps, rock gardens, stairs, roots, rough terrain, steep or tight switchbacks, rocky terrain and tight vegetation passages. The user can also force specific passages into the result, including atmospheric material or any other explicitly mandatory segment.

**AutoCut is the system.** Analysis, feature extraction, dataset generation, model inference, candidate selection, timeline construction, audio handling and rendering are internal components of that system.

The principal user workflow is:

```text
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

The refactored project should converge toward:

```text
MTB-Video-Tools/
├── autocut/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline.py
│   ├── config.py
│   ├── errors.py
│   ├── logging.py
│   ├── annotations.py
│   ├── models.py
│   ├── manifests.py
│   ├── video/
│   │   ├── probe.py
│   │   ├── reader.py
│   │   └── renderer.py
│   ├── motion/
│   │   ├── flow.py
│   │   └── features.py
│   ├── dataset/
│   │   ├── windows.py
│   │   ├── targets.py
│   │   ├── builder.py
│   │   └── audit.py
│   ├── ml/
│   │   ├── model.py
│   │   ├── scaling.py
│   │   ├── checkpoint.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── selection/
│   │   ├── candidates.py
│   │   ├── ranking.py
│   │   ├── budget.py
│   │   └── timeline.py
│   ├── audio/
│   │   └── music.py
│   └── cache.py
├── tests/
├── docs/
├── models/
├── output/
└── pyproject.toml
```

The exact file layout may evolve; the architectural responsibilities must remain stable.

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

Serialization formats such as CSV, NPZ and JSON are adapters around these domain objects, not the definition of the domain model.

---

## 8. Video Inspection

Before processing, AutoCut shall inspect every referenced video and establish duration, frame rate, frame count where available, dimensions, pixel format where relevant, audio presence and relevant audio properties.

Video probing is centralized. Downstream modules receive typed video information rather than independently probing the same source. Invalid or inaccessible media must produce actionable errors.

---

## 9. Motion and Feature Analysis

The current proven baseline uses optical-flow-derived motion features at **2 FPS**, approximately one sample every 0.5 seconds.

The current temporal baseline is a **4.0-second window** with **2.0-second stride**. Feature extraction is modular so the feature representation can evolve without changing annotation parsing, selection or rendering.

Feature artifacts must identify the source, sampling configuration, feature schema/version, feature order and extraction configuration. Dimensions alone do not establish compatibility.

---

## 10. Temporal Windowing

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

## 11. Training Data and Annotation Model

AutoCut uses **continuous MTB interestingness**, not five independent feature labels.

| Target | Meaning |
|---:|---|
| 0 | Irrelevant/non-riding: stopped, mounting/dismounting, walking the bike, camera manipulation, waiting, etc. |
| 1 | Boring/ordinary riding |
| 2 | Middling/borderline interest |
| 3 | Clearly interesting/keep: good drops, rock gardens, technical sections, etc. |

The scale is a scalar interest measure. Annotation descriptions are audit metadata, not independent prediction targets.

---

## 12. Window Target Generation

A window may overlap several annotation intervals. Its target is therefore derived from temporal overlap rather than from a single point or arbitrary annotation.

The accepted formulation is duration-weighted interest:

```text
target = sum(overlap_duration_i * annotation_value_i) / window_duration
```

Accordingly, generated targets may be intermediate values between 0 and 3. This represents continuous temporal interest even though the human annotation scale uses integer levels.

---

## 13. Dataset Generation

Dataset generation shall load validated annotations, inspect source videos, load or generate motion/features, create windows, calculate duration-weighted targets, preserve provenance, validate schemas and write versioned dataset artifacts.

The conceptual dataset shape is:

```text
X: [N, T, F]
y: [N]
```

where `N` is the number of windows, `T` the samples per window and `F` the feature dimension. The current baseline has `T = 8`; feature dimensions remain schema-defined rather than permanently hard-coded.

---

## 14. Dataset Audit

Audit is a first-class development capability. It shall expose annotation count/duration, target distributions, generated window counts, source-video distribution, exact/intermediate targets, temporal coverage, annotation transitions and malformed/missing source data.

It must be possible to inspect the actual generated examples around transitions, including source video, window timestamps, target and annotation overlap. Audit output should be deterministic and regression-testable.

---

## 15. ML Model

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

## 16. Data Scaling

Normalization/scaling parameters must be learned from training data only. They are part of the model/data contract.

A checkpoint/model package must identify feature schema and order, feature mask where applicable, scaling method and parameters, and model architecture/configuration. Inference must reject incompatible feature data rather than silently applying an incorrect transformation.

---

## 17. Training and Validation

Because overlapping windows cause substantial temporal leakage, random window-level splitting is not acceptable as the primary validation strategy.

All windows from one source video must remain in one split. Preferred development evaluation includes grouped video-level splits and **Leave-One-Video-Out (LOVO)** validation.

After model configuration is frozen, a production model may be trained on the approved complete corpus. Training must record dataset identity, configuration, scaling, seed, validation results and source revision where available.

---

## 18. Inference

Inference converts feature windows into continuous interest predictions. Each prediction must retain enough information to map back to source video, temporal window, model/checkpoint and feature schema.

Inference produces evidence; it does not decide the final highlight timeline. This separation allows predictions to be inspected independently of editing policy.

---

## 19. Candidate Generation

Candidate generation converts interest estimates into candidate intervals. Candidates should contain source video, start, end, score, provenance, mandatory status and optional reason/selection metadata.

Candidate generation may use thresholds, local maxima, contiguous high-interest regions, temporal smoothing and padding. It remains separate from model inference.

---

## 20. Highlight Selection

Selection transforms candidates into a final timeline subject to target duration and user constraints.

It must account for mandatory clips, target duration, minimum duration, padding, maximum merge gap, overlap/deduplication, score, diversity, temporal distribution and multiple source videos.

Mandatory duration is accounted for first:

```text
automatic_budget = max(0, target_duration - mandatory_duration)
```

The selector must never create duplicate or overlapping mandatory material.

---

## 21. Human-Interest Ranking

Model score is evidence of interest, not a complete definition of a good highlight. Selection should distinguish model interest from temporal coherence, clip usability, redundancy, diversity and mandatory status.

A slightly lower-scoring clip can therefore be preferable to a near-duplicate of a higher-scoring clip. Selection should optimize the final collection rather than independently taking the highest-scoring windows.

---

## 22. Diversity and Temporal Distribution

The selector should avoid spending the entire budget on one small region unless explicitly configured to do so.

Selection should consider temporal spacing, overlap, similarity/redundancy, source-video balance and preservation of strong regions later in long recordings. This supersedes simplistic sequential accumulation from the start of a video.

---

## 23. Multi-Video Assembly

AutoCut may process multiple source videos in one invocation. Every timeline clip retains its source identity.

Selection must not assume that different videos share a time axis. Budgeting may be global or configurable, but it must be explicit and deterministic. Mandatory clips from every source are protected.

---

## 24. Timeline Data Model

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

## 25. Rendering

Rendering converts the final timeline into the output video. It validates boundaries, extracts requested intervals, concatenates them in timeline order, applies audio policy and reports the actual duration.

**FFmpeg is the underlying media engine.** Business logic must not depend on MoviePy-specific objects. A Python media wrapper may be used where useful, but the domain model remains media-engine independent.

---

## 26. Audio

The manifest may specify background music. The audio stage supports original audio only, original audio mixed with background music, or background music replacing original audio.

Audio handling belongs at the audio/rendering boundary, not in ML selection logic. The selected policy must be recorded in the result manifest.

---

## 27. Caching

Expensive stages should be cacheable, including video probing, optical flow, feature generation, dataset generation and model inference.

Cache identity must incorporate all relevant source identity, configuration, feature schema, dataset schema, model/checkpoint and software/schema information. Deleting the cache must not change logical results.

Cache entries must be invalidated when their defining contract changes.

---

## 28. Intermediate Artifacts

Useful intermediate artifacts include feature data, datasets, audit reports, checkpoints, predictions, candidate intervals, timelines and result manifests.

Artifacts should be inspectable and versioned, while avoiding unnecessary permanent files in normal use. Debug/development modes may preserve additional intermediates.

---

## 29. Result Manifest

Every completed run should produce a machine-readable result manifest containing AutoCut version, configuration, source videos and identities, annotation manifest, model/checkpoint, feature schema, dataset identity where relevant, selected clips, mandatory clips, final duration, audio configuration, output path, artifact identities and Git revision where available.

The result manifest is the primary traceability record for a generated result.

---

## 30. Logging and Diagnostics

Logging must communicate progress through major stages, warnings, final selections, output location and duration during normal use.

Debug logging should allow diagnosis of annotation parsing, probing, feature extraction, model loading, schema mismatches, candidate generation, budget allocation, timeline construction and rendering failures. Errors must not be silently hidden.

---

## 31. Error Handling

Errors shall be explicit, actionable and associated with the relevant stage.

Examples include missing sources, invalid annotations, invalid time ranges, duration mismatches, unsupported schemas, incompatible checkpoints, missing scaling metadata, corrupted datasets, permissions failures and FFmpeg failures.

Fail early when continuation is unsafe. Do not silently substitute defaults for required information.

---

## 32. Testing Strategy

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

## 33. Golden Reference Videos

The current reference set includes **Mentorella, Rosara, Cascata and Ascoli**.

These are useful for selection, timeline, rendering and qualitative regression evaluation. Ascoli must be interpreted carefully for generalization if it is part of the training corpus.

Golden tests should preserve expected behavior rather than relying exclusively on pixel-identical encoded output.

---

## 34. Known Failure Modes

Known failure modes include optical-flow false positives caused by rider struggles or camera motion, a rider stuck in a rut, disentangling from vegetation, short interesting events requiring padding, repeated high-motion patterns, temporal leakage from overlapping windows, early concentration of the selection budget and duplicate candidates.

Hard negatives are important because large motion is not necessarily interesting riding.

---

## 35. Current Baseline and Refactoring Strategy

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

## 36. Refactoring Rules

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

---

## 37. Public vs Internal API

The public API should remain deliberately small. The primary public entry point is the AutoCut CLI.

Python APIs may be exposed later, but only intentionally and with documentation. Internal modules may change during refactoring without becoming compatibility commitments.

Versioned artifacts are stronger compatibility boundaries than incidental Python function signatures.

---

## 38. Versioning

The following should be versioned independently where appropriate:

- annotation format;
- dataset schema;
- feature schema;
- model/checkpoint schema;
- result manifest schema;
- timeline schema.

Schema versions describe contracts, not merely software releases.

---

## 39. Compatibility

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

## 40. Reproducibility

A run should be reproducible from source media, annotation manifest, configuration, model checkpoint, feature/schema versions, scaling parameters, software revision and random seed where applicable.

Training must explicitly control randomness. Production inference and selection should be deterministic. Where media encoding has unavoidable nondeterminism, the logical timeline is the reproducibility target.

---

## 41. Platform Requirements

The initial target platform is **Windows**, with reliable PowerShell use.

Paths must be handled in a platform-safe manner. Developer-specific hard-coded paths should be avoided. External tools such as FFmpeg must be detected and reported clearly when unavailable.

---

## 42. Python and Dependencies

The project is Python-based. Dependencies should be declared centrally.

The implementation should prefer standard library facilities where sufficient, NumPy for numerical arrays, OpenCV for video/motion processing where appropriate, PyTorch for ML and FFmpeg for media processing.

Specific third-party dependencies should not leak into domain models. The final dependency set must reflect the actual implementation rather than historical prototype requirements.

---

## 43. Security and File Handling

AutoCut operates on user-supplied local files. File handling must validate paths, avoid accidental overwrites unless explicitly requested, use safe temporary directories, clean temporary artifacts appropriately, avoid unsafe shell construction and report permission failures clearly.

Media paths must not be interpolated unsafely into shell commands.

---

## 44. Performance

The architecture must preserve performance improvements already achieved in the prototype.

Expensive work should be streamed or batched where appropriate, performed at the required sampling rate rather than unnecessarily decoding every frame, and cached when reuse is possible.

The current optimized optical-flow sampling approach must not regress to full-rate decoding without justification. Performance-sensitive modules should expose measurable timings in diagnostics.

---

## 45. Acceptance Criteria

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

---

## 46. Architectural Definition of Done

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
- the normal user workflow remains simple.

---

## 47. Guiding Principle

> **AutoCut is the system. The individual analysis, dataset, inference, selection and rendering components are internal parts of that system.**

The architecture should make the whole pipeline coherent without making the internals opaque.

A user should be able to provide an annotation manifest and receive a finished video.

A developer should be able to inspect every major intermediate stage, test it independently, replace an implementation without rewriting unrelated components, and understand why a particular clip was selected.

The refactor succeeds when those two properties coexist:

**simple use for the user, explicit structure for the developer.**
