# AutoCut Refactoring Phases

- [x] **Phase 1 — Packaging / src layout**
  - packaging / src layout
        ↓
- [x] **Phase 2 — Configuration + errors + logging**
  - configuration + errors + logging
        ↓
- [x] **Phase 3 — Annotations + domain models**
  - annotations + domain models
        ↓
- [x] **Phase 4 — Video probing / reading**
  - video probing / reading
        ↓
- [x] **Phase 5 — Motion / features**
  - motion / features
        ↓
- [x] **Phase 6 — Dataset / audit**
  - dataset / audit
        ↓
- [x] **Phase 7 — ML / scaling / checkpoint**
  - ML / scaling / checkpoint
        ↓
- [x] **Phase 8 — Inference**
  - inference
        ↓
- [x] **Phase 9 — Candidate generation**
  - candidate generation
        ↓
- [x] **Phase 10 — Selection / budget / timeline**
  - selection / budget / timeline
        ↓
- [x] **Phase 11 — Audio / rendering**
  - audio / rendering
        ↓
- [ ] **Phase 12 — Pipeline + CLI**
  - pipeline + CLI

## 12

Complete the end-to-end pipeline.
Connect the modules and establish a repeatable training and inference workflow. The orchestration should support the training process without prematurely depending on a final model.

autocut
├── run         # Full highlight-generation pipeline using a trained checkpoint
├── audit       # Inspect annotations, features, datasets, and splits
├── train       # Train a model and save its checkpoint and training metadata
├── validate    # Evaluate a model using defined video-level holdouts
├── test        # Run automated software tests, typically through pytest
├── lovo        # Orchestrate multiple train/evaluate runs, holding out one entire video at a time.
└── models      # Inspect checkpoints, model architecture, feature-schema compatibility, and training metadata.

run is the default mode, so autocut and autocut run should invoke the same full pipeline.
A few important distinctions to preserve:
- audit inspects the data and dataset without silently training a model.
- train produces a checkpoint and its training metadata.
- validate evaluates defined video-level holdouts.
- lovo orchestrates repeated training/evaluation runs, each holding out a different entire video.
- models inspects available models and their compatibility; it shouldn't simply list checkpoint filenames.
- test invokes the software test suite, normally through pytest

Proposed command convention

autocut [MODE] <input> <configuration.yaml>

MODE is optional only when it is run.

Command	              Input	                            Purpose
autocut          annotations.txt config.yaml	Annotation file	Generate highlights using a trained checkpoint
autocut audit    annotations.txt config.yaml	Annotation file or another supported audit input	Inspect annotations, features, datasets and splits
autocut train    dataset         config.yaml	Dataset or dataset specification	Train a model and save checkpoint plus metadata
autocut validate dataset         config.yaml	Dataset or validation specification	Evaluate on defined video-level holdouts
autocut test                     config.yaml	Test configuration, if needed	Run automated software tests
autocut lovo     dataset         config.yaml	Dataset or dataset specification	Run leave-one-video-out experiments
autocut models   config.yaml	                Model directory or model specification	Inspect checkpoints and compatibility

These are illustrative command forms. We should settle the precise input expected by each mode before implementing the CLI. In particular, train, validate, and lovo may need a dataset manifest or specification rather than a single annotations.txt file.
config.yaml is optional and used for configuration, while retaining the existing TXT annotation format. They serve different purposes:

- annotations.txt describes what was annotated.
- config.yaml describes how AutoCut should process it.

For example:

"""
\# config.yaml

paths:
  data_dir: data
  models_dir: models
  output_dir: output

video:
  sampling_fps: 2.0

features:
  roi:
    x: 0.10
    y: 0.10
    width: 0.80
    height: 0.65

selection:
  min_clip: 3.0

model:
  checkpoint: models/best_model.pt

audio:
  sync_mode: off
  fade_duration: 2.0

logging:
  level: INFO
"""

1. One configuration file per invocation. All tunable parameters live in YAML, not in a long list of command-line flags.
2. Sensible defaults. The user should not have to specify values that are already defined by the application.
3. A single configuration loader. Validate the YAML once, then pass typed configuration objects to the relevant modules.
4. Explicit mode-specific inputs. Each mode documents what its input represents; we should not overload one file argument to mean unrelated things.
5. Clear errors. Missing files, invalid YAML, unsupported settings and incompatible checkpoints should produce actionable messages.
6. Reproducibility. Training and validation metadata should record the effective configuration, dataset/split information and model details.
7. No duplicated settings. Command-line flags should not silently override values in YAML.

One small exception is useful: autocut --help and autocut MODE --help should work without requiring an input or configuration file. That doesn't compromise the clean interface.

## 13

Dataset and training readiness

Audit the additional training videos, check annotation quality and class balance, and confirm that the feature extractor and training code agree on the exact input schema.
1. Check that annotations are consistent with the three-class target.
2. Regenerate features and windows using the finalized extractor.
3. Audit class distribution, video coverage, and overlapping windows.
4. Make sure validation splits are grouped by video to prevent leakage.
5. Preserve the original held-out-video results when comparing recipes.

## 14 - LOVO validation and model-recipe selection

Compare candidate training recipes using held-out videos. Select the recipe based on generalization, not merely its training score.

### LOVO or the LSTM?

Use LOVO for validation and recipe selection, then train the final LSTM production model.

- LOVO — Leave-One-Video-Out validation: hold out one entire video at a time, train on the others, and evaluate on the unseen video. This helps reveal whether the model generalizes to new rides rather than memorizing characteristics of particular videos.
- LSTM — the model architecture we train and ultimately use for inference.
- Production training — after choosing the recipe, use `train_production_model.py` to train on all eight annotated datasets. The production checkpoint is what the extractor will use.

## 15 - Train the production model

Once the recipe is chosen, train the final model on all eight annotated datasets and use that checkpoint for AutoCut inference.
Generate and audit the dataset.
Confirm the target labels and feature schema.
Train the intended 3-class LSTM.
Evaluate using leakage-safe, grouped or leave-one-video-out validation.
Save and verify the production checkpoint and its metadata.

## 16 - End-to-end testing and validation

Verify the CLI can take the videos, annotations, model checkpoint, and music through to a finished highlight video.
Run the complete workflow with a trained model, first without music and then with music. Validate the final rendered videos.
The validation experiments should compare the relevant choices, including 2 FPS versus 10 FPS, class weighting, and retrieval behavior for the MTB class. Keep each held-out video out of training for its evaluation. Do not use the held-out results to repeatedly tune against that same video.
