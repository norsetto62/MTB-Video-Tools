# AutoCut — Motion Processing Optimization Notes

## Purpose

This document records performance-related design decisions and optimization work in the AutoCut motion-processing pipeline.

The goal is not to optimize individual operations blindly, but to reduce computational cost while preserving the motion information useful for MTB video-interest detection.

Each optimization should record:

- the problem
- the chosen approach
- the reason for the decision
- alternatives considered
- measurements, when available

---

# 1. Region of Interest (ROI)

## Problem

Optical flow is computationally expensive, and large parts of an MTB video frame are of limited value for detecting interesting riding.

Typical sources of unnecessary computation/noise include:

- extreme left/right image areas containing trees and peripheral scenery
- lower portions of the image containing the bike, handlebars, arms, etc.
- areas that contribute little useful information about the trail ahead

The original processing operated on approximately:

```text
480 × 270 = 129,600 pixels
```

A proposed ROI may reduce this substantially, for example:

```text
288 × 135 = 38,880 pixels
```

which is approximately 30% of the original pixel count.

## Decision

Use a **hard rectangular ROI before optical-flow calculation**.

The ROI is therefore applied to the grayscale frame before Farneback rather than calculating full-frame optical flow and masking the result afterwards.

This reduces the amount of image data processed by the expensive optical-flow algorithm.

The ROI is expressed as:

```text
(x0, y0, width, height)
```

using normal image coordinates:

```text
x = 0  left
x = 1  right

y = 0  top
y = 1  bottom
```

The ROI is established once for a processing session/video and does not change from frame to frame.

Different video formats may require different ROI settings.

## Architectural consequence

ROI handling belongs **upstream of `flow.py`**.

`flow.py` receives two grayscale arrays that already represent the selected processing region.

It does not:

- select the ROI
- validate the ROI
- convert BGR to grayscale
- perform masking

This keeps the optical-flow function on the computational hot path as small as possible.

## Important distinction

ROI is different from feature weighting.

### Hard ROI

Applied before Farneback:

```text
full frame
    ↓
hard rectangular crop
    ↓
Farneback
```

Purpose:

- reduce computation
- remove obviously irrelevant image regions

### Soft weighting

Applied later during feature calculation:

```text
OpticalFlow
    ↓
spatial weighting
    ↓
features
```

Purpose:

- give more importance to useful regions
- gradually reduce the contribution of less useful regions

These are separate mechanisms and should not be conflated.

## Open question

The exact ROI dimensions should be determined empirically.

In particular, we should verify that reducing the ROI does not remove motion information that is useful for distinguishing interesting MTB sections.

# 1.1 Region of Interest (ROI) Boundaries

Farneback estimates motion using image neighborhoods at multiple pyramid levels. When we crop first, pixels outside the crop are unavailable to those calculations. Near the crop boundary, this can affect the estimated flow, especially when moving objects cross the boundary or when the optical-flow neighborhood extends beyond it.

One important distinction: this doesn't mean all flow near the boundary is necessarily wrong. The concern is that estimates there may be less reliable because the algorithm has less surrounding image information. The severity depends on the parameters, image content, and motion.

## How to improve it

Keep the current target ROI unchanged and calculate flow on a larger context region around it. After calculating flow, we crop the flow arrays to the target ROI before extracting features.

Target ROI — features we keep

Horizontal: 15%–85% · Vertical: 10%–50%

Context ROI — where we calculate flow

Horizontal: 10%–90% · Vertical: 0%–60%

The extra pixels give Farneback surrounding image information. They do not become part of the feature calculations.

### Why this is a good compromise

* It preserves the meaning and dimensions of our existing 54 features.

* It reduces the artificial boundary effect without calculating flow across the entire frame.

* It keeps the same target region, so the annotations and their temporal alignment do not need to change.

* It increases computation compared with the current ROI crop, but less than using the entire resized frame.

There is one caveat: the proposed margins are a reasonable starting point, not a guarantee. Farneback uses multiple pyramid levels, so a margin that looks generous at full resolution becomes smaller at coarser levels. We should test whether the improvement is sufficient.

## Implementation notes

1. Change the pipeline consistently: production feature extraction and the audit's extraction code must use the same context-then-crop approach. Ideally, we should avoid maintaining two independent implementations of that preprocessing.

2. Compare old and new flow: use a few representative clips, particularly where riders or obstacles cross the target ROI boundary.

3. Inspect the differences: compare the resulting flow fields and 54-feature sequences. The features should remain the same in definition, even though their numerical values may change.

4. Regenerate the cached NPZ files: they contain the old features, so they must not be mixed with the new results.

5. Rerun the feature-sequence audit, then proceed to statistical sanity.

One important consequence: if we change the feature-extraction method, we must eventually generate the training dataset using that same method. Otherwise, the model would be trained on features produced differently from those it sees during inference.

# 2. Farneback Optical Flow

## Current implementation

AutoCut currently uses OpenCV's dense Farneback optical-flow implementation:

```text
cv2.calcOpticalFlowFarneback()
```

`flow.py` deliberately provides only a thin wrapper around this OpenCV operation.

The function receives two already-prepared grayscale ROI frames and returns the AutoCut `OpticalFlow` model.

Conceptually:

```text
grayscale ROI frame 1
          +
grayscale ROI frame 2
          ↓
cv2.calcOpticalFlowFarneback()
          ↓
OpticalFlow
```

## Current parameters

The current parameters originate from the legacy implementation:

```text
pyr_scale = 0.5
levels    = 3
winsize   = 15
iterations = 3
poly_n    = 5
poly_sigma = 1.2
flags     = 0
```

These values are **not considered final**.

They were inherited from the previous implementation and should be treated as a baseline for experimentation.

## Why the parameters need to be revisited

The new pipeline changes the input to Farneback by applying a reduced ROI before optical-flow calculation.

Consequently, the spatial scale seen by Farneback is different from the legacy implementation.

Parameters such as:

- `winsize`
- `levels`
- `iterations`

may therefore have a different performance/quality trade-off.

The objective is not to obtain theoretically optimal optical flow.

The objective is:

> obtain motion features that are useful for MTB video-interest detection at the lowest practical computational cost.

## Planned investigation

Start by varying the parameters with the largest expected effect:

1. `winsize`
2. `levels`
3. `iterations`

while initially keeping:

```text
pyr_scale = 0.5
poly_n = 5
poly_sigma = 1.2
flags = 0
```

as the baseline.

Candidate values should be tested on representative MTB footage.

Each configuration should be evaluated for both:

- processing time
- usefulness/quality of the resulting motion features

A faster configuration is not automatically better if it removes information needed by the downstream regression model.

## Benchmark principle

Farneback parameters should ultimately be selected using the actual AutoCut task rather than a generic optical-flow benchmark.

The relevant metric is the complete trade-off:

```text
computational cost
        ↕
quality of motion signal
        ↕
downstream prediction performance
```

---

# 3. Future Optimizations

Additional performance work should be recorded here as it is investigated.

Possible areas include:

- frame decoding / sampling
- grayscale conversion and frame caching
- memory allocation
- optical-flow calculation
- feature extraction
- temporal derivatives
- spatial aggregation
- dataset generation
- inference batching

Optimizations should be added only when there is a measurable or well-understood reason to do so.

---

# 4. General Principle

The motion pipeline should avoid repeating work.

In particular:

- frames should be converted to grayscale once
- grayscale frames should be reusable/cached where practical
- ROI should be applied upstream
- invariant configuration should be established once
- expensive optical-flow processing should not perform unnecessary validation
- downstream feature extraction should operate directly on the resulting optical-flow arrays

The hot path should contain as little application-level overhead as reasonably possible.