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

## Configurable target ROI

The target rectangle is configured with four normalized coordinates:

| Parameter | Meaning | Default |
|---|---|---:|
| `roi_left` | Left edge, fraction of frame width | `0.15` |
| `roi_right` | Right edge, fraction of frame width | `0.85` |
| `roi_top` | Top edge, fraction of frame height | `0.10` |
| `roi_bottom` | Bottom edge, fraction of frame height | `0.50` |

Every coordinate must be finite and in `[0.0, 1.0]`; left must be less than right, and top less than bottom. In the training configuration these keys belong under `data`; in the production configuration they belong under `video`. Keep the values identical between training and inference: the runtime rejects a checkpoint with missing ROI metadata or different ROI bounds.

The context crop adds 5% of frame width on each horizontal side and 10% of frame height above and below the target, clipped to the frame boundary. Farneback runs on that context crop; the flow is then cropped to the target ROI before the existing 54 features are calculated.

Changing any ROI coordinate changes the feature-cache identity. The feature-cache version is also incremented, so caches created before ROI support are not reused. Sampling FPS is part of the same cache identity. Window duration and stride do not affect raw optical-flow features and are not stored in a separate window cache: window tensors are rebuilt from the cached raw sequence whenever training or inference runs.

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

# 3. LightGBM Optimisation

Ottimizzare un modello LightGBM richiede di bilanciare la capacità della rete di apprendere pattern complessi con la regolarizzazione per evitare l'overfitting. Avendo elencato i parametri chiave, possiamo raggrupparli in categorie logiche e definire una strategia di tuning efficace.

1. Controllare la Capacità degli Alberi (Complessità)
Questi parametri definiscono quanto "grande" e profondo può crescere ogni singolo albero:

num_leaves (Il parametro più critico): Determina il numero massimo di foglie in un albero. Poiché LightGBM cresce gli alberi leaf-wise (per foglia) anziché level-wise (per livello), num_leaves è il fattore principale che controlla la complessità. Regola d'oro: Poiché ogni albero binario profondo d ha al massimo 2^d foglie, imposta sempre num_leaves inferiore a 2^{max_depth} (es. con max_depth=7, usa num_leaves tra 30 e 63) per evitare un overfitting estremo.

max_depth: Limita la profondità massima dell'albero. È utile impostarlo come barriera di sicurezza per impedire a num_leaves di generare alberi troppo sbilanciati o profondi.

min_data_in_leaf: Fondamentale per gestire dataset rumorosi o con poche istanze. Aumentarlo (es. a valori tra 20 e 100+) impedisce all'albero di creare foglie basate su pochissimi campioni. Valori bassi rischiano l'overfitting, valori alti portano a underfitting.

min_split_gain: Rappresenta il guadagno minimo (information gain) necessario per effettuare una divisione (split). Aumentarlo filtra i tagli "inutili" o guidati dal rumore, semplificando l'albero.

2. Velocità di Apprendimento e Iterazionilearning_rate (o eta) e num_iterations (o n_estimators): 

Sono indissolubili. Un learning_rate basso (es. 0.01 o 0.03) associato a un numero alto di iterazioni produce modelli generalmente più accurati e generalizzabili. Best practice: Invece di fissare num_iterations, imposta un numero alto (es. 1000 o 2000) e usa l'Early Stopping (interrompi il training se la validation loss non migliora per es. 50-100 iterazioni).

3. Regolarizzazione e Campionamento (Stocastica)

Introdurre casualità aiuta gli alberi a non dipendere troppo dalle stesse feature o dagli stessi dati.

feature_fraction (Subsampling delle colonne): Frazione di feature scelte casualmente per ogni albero (es. 0.7 - 0.9). Riduce la correlazione tra gli alberi e combatte l'overfitting.

bagging_fraction e bagging_freq: Frazione di dati campionati a ogni iterazione. Va usato insieme a bagging_freq > 0 (che definisce ogni quanti step effettuare il bagging). Accelera il training e migliora la regolarizzazione.

L1 (reg_alpha) e L2 (reg_lambda): Penalizzano i pesi delle foglie. Aggiungono regolarizzazione pura, molto utile se il modello tende a sovradattarsi sui dati di training.

Un ordine di priorità logico per la ricerca dei parametri è:

Fase 1 (Struttura): Ottimizza num_leaves, max_depth e min_data_in_leaf.
Fase 2 (Stocastica): Ottimizza feature_fraction e bagging_fraction.
Fase 3 (Regolarizzazione): Ottimizza reg_alpha (L1) e reg_lambda (L2).
Fase 4 (Fine-tuning): Abbassa il learning_rate e attiva l'Early Stopping per trovare il numero ottimale di alberi.

Le Regole d'Oro per l'HPO in questo scenario
Usa sempre l'Early Stopping: Non cercare il numero esatto di alberi (num_iterations). Imposta un numero alto (es. 1500) e fai fermare il training se la validation loss non migliora per 50-100 round. In questo modo permetti a Optuna di testare learning rate diversi senza preoccuparsi di bruciare tentativi con troppi o pochi alberi.

Usa scale logaritmiche per i parametri di scala: Parametri come learning_rate, reg_alpha (L1) e reg_lambda (L2) coprono ordini di grandezza differenti (da 0.001 a 0.1). Usare una scala logaritmica in Optuna (log=True) esplora lo spazio in modo molto più efficiente.

Occhio al bilanciamento della complessità: num_leaves e max_depth vanno legati mentalmente. Se permetti alberi molto profondi (max_depth=10), rischi un overfitting pazzesco sui tuoi 8 gruppi se non alzi min_data_in_leaf.

Il Search Space Consigliato per Optuna

        'objective': 'multiclass'
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        
        # 1. Complessità dell'albero
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127), # Regola: tieni num_leaves < 2^max_depth
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        
        # 2. Learning Rate e Iterazioni (con Early Stopping esterno)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        
        # 3. Subsampling (Stocastica per regolarizzare)
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1, # Obbligatorio se usi bagging_fraction > 0
        
        # 4. Regolarizzazione L1 e L2
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        
        # 5. Min split gain (taglia i rami deboli)
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0)

Per hpo **La metrica ideale è NDCG (Normalized Discounted Cumulative Gain)**

- *Perché:* Spesso, quando fai un montaggio video, non ti interessa che il modello azzecca la soglia esatta al millesimo, ma ti interessa che **i 10 momenti migliori in assoluto finiscano in cima alla classifica**.
- LightGBM supporta nativamente il ranking (`objective='lambdarank'`). Usare l'NDCG come metrica di Optuna significa dire all'algoritmo di ottimizzare i parametri affinché i momenti più spettacolari siano *sempre* ai primi posti della lista generata per ogni video.

## Configurazione HPO iniziale

Partirei da questi parametri:

| Parametro         | Scelta iniziale                                      |
| ----------------- | ---------------------------------------------------- |
| Obiettivo         | `lambdarank`                                         |
| Metrica           | `ndcg`                                               |
| Cutoff principale | `NDCG@10`                                            |
| Unità di ranking  | Video                                                |
| Validazione       | Grouped LOVO                                         |
| Obiettivo Optuna  | Massimizzare la media di NDCG\@10 sui video held-out |

Terrei anche `NDCG@5` e `NDCG@20` come metriche secondarie, per capire se il modello privilegia davvero i primissimi segmenti oppure produce classifiche buone su un insieme più ampio.
