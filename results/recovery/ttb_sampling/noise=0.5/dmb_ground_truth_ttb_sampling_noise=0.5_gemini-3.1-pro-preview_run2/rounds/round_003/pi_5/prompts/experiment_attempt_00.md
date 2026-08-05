# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Exponentially Weighted Additive Model: Subjects integrate all available features in a compensatory manner, but they apply a steep non-linear (exponential) transformation to the feature validities. This causes the most valid features to heavily dominate the decision, effectively mimicking the non-compensatory 'Take The Best' heuristic while remaining mathematically compensatory. The steepness of this transformation dictates how closely the strategy approximates strict lexicographic choice.

**Parameters:**
- gamma: [0.5, 20.0]
- beta: [0.05, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Steep non-linear transformation of validities to weights
    weights = np.exp(gamma * validities)
    
    # Calculate option scores as weighted sums
    scores = stim @ weights
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design quantitatively dissociates Tallying (which counts strict feature-wise wins irrespective of validities) from WADD (which computes a validity-weighted sum). The selected validities include a combination where the sum of three lower validities exactly equals the sum of two higher validities (0.55 + 0.55 + 0.50 = 1.60; 0.95 + 0.65 = 1.60), allowing trials where WADD predicts a perfect tie while Tallying strongly favors the option with more winning features. Other trials pit a small number of highly valid features against a larger number of less valid features, causing Tallying and WADD to predict opposite choices.

[1] This design quantitatively dissociates WADD (Weighted Additive) from Tallying using 5 features with carefully chosen validities. Tallying simply counts the number of winning features, ignoring their validities, whereas WADD computes a validity-weighted sum. By pitting a small number of high-validity features against a larger number of low-validity features, we create clear dissociations. For example, in the first trial type, Option A wins on 2 high-validity features while Option B wins on 3 low-validity features; Tallying favors B (3 > 2) whereas WADD strongly favors A (due to the high weights). Other trials include cases where Tallying predicts a perfect tie but WADD strongly prefers one option, as well as congruent trials to ensure task engagement.

[2] This design aims to quantitatively dissociate the Take The Best (TTB) heuristic from the Weighted Additive (WADD) rule. TTB is a non-compensatory strategy that makes decisions based solely on the single most valid discriminating feature, completely ignoring all other features. In contrast, WADD is a compensatory strategy that integrates all features weighted by their validities, allowing multiple lower-validity features to outweigh a single high-validity feature. The chosen validities are [0.9, 0.8, 0.7, 0.6, 0.5]. The trial pairs are specifically constructed so that the option favored by the highest-validity discriminating feature (which TTB will choose) is always opposed by a larger number of lower-validity features that collectively sum to a higher weight (which WADD will choose). By examining the choice proportions on these specific trials, we can cleanly separate the use of lexicographic (TTB) versus compensatory (WADD) decision strategies.

[3] To quantitatively dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, we use a set of validities where the highest validity feature can be outweighed by the sum of several lower-validity features. The trial pairs are constructed such that TTB always stops at the first discriminating feature (ordered by validity) and chooses the option with that feature. In contrast, WADD integrates all features, and the trials are designed so that the option with a greater number of lower-validity features consistently yields a higher weighted sum than the option possessing only the single highest-validity discriminating feature. This creates a stark divergence in the predicted choices between the two models across multiple levels of feature comparisons.

[4] This design quantitatively dissociates Take The Best (TTB) from Bounded Sequential Feature Sampling. TTB evaluates features in descending order of validity, stops at the first discriminating feature, and assigns a constant internal score difference (1 vs 0) regardless of the remaining features, with behavior softened by a wide range of response noise (beta in [0.01, 5.0]). The competing Bounded Sequential Feature Sampling accumulates unweighted evidence across features until a threshold (1, 2, or 3) is met, and operates with much lower response noise (beta in [5.0, 20.0]). By pitting the first discriminating feature against varying amounts of opposing evidence from subsequent features, we create trials where the competing model's accumulated evidence E systematically varies (e.g., E = -2, -1, 0, 1, 2). Consequently, the competing model is forced to predict either extreme deterministic choices (when |E| >= 1) or perfect ties (when E = 0). In contrast, TTB consistently favors the option with the best discriminating feature and can accurately capture moderate choice probabilities (e.g., 75%) across all these trial types due to its unconstrained noise parameter.

[5] To quantitatively dissociate Bounded Sequential Feature Sampling (BSFS) from Take The Best (TTB), we exploit their different stopping rules. TTB always stops at the first discriminating feature, ignoring all subsequent information. BSFS, on the other hand, accumulates unweighted evidence across features until a decision threshold (1, 2, or 3) is reached. By creating trials where the highest-validity feature points to Option A, but multiple subsequent features point to Option B, BSFS (if its threshold is > 1) will continue sampling, eventually overturning the initial evidence and choosing B. TTB will invariably choose A on these same trials. We vary the amount of opposing evidence to distinguish between different possible thresholds in the BSFS model.

[6] To quantitatively dissociate the Take The Best (TTB) heuristic from the Exponentially Weighted Additive Model, this design exploits the upper bound on the Exponential model's 'gamma' parameter (max 20.0). By choosing feature validities that are extremely close to one another, the exponential transformation is mathematically unable to make the single highest-validity feature outweigh the sum of the remaining features. Consequently, the Exponential model is forced to make compensatory predictions on critical trials. In contrast, TTB strictly follows the lexicographic order and will consistently choose the option favored by the single highest-validity discriminating feature, regardless of how many lower-validity features oppose it.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Exponentially Weighted Additive Model from the Take The Best (TTB) heuristic by exploiting the Exponential model's ability to act in a compensatory manner when its gamma parameter is low. TTB is strictly non-compensatory: it will always choose the option favored by the single highest-validity discriminating feature, regardless of how many lower-validity features oppose it. By constructing trials where the highest-validity feature points to Option A, but multiple subsequent features point to Option B, we create a situation where TTB rigidly predicts a preference for A. The Exponential model, however, can predict a preference for B (capturing compensatory human behavior) or A (mimicking non-compensatory behavior), depending on the steepness of its exponential weighting. A gradient of such opposing evidence is used to fully map out this divergence.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3275,
  "candidates_token_count": 521,
  "total_token_count": 5165
}
```
