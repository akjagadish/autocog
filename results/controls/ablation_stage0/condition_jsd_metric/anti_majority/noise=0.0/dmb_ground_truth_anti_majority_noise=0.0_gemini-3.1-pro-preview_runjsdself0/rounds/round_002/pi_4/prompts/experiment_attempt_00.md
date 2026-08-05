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
**Description:** People make decisions by sequentially accumulating evidence from cues in order of their validity. Evidence is weighted by the log-odds of each cue's validity. The accumulation stops and a choice is made as soon as the absolute difference in accumulated evidence between the two options exceeds an internal threshold. A low threshold leads to frugal 'Take The Best' behavior, while a high threshold leads to compensatory 'Weighted Additive' behavior. This unified mechanism accounts for varying degrees of cue integration depending on task context and individual differences.

**Parameters:**
- threshold: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Monotonic transformation of validities to log-odds
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Cues are consulted in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    a, b = stim[0], stim[1]
    score_a = 0.0
    score_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        score_a += weights[j] * a[j]
        score_b += weights[j] * b[j]
        diff = abs(score_a - score_b)
        # Stop if the accumulated difference exceeds the threshold
        # (and ensure we don't stop on a zero difference if threshold is 0)
        if diff >= threshold and diff > 1e-6:
            break

    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People integrate all available evidence by weighting each feature according to its validity. Specifically, they compute a weighted sum of the features for each option, where the weights are the log-odds of the cue validities. This allows for compensatory decision making, where multiple weaker cues can jointly override a single stronger cue. Choice probabilities are generated via a softmax function over these weighted sums, accommodating response noise, along with an independent lapse rate for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Monotonic transformation of validities to log-odds
    # Clip to avoid division by zero or infinite weights
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Calculate weighted additive score for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use 5 features with strictly descending validities. The trials are designed such that the option favored by the highest-validity discriminating cue (the TTB choice) is often opposed by the option that wins on the simple count of cues (the Tallying choice). Additionally, we include trials where Tallying predicts a perfect tie (guessing) while TTB makes a deterministic prediction based on the single highest-validity discriminating feature.

[1] To strongly dissociate Tallying from Take The Best (TTB), we use 4 features with descending validities. We design trials where the highest-validity discriminating cue favors one option, while the simple count of winning features (Tallying) favors the other option. We also include trials where Tallying predicts a perfect tie (guessing) because both options win on an equal number of features, whereas TTB makes a deterministic prediction based on the single highest-validity discriminating feature. This provides a robust quantitative dissociation between the two models.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use 5 features with validities chosen such that the log-odds weight of any single feature is smaller than the combined log-odds weights of the features ranked below it. In the critical trials, the option favored by the highest-validity discriminating cue (the TTB choice) is opposed by an option that has a greater number of lower-validity cues in its favor. Because WADD integrates all evidence via a weighted sum (compensatory), it predicts that the accumulation of weaker cues will override the single best cue. TTB, being non-compensatory, strictly follows the single highest-validity discriminating cue. We also include agreement trials where the two models make the same choice but with different confidence margins.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we configure 5 features with validities such that the log-odds weight of the most valid cue is smaller than the combined log-odds weights of multiple lower-validity cues. In our critical trials, the option favored by the single highest-validity discriminating cue (the TTB choice) is opposed by the option that possesses a greater number of lower-validity cues. Because WADD integrates all evidence via a compensatory weighted sum, it predicts that the accumulation of weaker cues will override the single best cue. TTB, being strictly non-compensatory, will always choose the option favored by the first discriminating cue regardless of the lower-validity cues.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the sequential evidence accumulation model (advocated theory) from the Weighted Additive model (WADD, competing theory), we utilize a 5-feature design where the validities are spread such that the first cue is highly valid, but its log-odds weight is smaller than the sum of the remaining cues' weights. In critical trials, one option is favored by the single best cue, while the other option is favored by all remaining cues. The advocated theory, operating with a moderate-to-low evidence threshold, will stop accumulating evidence after the first cue and choose the option it favors. In contrast, WADD integrates all cues without a stopping rule and will thus choose the option favored by the aggregate of the weaker cues. We also include agreement trials where both early stopping and full integration lead to the same choice, providing a robust baseline to estimate noise and threshold parameters.",
  "validities": [
    0.88,
    0.82,
    0.76,
    0.7,
    0.65
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
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
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
      0,
      1,
      1,
      1
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
      1,
      0,
      1,
      1
    ],
    [
      1,
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
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
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
      1,
      1
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
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2838,
  "candidates_token_count": 599,
  "total_token_count": 6096
}
```
