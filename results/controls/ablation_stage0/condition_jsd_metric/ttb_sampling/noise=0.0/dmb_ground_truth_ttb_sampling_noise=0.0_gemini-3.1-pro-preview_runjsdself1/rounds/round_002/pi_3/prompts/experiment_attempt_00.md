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
**Description:** Take-The-Best (TTB) heuristic: Individuals make decisions by sequentially searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the choice is based entirely on that single feature, ignoring all others. If no feature discriminates, a random guess is made.

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
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            scores = np.array([0.0, 1.0])
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
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
**Description:** Two-Stage Heuristic (TTB with Tallying Fallback): Decision-makers initially employ a non-compensatory Take-The-Best (TTB) strategy, searching sequentially through features in descending order of validity. However, they only trust features whose validity exceeds a certain subjective threshold. If a discriminating feature is found above this threshold, the choice is based entirely on it. If no such feature exists (either because all high-validity features are tied or none meet the threshold), the decision-maker abandons the sequential search and falls back to a compensatory Tallying strategy, weighing all features equally and choosing the option with the most winning features.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = None
    # Stage 1: Take-The-Best for features above the validity threshold
    for idx in order:
        if validities[idx] >= threshold:
            if a[idx] > b[idx]:
                scores = np.array([1.0, 0.0])
                break
            elif b[idx] > a[idx]:
                scores = np.array([0.0, 1.0])
                break
        else:
            # Stop searching if we hit features below the threshold
            break
            
    # Stage 2: Fallback to Tallying if no high-validity cue discriminated
    if scores is None:
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
[0] To quantitatively dissociate Tallying from WADD, we manipulate the distribution of feature validities. By using a steep validity gradient (two very high-validity features and three lower-validity features), we can construct trials where one option possesses fewer but highly valid features, while the other option possesses a greater number of lower-validity features. Tallying will strictly prefer the option with the higher count of winning features, whereas WADD will prefer the option with the higher validity sum. Additionally, we include trials where Tallying results in a tie (predicting random guessing) but WADD strongly prefers one option.

[1] To quantitatively dissociate WADD from Tallying, we use a 5-feature design with two highly valid features and three features of low validity. This enables the creation of 'compensatory crossover' trials: Option A wins on the two high-validity features, while Option B wins on the three low-validity features. Tallying purely counts the number of winning features, strictly preferring Option B (3 wins vs 2 wins). WADD computes a validity-weighted sum, strictly preferring Option A because the combined weight of the two highly valid features exceeds the sum of the three low-validity ones. We also include trials where Tallying results in a tie (e.g., 1 win vs 1 win) but WADD strongly prefers the option with the higher-validity feature, further separating the models' predictions.

[2] To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive rule (WADD), we utilize a 4-feature design with a linear descending validity gradient. Crucially, we introduce 'compensatory' trials where the single highest-validity feature favors one option, but all remaining lower-validity features favor the competing option. TTB relies solely on the first discriminating feature (the highest-validity one) and will thus reliably choose the first option. In contrast, WADD integrates all features; because the combined weighted sum of the multiple lower-validity features exceeds the single highest-validity feature, WADD will choose the second option. We also include trials where both models agree, and trials where the highest validities are tied to test secondary feature evaluation.

[3] To quantitatively dissociate the Weighted Additive rule (WADD) from Take-The-Best (TTB), we use a 5-feature design with a linearly descending validity gradient. We construct a series of 'compensatory' trials where Option A always possesses the single most valid discriminating feature, but Option B possesses all the remaining, less valid features. Because TTB stops searching at the first discriminating feature (ordered by validity), it will consistently choose Option A across these trials. In contrast, WADD computes a validity-weighted sum over all features. By ensuring that the sum of the validities of the features favoring Option B strictly exceeds the validity of the single feature favoring Option A, WADD will consistently choose Option B. This provides a stark, qualitative dissociation between the one-reason stopping rule of TTB and the fully compensatory integration of WADD.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Take-The-Best (TTB) from the Two-Stage Heuristic (TTB with Tallying Fallback), we must exploit the latter's 'threshold' parameter. The Two-Stage model behaves exactly like TTB as long as the first discriminating feature's validity is above a certain subjective threshold; however, if all features above the threshold are tied, it abandons sequential search and falls back to unweighted Tallying. By systematically varying which feature is the most valid discriminating one (while tying the features above it), and ensuring that the overall Tallying count strongly favors the opposite option, we can pinpoint the threshold. TTB will consistently follow the highest valid discriminating feature regardless of its absolute validity, while the Two-Stage model will suddenly switch its preference to the option with more winning features once the discriminating validity drops below its threshold.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
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
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
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
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
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
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2993,
  "candidates_token_count": 584,
  "total_token_count": 6236
}
```
