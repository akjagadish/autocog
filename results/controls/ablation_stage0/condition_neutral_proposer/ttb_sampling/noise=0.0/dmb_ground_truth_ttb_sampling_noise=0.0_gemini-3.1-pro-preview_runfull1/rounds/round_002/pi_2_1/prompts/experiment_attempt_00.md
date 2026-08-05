# experiment_attempt_00

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Use the log-odds of objective validities as the subjective weights
    validities = np.asarray(parameters["validities"], dtype=float)
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)  # Prevent division by zero or log(0)
    weights = np.log(validities / (1.0 - validities))
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## THEORY 2
**Description:** Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To distinguish Tallying from WADD, we need pairs where the heuristic that counts feature wins (Tallying) diverges from the heuristic that weights features by their validities (WADD). By using a skewed set of validities, we can create 'compensatory' trials where a small number of highly valid features outweighs a larger number of less valid features. Tallying will consistently prefer the option with more winning features, ignoring validities, while WADD will prefer the option with the higher weighted sum. We also include trials where Tallying produces a tie (equal number of wins) but WADD has a strong preference, which further isolates the effect of cardinal weighting.

[1] To maximally distinguish WADD from Tallying, we exploit the constraints of the [0.5, 1.0] validities range. Tallying purely counts strict feature-level wins, ignoring validities. WADD computes a weighted sum. By setting two very high validities (e.g., 0.99, 0.95) and three low validities (e.g., 0.55, 0.52, 0.50), we can construct trials where Option A wins on 3 low-validity features but Option B wins on the 2 high-validity features. Tallying will prefer Option A (3 vs 2 wins), whereas WADD will prefer Option B (since the sum of the two high validities exceeds the sum of the three low validities). We also include tie-breaking trials where the number of wins is equal, leaving Tallying to guess indiscriminately, while WADD shows a strong preference based on validities.

[2] This design carefully selects validities to produce specific log-odds weights (~2.0 for the first feature, ~1.0 for the next two, and ~0.5 for the last three). This allows us to construct trials that precisely dissociate Tallying (which simply counts feature wins) from WADD (which weights by log-odds of validities). The trials include pure reversals (where one heuristic strongly prefers A and the other B), trials where Tallying predicts a perfect tie while WADD has a strong preference, and trials where WADD predicts a perfect tie while Tallying has a clear preference.

[3] By carefully selecting validities, we engineer log-odds weights that allow for precise dissociations between WADD (which sums log-odds weights) and Tallying (which counts strict feature wins). We use validities of [0.90, 0.80, 0.70, 0.60, 0.5333], which yield approximate weights of 2.20, 1.39, 0.85, 0.41, and 0.13. Crucially, the sum of the last three weights (1.39) perfectly equals the second weight. This allows us to create trials where WADD is indifferent but Tallying has a strong preference (3 wins vs 1 win). The design also includes pure reversals where Tallying prefers options with more wins on weak features, but WADD prefers options with fewer wins on strong features, as well as trials where Tallying is indifferent (equal wins) but WADD has a strong preference.

[4] To maximally distinguish Take The Best (TTB) from Weighted Additive (WADD), we exploit their fundamental difference: TTB is non-compensatory and relies solely on the first discriminating feature, whereas WADD computes a compensatory weighted sum based on log-odds of validities. We use a set of validities where the highest validity feature is strong, but the sum of the log-odds of the remaining features exceeds it. By pitting the most valid feature against a combination of less valid features, we create pure reversals where TTB deterministically chooses the option favored by the single best cue, while WADD chooses the option with multiple smaller advantages.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To maximally distinguish Take The Best (TTB) from Weighted Additive (WADD), we exploit the compensatory nature of WADD versus the non-compensatory stopping rule of TTB. We define a set of 5 validities such that the subjective weight (log-odds) of any single feature is outweighed by the sum of the weights of the features below it. We construct trials that systematically pit the most valid discriminating feature against all remaining less valid features. This creates pure reversals at various depths of the TTB search process: trials where TTB stops at the first feature but WADD chooses the opposite option based on the sum of features 2, 3, and 4; trials where the first feature is tied so TTB stops at the second feature, but WADD prefers the sum of features 3, 4, and 5; and so on.",
  "validities": [
    0.85,
    0.75,
    0.7,
    0.65,
    0.6
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
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
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
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3128,
  "candidates_token_count": 461,
  "total_token_count": 5603
}
```
