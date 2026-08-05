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
**Description:** Take-The-Best (TTB) proposes that individuals use a non-compensatory lexicographic heuristic for decision making. They search through features in descending order of their validity and base their choice entirely on the first feature that discriminates between the two options, ignoring all remaining features. If no feature discriminates, they guess. Response noise is incorporated via a softmax function over the resulting binary scores and a lapse rate for random errors.

**Parameters:**
- beta: [0.001, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tallying (Equal Weights) heuristic: Decision makers ignore cue validities entirely and simply count the number of positive features for each option. They choose the option with the higher tally (i.e., the greatest number of positive cues). This represents a fast-and-frugal compensatory strategy where all cues are weighted equally, minimizing cognitive effort while still integrating all available features. Response noise is incorporated via a softmax function over the tallies, alongside a uniform lapse rate for random errors.

**Parameters:**
- beta: [0.001, 10.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    scores = np.array([np.sum(a), np.sum(b)], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design systematically contrasts Tallying (which counts the unweighted number of features favoring each option) with WADD (which weights each feature by its validity). By using validities of [0.9, 0.8, 0.6, 0.5, 0.5], we can construct adversarial trials where one option is favored by a majority of low-validity features (thus winning the Tally) but is outweighed by a minority of high-validity features (thus losing the WADD score). We also include trials where Tallying predicts a tie (equal number of winning features) while WADD predicts a strict preference, uniquely isolating the compensatory weighting effect.

[1] This design isolates the compensatory weighting mechanism of WADD from the unweighted feature-counting of Tallying. By employing validities that are highly skewed (two very high, three very low), we create trials where an option favored by a majority of low-validity features wins the Tallying count but loses the WADD score. We also include trials where Tallying predicts a perfect tie (equal number of winning features) while WADD predicts a strict preference based on the validities of the specific features won, uniquely dissociating WADD's cardinal weighting from Tallying's ordinal counting.

[2] To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) model, we exploit TTB's strict non-compensatory nature versus WADD's compensatory weighting. We use five features with linearly decreasing validities. The critical trials are designed such that the single highest-validity discriminating feature favors one option (which TTB will categorically choose), while the sum of multiple lower-validity discriminating features strongly favors the other option (which WADD will choose, provided the cumulative weight overcomes the single best cue). We also include trials where both models agree to serve as a baseline for response noise.

[3] To quantitatively dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take-The-Best (TTB) heuristic, we use a 5-feature design with a steep but surmountable validity gradient. The critical trials pit the single highest-validity feature against a coalition of multiple lower-validity features. In these adversarial trials, TTB's lexicographic stopping rule forces it to choose the option favored by the single best discriminating feature, ignoring the rest. WADD, however, integrates all features and will choose the opposite option because the sum of the lower-validity features outweighs the single best feature. We also include congruent trials where both models agree, and partially conflicting trials where the coalition is not strong enough to overcome the best feature, providing a precise measure of the compensatory threshold and response noise.

[4] To dissociate Take-The-Best (TTB) from the Two-Stage Heuristic, we exploit the latter's fallback mechanism which triggers Tallying when the top two discriminating cues have a validity difference smaller than delta_v (up to 0.1). We set the two highest validities to be very close (0.90 and 0.88). In critical trials, these two top cues conflict, and the remaining lower-validity cues heavily favor the option supported by the slightly weaker cue. TTB strictly follows the single best cue, whereas the Two-Stage model detects the small validity difference, falls back to tallying, and chooses the option with the greater total number of winning features.

[5] To dissociate the Two-Stage Heuristic from Take-The-Best (TTB), we exploit the Two-Stage model's fallback mechanism, which switches to tallying when the validity difference between the top two discriminating cues is less than delta_v (which ranges up to 0.1). We use validities [0.92, 0.88, 0.86, 0.60, 0.50], creating small differences between the top cues (0.04 and 0.02). In critical trials, the highest validity discriminating cue favors Option A, but the majority of remaining features favor Option B. TTB strictly chooses Option A based on the single best cue. In contrast, the Two-Stage Heuristic detects the small validity gap, falls back to tallying, and chooses Option B due to the higher number of winning features.

[6] This design systematically contrasts the non-compensatory Take-The-Best (TTB) model with the fully compensatory Weighted Additive (WADD) model using log-odds weights. By setting validities to [0.95, 0.85, 0.80, 0.75, 0.65], we create a steep but surmountable gradient in log-odds weights. TTB will always choose the option favored by the single highest-validity discriminating feature. WADD, however, integrates all features, allowing a coalition of lower-validity features to outweigh a single higher-validity feature. The trials include extreme conflicts (where the best feature is pitted against all others), moderate conflicts (where the best feature is pitted against a subset of lower features), and congruent trials where both models agree. This provides a robust quantitative dissociation of the two models.

[7] To quantitatively dissociate the Weighted Additive (WADD) model from Take-The-Best (TTB), this design uses five features with validities that produce a specific profile of log-odds weights. TTB makes decisions based solely on the highest-validity discriminating feature, entirely ignoring the rest. WADD, using log-odds transformations of the validities, integrates all information. By carefully selecting the validities and constructing trials where the single best feature favors one option but a coalition of lower-validity features favors the other, we can create strong adversarial conditions. For example, WADD's compensatory nature allows the sum of the second, third, and fourth features to outweigh the first feature, whereas TTB will strictly follow the first feature. We also include trials where the models agree to help constrain the estimates of decision noise.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design systematically contrasts Take-The-Best (TTB) with Tallying (Equal Weights). TTB relies on a lexicographic search based on cue validities, completely ignoring the total number of positive features if a high-validity cue discriminates. Tallying, conversely, ignores cue validities entirely and simply sums the positive features. We use five features with a clear validity gradient. Critical trials pit the single highest-validity discriminating feature against the sheer number of remaining lower-validity features. In these adversarial trials, TTB chooses the option favored by the top discriminating cue, while Tallying chooses the option with the higher total count of positive features. We also include trials where Tallying predicts a tie but TTB does not, and trials where both models agree, to robustly estimate response noise and lapse rates.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
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
      1,
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
      0,
      1,
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
      1,
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
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
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
  "prompt_token_count": 3368,
  "candidates_token_count": 523,
  "total_token_count": 5127
}
```
