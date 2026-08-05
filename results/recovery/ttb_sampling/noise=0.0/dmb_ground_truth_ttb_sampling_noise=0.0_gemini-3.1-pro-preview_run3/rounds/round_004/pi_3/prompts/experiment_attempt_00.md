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
**Description:** Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

**Parameters:**
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
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers primarily rely on the deterministic, non-compensatory Take-The-Best heuristic, sequentially checking features ordered by validity and stopping at the first discriminating feature. A small subset of decisions relies on a compensatory Tallying strategy. By restricting the tallying mixture weight to a low range, the model maintains overwhelming TTB dominance, capturing high TTB adherence and the lack of a strong compensatory effect from the number of favoring features, while still allowing structured deviations from pure TTB.

**Parameters:**
- w_tally: [0.0, 0.15]
- epsilon: [0.0, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # TTB Strategy
    order = np.argsort(validities)[::-1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Strategy
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tal = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tal = np.array([0.0, 1.0])
    else:
        p_tal = np.array([0.5, 0.5])
        
    # Mixture of TTB and Tallying
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tal
    
    # Uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To strongly dissociate Tallying (which counts unweighted feature-wise wins) from WADD (which computes a validity-weighted sum), we use a 6-feature design with a steep drop-off in validities. In several critical trials, Option A wins on a larger number of low-validity features (e.g., 3 wins), while Option B wins on fewer but much higher-validity features (e.g., 2 wins). Tallying strictly predicts Option A because it ignores cardinal validities and simply counts wins, whereas WADD predicts Option B because the sum of the validities for B's winning features exceeds A's. We also include agreement trials where both models predict the same option, ensuring the experiment covers a range of trade-offs.

[1] To strongly dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities. In critical trials, one option wins on a larger number of low-validity features (e.g., 3 wins out of 5), while the other option wins on fewer but much higher-validity features (e.g., 2 wins out of 5). Tallying strictly predicts the option with more wins because it ignores cardinal validities, whereas WADD predicts the option with fewer wins because the sum of the validities for its winning features is greater. We also include agreement trials where both models predict the same option, to ensure a broad coverage of the decision space.

[2] This design strictly dissociates Tallying from WADD. Tallying relies exclusively on the number of features where one option beats the other, completely ignoring cardinal validities. WADD, in contrast, computes a validity-weighted sum. By using a 5-feature design with a steep drop-off in validities (two highly valid features and three lower-validity features), we can construct critical trials where Option A wins on more features (e.g., 3 wins vs 2 wins) but Option B wins on the higher-validity features. In these cases, Tallying predicts Option A while WADD predicts Option B. We also include agreement trials and ties to thoroughly span the decision space and ensure robust model recovery.

[3] To strongly dissociate WADD from Tallying, we use a 5-feature design with a sharp drop-off in validities between the top two and the bottom three features. Tallying simply counts the number of features on which an option wins, while WADD computes a validity-weighted sum. By pitting the two high-validity features against the three low-validity features, we create critical trials where Option A wins on fewer but highly valid features, leading WADD to predict Option A while Tallying (counting more wins for Option B) predicts Option B. We also include trials where both models agree, as well as trials where Tallying produces a tie but WADD has a strong preference, ensuring robust parameter recovery and model dissociation.

[4] To strongly dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) rule, we construct a 5-feature design with a clear validity hierarchy. TTB strictly relies on the most valid discriminating feature, completely ignoring the rest of the feature profile. WADD, on the other hand, computes a compensatory sum, where multiple lower-validity features can outweigh a single higher-validity feature. The critical trials pit the highest-validity feature against a coalition of lower-validity features (e.g., Option A wins on the single best feature, but Option B wins on the remaining four). In these compensatory trials, TTB deterministically predicts Option A, while WADD predicts Option B. We also include agreement trials and trials where the highest-validity feature is tied, forcing TTB to drop down to the second most valid feature, ensuring a robust test of the lexicographic stopping rule against compensatory integration.

[5] To strongly dissociate the Weighted Additive (WADD) rule from Take-The-Best (TTB), we use a 5-feature design with a linearly spaced set of validities. TTB strictly relies on the most valid discriminating feature and ignores the rest. WADD computes a compensatory sum (weighted by validity^gamma), allowing multiple lower-validity features to outweigh a single higher-validity feature, depending on the gamma parameter. We construct critical trials where one option wins on the single most valid discriminating feature (favored by TTB), while the other option wins on all or most of the remaining lower-validity features (favored by WADD for low-to-moderate gamma values). We also include trials where the best feature is tied, forcing TTB to drop down to the second-best feature, while WADD integrates across the entire profile. Agreement trials are included to balance the design and ensure robust parameter recovery.

[6] To strongly dissociate Take-The-Best (TTB) from Probabilistic Take-The-Best (PTTB), we exploit their differing sensitivity to multiple lower-validity discriminating features. TTB deterministically selects the option favored by the single highest-validity discriminating feature, completely ignoring all other features. PTTB, however, stochastically samples from all discriminating features with probabilities proportional to an exponential function of their validities. By creating critical trials where Option A is favored by the single most valid discriminating feature, but Option B is favored by several slightly less valid features, TTB strongly predicts Option A. In contrast, PTTB's choice probabilities will be much more graded and can even favor Option B if the sum of the exponential weights of B's features exceeds A's (for low to moderate gamma values). We closely space the validities to maximize PTTB's sensitivity to lower-validity features, ensuring a stark quantitative dissociation from TTB's deterministic, single-feature focus.

[7] To strongly dissociate Probabilistic Take-The-Best (PTTB) from deterministic Take-The-Best (TTB), we use a 5-feature design with closely spaced validities. TTB strictly chooses the option favored by the single most valid discriminating feature, completely ignoring all other features. In contrast, PTTB stochastically selects a discriminating feature with a probability proportional to an exponential function of its validity. By pitting the single best feature against a coalition of several slightly less valid features, TTB predicts a deterministic choice for the option with the best feature, while PTTB's choice probabilities will be much more graded and can even favor the opposing option (for low to moderate gamma values) because the sum of the exponential weights of the opposing features exceeds that of the single best feature.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To strongly dissociate pure Take-The-Best (TTB) from a mixture of TTB and Tallying, we use a 5-feature design and manipulate the alignment between the single best discriminating feature and the total number of favoring features. Pure TTB predicts a constant choice probability for the option favored by the highest-validity discriminating feature, regardless of how many total features favor it. The competing mixture model, however, predicts a slightly lower choice probability when Tallying contradicts TTB, and a slightly higher probability when Tallying aligns with TTB. By including trials where TTB and Tallying strongly conflict (e.g., one option wins on the best feature, the other wins on the remaining four), trials where they agree, and trials where Tallying is tied, we can detect the subtle graded choice probabilities predicted by the mixture model that pure TTB cannot capture.",
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
      0,
      1,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3589,
  "candidates_token_count": 529,
  "total_token_count": 5405
}
```
