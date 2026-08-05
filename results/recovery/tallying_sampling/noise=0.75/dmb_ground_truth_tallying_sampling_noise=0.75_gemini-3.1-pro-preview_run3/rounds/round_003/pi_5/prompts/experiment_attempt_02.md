# experiment_attempt_02

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
**Description:** Weighted Additive (WADD) Decision Theory with Flexible Sigmoid Subjective Validity Transformation

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.1, 5.0]
- delta: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    
    # Transform raw probabilities into subjective weights using a sigmoid function
    w = 1.0 / (1.0 + np.exp(-gamma * (validities - delta)))
    
    # Calculate the overall score for each option by multiplying cue values by subjective weights
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice based on the weighted scores
    z = beta * scores
    z = z - np.max(z)  # for numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate for uniform guessing
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

**Parameters:**
- beta: [0.01, 1.0]
- epsilon: [0.3, 0.8]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To robustly dissociate Take The Best (TTB) from the Weighted Additive Rule (WADD), we must exploit their fundamental difference: TTB is non-compensatory (a single high-validity cue dictates the choice), whereas WADD is compensatory (multiple lower-validity cues can outweigh a single high-validity cue). We use 5 features with linearly decreasing validities. The design includes 'compensatory' trials where the option favored by the single best discriminating cue is opposed by all (or most) remaining lower-validity cues. In these trials, TTB strictly chooses the option with the best cue, while WADD chooses the option supported by the sum of the lower cues. We also include 'non-compensatory' (agreement) trials as a baseline to ensure subjects are paying attention and to constrain noise parameters.

[1] To robustly dissociate the Weighted Additive (WADD) rule from Take The Best (TTB), we use four features with descending validities. We design 'compensatory' trials where the option favored by the single highest discriminating cue (which TTB strictly follows) is opposed by the sum of all remaining lower-validity cues (which WADD favors). By varying which cue is the highest discriminating one (e.g., the first vs. the second cue), we ensure that TTB's non-compensatory stopping rule is tested at multiple levels of the cue hierarchy. We also include agreement trials to establish baseline performance and constrain noise parameters.

[2] To robustly dissociate Tallying (which simply counts the number of positive features, ignoring validities) from the Weighted Additive rule (WADD, which weights each feature by its validity), we use 5 features with a steep drop-off in validities. The critical dissociation trials pit an option with a greater number of low-validity features (favored by Tallying) against an option with fewer, but much higher-validity features (favored by WADD). We also include tie-breaking trials where Tallying sees equal feature counts but WADD has a strict preference, as well as agreement trials to anchor the baseline and constrain noise parameters.

[3] To robustly dissociate the Weighted Additive (WADD) rule from the Tallying heuristic, we employ a 5-feature design with a steep validity curve. Tallying simply counts the number of positive features, whereas WADD weights each feature by its validity. The critical dissociation trials pit an option with fewer but high-validity features (favored by WADD) against an option with more but low-validity features (favored by Tallying). Additionally, we include trials where Tallying predicts a tie (equal number of positive features) but WADD has a strict preference, to further highlight the use of validity weights.

[4] To robustly dissociate Tallying from Take-The-Best (TTB), we must pit the sheer quantity of positive features against the highest-validity discriminating feature. Tallying ignores validities and simply counts the number of 1s, while TTB searches through features strictly in order of validity and stops at the first difference. The trials are designed such that one option has a clear advantage in the total number of positive features (favored by Tallying), but the other option possesses the single highest-validity feature that discriminates between the two (favored by TTB). We also include tie-breaking trials where Tallying sees an equal count but TTB has a strict preference, to further highlight the difference in their decision rules.

[5] To robustly dissociate Take-The-Best (TTB) from Tallying, we pit the single highest-validity discriminating feature against the overall sum of positive features. TTB searches through features in descending order of validity and stops at the first difference, completely ignoring the remaining features. Tallying ignores validities and simply counts the total number of positive features. The critical dissociation trials are designed such that one option is favored by the highest-validity discriminating feature (chosen by TTB), while the other option possesses a greater total number of lower-validity positive features (chosen by Tallying). We also include trials where Tallying predicts a tie (equal number of positive features) but TTB has a strict preference, as well as agreement trials to serve as a baseline.

[6] To effectively dissociate the Tallying heuristic from the Weighted Additive (WADD) model with a flexible sigmoid transformation, we use a 5-feature design with a linear spread of validities. Tallying simply counts the total number of positive features, whereas WADD weights them according to their subjective validities. Critical trials pit an option with fewer, but high-validity features (favored by WADD) against an option with more, but lower-validity features (favored by Tallying). Additionally, we include trials where Tallying predicts a tie (equal number of positive features) but WADD has a strong preference, to expose the validity-weighting mechanism.

[7] To effectively dissociate the Weighted Additive (WADD) model with flexible sigmoid transformation from the Tallying heuristic, we use a 5-feature design with a steep drop-off in validities. Tallying simply counts the positive features, ignoring their validities, whereas WADD weights them according to subjective validity. Critical trials pit an option with fewer but high-validity features against an option with more but lower-validity features. Additionally, we include trials where Tallying predicts a tie (equal number of positive features) but WADD has a strong preference, to expose the validity-weighting mechanism.

[8] To robustly dissociate the Weighted Additive (WADD) model with a flexible sigmoid transformation from the Tallying heuristic, we employ a 5-feature design with a linear spread of validities. Tallying simply counts the total number of positive features, completely ignoring validities. WADD, however, weights features by their subjective validities. By using a sigmoid transformation, WADD can heavily emphasize the top validities while suppressing the lower ones. The critical dissociation trials pit an option with fewer but highest-validity features (favored by WADD) against an option with more but lower-validity features (favored by Tallying). We also include trials where Tallying predicts a tie (equal number of positive features) but WADD has a strong preference, to expose the validity-weighting mechanism.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To robustly dissociate the Weighted Additive (WADD) model with a flexible sigmoid transformation from the Tallying heuristic, we employ a 6-feature design with validities spanning from 0.95 down to 0.52. Tallying simply counts the total number of positive features, whereas WADD weights them by subjective validities. By using 6 features, we can create stark contrasts: in critical trials, one option has only 2 high-validity features while the other has 3 or 4 low-validity features. Tallying strongly prefers the latter, while WADD (especially with a sigmoid that amplifies high validities) prefers the former. Tie trials for Tallying (equal counts) where WADD has a strong preference further separate the models.",
  "validities": [
    0.95,
    0.88,
    0.79,
    0.7,
    0.61,
    0.52
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
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
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0
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
      0,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3286,
  "candidates_token_count": 491,
  "total_token_count": 4477
}
```
