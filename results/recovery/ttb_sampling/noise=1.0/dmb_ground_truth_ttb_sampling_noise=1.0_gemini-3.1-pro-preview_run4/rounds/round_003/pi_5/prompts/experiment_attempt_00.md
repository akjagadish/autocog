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
**Description:** Random Feature Heuristic (Minimalist): Subjects do not integrate multiple features or weight them by validity. Instead, on each trial, they randomly sample exactly one feature uniformly at random. They choose the option with the higher value on that sampled feature. If the options tie on that feature, they guess randomly. This acts as a highly noisy but boundedly rational alternative to pure guessing, predicting choices that slightly favor the option with more positive features overall. A lapse rate (epsilon) accounts for additional purely random choices.

**Parameters:**
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # The marginal probability of choosing A when picking one feature at random:
    # P(A) = 1/n * sum(a_i > b_i) + 0.5/n * sum(a_i == b_i)
    # For binary features, this simplifies exactly to 0.5 + 0.5 * (sum(a) - sum(b)) / n
    diff = np.sum(a) - np.sum(b)
    p_a_core = 0.5 + 0.5 * diff / n_features
    p_b_core = 1.0 - p_a_core
    
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b_core + epsilon * 0.5
    
    return np.array([p_a, p_b])
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
**Description:** Random Guessing: Due to task complexity, lack of trial-by-trial correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial.

**Parameters:**
(none)

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with Weighted Additive (WADD) (which scales feature magnitudes by their validities). To create a quantitative dissociation, we use a set of 6 features where a small number of highly valid features can outweigh a larger number of less valid features. In critical trials, Option A is superior on many low-validity features, leading Tallying to predict A, while Option B is superior on fewer but highly valid features, leading WADD to predict B. Other trials present ties for Tallying where WADD has a clear preference, or scenarios where both heuristics agree but with differing confidence levels.

[1] This design quantitatively dissociates Weighted Additive (WADD) from Tallying by manipulating the distribution of feature validities. We use 5 features with a steep drop-off in validity: two highly valid features and three weakly valid features. In critical trials, one option wins on the two highly valid features (strongly preferred by WADD) while the other option wins on the three weakly valid features (preferred by Tallying, as it simply counts wins). We also include trials where Tallying predicts indifference (equal number of wins) but WADD strongly prefers one option due to validity differences, and baseline trials where both models agree.

[2] This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive (WADD) model by exploiting the non-compensatory nature of TTB versus the compensatory nature of WADD. We use 5 features with linearly decreasing validities. In the critical trials, one option is favored by the single most valid discriminating feature, while the other option is favored by a larger number of less valid features whose combined weight exceeds that of the single best feature. TTB will strictly follow the single highest-validity cue that discriminates between the options, ignoring all others. Conversely, WADD integrates all available information and will choose the option with the higher weighted sum of features, consistently favoring the option with multiple lower-validity cues.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic. We use 5 features with linearly decreasing validities. In the critical trials, one option is favored by the single most valid discriminating feature (which TTB relies on exclusively), while the other option is favored by a larger number of less valid features whose combined weight exceeds that of the single best feature. Because WADD integrates all available information, it will consistently favor the option with multiple lower-validity cues, directly opposing TTB's choices. We also include trials where the best discriminating feature is tied, forcing TTB to look at the second or third best feature, while WADD still integrates the entire profile.

[4] This design aims to dissociate Random Guessing from the Weighted Additive (WADD) model. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the feature values. WADD, in contrast, predicts that choice probabilities will scale with the difference in the weighted sums of features between the two options. We use a set of 4 features with validities [0.9, 0.8, 0.7, 0.6] and construct a gradient of trials: some where WADD predicts exact indifference (e.g., A wins on features 1 and 4, B wins on 2 and 3, both summing to 1.5), some with weak WADD preferences (e.g., A wins on feature 1, B wins on feature 2, difference of 0.1), and some with strong WADD preferences (e.g., A wins on all features). If Random Guessing is correct, choice proportions will be near 0.5 for all these trial types. If WADD is correct, choice proportions will systematically diverge from 0.5 as the difference in weighted sums increases.

[5] This design quantitatively dissociates the Weighted Additive (WADD) model from Random Guessing by systematically varying the difference in weighted sums between options across a wide range. Random Guessing predicts a flat 50% choice probability for all trials, irrespective of the feature values. WADD, on the other hand, predicts that choice probabilities will track the difference in weighted sums (Option A - Option B). We use 5 features with validities [0.95, 0.80, 0.75, 0.60, 0.50] to construct a parametric gradient of trials: some where WADD predicts exact indifference (difference = 0), some with weak preferences (difference = +/- 0.10 or 0.20), moderate preferences (difference = +/- 1.00), and strong preferences (difference = +/- 3.60). If subjects use WADD, their choice proportions will form a sigmoidal curve as a function of the weighted sum difference. If they randomly guess, the curve will be completely flat at 0.5.

[6] This design quantitatively dissociates Random Guessing from the Random Feature Heuristic by systematically varying the difference in the total number of positive features between Option A and Option B. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the feature values. The Random Feature Heuristic, however, predicts that choice probabilities will track the difference in the unweighted sum of features (since picking a random feature favors the option with more positive features). We use 5 features and construct trials with maximum differences (5 vs 0), moderate differences (4 vs 1, 3 vs 0, 5 vs 2), and ties (2 vs 2, 3 vs 3). If subjects are purely guessing, choice proportions will remain near 0.5 for all trial types. If they use a Random Feature Heuristic, their choice proportions will systematically diverge from 0.5 as the difference in the number of positive features increases.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Random Feature Heuristic from Random Guessing by systematically varying the difference in the total number of positive features between Option A and Option B across 7 features. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the feature values. The Random Feature Heuristic, however, predicts that choice probabilities will track the difference in the unweighted sum of features. We construct a parametric gradient of trials with differences ranging from +7 to -7. If subjects are purely guessing, choice proportions will remain near 0.5 for all trial types. If they use a Random Feature Heuristic, their choice proportions will systematically and linearly diverge from 0.5 as the difference in the number of positive features increases.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
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
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
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
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
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
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
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
  "prompt_token_count": 2985,
  "candidates_token_count": 661,
  "total_token_count": 4844
}
```
