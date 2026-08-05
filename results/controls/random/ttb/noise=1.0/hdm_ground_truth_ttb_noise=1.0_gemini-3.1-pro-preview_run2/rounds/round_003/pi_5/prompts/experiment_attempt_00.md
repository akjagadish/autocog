# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, `rating_max`]. The features are NOT necessarily binary: if `rating_max=1` the rating is binary (0 / 1), but if `rating_max>=2` the rating is a cardinal score on a 0..`rating_max` scale and ratings are displayed as a horizontal filled bar with the numeric value (e.g. "4/7") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** High-Temperature Weighted Additive (WADD) Theory: Subjects attempt to integrate all available features by computing a weighted sum of feature differences, using subjective validities as weights. However, their decision process is subject to extremely high cognitive noise (a very low inverse temperature, beta, in the softmax choice rule). This results in behavior that appears nearly random, capturing the high lapse rates observed across experiments, while still retaining a slight sensitivity to large aggregate differences in option quality.

**Parameters:**
- beta: [0.0, 0.005]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute scores as weighted sum of features
    scores = stim @ validities
    
    # Extremely high cognitive noise (low beta)
    beta = float(parameters["beta"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
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
**Description:** Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

**Parameters:**
- bias: [0.48, 0.52]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])
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
[0] To dissociate Tallying from WADD, we exploit two key differences: Tallying ignores feature validities and feature magnitudes, whereas WADD integrates both. By setting rating_max to 10, we can create choices where one option wins on a majority of features (thus favored by Tallying) but loses on the weighted sum due to large deficits on high-validity features (thus the other option is favored by WADD). The design also includes ties in Tallying where WADD has a strong preference, to further distinguish the models.

[1] To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, this design uses cardinal ratings (rating_max=10) and varying expert validities. Tallying only counts strict feature-wise wins, ignoring both the magnitude of the difference and the validity of the feature. WADD, in contrast, integrates both cardinal magnitudes and validity weights. The trial pairs are constructed to pit these mechanisms against each other: in several trials, one option wins on a majority of features by a small margin (favored by Tallying) while the other option wins on fewer, higher-validity features by a large margin (favored by WADD). Additionally, trials where Tallying results in a tie but WADD shows a strong preference due to validity weighting are included to further separate the predictions.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the non-compensatory nature of TTB versus the compensatory nature of WADD. We use 3 features with varying validities (0.9, 0.75, 0.6) and a large rating scale (rating_max=10). In the designed trials, one option always has a small advantage on the most valid feature (cue 1), which TTB uses to exclusively determine its choice (assuming a low threshold). However, the other option has massive advantages on the remaining, slightly less valid features. Because WADD integrates all feature values weighted by their validities, it will strongly prefer the option that compensates for a small loss on the most valid cue with large gains on the others. This creates diametrically opposed predictions between the two models.

[3] To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we must target the core difference in how they process feature magnitudes and validities. TTB acts lexicographically: it searches for the first feature (ordered by validity) where the difference between options exceeds a threshold, ignoring all lower-validity cues. WADD is compensatory, integrating all features weighted by their validities. In this design, we use 4 cardinal features (rating_max=10). We present trials where Option A holds a varying advantage on the single highest-validity feature, while Option B holds a massive advantage on all remaining lower-validity features. For WADD, the large advantages on multiple lower-validity features easily compensate for a small-to-moderate loss on the highest-validity feature, leading to a strong preference for Option B. For TTB, if the difference on the highest-validity feature exceeds its threshold, it will decisively choose Option A, completely ignoring Option B's overwhelming advantage elsewhere. By systematically varying the difference on the highest-validity feature across trials, we can precisely identify TTB's threshold parameter while observing WADD's smooth, compensatory choice probabilities.

[4] To quantitatively dissociate Take The Best (TTB) from Random Guessing, we exploit TTB's deterministic, lexicographic search and its sensitivity to the cue difference threshold. We use 3 cardinal features and systematically vary the difference between Option A and Option B on the most valid cue (from 1 to 5 points), while the less valid cues strongly favor the opposite option. For TTB, if the difference on the top cue exceeds its internal threshold, it will decisively choose the option favored by that cue (nearly 100% probability, minus lapse rate). If the difference falls below the threshold, it will move to the second cue and decisively choose the opposite option. This creates a sharp step-function in choice probabilities across trials. The Random Guessing model, in stark contrast, predicts a flat ~50% choice probability across all trials regardless of feature differences.

[5] To quantitatively dissociate Random Guessing from Take The Best (TTB), we exploit TTB's lexicographic stopping rule and its threshold parameter. TTB searches through features by validity and stops at the first feature where the absolute difference exceeds its threshold (which is bounded between 0 and 50). By using a large rating scale (rating_max=100) and ensuring the difference on the highest-validity feature is always greater than 50, we force TTB to deterministically choose the option favored by the most valid cue on every trial, completely ignoring all other features. The Random Guessing model, on the other hand, predicts a flat ~50% choice probability regardless of the feature values. This creates a stark, qualitative and quantitative dissociation: TTB predicts strong alternation of choices (A, B, A, B...) tracking the highest-validity cue, while Random Guessing predicts uniform indifference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate the High-Temperature WADD theory from the Random Guessing baseline, we must exploit the residual sensitivity of High-Temperature WADD to massive aggregate differences in feature quality. While Random Guessing predicts a flat, near-50% choice probability across all trials regardless of the stimuli, High-Temperature WADD (despite its low beta parameter) will show a slight but systematic deviation from 50% that monotonically tracks the weighted sum of feature differences. By using a large rating scale (rating_max=100) and varying the weighted difference across trials from extreme to moderate, we create a clear quantitative dissociation: WADD predicts a graded, symmetric response curve tracking the objective differences, whereas Random Guessing predicts a completely flat line.",
  "validities": [
    1.0,
    0.8,
    0.6,
    0.5
  ],
  "rating_max": 100,
  "trial_a_ratings": [
    [
      100,
      100,
      100,
      100
    ],
    [
      0,
      0,
      0,
      0
    ],
    [
      100,
      100,
      0,
      0
    ],
    [
      0,
      0,
      100,
      100
    ],
    [
      100,
      0,
      0,
      0
    ],
    [
      0,
      100,
      100,
      100
    ],
    [
      50,
      50,
      50,
      50
    ],
    [
      100,
      50,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0
    ],
    [
      100,
      100,
      100,
      100
    ],
    [
      0,
      0,
      100,
      100
    ],
    [
      100,
      100,
      0,
      0
    ],
    [
      0,
      100,
      100,
      100
    ],
    [
      100,
      0,
      0,
      0
    ],
    [
      50,
      50,
      50,
      50
    ],
    [
      0,
      50,
      100,
      100
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3003,
  "candidates_token_count": 520,
  "total_token_count": 4769
}
```
