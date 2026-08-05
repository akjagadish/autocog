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
**Description:** Single Random Feature heuristic: On each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all. An epsilon parameter allows blending this strategy with pure random guessing.

**Parameters:**
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    epsilon = float(parameters["epsilon"])
    n_features = stim.shape[1]
    
    wins = (stim[0] > stim[1]).sum()
    losses = (stim[0] < stim[1]).sum()
    ties = (stim[0] == stim[1]).sum()
    
    p_A_core = (wins + 0.5 * ties) / n_features
    p_B_core = (losses + 0.5 * ties) / n_features
    
    probs = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * probs + epsilon * 0.5
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
**Description:** In complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. Thus, behavior resembles zero-intelligence random guessing.

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
[0] To dissociate Tallying from WADD, we use cardinal features (rating_max = 5) and vary both the magnitude of feature differences and the validity of the features. Tallying makes decisions based purely on the count of winning features, ignoring the size of the differences and their validities. WADD, in contrast, computes a weighted sum that is sensitive to both validities and magnitudes. By setting up trials where one option wins on more features (favored by Tallying) but by very small margins on low-validity cues, while the other option wins on fewer features but by large margins and/or on high-validity cues (favored by WADD), we can perfectly separate the predictions of the two models.

[1] This design perfectly dissociates the Weighted Additive (WADD) model from Tallying by manipulating the cardinal magnitudes of feature differences and their validities. Tallying relies exclusively on the sign of feature-wise comparisons, ignoring both the size of the difference and the validity of the features. WADD computes a compensatory weighted sum sensitive to both validities and magnitudes. We use a 10-point scale (rating_max=10) and three features with validities [0.9, 0.7, 0.6]. The trials are crafted such that in many pairs, one option wins on a majority of features (thus preferred by Tallying) but by very small margins and/or on lower-validity features, while the other option wins on a minority of features (thus losing the tally) but by massive margins on high-validity features (thus preferred by WADD). Other trials introduce ties for Tallying where WADD has a strong preference, ensuring a robust quantitative dissociation.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit their fundamental difference in processing: TTB relies exclusively on the highest-validity discriminating feature and ignores both the magnitudes of differences and all lower-validity features. WADD, conversely, computes a compensatory weighted sum that integrates all features, sensitive to both their validities and their cardinal magnitudes. By setting a high rating maximum (rating_max = 10) and using three features with clear validity differences (0.9, 0.7, 0.6), we can construct trials where one option has a minimal advantage on the highest-validity cue (thus chosen by TTB), while the other option has massive advantages on the remaining cues (thus chosen by WADD). We also include trials where the highest validity cue is tied, pushing TTB to the second cue, which is then outweighed by the third cue in WADD's calculation.

[3] This design quantitatively dissociates WADD from TTB by exploiting TTB's non-compensatory, lexicographic nature and its insensitivity to cardinal magnitudes. By using a large rating scale (rating_max = 10), we create trials where one option has a negligible advantage on the highest-validity cue (thus deterministically chosen by TTB) but massive disadvantages on lower-validity cues (thus rejected by WADD). We also include trials featuring ties on the primary cue, pushing TTB to the secondary cue, where again a small advantage dictates TTB's choice while WADD integrates the massive advantage of the alternative option on the tertiary cue.

[4] To strictly dissociate the Random Guessing theory from the Weighted Additive (WADD) model, we must construct a gradient of choice difficulty. WADD predicts that choice probability will scale monotonically with the weighted sum difference between options: it predicts near-deterministic choices when one option completely dominates the other by a large margin (e.g., 10s across all features vs 0s), and approaches chance only as the options become perfectly balanced. In contrast, the Random Guessing theory posits that subjects completely fail to integrate these cues and will guess uniformly at random (50/50) across all trials, regardless of whether the WADD difference is massive, moderate, or zero. By including transparently dominant options alongside complex compensatory tradeoffs, we can test whether subjects show any sensitivity to the magnitude of evidence (as WADD predicts) or if they remain at chance everywhere (as Random Guessing predicts).

[5] To robustly dissociate the Weighted Additive (WADD) model from Random Guessing, we construct a parametric gradient of expected value differences. WADD predicts a smooth psychometric function where choice probability for option A scales with the validity-weighted difference between the options' features. In contrast, Random Guessing predicts a flat 50% choice rate across all trials regardless of the evidence. We include completely dominating options (where WADD predicts near 100% deterministic choice), completely balanced options (where WADD predicts 50%), and compensatory trials where high-validity features pit against multiple lower-validity features to ensure subjects are actually integrating validities rather than just summing features.

[6] To quantitatively dissociate Random Guessing from the Single Random Feature heuristic, we systematically vary the proportion of features on which Option A strictly dominates Option B. The Random Guessing theory predicts a flat 50% probability of choosing A across all trials, completely ignoring the feature values. The Single Random Feature theory assumes the subject randomly samples one feature and decides based on that feature alone, predicting that the probability of choosing A will scale linearly with the number of features on which A is superior to B (e.g., 100% if A wins all features, 75% if A wins 3 out of 4, 25% if A wins 1 out of 4). By including trials that range from A dominating all features to B dominating all features, alongside tie trials, we can perfectly distinguish the flat prediction of Random Guessing from the monotonic step-wise prediction of Single Random Feature.

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
  "rationale": "To quantitatively dissociate the Single Random Feature (SRF) heuristic from the Random Guessing theory, we use a 5-feature design and systematically manipulate the number of features on which Option A wins, loses, or ties against Option B. The Random Guessing theory predicts a flat 50% probability of choosing A across all trials, completely ignoring the feature values. The SRF theory assumes the subject randomly samples exactly one feature and decides based on that feature alone (guessing if tied). Thus, SRF predicts that the probability of choosing A will scale linearly with the number of features on which A beats B, plus half the number of tied features. We construct a gradient of choice probabilities predicted by SRF (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0) by varying the win/loss/tie counts, while Random Guessing rigidly predicts 0.5 for every single trial.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      8,
      8,
      8,
      8,
      8
    ],
    [
      8,
      8,
      8,
      8,
      2
    ],
    [
      8,
      8,
      8,
      2,
      2
    ],
    [
      8,
      8,
      5,
      2,
      2
    ],
    [
      8,
      8,
      2,
      2,
      2
    ],
    [
      8,
      2,
      2,
      2,
      2
    ],
    [
      2,
      2,
      2,
      2,
      2
    ],
    [
      8,
      8,
      8,
      8,
      5
    ]
  ],
  "trial_b_ratings": [
    [
      2,
      2,
      2,
      2,
      2
    ],
    [
      2,
      2,
      2,
      2,
      8
    ],
    [
      2,
      2,
      2,
      8,
      8
    ],
    [
      2,
      2,
      5,
      8,
      8
    ],
    [
      2,
      2,
      8,
      8,
      8
    ],
    [
      2,
      8,
      8,
      8,
      8
    ],
    [
      8,
      8,
      8,
      8,
      8
    ],
    [
      2,
      2,
      2,
      2,
      5
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3073,
  "candidates_token_count": 565,
  "total_token_count": 4659
}
```
