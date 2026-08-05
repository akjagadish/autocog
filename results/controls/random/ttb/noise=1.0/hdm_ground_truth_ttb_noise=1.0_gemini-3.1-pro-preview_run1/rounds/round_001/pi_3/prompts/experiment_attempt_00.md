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
**Description:** Take The Best (TTB) heuristic: People make decisions between two options by sequentially inspecting features in descending order of their subjective or objective validity. The process stops at the first feature that discriminates between the options (i.e., one option has a higher value than the other). The option with the higher value on this discriminating feature is chosen, and all remaining features are ignored, regardless of their magnitudes. If no features discriminate, the decision maker guesses. Response noise is modeled via a softmax function on the binary outcome and an epsilon lapse rate.

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
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(validities)[::-1]
    
    # Default scores if no cue discriminates (leads to 50/50 after softmax)
    scores = np.array([0.0, 0.0])
    
    # Lexicographic search
    for cue in cue_order:
        if stim[0, cue] > stim[1, cue]:
            scores = np.array([1.0, 0.0])
            break
        elif stim[1, cue] > stim[0, cue]:
            scores = np.array([0.0, 1.0])
            break
            
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
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    return int(np.argmax(probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Tallying from WADD, we use cardinal features (rating_max = 5) and vary both the magnitude of feature differences and the validity of the features. Tallying makes decisions based purely on the count of winning features, ignoring the size of the differences and their validities. WADD, in contrast, computes a weighted sum that is sensitive to both validities and magnitudes. By setting up trials where one option wins on more features (favored by Tallying) but by very small margins on low-validity cues, while the other option wins on fewer features but by large margins and/or on high-validity cues (favored by WADD), we can perfectly separate the predictions of the two models.

[1] This design perfectly dissociates the Weighted Additive (WADD) model from Tallying by manipulating the cardinal magnitudes of feature differences and their validities. Tallying relies exclusively on the sign of feature-wise comparisons, ignoring both the size of the difference and the validity of the features. WADD computes a compensatory weighted sum sensitive to both validities and magnitudes. We use a 10-point scale (rating_max=10) and three features with validities [0.9, 0.7, 0.6]. The trials are crafted such that in many pairs, one option wins on a majority of features (thus preferred by Tallying) but by very small margins and/or on lower-validity features, while the other option wins on a minority of features (thus losing the tally) but by massive margins on high-validity features (thus preferred by WADD). Other trials introduce ties for Tallying where WADD has a strong preference, ensuring a robust quantitative dissociation.

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
  "rationale": "To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit their fundamental difference in processing: TTB relies exclusively on the highest-validity discriminating feature and ignores both the magnitudes of differences and all lower-validity features. WADD, conversely, computes a compensatory weighted sum that integrates all features, sensitive to both their validities and their cardinal magnitudes. By setting a high rating maximum (rating_max = 10) and using three features with clear validity differences (0.9, 0.7, 0.6), we can construct trials where one option has a minimal advantage on the highest-validity cue (thus chosen by TTB), while the other option has massive advantages on the remaining cues (thus chosen by WADD). We also include trials where the highest validity cue is tied, pushing TTB to the second cue, which is then outweighed by the third cue in WADD's calculation.",
  "validities": [
    0.9,
    0.7,
    0.6
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      6,
      0,
      0
    ],
    [
      5,
      10,
      10
    ],
    [
      8,
      2,
      10
    ],
    [
      0,
      8,
      8
    ],
    [
      10,
      0,
      0
    ],
    [
      5,
      5,
      5
    ],
    [
      2,
      10,
      10
    ],
    [
      7,
      8,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      5,
      10,
      10
    ],
    [
      6,
      0,
      0
    ],
    [
      8,
      3,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      9,
      9,
      9
    ],
    [
      6,
      0,
      0
    ],
    [
      3,
      2,
      2
    ],
    [
      7,
      7,
      10
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3114,
  "candidates_token_count": 456,
  "total_token_count": 4910
}
```
