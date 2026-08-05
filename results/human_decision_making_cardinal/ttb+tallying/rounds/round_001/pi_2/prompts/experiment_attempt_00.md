# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Heuristic Decision Making domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0] (they are stored **descending**, highest first) — to fix each expert's advertised accuracy; subjects are told these values up front. Choose `rating_max` (>= 1) to set the upper bound of each rating value: rating_max=1 gives binary ratings (but note that Tallying ≡ Equal-Weight on binary features), while rating_max >= 2 gives cardinal ratings that dissociate magnitude-based heuristics (EW, WADD) from sign-based ones (Tallying). Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, rating_max]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities, rating_max, and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, `rating_max`]. The features are NOT necessarily binary: if `rating_max=1` the rating is binary (0 / 1), but if `rating_max>=2` the rating is a cardinal score on a 0..`rating_max` scale and ratings are displayed as a horizontal filled bar with the numeric value (e.g. "4/7") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no trial-by-trial feedback, but at the end of the task subjects see a single summary screen with the percentage of their choices that matched the Lee & Cummins (2004) Bayesian normative benchmark. Subjects are instructed to keep the experiment tab in focus throughout; cumulative blur time is tracked by the BotDetection layer and used post-hoc to exclude participants who switched away for too long. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    return int(np.argmax(probabilities))
```


## COMPETING THEORY
**Description:** Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

**Parameters:**
- beta: [0.01, 20.0]
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
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue (or the first discriminating cue) points to one option, while the simple majority of features point to the other. TTB decides solely based on the highest-validity discriminating cue, ignoring all others. Tallying, on the other hand, ignores validities and simply counts the number of features on which each option is superior. We include trials where TTB and Tallying make strictly opposing predictions, as well as trials where the top cue ties and the conflict cascades to the second cue.

[1] To quantitatively dissociate Tallying from Take The Best (TTB), we present a 5-feature design using cardinal ratings (0-5 scale). TTB is a non-compensatory, one-reason decision rule that stops at the first discriminating feature, ordered by validity. Tallying ignores both the feature validities and the cardinal magnitude of the differences, counting only the number of features on which an option is strictly better. We construct trials where the highest-validity discriminating cue points to one option, but a simple majority of the remaining lower-validity cues point to the other option. By including trials where the top cue discriminates, as well as trials where the top cues tie and the conflict cascades to the 2nd or 3rd cue, we can isolate whether subjects are using a single-feature TTB strategy or a multi-feature Tallying strategy.

[2] To quantitatively dissociate WADD from Tallying, we exploit WADD's sensitivity to both feature validities and cardinal rating magnitudes. Tallying, by contrast, only counts strict feature-wise wins, ignoring both the sizes of the differences and the validities of the features. We use a 4-feature design with a high rating_max (10) and varied validities. The trials are constructed such that one option wins on a single feature by a large margin (strongly favored by WADD), while the other option wins on multiple features by small margins (favored by Tallying). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to magnitude and validity differences.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]. On load, the list is reordered to **descending** by validity (highest first) and the same permutation is applied to every `trial_a_ratings` / `trial_b_ratings` row, so Expert 1 is always the most valid. Length determines n_features and must match every rating list. Pick a spread (at least one high and one low) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule.
- rating_max: Upper bound (inclusive) of each rating value. Ratings are integers in [0, rating_max]. Choose rating_max=1 for binary ratings (note: Tallying and Equal-Weight are behaviorally identical on binary features — if you want to dissociate magnitude-based from sign-based heuristics, choose rating_max >= 2 for cardinal ratings).
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is an integer in [0, rating_max]. `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design with cardinal ratings (up to 10) and varying validities. Tallying ignores both the magnitude of the differences between ratings and the validities of the features, simply counting the number of features on which one option is strictly better than the other. WADD, however, integrates the cardinal magnitudes weighted by the validities. We design trials where one option has a small advantage on a majority of features (favored by Tallying) but the other option has a massive advantage on one or two high-validity features (favored by WADD). We also include trials where Tallying predicts a tie but WADD strongly prefers one option, as well as baseline trials where both models agree.",
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
      5,
      5,
      5,
      5,
      5
    ],
    [
      9,
      9,
      4,
      4,
      4
    ],
    [
      10,
      0,
      10,
      0,
      10
    ],
    [
      8,
      8,
      2,
      2,
      5
    ],
    [
      3,
      3,
      3,
      3,
      3
    ],
    [
      1,
      9,
      9,
      9,
      9
    ],
    [
      7,
      2,
      7,
      2,
      7
    ],
    [
      5,
      6,
      5,
      6,
      5
    ],
    [
      4,
      8,
      4,
      8,
      4
    ],
    [
      8,
      8,
      8,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      10,
      4,
      4,
      4,
      4
    ],
    [
      5,
      5,
      5,
      5,
      5
    ],
    [
      0,
      10,
      0,
      10,
      0
    ],
    [
      2,
      2,
      8,
      8,
      5
    ],
    [
      2,
      2,
      2,
      2,
      10
    ],
    [
      10,
      8,
      8,
      8,
      8
    ],
    [
      6,
      8,
      6,
      8,
      6
    ],
    [
      6,
      5,
      6,
      5,
      6
    ],
    [
      5,
      2,
      5,
      2,
      5
    ],
    [
      9,
      9,
      9,
      10,
      10
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3143,
  "candidates_token_count": 595,
  "total_token_count": 5589
}
```
