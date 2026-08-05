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
**Description:** Dynamic Strategy Selection (Threshold Model): Decision-makers dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy on a trial-by-trial basis. The choice of strategy depends on the maximum cardinal difference between the options across all features. If this difference exceeds a certain threshold, the decision-maker is more likely to use WADD to account for the large magnitude; otherwise, they default to the simpler Tallying heuristic. This is modeled as a probabilistic mixture where the weight of WADD is a soft step function (sigmoid) of the maximum feature difference.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 10.0]
- steepness: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD score computation
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a_wadd = np.sum(a * val) / sum_val
        score_b_wadd = np.sum(b * val) / sum_val
    else:
        score_a_wadd = np.sum(a)
        score_b_wadd = np.sum(b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # Tallying score computation
    score_a_tally = float(np.sum(a > b))
    score_b_tally = float(np.sum(b > a))
    scores_tally = np.array([score_a_tally, score_b_tally])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Dynamic strategy selection based on max cardinal difference
    max_diff = np.max(np.abs(a - b))
    w_wadd = 1.0 / (1.0 + np.exp(-steepness * (max_diff - threshold)))
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
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

[3] To quantitatively dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design with cardinal ratings (up to 10) and varying validities. Tallying ignores both the magnitude of the differences between ratings and the validities of the features, simply counting the number of features on which one option is strictly better than the other. WADD, however, integrates the cardinal magnitudes weighted by the validities. We design trials where one option has a small advantage on a majority of features (favored by Tallying) but the other option has a massive advantage on one or two high-validity features (favored by WADD). We also include trials where Tallying predicts a tie but WADD strongly prefers one option, as well as baseline trials where both models agree.

[4] To dissociate the purely compensatory Weighted Additive (WADD) theory from the Dynamic Strategy Selection (Threshold) model, we manipulate the maximum cardinal difference ('max_diff') between options across features. The Threshold model predicts that subjects will use Tallying when the maximum feature difference is small (falling below their threshold) and switch to WADD only when a feature difference is large enough to cross the threshold. In contrast, the pure WADD model integrates validity-weighted cardinal differences on every trial, regardless of the maximum difference. We construct a 5-feature design where WADD and Tallying consistently make opposing predictions: one option wins on the two most valid cues, while the other option wins on the three less valid cues. By parametrically varying the magnitude of the differences (max_diff = 1, 2, 4, and 8), we can observe whether subjects' choices flip from Tallying to WADD as the difference grows (supporting the Threshold model) or remain consistent with WADD throughout (supporting the advocated pure WADD theory).

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
  "rationale": "To quantitatively dissociate the Dynamic Strategy Selection (Threshold) model from the pure Weighted Additive (WADD) model, we construct a 5-feature design where the two models make diverging predictions depending on the maximum cardinal difference between the options. We design conflict trials where Option A wins on the two most valid features (sum of validities = 1.9), while Option B wins on the three least valid features (sum of validities = 1.6). Because WADD integrates validity-weighted magnitudes, it consistently prefers Option A on these trials regardless of the scale of the difference. Tallying, conversely, simply counts feature wins and consistently prefers Option B (3 wins to 2). By parametrically varying the magnitude of the difference (max_diff from 1 to 10), we can test the Threshold model's prediction: subjects will prefer Option B (Tallying) when the difference is small and below their threshold, but switch to Option A (WADD) when the difference is large. The pure WADD model predicts a consistent preference for Option A across all magnitudes.",
  "validities": [
    1.0,
    0.9,
    0.6,
    0.5,
    0.5
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      5,
      5,
      4,
      4,
      4
    ],
    [
      6,
      6,
      4,
      4,
      4
    ],
    [
      7,
      7,
      3,
      3,
      3
    ],
    [
      8,
      8,
      2,
      2,
      2
    ],
    [
      9,
      9,
      1,
      1,
      1
    ],
    [
      10,
      10,
      0,
      0,
      0
    ],
    [
      5,
      5,
      5,
      4,
      4
    ],
    [
      10,
      10,
      10,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      4,
      4,
      5,
      5,
      5
    ],
    [
      4,
      4,
      6,
      6,
      6
    ],
    [
      3,
      3,
      7,
      7,
      7
    ],
    [
      2,
      2,
      8,
      8,
      8
    ],
    [
      1,
      1,
      9,
      9,
      9
    ],
    [
      0,
      0,
      10,
      10,
      10
    ],
    [
      4,
      4,
      4,
      5,
      5
    ],
    [
      0,
      0,
      0,
      10,
      10
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3781,
  "candidates_token_count": 591,
  "total_token_count": 6443
}
```
