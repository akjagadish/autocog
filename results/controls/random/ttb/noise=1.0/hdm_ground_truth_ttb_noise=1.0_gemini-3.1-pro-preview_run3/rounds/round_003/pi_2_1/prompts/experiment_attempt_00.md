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
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
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

    # Weighted sum per option (dot product with per-feature subjective weights).
    scores = stim @ (validities * w)

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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

**Parameters:**
- beta: [0.0, 0.01]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
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
[0] To dissociate Tallying from WADD, we exploit two core differences: Tallying ignores both feature validities and cardinal feature magnitudes, relying solely on the count of strict feature-wise wins. By using a rating_max of 10, we can create trials where Option A wins on a majority of features by a small margin, leading Tallying to strongly prefer A. However, Option B can be given a massive magnitude advantage on the remaining high-validity features, causing WADD to strongly prefer B. We also include trials where Tallying sees a tie (2 wins each) but WADD has a strong preference due to magnitudes and validities, isolating the compensatory nature of WADD.

[1] This design systematically dissociates the Weighted Additive (WADD) strategy from the Tallying heuristic. Tallying chooses strictly based on the count of features where one option exceeds another, ignoring both the validity of the features and the magnitude of the difference. WADD, by contrast, considers both feature validities and the cardinal magnitude of differences. The trials are constructed such that in several cases, Option A has a small advantage on a majority of features (causing Tallying to strongly prefer A), but Option B has a massive magnitude advantage on the highest-validity feature (causing WADD to strongly prefer B). We also include tie-trials for Tallying (equal number of feature wins) where WADD has a strong preference due to validities, and vice versa.

[2] To cleanly dissociate Tallying from WADD, this design exploits Tallying's insensitivity to both feature validities and cardinal feature magnitudes. By setting rating_max to 10 and providing four features with decreasing validities, we construct several distinct trial types. In some trials, one option wins on three out of four features by a small margin (strongly preferred by Tallying), but the other option wins on the most valid feature by a massive margin (strongly preferred by WADD). We also include trials where the models make orthogonal predictions: trials where Tallying sees a tie (2 wins each) but WADD strongly prefers one option due to high validities/magnitudes, and trials where WADD sees a tie (equal weighted sums) but Tallying strongly prefers one option due to it winning on more features.

[3] To cleanly dissociate the Weighted Additive (WADD) strategy from the Tallying heuristic, we leverage two fundamental differences: Tallying's insensitivity to both feature validities and cardinal feature magnitudes. By setting rating_max to 10 and using 5 features with varying validities, we can construct trials where Tallying's count of strict feature-wise wins points in the opposite direction of WADD's validity-weighted cardinal sum. For example, Option A can win on a majority of features by a small margin (triggering a Tallying preference for A), while Option B wins on the highest-validity features by a massive margin (triggering a WADD preference for B). We also include trials where Tallying sees a tie but WADD strongly prefers one option due to validities and magnitudes, as well as trials where WADD is nearly indifferent but Tallying has a strong preference due to the sheer number of feature wins.

[4] To dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use cardinal ratings and exploit TTB's non-compensatory stopping rule. We construct trials where one option has a small advantage on the highest-validity cue (causing TTB to stop and choose it), while the other option has massive advantages on all lower-validity cues (causing WADD to choose the latter due to compensatory weighted sums). We also include trials where the highest-validity cues are tied, forcing TTB to evaluate the next cue, where again a small advantage dictates TTB's choice but is overridden by remaining features in WADD.

[5] This design systematically dissociates the Weighted Additive (WADD) strategy from Take The Best (TTB). TTB relies on a non-compensatory stopping rule, making its decision based entirely on the first feature (ordered by validity) that discriminates between the two options. WADD, on the other hand, integrates all features in a compensatory manner, weighting them by their validities. By setting a high rating_max (10), we can create scenarios where one option holds a minor advantage on the highest-validity feature (causing TTB to choose it), while the other option possesses massive advantages on all remaining lower-validity features (overwhelming the small deficit and causing WADD to choose it). We also include trials where the highest-validity features are tied, forcing TTB to stop at the second or third feature, where again its single-feature focus leads to an opposite prediction compared to WADD's holistic integration.

[6] This design tests the advocated 'Tallying with Extreme Noise' theory, which predicts near-chance (50/50) responding across all conflict trials due to a near-zero inverse temperature, against the competing WADD theory, which predicts strong deterministic preferences driven by validity-weighted cardinal differences. We construct trials where Option A has a massive magnitude advantage on a minority of high-validity features, while Option B has small advantages on a majority of low-validity features. WADD predicts a strong preference for the option with the large magnitude advantage, whereas the advocated theory predicts choices will remain uniformly random regardless of the magnitude or validity differences.

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
  "rationale": "To conclusively dissociate the advocated Weighted Additive (WADD) theory from the competing 'Tallying with Extreme Noise' theory, we exploit the fundamental difference in their predicted choice certainty. The competing theory posits that choices are driven by a Tallying heuristic heavily corrupted by extreme noise (beta near zero), which strictly bounds the maximum possible log-odds difference to the number of features and guarantees near-50/50 choice probabilities on all trials. By using a high rating maximum (10) and generating trials with massive cardinal magnitude differences, WADD predicts highly deterministic choices (approaching 100% certainty for the superior option) due to the large differences in weighted sums, whereas the competing theory remains constrained to predict near-chance behavior.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      10,
      10,
      10,
      10,
      10
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      10,
      10,
      10,
      0,
      0
    ],
    [
      0,
      0,
      10,
      10,
      10
    ],
    [
      9,
      9,
      9,
      9,
      9
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      10,
      0,
      10,
      0,
      10
    ],
    [
      0,
      10,
      0,
      10,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      10,
      10,
      10,
      10,
      10
    ],
    [
      0,
      0,
      0,
      10,
      10
    ],
    [
      10,
      10,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      9,
      9,
      9,
      9,
      9
    ],
    [
      0,
      10,
      0,
      10,
      0
    ],
    [
      10,
      0,
      10,
      0,
      10
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3559,
  "candidates_token_count": 545,
  "total_token_count": 4908
}
```
