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
**Description:** People make decisions by computing a weighted sum of the features for each option, but the weights are non-linear transformations of the provided cue validities. Specifically, subjects exponentiate the validities by a parameter gamma, which allows the strategy to continuously interpolate between Tallying (gamma = 0) and Take-The-Best (gamma -> infinity). The option with the higher total weighted score is preferred, subject to decision noise (softmax) and occasional lapses.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- gamma: [-1.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To maximally dissociate Take The Best (TTB) from Tallying, we use 5 binary features with monotonically decreasing validities. This allows us to construct 'compensatory' trials where the single highest-validity discriminating cue favors one option, but a larger number of lower-validity cues favor the other option. In these critical trials, TTB will choose the option favored by the top cue, whereas Tallying will choose the opposite option based on the sheer count of winning features. We also include trials where Tallying ties (equal number of winning features) but TTB has a strong preference, as well as baseline trials where both models agree.

[1] To cleanly dissociate Tallying from Take The Best (TTB), we use 4 features with descending validities. We design trials where the highest valid discriminating cue points to one option, but a larger number of lower-validity cues point to the other. In these cases, TTB chooses based on the single most valid cue, while Tallying sums the feature-wise wins and chooses the opposite option. We also include trials where Tallying results in a tie (equal number of feature wins) but TTB has a strict preference, allowing us to capture differences in both deterministic choice and guessing behavior.

[2] To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, we exploit their divergent treatment of cue validities. Tallying treats all features equally, merely counting the number of features on which one option strictly dominates the other. WADD, by contrast, weights each feature's contribution by its explicitly provided validity. We employ 5 binary features with a steep drop-off in validities (e.g., two highly valid cues and three less valid cues). The critical trials pair an option that wins on the two highly valid cues against an option that wins on the three less valid cues. Tallying predicts a preference for the latter (3 wins vs 2 wins), whereas WADD predicts a preference for the former due to the larger summed validity. We also include trials where the options tie in the number of winning features (leading Tallying to guess uniformly) but differ in their weighted sums, allowing WADD to make a deterministic prediction.

[3] To quantitatively dissociate Tallying from WADD, we use 5 binary features where two cues have very high validity (0.95) and three have low validity (0.55). Tallying ignores validities and simply counts the number of winning features, whereas WADD integrates the validities into a weighted sum. The critical dissociation trials pit an option that wins on the three low-validity features against an option that wins on the two high-validity features. Tallying predicts a preference for the former (3 wins vs 2 wins), whereas WADD predicts a strong preference for the latter (weighted sum 1.90 vs 1.65). We also include trials where Tallying results in a tie (equal number of winning features) but WADD has a strong preference, as well as baseline trials where both models agree.

[4] To robustly dissociate Take The Best (TTB) from Tallying, we employ 5 binary features with monotonically decreasing validities. TTB decides based solely on the single most valid discriminating cue, ignoring all others. Tallying, on the other hand, ignores validities and simply counts the number of features favoring each option. We designed a set of trials where the highest-validity discriminating cue points to one option, but a strictly greater number of lower-validity cues point to the other option. In these 'compensatory' trials, TTB and Tallying will choose opposite options. We also include trials where Tallying results in a tie (equal number of winning features) but TTB has a strict preference, yielding differences in deterministic choice versus guessing.

[5] This design explicitly pits Tallying against Take The Best (TTB) by systematically manipulating the number of winning features versus the validity of the highest discriminating feature. We employ 5 features with monotonically decreasing validities. In the critical dissociation trials, one option is favored by the single most valid cue, but the other option is favored by a strictly greater number of lower-validity cues. TTB will choose the former, while Tallying (which ignores validities and simply counts feature-wise wins) will choose the latter. We also include trials where the options tie in the number of winning features (leading Tallying to predict a 50/50 guess) but differ on the highest validity cue (leading TTB to make a deterministic choice). Finally, baseline trials where both models agree are included to ensure basic task engagement.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate the advocated WADD theory (with its flexible gamma parameter) from Tallying, we use 5 features with linearly decreasing validities. Tallying simply counts the number of winning features, ignoring validities. WADD computes a weighted sum where weights are validities exponentiated by gamma. We include 'compensatory' trials where one option wins on fewer but more valid features (e.g., the top 2) while the other wins on more but less valid features (e.g., the bottom 3). Tallying strictly predicts the latter, whereas WADD predicts the former for typical positive gamma values. We also include trials that result in a tie under Tallying but yield a strict preference under WADD, and trials that differentiate low gamma (close to Tallying) from high gamma (close to Take-The-Best) to fully exercise the advocated model's parameter space.",
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
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0
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
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3256,
  "candidates_token_count": 471,
  "total_token_count": 5171
}
```
