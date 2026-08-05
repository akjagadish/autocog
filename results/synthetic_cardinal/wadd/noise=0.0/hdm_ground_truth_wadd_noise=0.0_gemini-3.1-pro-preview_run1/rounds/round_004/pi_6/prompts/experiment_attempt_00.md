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
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see a set of ratings for each product. Each rating comes from an expert and is an integer between 0 and an upper bound fixed for this experiment (higher = more positive). The experts differ in how well they typically predict product quality, and their accuracies are communicated before the experiment begins; they are always shown in the same order. On every trial, decide which product you believe is of higher quality and press F for product A or J for product B. There is no time limit, but please respond as accurately as you can. There is no feedback during the task: try to make the best inference you can from the ratings and the experts' accuracies. Press SPACE when you are ready to begin.

## ADVOCATED THEORY
**Description:** Equal Weights (EQW) heuristic: People evaluate multi-attribute options by taking an unweighted sum of their cardinal feature values, ignoring the provided cue validities. Unlike Tallying, which ignores magnitudes and only counts strict superiorities, EQW uses the full cardinal information but treats all features as equally important. Options are compared based on their total sum of feature values, and choices are made probabilistically via a softmax rule over these sums.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"EQW expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Unweighted sum of cardinal feature values
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
**Description:** People evaluate options by taking a weighted sum of their feature values, where the weights correspond to the subjective validities of the cues (Weighted Additive rule). This allows for fully compensatory decision making, where many small advantages can outweigh a single large advantage, and is sensitive to the cardinal magnitude of feature differences, unlike simple counting (Tallying) or one-reason heuristics (Take The Best).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    if len(val) != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Compute the weighted sum of feature values for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return int(np.argmax(probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design strongly dissociates Take The Best (TTB) from Tallying by pitting the single most-valid discriminating cue against the sheer number of winning cues. On every trial, the option that wins on the highest-validity discriminating feature (which TTB chooses) explicitly loses on the majority of the remaining lower-validity features (which Tallying heavily weights). By using 5 features, we ensure that Tallying will firmly select the option with the most feature-wise wins, while TTB ignores those counts and relies solely on the highest-validity cue. A rating_max of 5 is used to allow for cardinal variance, ensuring the design can also be used to evaluate magnitude-sensitive heuristics like WADD in future analyses.

[1] This design systematically dissociates Tallying from Take The Best (TTB) by exploiting both preference reversals and predicted indifference. In half of the trials, the highest-validity discriminating cue favors one option while the simple count of winning cues favors the other, causing a strict preference reversal between TTB and Tallying. In the other half of the trials, the options tie in the total number of winning cues (e.g., 1 win vs 1 win, with the rest tied), leading Tallying to predict complete indifference (guessing), whereas TTB still predicts a deterministic choice based on the single most valid discriminating cue. This provides a clear quantitative divergence: Tallying predicts 50/50 choice probabilities on trials where TTB predicts near 100/0.

[2] This design strictly dissociates the Weighted Additive (WADD) rule from Tallying by manipulating both cardinal feature magnitudes and feature validities. Tallying relies solely on counting the number of features where one option strictly beats the other, ignoring the size of the difference and the validities of the features. WADD, in contrast, computes a validity-weighted sum of the cardinal values. The trials are constructed such that in some cases Tallying predicts a tie (because both options win on an equal number of features) while WADD predicts a strong preference due to magnitude differences. In other cases, Tallying strongly prefers one option because it wins on more features (e.g., 3 wins vs 1 win), but WADD prefers the opposite option because its single win is either on a highly valid feature or involves a massive magnitude advantage.

[3] This design strictly dissociates Tallying from the Weighted Additive (WADD) rule by manipulating both the number of winning features and the cardinal magnitudes/validities of those features. Tallying ignores magnitude and validities, simply counting the number of features where one option is strictly greater than the other. WADD computes a validity-weighted sum of the cardinal values. The trials pit options that win on a majority of features but by small margins or on low-validity features against options that win on fewer features but by massive margins or on high-validity features. This creates stark preference reversals between the two theories.

[4] This design quantitatively dissociates the Weighted Additive (WADD) rule from the Equal Weight (EQW) heuristic. Both rules sum cardinal feature values, but WADD weights them by validities whereas EQW ignores validities and treats all features equally. By using highly dispersed validities (0.9, 0.6, 0.5) and a large rating scale (0-10), we construct three types of dissociating trials: 1) Preference reversals, where one option has a higher unweighted sum but a lower validity-weighted sum; 2) EQW indifference, where unweighted sums are equal but weighted sums strongly favor one option; and 3) WADD indifference, where validity-weighted sums are perfectly balanced but unweighted sums favor one option. These trial types ensure that the two theories predict distinctly different choice patterns and indifference points.

[5] To distinctively separate Equal Weight (EQW) from Weighted Additive (WADD) without duplicating previous 3-feature designs, this experiment uses 4 features with a wide spread of validities [0.9, 0.8, 0.6, 0.5] and a rating maximum of 10. We construct trials that systematically manipulate the locus of the cardinal advantage. In 'preference reversal' trials, one option has a large advantage on low-validity features (maximizing the unweighted sum for EQW), while the other has a moderate advantage on high-validity features (maximizing the weighted sum for WADD). We also include trials where one theory predicts exact indifference while the other predicts a strong preference. This provides a robust quantitative dissociation across multiple distinct trial types.

[6] This design quantitatively dissociates the Weighted Additive (WADD) rule from the Tallying heuristic. Tallying ignores both the validities of the features and the cardinal magnitudes of their values, simply counting the number of features on which an option is strictly superior. WADD, conversely, computes a validity-weighted sum of the cardinal feature values. The trials are constructed to include stark preference reversals (where one option wins on more features but the other has a massive magnitude advantage on a highly valid feature), as well as orthogonal indifference points. In some trials, Tallying predicts a tie (equal number of winning features) while WADD predicts a strong preference due to cardinal differences. In other trials, WADD predicts a tie (weighted sums are equal) but Tallying predicts a strong preference (one option wins on 3 out of 4 features).

[7] This design strictly dissociates Tallying (a sign-based, unweighted heuristic) from the Weighted Additive (WADD) rule (a magnitude-based, validity-weighted heuristic). To do this, we use 4 features with dispersed validities and a cardinal rating scale (0-10). The trials are carefully designed to capture three quantitative divergences: 1) Preference reversals, where one option wins on the majority of features by a small margin (favored by Tallying), but the other wins on a single high-validity feature by a massive margin (favored by WADD); 2) Tallying indifference, where options win on an equal number of features (leading Tallying to predict 50/50 guessing), but one option has a much larger cardinal advantage on its winning features (leading WADD to strongly prefer it); 3) WADD indifference, where the validity-weighted sums of the two options are exactly equal (causing WADD to predict 50/50 guessing), but one option wins on 3 out of 4 features (leading Tallying to strongly prefer it).

[8] This experiment is designed to strictly dissociate the Weighted Additive (WADD) rule from the Equal Weight (EQW) heuristic using a 5-feature design. Both theories compute a sum of cardinal feature values, but EQW ignores validities while WADD weights the features by their validities. By using a broad spread of validities [1.0, 0.9, 0.8, 0.6, 0.5] and a rating max of 10, we construct specific trials that pull the two theories apart in three ways: 1) Preference reversals, where one option has a much higher unweighted sum but a lower weighted sum due to poor performance on top-validity features; 2) EQW indifference, where unweighted sums are identical but weighted sums strongly favor one option; and 3) WADD indifference, where the validity-weighted sums are perfectly balanced, but the unweighted sums favor the option that relies on many low-validity features. This 5-feature structure ensures we do not duplicate previously explored 3- and 4-feature designs while robustly separating the two models.

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
  "rationale": "This experiment is designed to strictly dissociate the Equal Weight (EQW) heuristic from the Weighted Additive (WADD) rule using a novel 6-feature design. Both theories compute a sum of cardinal feature values, but EQW ignores validities while WADD weights the features by their validities. By using a broad spread of validities [0.9, 0.8, 0.8, 0.6, 0.6, 0.5] and a rating max of 10, we construct specific trials that pull the two theories apart in three ways: 1) Preference reversals, where one option has a much higher unweighted sum but a lower weighted sum due to poor performance on top-validity features; 2) EQW indifference, where unweighted sums are identical but weighted sums strongly favor one option; and 3) WADD indifference, where the validity-weighted sums are perfectly balanced, but the unweighted sums favor the option that relies on many low-validity features. The 6-feature structure ensures we do not duplicate previously explored 3-, 4-, and 5-feature designs while robustly separating the two models.",
  "validities": [
    0.9,
    0.8,
    0.8,
    0.6,
    0.6,
    0.5
  ],
  "rating_max": 10,
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      10,
      10,
      10
    ],
    [
      10,
      10,
      0,
      0,
      0,
      0
    ],
    [
      8,
      0,
      0,
      0,
      0,
      0
    ],
    [
      5,
      5,
      5,
      0,
      0,
      0
    ],
    [
      5,
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
      0,
      0,
      0
    ],
    [
      9,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      10,
      10,
      2,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      10,
      10,
      0
    ],
    [
      0,
      0,
      0,
      10,
      2,
      0
    ],
    [
      0,
      0,
      0,
      6,
      6,
      6
    ],
    [
      10,
      10,
      10,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      10,
      0,
      4
    ],
    [
      0,
      0,
      0,
      4,
      4,
      4
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3718,
  "candidates_token_count": 631,
  "total_token_count": 6093
}
```
