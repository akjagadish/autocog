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
**Description:** Tallying under Overload (Equal Weights): Under cognitive overload without trial-by-trial feedback, subjects abandon complex integration of cue validities. Instead, they fall back on a highly simplified Equal Weights heuristic, merely tallying the total number of positive features (1s) for each option. Even with this simplification, the high cognitive demand leads to near-random choice behavior, which is captured by extreme softmax noise and a very high lapse rate.

**Parameters:**
- beta: [0.0, 0.2]
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Equal Weights / Tallying: count the number of positive features (1s) for each option
    a, b = stim[0], stim[1]
    a_score = np.sum(a)
    b_score = np.sum(b)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Apply high lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload. Instead of systematically integrating cue validities and feature vectors, they abandon structured decision strategies and resort to random guessing. Choice behavior is driven entirely by this stochasticity, with only a potential slight bias toward one spatial position (e.g., Option A or Option B) over the other.

**Parameters:**
- side_bias: [0.3, 0.7]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Under cognitive overload, subjects ignore the state (features) and just guess.
    # The choice probability is determined only by an intrinsic side bias.
    p_b = float(parameters.get('side_bias', 0.5))
    p_a = 1.0 - p_b
    return np.array([p_a, p_b])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design aims to maximally dissociate Take The Best (TTB) from Tallying using a 5-feature choice task. The validities are strictly descending (0.9, 0.8, 0.7, 0.6, 0.5). TTB predicts that choices will be driven entirely by the single most valid cue that discriminates between the two options, ignoring all other cues. Tallying predicts that choices will be driven by the simple count of how many features each option wins on, completely ignoring validities. The trial pairs are constructed such that the option winning on the highest-validity discriminating cue (the TTB choice) loses on the simple count of discriminating features (the Tallying choice), or one model predicts a strict preference while the other predicts a guess (due to equal counts).

[1] This design quantitatively dissociates Tallying from Take The Best (TTB) using 4 binary features with distinct, descending validities. Tallying simply counts the number of features each option wins on, completely ignoring validities. TTB acts lexicographically, choosing based entirely on the most valid discriminating feature. The trial pairs are constructed such that the option favored by the highest validity discriminating feature (TTB's choice) is often outnumbered by the sheer quantity of lower-validity features favoring the alternative (Tallying's choice). Furthermore, several trials create a tie for Tallying (equal number of winning features) while TTB makes a deterministic prediction, perfectly separating the two theories.

[2] This design quantitatively dissociates the non-compensatory Take The Best (TTB) heuristic from the compensatory Weighted Additive (WADD) rule. We use 5 features with a wide spread of validities. Crucially, the competing WADD model is restricted to a low inverse temperature (beta <= 1.0), meaning it can only predict highly stochastic choices (probabilities near 0.5) even when the weighted score difference is maximal. TTB's beta can range up to 20.0, allowing it to predict deterministic, highly confident choices. The trial pairs are constructed as 'compensatory vs. non-compensatory' conflicts: one option always wins on the single highest-validity discriminating cue (the TTB choice), while the other option wins on a multitude of lower-validity cues, giving it a higher overall weighted sum (the WADD choice). Thus, TTB predicts strong choices for Option A on most trials, while WADD predicts weak choices for Option B.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic. The trial pairs systematically pit the single highest-validity discriminating cue against a coalition of lower-validity cues. In every trial, TTB predicts choice for the option that wins on the earliest discriminating cue, completely ignoring the rest of the features. Conversely, WADD integrates all features and consistently predicts choice for the opposite option, which has a higher total weighted sum due to winning on multiple lower-validity cues. By varying which cue is the highest discriminating one (e.g., the first vs. the second vs. the third) and balancing the left/right presentation, we create a robust set of choices where the two models make diametrically opposed predictions across the entire experiment.

[4] To dissociate the Cognitive Overload theory (which predicts choice probabilities are completely invariant to stimulus features and driven only by a constant side bias) from the Weighted Additive theory (which predicts that choice probabilities, despite being noisy, will systematically track the weighted feature differences between options). By presenting trials with maximal and intermediate differences in weighted sums favoring either Option A or Option B, we can observe whether the empirical choice rates shift in the direction of the superior option (supporting WADD) or remain flat across all trial types (supporting Cognitive Overload).

[5] To quantitatively dissociate the highly-stochastic Weighted Additive (WADD) theory from the Cognitive Overload (random guessing) theory, we must test whether choice probabilities remain completely flat across trials or whether they track the weighted feature differences, even if dampened by high noise. We use 6 features to create a wide spectrum of weighted score differences between Option A and Option B, ranging from subtle (+0.6) to extreme (+4.5), and their exact mirrors. WADD predicts a monotonic relationship between the score difference and the probability of choosing A, whereas Cognitive Overload predicts a completely flat choice rate driven only by a constant side bias.

[6] To quantitatively dissociate the Cognitive Overload theory (which predicts completely flat choice probabilities driven only by a spatial side bias) from the Tallying under Overload theory (which predicts that choice probabilities, despite being extremely noisy, will systematically track the difference in the unweighted sum of features). We present trials with maximal variation in the tally difference between Option A and Option B (ranging from +5 to -5). The Tallying theory predicts a monotonic relationship between the tally difference and the probability of choosing A, whereas the Cognitive Overload theory predicts a completely flat choice rate across all trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Tallying under Overload theory from the pure Cognitive Overload (random guessing) theory, we must test whether choice probabilities remain completely flat across trials or whether they track the unweighted feature tally differences, even if heavily dampened by extreme noise and lapse rates. We use 8 features to create a wide spectrum of tally differences between Option A and Option B, ranging from +8 to -8. The Tallying under Overload theory predicts a slight but systematic monotonic relationship between the tally difference and the probability of choosing A, whereas the pure Cognitive Overload theory predicts a completely flat choice rate driven only by a constant spatial side bias.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.6,
    0.55,
    0.52,
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
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
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
      0,
      1,
      1
    ],
    [
      0,
      0,
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
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
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
      1,
      0
    ],
    [
      1,
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
  "prompt_token_count": 2845,
  "candidates_token_count": 699,
  "total_token_count": 4740
}
```
