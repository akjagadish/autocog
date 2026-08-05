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
**Description:** People use a Weighted Additive (WADD) strategy when comparing two options, but they subjectively scale the stated cue validities. By applying a non-linear transformation to the validities (exponentiating them by a parameter gamma), decision-makers can amplify the differences between cues (mimicking Take The Best) or compress them (mimicking Tallying). Expanding the range of gamma allows for extreme weight disparities, accommodating individuals who rely heavily on the most valid cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    gamma = float(parameters["gamma"])
    val = np.asarray(parameters["validities"], dtype=float) ** gamma
    
    # Compute the weighted sum of features for each option
    ev_a = np.sum(val * a)
    ev_b = np.sum(val * b)
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Random Guessing / High-Noise Tallying: Due to the lack of trial-by-trial feedback and potentially low engagement, subjects largely ignore cue validities and feature values. Instead, they make choices that are nearly indistinguishable from uniform random guessing, which can be modeled as a simple unweighted tally of features heavily diluted by extreme decision noise and high lapse rates. This explains the ~0.5 agreement with Take The Best, ~0.44 agreement with Tallying, and the near-zero variance in choice proportions across different trial types.

**Parameters:**
- beta: [0.0, 0.05]
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of winning features
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    # Blend with uniform guessing based on epsilon
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design strictly dissociates Take The Best (TTB) from Tallying by manipulating the distribution of feature-wise wins across cues of varying validities. In trials 1-6, one option wins on the single highest-validity cue available, while the other option wins on a greater number of lower-validity cues. TTB predicts choice strictly based on the highest-validity discriminating cue, whereas Tallying predicts choice based on the simple count of feature wins. Trials 7-8 represent cases where Tallying predicts a tie (guessing) due to equal numbers of feature wins, but TTB predicts a deterministic choice based on the highest-validity cue.

[1] This design systematically dissociates Tallying from Take The Best (TTB) across a 5-feature environment. It includes 'opposition' trials where one option is favored by the highest-validity discriminating cue (triggering TTB) while the other option has a strictly greater number of feature wins (triggering Tallying). It also includes 'tie' trials where Tallying predicts a 50/50 guess because both options win on an equal number of features, yet TTB predicts a deterministic choice because one option wins on the highest-validity cue among the discriminating ones.

[2] This design quantitatively dissociates Take The Best (TTB) from the generalized Weighted Additive (WADD) strategy by exploiting their different confidence (choice probability) predictions. TTB generates a binary internal score (1 for the winner, 0 for the loser) based solely on the first discriminating cue, completely ignoring the magnitude of the validities and the number of opposing cues. Consequently, TTB predicts a constant choice probability across all trials where a choice is made. By contrast, WADD computes a weighted sum. To mimic TTB's choices on trials where lower-validity cues outnumber the highest-validity cue (e.g., Trial 1), WADD requires a high 'gamma' parameter to heavily skew the weights. However, a high gamma causes the absolute difference in weighted sums to shrink exponentially for trials decided by lower-validity cues (e.g., Trial 4). Thus, WADD predicts highly variable choice probabilities across these trials (very confident when Cue 1 decides, near-guessing when Cue 4 decides), whereas TTB predicts uniform confidence.

[3] This design quantitatively dissociates the generalized Weighted Additive (WADD) strategy from Take The Best (TTB) by exploiting their divergent predictions regarding choice confidence. TTB relies entirely on the first discriminating cue, ignoring both the absolute validity of that cue and any opposing lower-validity cues. Consequently, TTB predicts a constant choice probability (confidence) across all trials where a choice is made. Conversely, WADD computes a weighted sum of all cues, with validities transformed by a gamma parameter. Even if gamma is high (mimicking TTB's choices), the absolute difference in weighted sums shrinks exponentially for trials decided by lower-validity cues. Thus, WADD predicts highly variable choice probabilities across trials depending on which cue decides the choice and how many cues oppose it, while TTB predicts uniform confidence.

[4] This design aims to completely dissociate the Random Guessing / High-Noise Tallying theory from the Weighted Additive (WADD) theory by including trials that vary from extreme evidence disparities to complex trade-offs. WADD predicts that choice probabilities will systematically track the weighted sum of validities (transformed by gamma), leading to highly deterministic choices (probabilities near 1.0 or 0.0) when one option dominates (e.g., all 1s vs all 0s) or when the weighted evidence strongly favors one side. In stark contrast, the advocated Random Guessing theory predicts that subjects largely ignore the validities and feature values due to high noise and lapse rates, resulting in choice proportions that remain universally close to 50/50 across all trials, with only a marginal pull toward the option with more unweighted feature wins.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To decisively dissociate the Weighted Additive (WADD) theory from the Random Guessing / High-Noise Tallying theory, this design spans a wide range of evidence disparities. The Random Guessing theory posits that subjects suffer from extreme noise (epsilon > 0.8) and low sensitivity (beta < 0.05), predicting that choice probabilities will remain stubbornly close to 50/50 across all trials, with only marginal biases toward options with more unweighted feature wins. In contrast, WADD predicts highly deterministic choices when one option strongly dominates or when high-validity cues heavily favor one side. By including complete dominance trials, single-cue difference trials of varying validities, and trials where a single high-validity cue opposes multiple low-validity cues, we can reveal whether subjects exhibit the high-confidence, validity-sensitive choices predicted by WADD, which the high-noise tallying model fundamentally cannot accommodate.",
  "validities": [
    0.95,
    0.8,
    0.7,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
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
      1
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
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
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
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
  "prompt_token_count": 2934,
  "candidates_token_count": 544,
  "total_token_count": 4233
}
```
