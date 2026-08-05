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
**Description:** Salience-Weighted Tallying with Near-Tie Threshold: Decision-makers primarily rely on a non-compensatory Tallying heuristic, counting the number of features where one option dominates the other. However, if the difference in tallies is small (below a specific threshold), a secondary compensatory evaluation is triggered. In this tie-breaking phase, the weights of the features are dynamically adjusted based on their salience rather than strictly their objective validity, allowing less predictive features to become disproportionately salient and overweighted during conflict resolution.

**Parameters:**
- beta: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [-5.0, 5.0]
- tie_scale: [0.1, 10.0]
- tally_threshold: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary stage: Tallying feature wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    tally_diff = np.abs(a_wins - b_wins)
    tally_threshold = float(parameters["tally_threshold"])
    
    if tally_diff > tally_threshold:
        scores = np.array([a_wins, b_wins])
    else:
        # Secondary stage: Salience-weighted tie-breaker
        val = np.asarray(parameters["validities"], dtype=float)
        gamma = float(parameters["gamma"])
        centered_val = val - 0.5
        
        # Non-linear scaling allows for negative gamma (overweighting less valid cues)
        w = np.sign(centered_val) * ((np.abs(centered_val) + 1e-4) ** gamma)
        
        tie_scale = float(parameters["tie_scale"])
        scores = tie_scale * np.array([np.sum(a * w), np.sum(b * w)])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] This design systematically pits Take The Best (TTB) against Tallying. By using five features with distinct, descending validities, we can construct trials where the option favored by the single most valid discriminating cue (TTB's choice) is opposed by the option that wins on a larger number of less valid cues (Tallying's choice). This creates a sharp quantitative dissociation: TTB predicts strong preference for the single-best-cue winner, while Tallying predicts preference for the option with more overall feature wins.

[1] This design systematically pits Tallying against Take The Best (TTB) using five features with distinct, descending validities. We construct trials where the option favored by the single most valid discriminating cue (TTB's choice) is opposed by the option that wins on a larger number of less valid cues (Tallying's choice). We also include trials where Tallying predicts a tie (equal number of feature wins) but TTB predicts a clear winner, providing a sharp quantitative dissociation.

[2] This design systematically pits the Weighted Additive Model (WADD) against Tallying by contrasting the number of feature wins with the validities of those features. Tallying strictly counts the number of features where one option dominates, completely ignoring the validities. In contrast, WADD scales the features by a non-linear transformation of their validities centered at chance. We constructed a set of 5 features with a steep drop-off in validity. The trial pairs include cases where one option wins on a single highly valid feature while the other wins on multiple less valid features (causing opposing predictions between WADD and Tallying), as well as cases where both options win on an equal number of features (causing Tallying to predict a tie while WADD predicts a strict preference).

[3] This design pits Tallying against the Weighted Additive Model (WADD). Tallying simply counts the number of features where an option is superior, entirely ignoring the validities of those features. WADD, on the other hand, scales validities non-linearly and computes a weighted sum. By using a set of validities with a steep drop-off (e.g., 0.95, 0.80, 0.60, 0.55, 0.55), we can construct trials where one option wins on a single highly valid feature while the other wins on multiple low-validity features. Tallying will consistently choose the option with more feature wins, whereas WADD will prefer the option with the higher-validity features. Furthermore, we include trials where both options win on an equal number of features (e.g., 2 vs 2), causing Tallying to predict a tie (guessing), while WADD predicts a deterministic preference based on feature validities.

[4] To dissociate the Advocated Theory (Tallying with a validity-weighted tie-breaker) from the Competing Theory (Pure Tallying), the design relies heavily on trials where both options win on an equal number of features. According to Pure Tallying, any tie in the number of feature wins results in a 50/50 guess, regardless of the validities of the features involved. In contrast, the Advocated Theory predicts that when tallies are tied, subjects will evaluate the options using a secondary compensatory mechanism that weights features by their validities (centered at chance). By configuring trials where A and B have the same number of feature wins but differ significantly in the cumulative validity of those winning features, we create a sharp quantitative dissociation: Pure Tallying predicts uniform probability (guessing), while the Advocated Theory predicts a deterministic preference for the option with higher-validity features. Filler trials where one option strictly wins the tally are also included to ensure subjects do not realize that tallying always results in a tie.

[5] To dissociate Pure Tallying (Advocated Theory) from Tallying with a validity-weighted tie-breaker (Competing Theory), we focus on trials where both options win on an equal number of features. Pure Tallying strictly counts feature wins and predicts a 50/50 guess whenever these tallies are tied, regardless of the validities of the winning features. The Competing Theory, however, uses a validity-weighted tie-breaker when tallies are tied, predicting a strong preference for the option whose winning features have higher validities. By pairing high-validity feature wins against low-validity feature wins in equal numbers, we create a sharp quantitative dissociation. Filler trials where one option wins on more features are included to prevent subjects from noticing that tallies are often tied and to ensure both models have trials where they agree.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To effectively dissociate Salience-Weighted Tallying with a Near-Tie Threshold from Pure Tallying, we must test 'near-ties' (tally differences of 1 or 2) rather than just exact ties. Pure Tallying invariably favors the option with more feature wins, regardless of the margin or the validities of the features involved. The Advocated theory, however, posits that if the tally difference falls within a specific threshold, a secondary compensatory tie-breaker evaluates the options based on feature salience. By pitting an option that wins on fewer but highly valid (salient) features against an option that wins on more but lower-validity features, we create trials where Pure Tallying strongly predicts a win for the tally-leader, but the Advocated theory predicts a reversal in preference due to the near-tie threshold activating the salience-weighted tie-breaker.",
  "validities": [
    0.95,
    0.9,
    0.65,
    0.6,
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
    ],
    [
      0,
      1,
      1,
      1,
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
      0
    ],
    [
      1,
      1,
      1,
      1,
      0
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
      1,
      1,
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
      1,
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
      1,
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
      0,
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3491,
  "candidates_token_count": 534,
  "total_token_count": 5832
}
```
