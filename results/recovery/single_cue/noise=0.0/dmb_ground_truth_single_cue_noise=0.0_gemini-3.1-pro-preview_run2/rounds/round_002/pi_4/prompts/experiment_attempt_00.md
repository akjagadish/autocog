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
**Description:** Hybrid Tallying: Decision-makers primarily rely on a frugal Tallying heuristic, counting the number of features where one option dominates another. However, they are not completely blind to cue validities; they give a slight premium to the single most valid cue. This premium acts strictly as a tie-breaker or a soft modulator and is never large enough to overcome a strict difference in tallies, blending the robustness of Tallying with a minimal sensitivity to cue hierarchy.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- premium: [0.0, 0.99]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid Tallying expects a (2, n_features) stimulus.")

    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    premium = float(parameters["premium"])
    
    a, b = stim[0], stim[1]
    
    # Pure tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Identify the most valid cue (stable sort to break ties by original index)
    top_cue = np.argsort(-val, kind="stable")[0]
    
    # Premium for winning the most valid cue
    a_top = float(a[top_cue] > b[top_cue])
    b_top = float(b[top_cue] > a[top_cue])
    
    score_a = a_wins + premium * a_top
    score_b = b_wins + premium * b_top
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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
[0] This design systematically pits Take The Best (TTB) against Tallying. By using five features with clearly separated validities, we can construct trials where the highest-validity discriminating cue strongly favors one option, while the sheer number of winning cues favors the other. TTB predicts choices based exclusively on the first discriminating cue in descending validity order, completely ignoring the count of lower-validity cues. In contrast, Tallying treats all features equally and chooses the option with the most winning cues, ignoring their validities. These opposing predictions will yield a stark quantitative dissociation between the two theories.

[1] By utilizing a 4-feature design, we construct scenarios where the option favored by the single highest-validity discriminating cue (Take The Best) is consistently opposed by the option with the greater total number of winning cues (Tallying). We also include trials where Tallying predicts a tie (guessing) because both options win on an equal number of cues, whereas TTB has a strict preference based on the validities of the discriminating cues. This mix of opposing predictions and differential tie-breaking guarantees a strong quantitative dissociation.

[2] This experiment is designed to strongly dissociate the Weighted Additive (WADD) model from the Tallying heuristic using a 5-feature binary choice task. WADD integrates all features by weighting them according to their validities, allowing a smaller number of highly valid cues to compensate for a larger number of less valid cues. Tallying, on the other hand, strictly counts the number of winning features for each option and ignores their validities. By setting a steep distribution of validities (e.g., two very high and three lower validities), we can create critical trials where one option is favored by WADD due to possessing the high-validity features, while the alternative is favored by Tallying because it possesses a greater total number of winning features. We also include trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a distinct preference, as well as baseline trials where both models agree.

[3] This design specifically pits Tallying against the Weighted Additive (WADD) model by using five features with a steep distribution of validities (two high, three low). Tallying simply counts the number of features favoring each option, treating all cues equally regardless of validity. WADD, conversely, integrates validities to form a compensatory sum. The trial set includes pairs where a smaller number of high-validity cues competes against a larger number of low-validity cues (producing opposing choices), as well as trials where both options are supported by an equal number of cues (producing a Tallying tie) but WADD strongly favors the option with the higher-validity cues. This ensures a robust quantitative dissociation across different choice profiles.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Hybrid Tallying from Pure Tallying, we must exploit the only condition where their predictions diverge: when the total number of winning features is tied. Pure Tallying predicts a 50/50 guess on any trial where both options win on an equal number of cues. Hybrid Tallying, however, applies a 'premium' strictly as a tie-breaker for the single most valid cue. Thus, on tally-tied trials where one option wins the highest-validity cue, Hybrid Tallying will predict a distinct preference for that option. We also include control trials where one option has a strict tally advantage but loses the most valid cue; because the premium is strictly less than 1, Hybrid Tallying and Pure Tallying will agree on these, confirming that the premium acts only as a soft modulator rather than a lexicographic rule.",
  "validities": [
    0.95,
    0.8,
    0.65,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2977,
  "candidates_token_count": 473,
  "total_token_count": 4921
}
```
