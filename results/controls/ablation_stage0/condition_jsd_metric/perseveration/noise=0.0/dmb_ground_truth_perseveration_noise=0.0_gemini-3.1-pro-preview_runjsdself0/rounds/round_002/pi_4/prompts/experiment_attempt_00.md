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
**Description:** Decision-makers use the Take-The-Best (TTB) heuristic, searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the choice is a random guess. To account for behavioral noise without parameter redundancy, deviations from the deterministic TTB rule are modeled using a single lapse rate parameter (epsilon), replacing the redundant softmax temperature found in prior TTB instantiations.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_core = np.array([0.5, 0.5])
    for idx in cue_order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters['epsilon'])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People make choices by computing a weighted sum of the features for each option (the Weighted Additive, or WADD, heuristic). Unlike Take The Best or Tallying, WADD scales the contribution of each feature by its validity. Furthermore, features are processed as bipolar evidence: the presence of a feature adds to the option's value proportionally to its validity, while its absence actively penalizes the option. The resulting option scores are translated into choice probabilities via a softmax function, with an independent lapse rate accounting for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # WADD (Weighted Additive) heuristic.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    
    # Center the stimulus features to -1 and 1 so absence of a feature penalizes
    stim_centered = stim * 2.0 - 1.0
    
    # Compute the weighted sum of features for each option.
    # The weights are directly proportional to the cue validities.
    scores = np.dot(stim_centered, val)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To cleanly dissociate Take The Best (TTB) from Tallying, this design uses 5 features with strictly descending validities. The trials are constructed such that the option favored by the single most valid discriminating cue (which TTB will invariably choose) is opposed by a larger number of less valid cues (which Tallying will count up to choose the opposite option). For example, in many trials, Option A has a 1 on the highest valid cue but 0s elsewhere, while Option B has a 0 on the highest valid cue but multiple 1s on lower valid cues. This creates a stark contrast between a non-compensatory strategy (TTB) and an unweighted compensatory strategy (Tallying).

[1] To quantitatively dissociate Tallying from Take The Best (TTB), this design employs 5 features with strictly descending validities. Tallying compares the simple count of features favoring each option, ignoring validities, while TTB relies solely on the most valid discriminating feature. The trial pairs are constructed to create direct conflicts: in many trials, one option is favored by the highest-validity discriminating cue (triggering TTB) while the other option is favored by a larger number of lower-validity cues (triggering Tallying). Furthermore, we include trials where Tallying predicts a perfect tie (equal number of wins for both options, leading to chance-level guessing) but TTB makes a strong deterministic prediction based on a single high-validity cue. This mix of opposing predictions and differential guessing behavior allows precise estimation of which heuristic is driving choices.

[2] To quantitatively dissociate WADD from Tallying, this design uses 5 features with a steep drop-off in validities. WADD computes a weighted sum of the features based on their validities, whereas Tallying simply counts the number of features favoring each option, ignoring validities. By pitting a small number of high-validity features against a larger number of low-validity features, we create clear conflicts. For instance, in trials where Option A possesses the top two features and Option B possesses the bottom three, WADD strongly favors Option A while Tallying favors Option B. We also include trials where Tallying predicts a tie (equal number of wins) but WADD makes a strong deterministic prediction, and baseline trials where both heuristics agree to ensure data quality.

[3] This design quantitatively dissociates Tallying from WADD using 6 features. Tallying counts unweighted feature-wise wins, ignoring validities and ties. WADD computes a validity-weighted sum where the absence of a feature actively penalizes the option's score. The trials are constructed to create specific dissociations: (1) WADD Tie / Tallying Strong: Option A has the two most valid features, while Option B has three lower-validity features whose combined weights exactly equal A's, resulting in a perfect tie for WADD but a clear win for Tallying. (2) Opposite Predictions: Option A has the top two features, while Option B has the bottom three. The penalty for missing features makes WADD favor A, while Tallying favors B's sheer quantity of wins. (3) Tallying Tie / WADD Strong: Both options win on an equal number of features, leading Tallying to guess, but WADD strongly prefers the option with higher-validity features.

[4] This design quantitatively dissociates the Weighted Additive (WADD) heuristic from Take-The-Best (TTB) using 5 features. TTB relies solely on the most valid discriminating feature, making deterministic predictions regardless of the opposing evidence. In contrast, WADD integrates all features weighted by their validities, with the absence of a feature actively penalizing the option. By constructing trials where the single highest-validity feature points to one option while multiple lower-validity features point to the other, we create opposing predictions (e.g., TTB chooses A, WADD chooses B). Additionally, by varying the net validity difference across trials where both heuristics agree, we can detect WADD's graded choice probabilities (via softmax) versus TTB's constant confidence level.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design isolates Take-The-Best (TTB) from the Weighted Additive (WADD) heuristic by creating opposing predictions and exploiting WADD's graded sensitivity to the magnitude of evidence. Using 6 features with a wide spread of validities, we construct conflict trials where TTB's strict non-compensatory reliance on the most valid discriminating cue predicts one choice, while WADD's compensatory integration of all features predicts the opposite. Furthermore, we include pairs where both heuristics agree on the direction but WADD predicts varying choice probabilities due to different net score differences, whereas TTB predicts a constant confidence level (1-epsilon) since it only considers the first discriminating cue.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
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
      1
    ],
    [
      0,
      0,
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
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
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
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
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
      0
    ],
    [
      0,
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2956,
  "candidates_token_count": 625,
  "total_token_count": 5955
}
```
