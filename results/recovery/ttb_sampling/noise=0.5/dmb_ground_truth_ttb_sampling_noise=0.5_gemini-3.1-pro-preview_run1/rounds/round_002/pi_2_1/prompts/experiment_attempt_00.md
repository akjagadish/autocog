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
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # Paper-faithful Weighted Additive rule (WADD)
    # Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. 
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    # Use the experiment-provided validities directly as the weights.
    validities = np.asarray(parameters["validities"], dtype=float)

    # Weighted sum per option (dot product with validities).
    scores = stim @ validities

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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take The Best (TTB) is a lexicographic, non-compensatory heuristic. Decision-makers evaluate options by comparing them sequentially on features, starting with the cue that has the highest subjective validity. The first feature that discriminates between the options determines the choice. If all features tie, the decision-maker guesses. Response noise is modeled as an independent lapse rate that occasionally results in a random choice.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    p_core = np.array([0.5, 0.5])  # Default to guessing if all features tie
    
    # Sequential comparison
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic TTB choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design uses six features with linearly spaced validities to quantitatively dissociate Tallying (which simply counts feature-wise wins) from WADD (which weights wins by feature validity). The trial pairs include cases where Tallying prefers one option while WADD prefers the other, as well as cases where one model is indifferent but the other strongly prefers an option. For example, comparing an option that wins on three lower-validity cues against one that wins on the two highest-validity cues creates a sharp divergence: Tallying favors the former (3 vs 2 wins), whereas WADD favors the latter (due to higher weighted sum).

[1] This design quantitatively dissociates the Weighted Additive (WADD) theory from Tallying using a five-feature structure with highly skewed validities. In trials 1 and 2, Tallying predicts a preference for the option that wins on three low-validity features, whereas WADD predicts a preference for the option that wins on the two highest-validity features. In trials 3 through 8, the options are tied in terms of the total number of winning features (either 2 vs 2 or 1 vs 1), leading Tallying to predict complete indifference (guessing). However, WADD strongly predicts preferences on these trials because the features won by one option have substantially higher validities than those won by the other.

[2] This design quantitatively dissociates Tallying from the Weighted Additive (WADD) rule using a five-feature structure. We use validities that are spread out, allowing WADD to strongly prefer options with fewer but highly valid features, while Tallying simply counts feature-wise wins. Some trials create a direct conflict (e.g., Option A wins on the two highest-validity features, while Option B wins on the three lowest, causing WADD to prefer A and Tallying to prefer B). Other trials result in an equal number of feature wins for both options (e.g., 1 vs 1 or 2 vs 2), causing Tallying to predict complete indifference, whereas WADD predicts strong preferences based on the asymmetric validities of the winning features.

[3] This design quantitatively dissociates the Weighted Additive (WADD) theory from Tallying using five features with skewed validities. Trial 1 pits the two highest-validity features against the three lowest, creating a direct conflict where WADD prefers the option with fewer, higher-validity features (Option A), while Tallying prefers the option with more feature wins (Option B). Trials 2, 3, and 4 feature ties in the number of feature wins (2 vs 2 or 3 vs 3), leading Tallying to predict complete indifference (guessing). In contrast, WADD predicts clear preferences on these trials based on the specific validities of the winning features. Trial 5 serves as a control where both models agree on the preferred option but diverge on the strength of preference.

[4] This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive rule (WADD) by exploiting the compensatory versus non-compensatory distinction. In these trials, one option always possesses the single most valid discriminating feature (which TTB relies on exclusively), while the other option possesses a greater number of slightly less valid features whose combined weight exceeds that of the single best feature (which WADD favors). By creating these direct conflicts across different validity levels and including trials with tied top features, we can cleanly distinguish the lexicographic stopping rule of TTB from the compensatory integration of WADD.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design strongly dissociates the non-compensatory Take The Best (TTB) heuristic from the compensatory Weighted Additive (WADD) rule. By assigning validities that are relatively spread out but still allow for compensation, we create trials where one option wins on the single most valid discriminating feature (which strictly dictates TTB's choice), while the other option wins on a larger number of slightly less valid features such that its overall weighted sum is higher (which dictates WADD's choice). The design includes conflicts at the first, second, and third most valid features by tying the higher-validity features, ensuring that the dissociation is robust across different stopping points of the lexicographic TTB process.",
  "validities": [
    0.95,
    0.75,
    0.65,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1
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
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
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
  "prompt_token_count": 3022,
  "candidates_token_count": 393,
  "total_token_count": 5195
}
```
