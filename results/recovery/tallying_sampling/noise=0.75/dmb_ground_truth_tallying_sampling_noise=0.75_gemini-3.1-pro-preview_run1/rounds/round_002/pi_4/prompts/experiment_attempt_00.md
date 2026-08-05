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
**Description:** Probabilistic Take-The-Best (TTB with evaluation error): Subjects evaluate features sequentially in order of validity, making a decision based on the first feature that discriminates between the options. However, feature evaluation is subject to noise. When a feature actually discriminates, there is a probability 'gamma' (evaluation error) that the subject fails to notice or utilize this discrimination, causing them to mistakenly proceed to the next feature in the hierarchy. If all features are exhausted without a decision, the subject guesses. A baseline lapse rate 'epsilon' accounts for completely random choices independent of the cue evaluation process.

**Parameters:**
- gamma: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    gamma = float(parameters['gamma'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    for j in cue_order:
        if a[j] > b[j]:
            p_a += p_reach * (1.0 - gamma)
            p_reach *= gamma
        elif b[j] > a[j]:
            p_b += p_reach * (1.0 - gamma)
            p_reach *= gamma
            
    # If no feature discriminates (or all discriminating features were missed), guess uniformly
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
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

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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
[0] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) rule, this design uses a 'compensatory' structure where the most valid cue often points to one option while the sum of the less valid cues points to the other. TTB is a non-compensatory heuristic: it stops at the first discriminating cue (ordered by validity) and ignores all others. In contrast, WADD computes a weighted sum of all features, allowing multiple weaker cues to outvote a single strong cue. By setting the validities such that the highest validity is less than the sum of the lower validities (e.g., 0.95 vs 0.70+0.65+0.60), we create trials where TTB and WADD make diametrically opposed predictions.

[1] To quantitatively dissociate the Weighted Additive (WADD) rule from Take The Best (TTB), this design uses a set of four cues with validities carefully chosen such that the most valid cue is outweighed by combinations of less valid cues. TTB always decides based on the single highest-validity cue that discriminates between the options. In contrast, WADD computes a compensatory weighted sum. By presenting pairs where the highest-validity cue favors one option but the sum of the remaining cues favors the other, we create strong opposite predictions. Additional trials where both models agree serve as internal controls to measure baseline noise and lapse rates independently of strategy differences.

[2] To dissociate Rank-Weighted Additive (Rank-WADD) from cardinal Weighted Additive (WADD) models, we exploit the difference between linear rank-based weights and actual validities. We use 5 features where the highest validity is much larger than the others, while the remaining validities decline gradually (0.95, 0.65, 0.60, 0.55, 0.50). In Rank-WADD, weights decay linearly (5, 4, 3, 2, 1). Thus, combinations of lower-ranked features can easily outweigh the top feature in rank-weight sums, but not in cardinal-validity sums. For instance, comparing [1, 0, 0, 0, 1] to [0, 1, 1, 0, 0]: Rank-WADD prefers Option B (rank sum 7 vs 6), while WADD prefers Option A (cardinal sum 1.45 vs 1.25). We include multiple such dissociative pairs in both directions alongside baseline pairs where models agree.

[3] To quantitatively dissociate the cardinal Weighted Additive (WADD) rule from the Rank-Weighted Additive theory, we exploit the non-linear gaps between feature validities. We use five features with validities [0.98, 0.70, 0.68, 0.66, 0.50]. In Rank-WADD, the weights are strictly linear based on rank (5, 4, 3, 2, 1). Therefore, the second and third features combined (rank sum 4 + 3 = 7) will outweigh the first and fifth features combined (rank sum 5 + 1 = 6). However, in the cardinal WADD model, the first feature is exceptionally strong (0.98), so the sum of the first and fifth features (0.98 + 0.50 = 1.48) outweighs the sum of the second and third (0.70 + 0.68 = 1.38). By including pairs where these combinations are pitted against each other, we create strong opposite predictions between the two models. Additional pairs are included where both models agree, serving as baseline controls.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Probabilistic Take-The-Best (TTB) from the Weighted Additive (WADD) rule, this design uses five features with a steep drop-off in validities. TTB is a non-compensatory heuristic that makes a decision based entirely on the first discriminating cue (in order of validity), whereas WADD aggregates all features compensatorily. We present pairs of options where the highest-validity cue(s) favor one option, but a larger number of lower-validity cues favor the other option. TTB will consistently choose the option favored by the highest-validity discriminating cue, whereas WADD will choose the other option because the combined weight of the numerous lower-validity cues outweighs the single top cue. Additional trials where the models agree are included as baseline controls to estimate lapse and noise parameters.",
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
      1,
      1,
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
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
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
  "prompt_token_count": 3276,
  "candidates_token_count": 523,
  "total_token_count": 5180
}
```
