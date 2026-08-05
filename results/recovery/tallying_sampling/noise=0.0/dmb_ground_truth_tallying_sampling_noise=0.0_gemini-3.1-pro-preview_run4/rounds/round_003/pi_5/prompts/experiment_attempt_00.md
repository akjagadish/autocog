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
**Description:** Weighted Additive Strategy (WADD) with Linearly Compressed Weights: Individuals compute a compensatory weighted sum of positive features for each option, but their subjective weights are compressed relative to objective validities. People interpolate between raw validities and equal weights (Tallying), flattening the weight distribution. This allows a large number of less valid cues to compensate for a highly valid cue, matching human behavior in compensatory environments.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- alpha: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Interpolate between raw validities and equal weights
    weights = alpha * validities + (1.0 - alpha) * 1.0
    
    # Weighted Additive Strategy
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD), we use a 4-feature design with a clear hierarchy of validities. The crucial trials pit the single most valid discriminating cue against a coalition of multiple lower-validity cues. TTB ignores the lower-validity cues entirely and always chooses the option endorsed by the highest-validity discriminating cue. In contrast, WADD integrates all cues, allowing a strong deficit on the top cue to be compensated by advantages on several lower-validity cues. By varying whether the top cue aligns with or opposes the sum of the remaining cues, we can uniquely identify which strategy subjects are using.

[1] To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we use a 5-feature design with linearly descending validities. Crucial trials pit the single highest-validity discriminating cue against a coalition of multiple lower-validity cues. TTB, being non-compensatory, will always choose the option endorsed by the first discriminating cue, completely ignoring all subsequent cues. WADD, being compensatory, will integrate all cues. By configuring trials where the sum of the lower-validity cues outweighs the single highest-validity cue, WADD will choose the opposite of TTB. We also include trials where the discriminating cue is not the first feature (e.g., the first feature ties), testing the identical logic further down the cue hierarchy.

[2] To quantitatively dissociate Tallying (Equal Weights) from the Weighted Additive (WADD) rule, this design uses a 6-feature environment where the validities are distinctly skewed: two high-validity cues and four low-validity cues. The critical trials pit an option with fewer, but higher-validity, positive features against an option with more, but lower-validity, positive features. Tallying simply counts the number of positive features and will reliably prefer the option with the greater sheer quantity of endorsements (the 'coalition' of lower-validity cues). WADD, conversely, weights each feature by its validity, leading it to prefer the option endorsed by the fewer but more predictive cues. Additional trials include equal-count feature comparisons to demonstrate that WADD can differentiate where Tallying must guess.

[3] To quantitatively dissociate the Weighted Additive (WADD) rule from Tallying (Equal Weights), we use a 6-feature design with a mix of high, medium, and low validities. Tallying ignores feature validities and simply sums the number of positive features, preferring the option with the higher count. WADD, on the other hand, weights each feature by its validity. By creating trials where one option has fewer positive features but higher validities (e.g., 2 positive features with high validities vs. 3 positive features with lower validities), Tallying will strictly prefer the option with more features, whereas WADD will prefer the option with the higher weighted sum. We also include trials where Tallying produces a tie (equal number of positive features) but WADD has a clear preference, demonstrating WADD's sensitivity to validity differences when feature counts are matched.

[4] Since the Thresholded Counting theory has a threshold parameter constrained to [0.0, 0.5] and validities are constrained to [0.5, 1.0], all features will always pass the threshold. Thus, Thresholded Counting is mathematically identical to Tallying in this environment. This design includes validities at the absolute minimum boundary (0.5) to test if any edge-case thresholding occurs, while providing standard feature-summing vs validity-weighting dissociations.

[5] Because the threshold parameter for Thresholded Counting is bounded in [0.0, 0.5] and validities are bounded in [0.5, 1.0], the formalized Thresholded Counting model mathematically reduces to Tallying (all features pass the threshold). However, conceptually, Thresholded Counting posits that subjects ignore low-validity features. This design pits high-validity features (0.9, 0.8) against a coalition of features at the absolute minimum validity bound (0.5). If subjects behaviorally ignore the 0.5 features (effectively employing a threshold slightly above 0.5), they will choose the option endorsed by the high-validity features, diverging from the pure Tallying prediction which simply counts all features equally.

[6] To quantitatively dissociate Tallying (Equal Weights) from WADD with Linearly Compressed Weights, we use a 7-feature design with three high-validity cues and four low-validity cues. Tallying simply counts the number of positive features, completely ignoring validity weights. The compressed WADD model computes a weighted sum, blending raw validities with equal weights (controlled by alpha). By pitting an option with 3 high-validity features against an option with 4 low-validity features, Tallying strictly prefers the latter (4 > 3). However, for any reasonable alpha > 0.55, the compressed WADD model will prefer the former due to the substantial validity advantage. We also include trials where feature counts are tied (which Tallying predicts as random choice) but WADD has a strong preference, to further highlight the lack of validity weighting in pure Tallying.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate WADD with Linearly Compressed Weights from Tallying, we use a 6-feature design with two highly valid cues and four low-validity cues. Tallying assumes all features are weighted equally (effectively alpha=0), predicting choices based purely on the count of positive features. Compressed WADD posits that individuals interpolate between equal weights and objective validities (alpha > 0). By including trials where Tallying predicts a tie (equal feature counts) but Compressed WADD predicts a distinct preference based on validity, and trials where an option with fewer but highly valid features overcomes an option with more but lower-validity features, we can cleanly separate the two models. If subjects show sensitivity to validities (preferring the higher-weighted option despite equal or slightly lower feature counts), it supports Compressed WADD over pure Tallying.",
  "validities": [
    0.98,
    0.92,
    0.6,
    0.55,
    0.52,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
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
    ]
  ],
  "trial_b_ratings": [
    [
      0,
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
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0
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
      0,
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
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3113,
  "candidates_token_count": 502,
  "total_token_count": 6138
}
```
