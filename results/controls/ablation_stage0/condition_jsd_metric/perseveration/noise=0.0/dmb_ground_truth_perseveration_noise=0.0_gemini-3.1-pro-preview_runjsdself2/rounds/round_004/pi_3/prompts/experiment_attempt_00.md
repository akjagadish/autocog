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
**Description:** People evaluate options by computing a weighted sum of their feature values, where the weights correspond to the subjective or objective validities of the cues (Weighted Additive rule, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), this compensatory strategy allows multiple lower-validity cues to jointly outweigh a single high-validity cue. Choices are made probabilistically via a softmax over the options' weighted sums, along with a lapse rate to account for random errors.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Sequential Cue Integration with Tallying Fallback (Compensatory Shift)

**Parameters:**
- threshold: [2.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)

    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    diff = 0.0
    crossed = False
    # Accumulate evidence sequentially
    for j in cue_order:
        diff += val[j] * (a[j] - b[j])
        if abs(diff) >= threshold:
            crossed = True
            break

    # If all cues are exhausted without crossing the threshold, fall back to tallying
    if not crossed:
        diff = float(np.sum(a > b) - np.sum(b > a))

    scores = np.array([diff, 0.0])
    
    z = beta * scores
    z -= np.max(z)  # numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, the experiment employs a set of five binary features with strictly descending validities. The trials are specifically constructed so that the option with the highest valid discriminating cue (favored by TTB) always has fewer total positive features than the alternative option (which is favored by Tallying). Under TTB, subjects will base their decision solely on the single most valid cue that differs between the options, ignoring the sheer number of features. Under Tallying, subjects will simply count the total number of winning features for each option, ignoring the validities entirely. These opposing predictions will yield a stark contrast in choice probabilities between the two models.

[1] To decisively dissociate Tallying from Take The Best (TTB), we design trials where one option wins on the single highest-validity discriminating cue (the choice favored by TTB), but the alternative option wins on a strictly greater number of lower-validity cues (the choice favored by Tallying). By systematically varying which cue is the highest discriminating cue (e.g., the 1st, 2nd, or 3rd most valid cue) and tying higher-validity cues, we can robustly verify whether subjects integrate across all features equally (Tallying) or stop at the first discriminating cue (TTB).

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), this design pits the single highest-validity discriminating cue against the combined weight of multiple lower-validity cues. In key trials, the option favored by TTB possesses the single best discriminating feature, while the alternative option possesses a greater sum of validities across the remaining features, making it the choice favored by WADD. We also include control trials where both heuristics agree to ensure baseline performance and task engagement.

[3] To quantitatively dissociate the Weighted Additive rule (WADD) from Take The Best (TTB), we design trials where the single best discriminating cue strongly favors one option (the choice predicted by TTB), while the sum of the remaining, lower-validity cues favors the alternative option (the choice predicted by WADD). We use a set of five features with validities spread out enough to allow the sum of lower validities to outweight the single highest validity. The design includes trials with varying top discriminating cues to ensure the dissociation is robust across different levels of the cue hierarchy, as well as control trials where both models agree.

[4] To dissociate the advocated threshold-based adaptive mixture theory from pure Weighted Additive (WADD) theory, we systematically vary the cue dispersion (the difference in WADD scores between options) across trials where Take The Best (TTB) and WADD make opposing predictions. The pure WADD theory predicts a smooth, monotonic increase in the probability of choosing the WADD-favored option as dispersion increases. In contrast, the advocated theory predicts a stark, non-linear step-like shift: at low dispersions (below the individual's threshold), choices will be strictly driven by TTB (often yielding high confidence in the direction opposite to WADD), and at high dispersions, choices will abruptly switch to WADD. By including a spectrum of dispersion levels where TTB and WADD disagree, we can detect this hard-switching mechanism.

[5] To quantitatively dissociate the advocated Weighted Additive (WADD) theory from the competing threshold-based TTB-WADD mixture theory, we present a spectrum of trials systematically varying the 'cue dispersion' (the difference in WADD scores between options). In all trials, the Take The Best (TTB) heuristic strongly favors Option A. However, the WADD score difference ranges smoothly from strongly favoring Option B (negative dispersion relative to A) to strongly favoring Option A (positive dispersion). The pure WADD theory predicts a smooth, monotonic sigmoid curve for choice probabilities as dispersion varies. Conversely, the competing theory posits a deterministic hard-switch: for dispersions below a certain threshold, subjects rely entirely on TTB, predicting a flat, high probability of choosing Option A regardless of the slight WADD variations. By densely sampling small and moderate dispersions, this design isolates the smooth probabilistic integration of WADD from the step-like, non-compensatory plateau predicted by the competing theory.

[6] To quantitatively dissociate the Soft Strategy Selection theory from pure Weighted Additive (WADD) theory, we systematically vary the 'cue dispersion' (absolute difference in WADD scores) across trials where Take The Best (TTB) and WADD make opposing predictions. In pure WADD, when the dispersion is very small, the choice probability approaches 50/50 because the weighted sums are nearly identical. In contrast, the Soft Strategy Selection theory predicts a non-linear shift: at low dispersions, the decision-maker will revert to the non-compensatory TTB heuristic, causing the choice probability to shoot up toward 100% for the TTB-favored option despite the small WADD difference. By providing a gradient of dispersions (from 0.15 to 1.85) where TTB favors Option A but WADD favors Option B, we can clearly observe whether choice probabilities smoothly approach 50/50 (pure WADD) or exhibit a sharp resurgence of TTB at low dispersions (Advocated theory).

[7] To quantitatively dissociate the pure Weighted Additive (WADD) theory from the Soft Strategy Selection theory, we systematically vary the 'cue dispersion' (absolute difference in WADD scores) across trials where Take The Best (TTB) and WADD make opposing predictions. According to the advocated pure WADD theory, when the dispersion is very small (e.g., 0.1), the choice probability will be near 50/50 because the weighted sums are nearly identical, and it will smoothly increase for the WADD-favored option as dispersion increases. In contrast, the competing Soft Strategy Selection theory predicts a non-linear shift: at low dispersions, the decision-maker will fall back on the non-compensatory TTB heuristic, causing the choice probability to sharply favor the TTB-predicted option. By providing a gradient of dispersions where TTB favors Option A but WADD favors Option B (and vice versa), we can clearly observe whether choice probabilities smoothly track the WADD score differences (pure WADD) or exhibit a sharp resurgence of TTB at low dispersions (Soft Strategy Selection).

[8] This design quantitatively dissociates the advocated Sequential Cue Integration (with Tallying Fallback) theory from the competing Weighted Additive (WADD) theory. The advocated theory posits that decision-makers accumulate evidence sequentially by validity and, if a threshold (minimum 2.0) is not crossed, fall back to simple Tallying (unweighted feature counting). We use six features with validities such that the maximum possible running sum of differences in our critical trials never reaches 2.0. Consequently, the advocated theory will ALWAYS fall back to Tallying on these trials. We meticulously construct trials where one option has fewer but highly valid features (winning under WADD), while the alternative has more but less valid features (winning under Tallying). This creates a stark, qualitative divergence: the advocated model predicts choices based strictly on feature counts, whereas the competing model predicts choices based on weighted sums.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the advocated Weighted Additive (WADD) theory from the competing Sequential Cue Integration theory, we exploit the competing theory's threshold parameter. The competing theory assumes that if the accumulated difference in cue validities never reaches a minimum threshold of 2.0, the decision-maker falls back to unweighted Tallying. By setting the validities and designing the trial pairs such that the maximum possible running difference is always strictly less than 2.0, we force the competing model to ALWAYS fall back to Tallying. We then pit the predictions of WADD (which integrates all validities compensatorily) against Tallying (which simply counts the number of winning features). In our critical trials, one option has fewer but highly valid features (favored by WADD), while the alternative has more but less valid features (favored by Tallying). This creates a stark qualitative dissociation between the two theories.",
  "validities": [
    0.95,
    0.85,
    0.65,
    0.55,
    0.5
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
      1,
      0,
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
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
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
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3630,
  "candidates_token_count": 473,
  "total_token_count": 7685
}
```
