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
**Description:** Weighted Additive Model with Power-Scaled Log-Odds (WADD-Gamma). Decision-makers compute a weighted sum of features for each option. The weights are derived from the log-odds of the cue validities, raised to a power gamma. This parameterization allows the model to smoothly interpolate between Tallying/Equal-Weighting (gamma = 0) and standard log-odds WADD (gamma = 1). Choices are then made via a softmax over the weighted sums, incorporating an independent lapse rate for noise.

**Parameters:**
- gamma: [0.0, 0.75]
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to ensure log-odds are strictly positive and well-defined
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Parameters:**
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To cleanly dissociate Take The Best (TTB) from Weighted Additive (WADD), we must construct 'compensatory' trials where the highest-validity discriminating cue points to one option, but the sum of the remaining lower-validity cues points to the other option. TTB, being a non-compensatory one-reason heuristic, will exclusively follow the highest-validity cue. WADD, being compensatory, will integrate all cues and can choose the option favored by multiple lower-validity cues if their summed weight exceeds that of the single highest-validity cue. We include a mix of these critical dissociation trials and agreement (catch) trials to accurately estimate response noise and model parameters.

[1] To quantitatively dissociate Weighted Additive (WADD) from Take The Best (TTB), we employ a 4-feature design where the highest validity cue strongly discriminates, but the sum of lower validity cues outweighs it. TTB strictly follows the highest validity cue, whereas WADD integrates all cues. We include compensatory trials where the models predict opposite choices, and non-compensatory trials where both models agree, ensuring robust estimation of response noise and specific model mechanisms.

[2] To quantitatively dissociate Tallying (Equal Weighting) from Weighted Additive (WADD), we must construct trials where the two models make divergent predictions based on their core mechanisms. Tallying assumes decision-makers count the number of positive features for each option, ignoring the validities (weights) of the cues. In contrast, WADD computes a weighted sum of features, heavily prioritizing cues with higher validities. By using a 5-feature design with a steep drop-off in validities, we can create 'conflict' trials where one option has fewer positive features (disfavored by Tallying) but those features have much higher validities (favored by WADD). We also include 'tie' trials where both options have the same number of positive features (Tallying predicts guessing) but differ in the validities of those features (WADD predicts a clear preference), as well as 'agreement' trials where both models favor the same option. This design provides maximum leverage to distinguish uniform evidence accumulation from validity-weighted compensatory integration.

[3] This design quantitatively dissociates the Weighted Additive (WADD) model from Tallying (Equal Weighting) by exploiting trials where the models diverge due to feature validity integration. Tallying simply counts the number of positive features, whereas WADD integrates both the presence of features and their respective validities. By using a 5-feature design with a steep drop-off in validities, we create strong dissociation trials: for instance, Option A has fewer positive features but they are highly valid (WADD favors A), while Option B has more positive features with lower validities (Tallying favors B). We also include 'tie' trials where Tallying predicts guessing but WADD predicts a clear preference, and agreement trials to properly estimate baseline noise and specific parameter distributions.

[4] To quantitatively dissociate Tallying from Probabilistic Search Take-The-Best (PS-TTB), we employ a 5-feature design with a steep gradient in cue validities. Tallying simply counts the number of positive features and ignores validities, whereas PS-TTB probabilistically searches through cues weighted by their validities and decides based on the first discriminating cue. The design includes 'conflict' trials where one option has more positive features (favored by Tallying) but the competing option is favored by the highest-validity cue (favored by PS-TTB). We also include 'tie' trials where both options have the same number of positive features (Tallying predicts guessing) but the features favoring one option have much higher validities (PS-TTB strongly favors this option), providing a robust quantitative dissociation.

[5] To quantitatively dissociate Probabilistic Search Take-The-Best (PS-TTB) from Tallying, we utilize a 5-feature design with a steep gradient in cue validities. Tallying decides based on the simple sum of positive features irrespective of their validities, whereas PS-TTB probabilistically searches cues weighted by their validities and decides on the first discriminating cue. We include 'conflict' trials where one option has fewer positive features but is favored by the highest-validity cues (Tallying favors the option with more features, PS-TTB favors the one with the highest-validity feature). We also include 'tie' trials where both options have an equal number of positive features (Tallying predicts guessing) but the features differ in validity (PS-TTB strongly favors the option with higher-validity features).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate WADD-Gamma from Probabilistic Search Take-The-Best (PS-TTB), we use a 5-feature design. WADD-Gamma is a compensatory model that integrates all features (with weights scaling by a power of log-odds), whereas PS-TTB sequentially searches cues weighted by validities and stops at the first discriminating feature. The critical dissociation trials are 'compensatory' trials where the single highest-validity cue points to Option B, but the sum of the remaining lower-validity cues strongly points to Option A. PS-TTB will reliably favor Option B because it finds the highest-validity cue first. WADD-Gamma, regardless of its gamma parameter (whether it acts more like Tallying or pure WADD), will favor Option A because the combined weight of the numerous lower-validity cues exceeds the single highest-validity cue.",
  "validities": [
    0.92,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
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
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
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
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
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
      0,
      0,
      0,
      1
    ],
    [
      0,
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
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3188,
  "candidates_token_count": 502,
  "total_token_count": 5559
}
```
