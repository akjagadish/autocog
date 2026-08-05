# experiment_attempt_01

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
**Description:** People make choices by computing a weighted sum of all available features for each option, but the weights they use are a non-linear transformation of the objective cue validities. By exponentiating the validities with a free parameter gamma, the strategy can smoothly interpolate between equal weighting (Tallying, gamma=0) and a strong reliance on the most valid cues (approximating Take The Best, gamma > 1). The choice is then made probabilistically based on the difference between the options' weighted sums, with a bounded inverse temperature beta to allow for more stochastic choices.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    subjective_weights = val ** gamma
    
    # Calculate weighted sums for both options
    a, b = stim[0], stim[1]
    score_a = np.sum(a * subjective_weights)
    score_b = np.sum(b * subjective_weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate response noise (lapse rate)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
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
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the highest-validity feature often points to one option while the majority of the remaining features point to the other. TTB decides solely based on the first discriminating cue (in descending order of validity), ignoring the rest. Tallying counts the number of strictly winning features for each option, ignoring validities. For instance, if Option A wins on the most valid feature but Option B wins on the three less valid features, TTB chooses A while Tallying chooses B. We also include trials where the top cue is tied, forcing TTB to look at the second cue, while Tallying counts the remaining features.

[1] To robustly dissociate Tallying from Take The Best (TTB), we use a 5-feature design. Tallying simply counts the number of winning features for each option, treating all features equally regardless of validity. TTB, on the other hand, strictly follows the validity hierarchy, stopping at the first discriminating cue. The trials are designed such that the highest-validity discriminating cue often points to one option, while a numerical majority of lower-validity cues points to the other. We also include trials where the top cue(s) are tied, forcing TTB to rely on the second or third best cue, which again opposes the overall tally. A few agreement trials are included as a baseline.

[2] To dissociate WADD from Tallying, we use a 5-feature design where features have a wide spread of validities. Tallying simply counts the number of features favoring each option, ignoring validities. WADD, however, weights features by a non-linear transformation of their validities. The trials are constructed so that one option has a numerical majority of winning features (favored by Tallying), while the other option wins on fewer but highly valid features (favored by WADD, depending on the gamma parameter). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers the option with higher-validity features.

[3] To robustly dissociate Tallying from WADD, we use a 5-feature design with a linear spread of validities (0.9 to 0.5). Tallying computes a simple sum of winning features for each option, completely ignoring the validities. In contrast, WADD computes a weighted sum where the validities are transformed by a non-linear parameter gamma. The trials are carefully chosen to pit the number of winning features against the validities of those features. For instance, some trials feature an option that wins on a single high-validity cue but loses on multiple lower-validity cues, leading Tallying to predict the latter and WADD (especially with gamma > 1) to predict the former. We also include several 'tie' trials where Tallying predicts equal preference (e.g., both options win on 2 features), but WADD strongly predicts the option with the higher-validity features.

[4] To quantitatively dissociate the pure Weighted Additive (WADD) model from the Mixture (Tallying + WADD) model, this design manipulates the level of agreement between Tallying (which counts strictly winning features) and WADD (which integrates non-linearly scaled validities). We include trials where Tallying predicts a strict tie (e.g., each option wins on exactly one feature) but WADD strongly favors one option due to cue validity differences. In the mixture model, a Tallying tie pulls the predicted probability strongly toward 0.5 (scaled by the mixture parameter alpha). By contrasting these tie trials with trials where Tallying strongly opposes the highest-validity cues, the mixture model is forced into a trade-off: to capture choices that align with Tallying, it must increase alpha, which inevitably dilutes its ability to capture extreme choice probabilities on the Tallying-tie trials. The pure WADD model, lacking the Tallying mixture, predicts consistent, validity-driven choices across both trial types without this artificial dilution.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the pure Weighted Additive (WADD) model (Advocated) from the Mixture model (Competing) by exploiting a structural difference in how they scale evidence. The Competing model normalizes the WADD score difference by the sum of all subjective weights before applying the inverse temperature 'beta'. By using 8 features with high validities, the sum of weights becomes very large (e.g., > 6.5). The pure WADD model does not normalize by the sum of weights. Consequently, for trials where the unnormalized score difference is moderate but the total sum of weights is large, the pure WADD model can easily predict highly deterministic choices (using its beta parameter up to 5.0). In contrast, the Competing model divides this score difference by ~6.5, shrinking it so much that even with its maximum beta of 10.0, it is mathematically incapable of predicting the same level of determinism. By including trials where Tallying predicts a tie (further drawing the Mixture model toward 0.5) alongside trials with varying WADD score differences, we create a regime where the Advocated model can capture high choice probabilities that the Competing model strictly underpredicts.",
  "validities": [
    1.0,
    0.95,
    0.9,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65
  ],
  "trial_a_ratings": [
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
      1,
      1,
      0,
      0,
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
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
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
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
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
  "prompt_token_count": 3408,
  "candidates_token_count": 661,
  "total_token_count": 7478
}
```
