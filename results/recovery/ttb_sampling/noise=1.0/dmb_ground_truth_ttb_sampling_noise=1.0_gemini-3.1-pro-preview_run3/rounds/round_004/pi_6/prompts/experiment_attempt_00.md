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
**Description:** Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying: Sum of active features for each option (ignoring validities)
    scores = np.sum(stim, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
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
**Description:** Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Tallying from WADD, we use an environment with 5 features where the first two validities are high and the remaining three are low. Tallying simply counts the number of features each option wins on, completely ignoring validities. WADD, however, weights each win by the feature's validity. By pitting an option that wins on the two most valid features (2 wins) against an option that wins on the three least valid features (3 wins), Tallying strongly prefers the latter while WADD strongly prefers the former. Adding trials where Tallying ties but WADD has a strict preference further teases apart the use of validity weights.

[1] To quantitatively dissociate WADD from Tallying, we use a 5-feature environment with a steep drop-off in validities (two highly valid features and three less valid features). Tallying simply counts the number of features each option wins on, completely ignoring validities. WADD computes a weighted sum using the validities. By pitting an option that wins on the two most valid features against an option that wins on the three least valid features, Tallying strictly prefers the latter (3 wins vs 2 wins), whereas WADD strictly prefers the former because the combined weight of the two high-validity features exceeds the combined weight of the three low-validity features. The design also includes trials where both models agree, and trials where Tallying ties but WADD has a strict preference, to robustly estimate the validity weighting parameter.

[2] To dissociate the non-compensatory Take-The-Best (TTB) model from the compensatory Weighted Additive (WADD) model, we use a 5-feature environment. The validities range from very high (0.95) to chance (0.50). We designed trials where the option favored by the single most valid cue is pitted against an option supported by multiple less valid cues. TTB, which stops at the first discriminating feature, will consistently choose the option with the highest valid cue. In contrast, WADD integrates all cues and will often prefer the option with multiple lesser cues because their combined weight exceeds that of the single best cue. Additional trials where the models agree or marginally disagree provide constraints on the noise and weighting parameters.

[3] To quantitatively dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Noisy Take-The-Best (TTB) model, we use a 5-feature environment with linearly decreasing validities. The design systematically pits the single most valid cue (or the top two) against a coalition of multiple less valid cues. Because TTB searches features in order of noisy validity and stops at the first discriminating cue, it will predominantly choose the option favored by the highest-validity feature. In contrast, WADD integrates all features, allowing the combined weight of several lower-validity features to easily overcome the single best feature. By varying the margin of WADD's preference and the probability that Noisy TTB samples the critical feature first, we can cleanly separate the two models.

[4] This design exploits the difference between 'Noisy TTB' (which assumes Gaussian noise on validities, causing features with similar validities to occasionally swap in the search hierarchy) and 'Strategy Mixture' (which assumes strict, noiseless TTB mixed with Tallying). By using validities where the top two features are very close (0.90 and 0.85) and the remaining features are lower, we can pit them against each other. In a trial where Option A has only feature 2 and Option B has only feature 1, Strategy Mixture predicts no preference for A (strict TTB chooses B, Tallying ties), while Noisy TTB predicts a significant preference for A due to validity swapping. Conversely, when Option A has feature 2 plus several lower-tier features and Option B has feature 1, Strategy Mixture predicts a jump in preference for A (since Tallying now strongly favors A), whereas Noisy TTB predicts the same choice probability as the first trial because the top two features still dominate the search process.

[5] This design leverages a structural invariance in the Strategy Mixture model to cleanly dissociate it from Noisy TTB. We use a 5-feature environment where the most valid feature is highly diagnostic (0.95), while the remaining four features are clustered tightly together (0.75, 0.73, 0.71, 0.69). In Trial 1, Option A is supported only by the top feature (0.95), while Option B is supported by all four lower features. In Trial 2, Option A is supported only by the second-best feature (0.75), while Option B is supported by the bottom three. According to the Strategy Mixture model, both trials present the exact same conflict: strict TTB chooses Option A, and Tallying chooses Option B. Thus, the Strategy Mixture model predicts identical choice probabilities for Option A across both trials. In contrast, Noisy TTB predicts a much higher choice probability for Option A in Trial 1 (because the 0.95 validity is far enough above the rest to resist noise-induced rank swapping) than in Trial 2 (where the 0.75 validity is easily swapped with the tightly clustered lower features, leading to frequent choices of Option B). Additional trials vary the tallying margins to further constrain parameters.

[6] To quantitatively dissociate Noisy Take-The-Best (TTB) from the highly noisy Weighted Additive (WADD) model, we exploit both the structural differences (non-compensatory vs. compensatory) and the distinct noise regimes parameterized for each theory. The competing WADD model is restricted to a very high lapse rate (epsilon >= 0.5) and low inverse temperature (beta <= 0.5), forcing its predictions to be highly compressed toward 0.5, while always leaning toward the option with the highest sum of validities. In contrast, the advocated Noisy TTB model can exhibit much sharper choice probabilities (epsilon as low as 0.0) and relies on sequential search. We design trials that pit the single most valid feature against a coalition of lesser features. In these trials, Noisy TTB strongly predicts a preference for the option favored by the top feature, whereas the WADD model predicts a slight preference (barely above chance) for the option with the coalition of lesser features. This opposing directionality combined with the stark difference in predicted choice extremity provides a robust dissociation.

[7] To quantitatively dissociate the highly noisy Weighted Additive (WADD) model from the Noisy Take-The-Best (TTB) model, we exploit both the compensatory vs. non-compensatory structural differences and their distinct noise assumptions. The advocated WADD model assumes decisions are compensatory but heavily smoothed by high lapse rates and low inverse temperatures, predicting a slight but consistent preference for the option with the highest sum of validities. In contrast, Noisy TTB predicts sharper preferences dictated primarily by the single most valid cue. By pitting a single highly valid feature against a coalition of multiple slightly less valid features, WADD predicts a muted preference for the coalition, whereas Noisy TTB predicts a strong preference for the single best feature. Observing a reliable but weak preference for the coalition would support the highly noisy WADD account over TTB.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Noisy Tallying model from the Noisy Weighted Additive (WADD) model, we exploit the fundamental difference in how they treat feature validities. Tallying ignores validities entirely, simply counting the number of active features for each option. WADD, conversely, weights each active feature by its validity. By using a 5-feature environment with a steep drop-off in validities (e.g., two highly valid features and three weakly valid features), we can pit an option that wins on the two most valid features against one that wins on the three least valid features. In this critical trial, Tallying strictly prefers the latter (3 wins vs. 2 wins), whereas WADD strictly prefers the former because the combined weight of the two high-validity features exceeds that of the three low-validity features. Even with high noise (high epsilon, low beta) compressing choice probabilities toward 0.5, the models predict opposite directionalities (p(A) > 0.5 vs. p(A) < 0.5). Additional trials where Tallying ties but WADD has a strict preference further isolate the effect of validity weighting.",
  "validities": [
    0.95,
    0.85,
    0.6,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
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
      1,
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
      1
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3550,
  "candidates_token_count": 529,
  "total_token_count": 6570
}
```
