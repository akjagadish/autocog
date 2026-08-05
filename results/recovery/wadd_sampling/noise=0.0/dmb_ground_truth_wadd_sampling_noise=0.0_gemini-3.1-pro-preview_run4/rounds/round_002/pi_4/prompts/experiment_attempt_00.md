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
**Description:** Decision-makers assign importance to features based on their ordinal rank in validity rather than their exact cardinal values. This Rank-Based Weighting heuristic avoids the extreme sensitivity to numerical validity differences seen in purely compensatory models, while still acknowledging that some cues are more diagnostic than others. Feature weights are computed as a power transformation of their inverse rank (e.g., 1 / rank^gamma). By restricting gamma to lower values, the model maintains a strong compensatory nature, ensuring that multiple lower-ranked cues can outweigh a single higher-ranked cue. Combined with a lower softmax temperature upper bound, it prevents overly deterministic choices and captures the noisier human behavior in conflicting trade-offs.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.6]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Based Weighting expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Determine the ordinal rank of each feature's validity (1 = highest validity)
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Weights are a power transformation of the inverse rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
**Description:** Decision-makers evaluate options by computing a weighted sum of their features, where the weights are a subjective power transformation of the cue validities. This Weighted Additive (WADD) model is fully compensatory, allowing the combined evidence of multiple lower-validity cues to potentially outweigh a single higher-validity cue, while still remaining sensitive to the varying diagnosticity of different features. The parameter governing the power transformation is constrained to prevent extreme exponentiation, ensuring the model remains compensatory rather than collapsing into a non-compensatory heuristic.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 3.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities into weights
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to compute choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To strongly dissociate Take The Best (TTB) from Tallying, we use a 5-feature design with strictly descending validities. The trials are constructed so that one option wins on the single highest-validity discriminating cue (the only cue TTB cares about), while the other option wins on a larger number of lower-validity cues (which Tallying simply counts). By pitting the most valid cue against a sheer numerical advantage in less valid cues, the two models make completely opposite predictions. We also include a trial where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction based on the higher-validity feature.

[1] To strongly dissociate Tallying from Take The Best (TTB), this design uses a 4-feature environment with strictly descending validities. The trials are crafted to pit the single highest-validity discriminating feature (which solely determines TTB's choice) against a larger number of lower-validity features (which dominate Tallying's count). The design includes trials where Tallying strongly prefers one option while TTB strongly prefers the other, as well as trials where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction based on the highest-validity discriminating cue.

[2] This design strictly dissociates the Weighted Additive (WADD) model from the Tallying heuristic by pitting a small number of highly valid features against a larger number of less valid features. Tallying ignores validities and simply counts the number of features on which each option is superior, leading to predictions that favor the option with a greater quantity of winning features, or predicting a tie when the counts are equal. WADD, being a compensatory model that weights features by their subjective validities, will instead favor the option with the higher weighted sum. The selected trials include cases where Tallying strongly prefers one option while WADD prefers the other, as well as trials where Tallying predicts a tie but WADD makes a deterministic choice based on the cumulative weight of the winning features.

[3] To quantitatively dissociate Tallying from the Weighted Additive (WADD) model, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. Tallying completely ignores validities and simply counts the number of features on which one option is strictly better than the other. In contrast, WADD computes a weighted sum where features are weighted by a power-transformation of their validities. The design includes trials where one option has more winning features (favored by Tallying) but the other option wins on the fewer, higher-validity features (favored by WADD). It also includes trials where both options win on an equal number of features (Tallying predicts a tie, resulting in random guessing), while WADD strongly prefers the option that wins on the more valid features.

[4] To quantitatively dissociate the Weighted Additive (WADD) model from Rank-Based Weighting, we use a 5-feature design where the top four validities are clustered closely together, while the fifth is much lower. In WADD, closely clustered validities result in similar weights, making the sum of multiple slightly-lower-validity features easily exceed a single higher-validity feature. In Rank-Based Weighting, weights are determined by inverse rank, which imposes a sharp, non-linear drop-off in importance regardless of the actual cardinal differences (e.g., the 1st feature is weighted twice as much as the 2nd feature, even if their validities are 0.95 and 0.90). This structural difference guarantees opposing predictions on trials where a single highest-validity feature is pitted against two slightly lower-validity features, or when comparing sums across different rank tiers.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Rank-Based Weighting from the Weighted Additive (WADD) model, we use a 5-feature design where the top four validities are clustered very closely together (e.g., 0.99, 0.97, 0.95, 0.93), while the fifth is much lower. In WADD, closely clustered validities result in similar cardinal weights, allowing the sum of two slightly-lower-validity features to easily exceed a single higher-validity feature. In Rank-Based Weighting, weights are determined by ordinal rank (e.g., 1/rank^gamma), which imposes a sharp, non-linear drop-off in importance regardless of the actual cardinal differences (e.g., the 1st feature is heavily favored over the 2nd feature, even though their validities are 0.99 and 0.97). This structural difference guarantees opposing predictions on trials where a single highest-validity feature is pitted against two slightly lower-validity features.",
  "validities": [
    0.99,
    0.97,
    0.95,
    0.93,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
      0,
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
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
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
      1
    ],
    [
      1,
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
  "prompt_token_count": 3091,
  "candidates_token_count": 436,
  "total_token_count": 5569
}
```
