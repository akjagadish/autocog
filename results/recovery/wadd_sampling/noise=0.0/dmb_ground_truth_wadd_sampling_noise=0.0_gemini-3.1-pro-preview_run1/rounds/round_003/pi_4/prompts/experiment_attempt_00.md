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
**Description:** Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory (Tallying-Biased with Softened Determinism): Decision-makers do not universally rely on a single compensatory mechanism. Instead, the population consists of a mixture of strategies using fast-and-frugal heuristics: 'Take-The-Best' (lexicographic) and 'Tallying' (unweighted sum of strict feature-wise wins). The population shows a stronger preference for Tallying over Take-The-Best, but choices are also somewhat stochastic. By softening the determinism of the individual heuristics, extreme choice probabilities are tempered, allowing the model to fit intermediate conflict trial outcomes more robustly without drastically shifting the population mixture.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.2]
- w_tally: [0.4, 0.9]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Order features by validity (highest first)
    order = np.argsort(val)[::-1]
    
    # Take-The-Best (TTB) evaluation
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying evaluation (count of strict feature-wise wins)
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    # TTB probabilities
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Response noise (epsilon-greedy)
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design pits the single most valid discriminating cue against a numerical majority of less valid cues. Take The Best (TTB) decides solely based on the highest-validity discriminating cue, entirely ignoring the lower-validity cues. Tallying, on the other hand, ignores cue validities and simply counts the number of features favoring each option. By presenting pairs where one option wins on the single most valid discriminating cue while the other option wins on multiple less valid cues, we create a strong quantitative dissociation between the two theories.

[1] To dissociate Tallying from Take The Best (TTB), this design pairs options such that the one favored by the highest-validity discriminating cue (the TTB choice) is consistently outnumbered by the total number of features favoring the alternative (the Tallying choice). Additionally, some trials feature a tie in the number of winning features (leading Tallying to guess) while maintaining a clear TTB preference. This provides a robust quantitative dissociation across different cue profiles.

[2] This design dissociates the Weighted Additive (WADD) strategy from Tallying by pitting a small number of high-validity features against a larger number of low-validity features. WADD computes a compensatory weighted sum, allowing strong cues to override a numerical majority of weaker cues. Tallying, by contrast, simply counts the number of winning features, ignoring validities entirely. The trial pairs include cases where WADD and Tallying make opposite predictions, cases where Tallying predicts a tie while WADD has a clear preference, and baseline cases where both models agree.

[3] This design dissociates Tallying from Weighted Additive (WADD) by contrasting the raw count of winning features against the sum of feature validities. By using five features with a steep drop-off in validity (two highly valid, three weakly valid), we can create scenarios where one option wins on a majority of features (favoring Tallying) while the other option wins on fewer but much more valid features (favoring WADD). We include trials where the models strongly disagree, trials where Tallying predicts a tie but WADD predicts a clear winner, and baseline trials where both models agree.

[4] To dissociate the Weighted Additive (WADD) strategy from the Non-linear Rank-Weighted Additive Strategy, we exploit the difference between true cue validities and their ordinal ranks. We use a set of validities where the top cues are closely clustered in validity, followed by a sharp drop-off to the lowest cue. WADD's compensatory sums will reflect these specific quantitative differences (e.g., heavily penalizing the lowest validity cue and treating the top ones as nearly interchangeable). In contrast, the rank-based model enforces a rigid ordinal structure (ranks 5, 4, 3, 2, 1) and scales them by a parameter gamma. This forces the rank model to either predict ties (when rank sums are equal, gamma=1), heavily favor the single highest-ranked cue (gamma > 1), or approximate tallying (gamma < 1), consistently failing to match WADD's continuous validity-summation in specific conflict trials.

[5] To dissociate the non-linear Rank-Weighted Additive Strategy from the standard Weighted Additive (WADD) model, we exploit the difference between true cue validities and their ordinal ranks. We use a set of validities where the top cues are closely clustered (e.g., 0.95, 0.90, 0.85, 0.80) with a drop-off to the lowest cue (0.50). WADD's compensatory sums will reflect these specific quantitative differences, often favoring an option with two closely-matched high-validity cues over an option with the single highest validity cue plus the lowest validity cue. In contrast, the rank-based model scales the integer ranks (5, 4, 3, 2, 1) by a power parameter gamma. When gamma > 1, the rank weights grow non-linearly, causing the highest-ranked cue to dominate. This allows the Rank-Weighted model to prefer options that WADD strictly rejects, providing a clear quantitative dissociation.

[6] To dissociate the Strategy Mixture Theory (Tallying + Take-The-Best) from the Non-linear Rank-Weighted Additive Strategy, we exploit how the two models handle ties in feature counts and varying cue combinations. By using pairs where Tallying predicts a tie while TTB has a clear preference, the Mixture model is forced to predict in the direction of TTB (since the Tallying component provides a uniform probability). In contrast, the Rank-Weighted model, depending on its gamma parameter, can show the opposite preference. For example, if Option A wins on ranks 4 and 1, and Option B wins on ranks 3 and 2, Tallying ties (2 vs 2) and TTB favors A (rank 4 > rank 3). Thus, the Mixture model strictly favors A. However, the Rank model with gamma < 1 (concave rank weighting) will favor B, because 3^g + 2^g > 4^g + 1^g for g < 1. We also include trials pitting a single high-rank cue against multiple lower-rank cues to capture divergence at higher gamma values.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate the Strategy Mixture Theory (Tallying + Take-The-Best) from the Non-linear Rank-Weighted Additive Strategy, we exploit how the two models handle feature count ties and the trade-off between feature quantity and rank quality. The Strategy Mixture model resolves feature-count ties strictly in the direction of the highest validity cue (TTB) and otherwise leans heavily toward the option with more winning features (Tallying). In contrast, the Rank-Weighted model interpolates between these extremes based on the non-linear scaling parameter gamma. For example, when Tallying predicts a tie (e.g., A wins ranks 5 and 1; B wins ranks 4 and 3), the Mixture model strictly favors A (via TTB), while the Rank-Weighted model with concave weighting (gamma < 1) favors B. Conversely, when an option has more winning features but lower ranks (e.g., A wins ranks 4, 2, 1; B wins ranks 5, 3), the Mixture model leans toward A (due to its Tallying bias), whereas the Rank-Weighted model with convex weighting (gamma > 1) strongly favors B.",
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
      1
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
      0,
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
      1,
      0,
      0,
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3504,
  "candidates_token_count": 496,
  "total_token_count": 6401
}
```
