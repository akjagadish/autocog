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
**Description:** Normalized Rank-Dependent Evidence Accumulation: Decision-makers use a unified compensatory process where the subjective weight of each feature decays exponentially based on its validity rank. To maintain consistent sensitivity across contexts with varying numbers of features, these rank-based weights are normalized to sum to 1.0 before evidence accumulation. This ensures the decision mechanism's determinism scales uniformly across different experimental environments.

**Parameters:**
- gamma: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Assign exponentially decaying weights based on rank
    weights = np.zeros_like(val)
    for rank, cue_idx in enumerate(cue_order):
        weights[cue_idx] = gamma ** rank
        
    # Normalize weights to sum to 1.0
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
        
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add random lapse noise
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** WADD / Take-The-Best Strategy Mixture Theory with Calibrated Determinism: Decision-makers probabilistically choose between a fully compensatory Weighted Additive (WADD) strategy and a non-compensatory Take-The-Best (TTB) strategy on any given trial. By mixing WADD and TTB, the model captures both dominance violations (via WADD) and choice probability plateaus (via TTB). Independent inverse temperatures allow each strategy to scale properly. To match the high level of equivocation in human data across multiple experiments, the determinism of these strategies is strongly bounded, ensuring predictions remain appropriately probabilistic.

**Parameters:**
- beta_wadd: [0.0, 3.0]
- beta_ttb: [0.0, 3.0]
- p_wadd_weight: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    p_wadd_weight = float(parameters["p_wadd_weight"])
    
    # --- WADD Prediction ---
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Take-The-Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Mixture & Noise ---
    p_mix = p_wadd_weight * p_wadd + (1.0 - p_wadd_weight) * p_ttb
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design strictly pits Take The Best (TTB) against Tallying by creating a complete dissociation between the two models. In every trial, one option is favored by the highest-validity discriminating cue (which TTB relies on exclusively), while the other option possesses a greater total number of positive cues (which Tallying simply counts). Consequently, TTB and Tallying will predict exactly opposite choices for every single trial. This allows for a robust quantitative evaluation of whether subjects rely on a single dominant cue or sum across all available cues.

[1] This design systematically varies the Tallying difference (number of features favoring A minus number of features favoring B) while holding the Take The Best (TTB) prediction constant. By including trials where Tallying strongly opposes TTB, weakly opposes TTB, ties, weakly agrees, and strongly agrees, we can test whether subjects' choice probabilities scale with the Tallying count difference (as predicted by Tallying with a softmax choice rule) or remain a flat step-function determined solely by the highest-validity cue (as predicted by TTB).

[2] This design pits WADD against Tallying by manipulating the distribution of feature validities. Tallying ignores validities and simply counts the number of winning features for each option, predicting a tie when the counts are equal and a strict preference when one option has more winning features. WADD, however, weights each feature by a subjective transformation of its validity. We include 'Tallying Tie' trials where Tallying predicts random guessing (50/50), but WADD strongly predicts the option with higher-validity features. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by WADD, depending on the gamma parameter). This creates a quantitative dissociation where WADD predicts a graded probability of choice based on the weighted sum, whereas Tallying predicts either a step-function preference based on counts or a complete tie.

[3] This design isolates the effect of cue validities to cleanly dissociate Tallying (which ignores validities and only counts feature wins) from WADD (which weights features by a subjective transformation of their validities). We use four features with a wide spread of validities. The trials include 'Tallying Ties' where each option wins on the same number of features but differs in validity, forcing Tallying to predict random guessing while WADD predicts a strong preference. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) while the other option wins on fewer but higher-validity features (favored by WADD, especially for higher values of gamma). This provides a rigorous quantitative test of whether subjects completely ignore validities as Tallying predicts.

[4] To decisively dissociate WADD from the Tallying/TTB Strategy Mixture, we introduce 'Dominance Violation' trials. In these critical trials, Option A possesses the single highest-validity cue (winning TTB) AND a greater total number of winning cues (winning Tallying). Because both strategies in the Mixture model agree, the Mixture model strictly predicts Option A. However, WADD with a non-linear validity transformation (gamma > 1) will predict Option B if Option B's fewer cues are collectively strong enough (e.g., the second and third highest validities) to outweigh Option A's combination of one top cue and several weak cues. This creates a qualitative divergence that no mixture of Tallying and TTB can explain. We also include standard conflict trials to calibrate the mixture weights and gamma.

[5] This design tests for a qualitative signature of the Strategy Mixture Theory that WADD cannot capture: the decoupling of determinism between TTB and Tallying. By varying the feature count difference (e.g., Option A having 1 vs 4 features, or 2 vs 3 features) while holding the TTB winner constant, WADD strictly predicts a monotonic change in choice probabilities because the weighted sum difference must change. In contrast, the Strategy Mixture Theory can predict perfectly flat (plateaued) choice probabilities across these trials if the Tallying inverse temperature (beta_tally) is high, because Tallying becomes deterministic as long as the count difference is non-zero, regardless of its magnitude. This allows the mixture model to predict identical choice proportions for pairs with drastically different feature-sum differences.

[6] This design strictly dissociates the WADD/TTB mixture model from the Tallying/TTB mixture model by creating 'Double Agreement' trials where TTB and Tallying both favor the same option, but WADD strongly favors the opposite option. In the Tallying/TTB model, if both heuristics agree, the model must predict that option with high probability. The WADD/TTB mixture, however, can predict the opposite option if WADD's compensatory weighting of multiple high-validity cues outweighs the single highest-validity cue and the numerous low-validity cues. We use a set of 7 features with a specific validity distribution (four very high validities, three very low validities) to construct these critical dissociations, along with conflict and tie trials to calibrate the mixture weights and inverse temperatures.

[7] This design isolates the quantitative signature of the Strategy Mixture Theory with Decoupled Determinism (Tallying + TTB) and strictly dissociates it from the WADD + TTB mixture. The core logic relies on creating pairs of trials where the Tallying count difference and the TTB winner are held perfectly constant, but the WADD score difference flips direction. According to the advocated theory, because Tallying only computes raw count differences and ignores validities, it will produce the exact same choice probability for both trials in the pair. The WADD + TTB theory, however, must predict drastically different choice probabilities across the pair because the compensatory WADD score shifts from favoring Option A to favoring Option B. By comparing trials where Tallying and TTB predictions are fixed but WADD predictions vary, we can test whether subjects' behavior reflects pure counting (Tallying) or weighted summation (WADD).

[8] This design introduces a strict quantitative dissociation between the Normalized Rank-Dependent model (Competing) and the WADD/TTB Strategy Mixture (Advocated). By structuring the validities with a steep drop-off between the top three and bottom two cues, we create critical 'Trap' trials (Pairs 1-4). In these trials, Option A wins on the highest available validity cue (winning TTB), ensuring the Rank-Dependent model strictly predicts P(A) >= 0.5 for all possible values of its exponential decay parameter (gamma). However, Option B's cues possess a higher sum of raw validities, meaning WADD strictly favors Option B. If subjects choose Option B more than 50% of the time, the Rank-Dependent model is mathematically falsified because it cannot produce P(A) < 0.5, whereas the Strategy Mixture model naturally captures this via a high WADD mixture weight. Additional conflict trials (Pairs 5-8) are included to calibrate the gamma and mixture parameters.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design strictly dissociates the Advocated Theory (Normalized Rank-Dependent) from the Competing Theory (WADD/TTB Mixture) by exploiting the difference between rank-based weighting and raw-validity weighting. We use a validity distribution with a cluster of high validities and a cluster of low validities: [0.95, 0.94, 0.93, 0.55, 0.50]. In critical trials (e.g., A wins on rank 0 and 3; B wins on rank 1 and 2), the Rank-Dependent model mathematically MUST favor Option A for any decay parameter gamma < 1, because 1 + gamma^3 > gamma + gamma^2. However, the Competing Theory's WADD component strongly favors Option B because the sum of raw validities for B (0.94 + 0.93 = 1.87) vastly exceeds A (0.95 + 0.55 = 1.50). If subjects choose B, the Advocated theory is falsified as it cannot predict P(A) < 0.5 on these trials.",
  "validities": [
    0.95,
    0.94,
    0.93,
    0.55,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      1,
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
      0,
      0,
      1
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
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4109,
  "candidates_token_count": 599,
  "total_token_count": 6914
}
```
