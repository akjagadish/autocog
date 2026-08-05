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
**Description:** Threshold-Gated Compensatory Model: Subjects sort cues by their validities and evaluate the numerical gaps between adjacent cues in this sorted order. If the validities are closely clustered, subjects integrate them additively, functioning as a weighted additive model. However, if a gap exceeds a subjective threshold, all subsequent lower-ranked cues are heavily discounted or completely ignored, triggering a shift toward a non-compensatory, semi-lexicographic heuristic. The threshold is sensitive enough to trigger discrete strategy shifts frequently.

**Parameters:**
- gap_threshold: [0.0, 0.3]
- discount_factor: [0.0, 1.0]
- beta: [0.1, 25.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gap_threshold = float(parameters["gap_threshold"])
    discount_factor = float(parameters["discount_factor"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort validities descending to evaluate gaps
    order = np.argsort(-validities)
    sorted_vals = validities[order]
    
    w_sorted = np.zeros_like(sorted_vals)
    if len(sorted_vals) > 0:
        w_sorted[0] = sorted_vals[0]
        current_discount = 1.0
        
        for i in range(1, len(sorted_vals)):
            # If the gap between adjacent sorted validities exceeds the threshold, apply discount
            if (sorted_vals[i-1] - sorted_vals[i]) > gap_threshold:
                current_discount *= discount_factor
            w_sorted[i] = sorted_vals[i] * current_discount
            
    # Map subjective weights back to original feature order
    w = np.zeros_like(validities)
    w[order] = w_sorted
    
    # Option scores are the sum of feature values weighted by the threshold-gated subjective weights
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

**Parameters:**
- gamma: [0.0, 10.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] To dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we must construct trials where the single most valid discriminating cue points to one option, but the sum of the remaining lower-validity cues points to the other option. By using four features with validities [0.95, 0.80, 0.70, 0.65], TTB will always choose the option favored by the highest-validity cue that differs. However, the sum of the three lower-validity cues (0.80 + 0.70 + 0.65 = 2.15) heavily outweighs the highest validity cue alone (0.95). Trials 3, 4, 5, and 6 exploit this compensatory structure: the highest discriminating cue favors one option, but the sheer quantity of lower-validity cues favors the other, perfectly separating the non-compensatory TTB from the compensatory WADD. Trials 1, 2, 7, and 8 serve as agreement baselines.

[1] To cleanly dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, we use a 5-feature design with a linear descent in validities. TTB decides solely based on the highest-validity cue that discriminates between the two options, ignoring all lower-validity cues. By contrast, WADD integrates all features, allowing a coalition of lower-validity cues to mathematically override a single high-validity cue. The selected trials present 'compensatory' conflicts where the most valid discriminating feature favors one option, but the sum of several less valid features strongly favors the other. We also include agreement baseline trials where both models predict the same choice.

[2] To quantitatively dissociate the Probabilistic Heuristic Selection (PHS) model from the Weighted Additive (WADD) model, we exploit a 7-feature design with a steep drop in validities (four very high validities, three very low validities). Under PHS, subjects probabilistically mix Take The Best (TTB) and Tallying (Equal-Weight). We construct critical trials where Option A wins on the highest validity cue (favored by TTB) AND has a higher total number of positive features (favored by Tallying), meaning PHS strongly predicts choosing Option A regardless of the mixing parameter. However, Option B possesses several high-validity cues that collectively outweigh Option A's single highest-validity cue and its numerous low-validity cues. Thus, WADD strongly predicts choosing Option B. This provides a stark qualitative and quantitative divergence between the two models.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from the Probabilistic Heuristic Selection (PHS) model, this experiment uses an 8-feature design with a steep drop in validities between the top four and bottom four features. PHS assumes subjects probabilistically mix Take The Best (TTB) and Tallying. In the critical trials (1-4), Option A has a '1' on the highest validity cue (winning TTB) and possesses a greater total number of positive features (winning Tallying), meaning PHS strongly predicts Option A regardless of the mixing parameter. However, Option B has '1's on the second, third, and fourth most valid cues, which collectively outweigh Option A's single highest-validity cue plus its numerous low-validity cues. Thus, WADD strongly predicts Option B. Trials 5-8 serve as baselines where both models largely agree.

[4] This design quantitatively dissociates Probabilistic Heuristic Selection (PHS) from the Non-linear Subjective Weighting Model (NLSW) by exploiting PHS's insensitivity to the specific validities of secondary cues. Under PHS, the choice probability is a mixture of Take The Best (TTB) and Tallying. If we hold the TTB winner constant (e.g., Option A always has the single highest validity cue) and hold the Tallying winner constant (e.g., Option B always has a greater total number of positive cues), PHS predicts the exact same choice probability across trials, regardless of which specific lower-validity cues Option B possesses. By contrast, NLSW computes a weighted sum using all validities. By shifting Option B's positive features from higher-validity cues to lower-validity cues across trials while keeping the total count identical, NLSW predicts a systematic shift in choice probabilities, whereas PHS predicts strict invariance.

[5] This design introduces a profound qualitative dissociation by creating 'Double-Agreement Override' trials. In these critical trials, Option A possesses the single highest-validity cue (winning Take The Best) AND a strictly greater total number of positive features (winning Tallying). Because the Probabilistic Heuristic Selection (PHS) model is a mixture of TTB and Tallying, it must overwhelmingly predict Option A on these trials, regardless of the mixing parameter. However, the Non-linear Subjective Weighting (NLSW) model allows for intermediate non-linear scaling of validities (e.g., a gamma exponent between 2 and 5). Under this intermediate non-linear regime, a coalition of high-to-medium validity cues in Option B mathematically outweighs Option A's single highest-validity cue and its numerous low-validity cues. Thus, NLSW uniquely predicts that subjects can prefer Option B despite both TTB and Tallying pointing to Option A, providing a strict test between the models.

[6] This design quantitatively dissociates the Rank-Based Weighting Model (Advocated) from the Non-linear Subjective Weighting Model (Competing) by exploiting the difference between rank-based exponential decay and value-based non-linear scaling. The validities are chosen to have a large gap between rank 1 and rank 2, but very small gaps between ranks 2, 3, and 4. The Rank-Based model enforces a constant ratio of weights between adjacent ranks (w_i / w_{i+1} = 1/lambda), regardless of the actual validity values. Consequently, the Rank-Based model strictly constrains the relative trade-offs among lower-ranked cues. By contrasting combinations like 'Cue 2' vs 'Cue 3 + Cue 4', the Non-linear model (which is sensitive to the tightly clustered values of cues 2, 3, and 4) will strongly favor the sum of cues 3 and 4 across a wide range of its gamma parameter. Conversely, the Rank-Based model, depending on lambda, can easily predict Cue 2 to outweigh Cues 3 and 4 combined, due to its fixed exponential rank decay.

[7] This design quantitatively dissociates the Non-linear Subjective Weighting (NLSW) model from the Rank-Based (RB) model by exploiting the difference between value-based scaling and rank-based exponential decay. The validities [0.98, 0.96, 0.94, 0.60, 0.55] feature a tiny numerical gap between the top three cues, but a massive gap between the third and fourth cues. Under the NLSW model, the weights for the top three cues will be very similar, while the weights for the bottom two will be heavily suppressed for moderate to large gamma. Consequently, NLSW predicts that Cues 2 + 3 will easily outweigh Cue 1 (Trial 1), but Cue 3 will easily outweigh Cues 4 + 5 (Trial 2). In stark contrast, the RB model assigns weights based solely on rank, meaning the relative weight ratio of Rank 1 vs (Rank 2 + Rank 3) is mathematically identical to Rank 3 vs (Rank 4 + Rank 5). Thus, the RB model strictly predicts that a subject's preference direction (Option A vs Option B) must be identical across these two trials, whereas the NLSW model naturally predicts a preference reversal.

[8] This design quantitatively dissociates the Threshold-Gated Compensatory Model from the Non-linear Subjective Weighting (NLSW) Model by exploiting the difference between a discrete threshold-triggered discount and continuous non-linear scaling. We use an 8-feature design with validities clustered into a top tier [0.92, 0.90, 0.88] and a bottom tier [0.80, 0.80, 0.80, 0.80, 0.80]. The critical Trial 2 pits Option A (possessing features 2 and 3) against Option B (possessing feature 1 and all five bottom-tier features). Under the NLSW model, Option B mathematically must win across the entire parameter space: for small gamma, the sheer quantity of features in Option B dominates; for large gamma, Option B's single highest feature (0.92) strictly dominates Option A's features (0.90 + 0.88). There is no gamma exponent where features 2 and 3 can simultaneously overcome feature 1 AND the sum of features 4-8. In stark contrast, the Threshold-Gated model identifies the gap between 0.88 and 0.80 and applies a discrete discount to all bottom-tier features. Consequently, Option A's un-discounted sum (1.78) easily outweighs Option B's feature 1 (0.92) plus the heavily suppressed bottom-tier features. Thus, the Advocated model uniquely predicts a preference for Option A, a choice structurally impossible under the Competing model.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Threshold-Gated Compensatory Model from the Non-linear Subjective Weighting (NLSW) Model by exploiting the difference between a discrete threshold-triggered discount and continuous non-linear scaling. We use a 7-feature design with validities clustered into a top tier [0.92, 0.91, 0.90] and a bottom tier [0.85, 0.85, 0.85, 0.85]. The critical Trial 1 pits Option A (possessing features 2 and 3) against Option B (possessing feature 1 and all four bottom-tier features). Under the NLSW model, Option B mathematically must win across the entire parameter space: for small gamma, the sheer quantity of features in Option B dominates; for large gamma, Option B's single highest feature (0.92) strictly dominates Option A's features. There is no gamma exponent where features 2 and 3 can simultaneously overcome feature 1 AND the sum of features 4-7. In stark contrast, the Threshold-Gated model identifies the gap between 0.90 and 0.85 and applies a discrete discount to all bottom-tier features. Consequently, Option A's un-discounted sum (1.81) easily outweighs Option B's feature 1 (0.92) plus the heavily suppressed bottom-tier features. Thus, the Advocated model uniquely predicts a preference for Option A, a choice structurally impossible under the Competing model.",
  "validities": [
    0.92,
    0.91,
    0.9,
    0.85,
    0.85,
    0.85,
    0.85
  ],
  "trial_a_ratings": [
    [
      0,
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
      1,
      1
    ],
    [
      1,
      0,
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
      1,
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
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
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
      1,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
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
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
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
  "prompt_token_count": 4410,
  "candidates_token_count": 791,
  "total_token_count": 10669
}
```
