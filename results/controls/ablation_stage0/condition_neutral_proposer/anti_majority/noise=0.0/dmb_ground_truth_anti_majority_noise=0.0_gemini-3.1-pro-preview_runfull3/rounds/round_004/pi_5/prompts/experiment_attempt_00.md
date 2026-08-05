# experiment_attempt_00

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** Decision-makers use an adaptive toolbox of strategies, dynamically selecting between Take The Best (TTB) and Weighted Additive (WADD) evaluation on a trial-by-trial basis. The probability of using TTB over WADD depends on the conflict between the cues: when the top discriminating cue is much stronger than the opposing cues, TTB dominates; when multiple opposing cues rival the top cue's validity, the decision-maker is more likely to switch to the compensatory WADD strategy. By applying independent scaling weights to the top cue's validity and the sum of opposing validities, the model can penalize a large number of weak opposing cues, keeping TTB dominant in those cases while still allowing WADD to differentiate between strong compensatory profiles.

**Parameters:**
- theta_top: [0.0, 20.0]
- theta_opp: [0.0, 20.0]
- bias_ttb: [-10.0, 10.0]
- gamma: [0.0, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    v_top = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            v_top = val[j]
            break
        if b[j] > a[j]:
            winner_ttb = 1
            v_top = val[j]
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
        p_ttb_weight = 0.5
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
        # Dynamic mixture weight based on conflict
        v_opp_sum = 0.0
        for j in range(len(val)):
            if winner_ttb == 0 and b[j] > a[j]:
                v_opp_sum += val[j]
            elif winner_ttb == 1 and a[j] > b[j]:
                v_opp_sum += val[j]
                
        theta_top = float(parameters["theta_top"])
        theta_opp = float(parameters["theta_opp"])
        bias_ttb = float(parameters["bias_ttb"])
        z_mix = theta_top * v_top - theta_opp * v_opp_sum + bias_ttb
        p_ttb_weight = 1.0 / (1.0 + np.exp(-z_mix))
        
    # --- Weighted Additive (WADD) ---
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores_wadd = np.array([score_a, score_b])
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # --- Strategy Mixture ---
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_wadd_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## THEORY 2
**Description:** Decision-makers evaluate options using a combination of the direct, unscaled linear sum of the mean-centered cue validities (Linear WADD) and a proportional tally of the cues. By centering validities around their mean rather than a fixed neutral point like 0.5, cues with below-average validity are treated as negative evidence (a dilution effect), meaning that adding weak cues to an option can actually decrease its overall evaluation. The tallying count is normalized into a proportion to put both strategies on a comparable numerical scale, allowing the mixing parameter to effectively balance the influence of the highest-validity cues against the sheer quantity of positive cues.

**Parameters:**
- w_tally: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_tally = float(parameters["w_tally"])
    
    # Linear WADD scores (mean-centered validities to capture dilution effects)
    centered_val = val - np.mean(val)
    wadd_a = np.sum(centered_val * a)
    wadd_b = np.sum(centered_val * b)
    
    # Tallying scores (proportion of positive features to fix scaling mismatch)
    tally_a = np.mean(a)
    tally_b = np.mean(b)
    
    # Combine scores
    score_a = (1.0 - w_tally) * wadd_a + w_tally * tally_a
    score_b = (1.0 - w_tally) * wadd_b + w_tally * tally_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Lapse noise
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] To distinguish Take The Best (TTB) from Tallying, we need trials where the option that is superior on the single most valid discriminating cue is inferior on the total number of winning cues. By setting up a 4-feature environment with descending validities, we construct pairs where Option A wins on the highest validity cue but Option B wins on multiple lower-validity cues. TTB will deterministically choose the option winning on the highest-validity cue, completely ignoring the total count of wins. Conversely, Tallying will simply count the number of winning features, ignoring validities, and choose the option with more winning features. This creates a stark quantitative dissociation between the models.

[1] To strongly dissociate Take The Best (TTB) from Tallying, we use a 5-feature environment with strictly descending validities. We design trials where one option dominates on the single highest-validity discriminating cue (triggering TTB to choose it), while the other option wins on a larger number of lower-validity cues (triggering Tallying to choose the latter). By varying which specific cues tie and the margins of the tallying wins (e.g., 3-to-1 vs. 3-to-2 vs. 4-to-1), we ensure the dissociation is robust across different levels of feature overlap.

[2] To maximally distinguish Take The Best (TTB) from Weighted Additive (WADD), we must exploit WADD's compensatory nature driven by its gamma parameter. While TTB strictly decides based on the single highest-validity discriminating cue, WADD integrates all cues weighted by a non-linear transformation of their validities. We propose a 5-cue environment with linearly spaced validities. The trial pairs are designed to pit the highest-validity discriminating cue (favoring Option A) against multiple lower-validity cues (favoring Option B). By varying the number and specific validities of the compensating cues, we create a gradient of thresholds where WADD will switch from Option B to Option A as gamma increases. TTB, on the other hand, will rigidly select Option A across all these critical pairs regardless of how many lower-validity cues are stacked against it.

[3] To robustly dissociate WADD from TTB, we use a 6-feature environment with a steady gradient of validities. The trials are constructed to pit the highest-validity discriminating cue (favoring Option A) against varying combinations of lower-validity cues (favoring Option B). TTB will deterministically choose Option A in all these critical trials because it relies solely on the first discriminating cue. In contrast, WADD's compensatory nature allows it to choose Option B when the sum of the subjectively weighted lower-validity cues exceeds the weight of the highest-validity cue. By varying the number and specific validities of the compensating cues, we map out the exact threshold of the gamma parameter at which WADD switches from a compensatory (Tallying-like) to a non-compensatory (TTB-like) strategy, creating a stark quantitative dissociation.

[4] To maximally distinguish the TTB/Tallying mixture model (THEORY 1) from the WADD model (THEORY 2), we exploit the fact that the mixture model relies only on the rank of the highest valid discriminating cue (for TTB) and the unweighted count of winning cues (for Tallying). WADD, on the other hand, is sensitive to the specific absolute validities of all cues. We use a 5-feature environment with linearly spaced validities and construct a series of pairs where the structural signals for TTB and Tallying are held perfectly constant. For example, in several pairs, Option A wins on the single highest-validity discriminating cue (triggering TTB), while Option B wins on exactly two lower-validity cues (triggering Tallying). THEORY 1 predicts identical choice probabilities across all these pairs because the TTB and Tallying outputs are identical. THEORY 2 (WADD) predicts systematically varying choice probabilities because the actual validities of the compensating cues change across pairs (e.g., cues 2 and 3 vs. cues 4 and 5).

[5] To perfectly dissociate THEORY 1 (WADD) from THEORY 2 (TTB/Tallying mixture), we construct a series of choice pairs where the structural signals for TTB and Tallying are held perfectly constant, while the absolute validities of the compensating cues vary. In all pairs, Option A is favored by the single highest-validity discriminating cue (triggering TTB to choose A with a constant score difference of 1 vs 0), while Option B is favored by exactly two lower-validity cues (triggering Tallying to choose B with a constant score difference of 2 vs 1). THEORY 2 will predict the exact same probability of choosing A across all these pairs because its components only care about rank (TTB) and unweighted counts (Tallying). Conversely, THEORY 1 (WADD) sums the non-linearly transformed validities, so it will predict systematically different choice probabilities as the actual validities of Option B's compensating cues shift from high to low.

[6] To maximally distinguish THEORY 1 (dynamic TTB/WADD mixture based on explicit cue conflict) from THEORY 2 (pure WADD model), we manipulate the identity of the top discriminating cue and the number/validity of opposing cues to decouple the pure WADD score difference from the explicit conflict signal. In THEORY 1, the probability of using TTB over WADD depends on the difference between the top cue's validity and the sum of the opposing cues' validities. By creating pairs where the WADD score difference is comparable but the absolute validities (v_top and v_opp_sum) vary significantly, THEORY 1 predicts abrupt shifts in choice probabilities due to strategy switching, whereas THEORY 2 predicts smoother, monotonic variations based on the non-linear compensatory sum.

[7] To maximally distinguish pure WADD (THEORY 1) from the adaptive TTB/WADD mixture (THEORY 2), we exploit the specific functional form of the strategy switch in THEORY 2. THEORY 2 transitions from TTB to WADD based on a linear combination of the top discriminating cue's validity (v_top) and the sum of the opposing cues' validities (v_opp_sum). By contrast, THEORY 1 uses a non-linear compensatory sum (via the gamma parameter) to evaluate all cues continuously. We construct a 6-feature environment and design trial pairs that decouple the non-linear WADD score differences from the linear v_opp_sum. For instance, we compare trials where a high-validity top cue is opposed by a few strong cues versus many weak cues. These configurations can yield similar WADD score differences (depending on gamma) but radically different v_opp_sum values, triggering a sharp strategy switch in THEORY 2 while producing only smooth probability changes in THEORY 1.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This experiment is designed to maximally dissociate THEORY 1 (TTB/WADD mixture) from THEORY 2 (Linear WADD with mean-centered validities + Tallying). THEORY 2 evaluates options using mean-centered validities, meaning that any feature with a validity below the average validity acts as *negative* evidence (a dilution effect). In contrast, THEORY 1 uses strictly positive validities (both in TTB and its WADD component), meaning that adding *any* positive feature to an option strictly increases its evaluation. By creating pairs where Option B is identical to Option A but with additional low-validity features, THEORY 1 will strictly prefer Option B (both TTB and WADD see positive evidence), whereas THEORY 2's WADD component will penalize Option B for these below-average features. The inclusion of Tallying in THEORY 2 ensures that the model can be fit to different degrees of this dilution effect, but the qualitative divergence on these 'added weak feature' trials provides a stark quantitative dissociation.",
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
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4518,
  "candidates_token_count": 700,
  "total_token_count": 7713
}
```
