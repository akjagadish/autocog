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
**Description:** Probabilistic Heuristic Selection: Subjects do not rigidly apply a single deterministic decision rule across all trials. Instead, they maintain a repertoire of strategies—specifically, a non-compensatory one-reason heuristic (Take The Best) and a compensatory strategy (Tallying/Equal-Weight). On each trial, a subject probabilistically selects which strategy to deploy based on a subject-specific mixing parameter (p_ttb). This trial-by-trial internal blending natively captures intermediate aggregate metric values while preserving low between-subject variance, as each individual exhibits a mixture of both choice patterns.

**Parameters:**
- p_ttb: [0.0, 0.9]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (Non-compensatory)
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner is None:
        p_ttb_arr = np.array([0.5, 0.5])
    else:
        p_ttb_arr = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Strategy 2: Tallying (Compensatory)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_arr + (1.0 - p_ttb) * p_tally_arr
    
    # Late-stage lapse noise
    epsilon = float(parameters["epsilon"])
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Probabilistic Heuristic Selection (PHS) from the Non-linear Subjective Weighting Model (NLSW) by exploiting PHS's insensitivity to the specific validities of secondary cues. Under PHS, the choice probability is a mixture of Take The Best (TTB) and Tallying. If we hold the TTB winner constant (e.g., Option A always has the single highest validity cue) and hold the Tallying winner constant (e.g., Option B always has a greater total number of positive cues), PHS predicts the exact same choice probability across trials, regardless of which specific lower-validity cues Option B possesses. By contrast, NLSW computes a weighted sum using all validities. By shifting Option B's positive features from higher-validity cues to lower-validity cues across trials while keeping the total count identical, NLSW predicts a systematic shift in choice probabilities, whereas PHS predicts strict invariance.",
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
      0,
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
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
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
      0,
      1,
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
  "prompt_token_count": 3164,
  "candidates_token_count": 554,
  "total_token_count": 5705
}
```
