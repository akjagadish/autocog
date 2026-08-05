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
**Description:** Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
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
**Description:** Probabilistic Cue Retrieval TTB: Decision-makers follow the non-compensatory, sequential search architecture of Take The Best (TTB), but execution is noisy at the cue level. When a discriminating cue is evaluated, there is a probability of skipping it or failing to retrieve its value. This skip probability is a function of a base skip rate and is inversely proportional to the cue's validity (operationalized as proportional to 1 - validity). If a cue is skipped, the decision-maker continues to the next most valid cue. This produces slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration.

**Parameters:**
- p_skip: [0.0, 0.4]
- epsilon: [0.0, 0.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='stable')
    
    p_skip_base = float(parameters["p_skip"])
    
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    # Sequential search with probabilistic cue retrieval
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of skipping this cue is proportional to its invalidity
            p_skip_cue = p_skip_base * (1.0 - validities[idx])
            p_eval = 1.0 - p_skip_cue
            
            if a[idx] > b[idx]:
                p_A += p_continue * p_eval
            else:
                p_B += p_continue * p_eval
                
            p_continue *= p_skip_cue
            
    # If all discriminating cues are skipped, or no cues discriminate, guess randomly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    # Global response noise
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To robustly dissociate Tallying from WADD, we use an environment with 5 features where the two highest validities are substantially greater than the three lowest. Tallying ignores these validity differences and strictly counts the number of features favoring each option, whereas WADD computes a weighted sum. By pitting the two highest-validity features against the three lowest-validity features, we create strong dissociations: Tallying prefers the option winning on the 3 lower-validity cues, while WADD prefers the option winning on the 2 higher-validity cues. We also include agreement trials (where both models prefer the same option) and trials that result in a tie for Tallying but a clear preference for WADD, ensuring a comprehensive test of the competing mechanisms.

[1] To strongly dissociate the Weighted Additive rule (WADD) from Tallying, we utilize a 5-feature environment with a steep drop-off in validities. WADD integrates both the number of positive features and their validities, whereas Tallying strictly counts the number of features favoring each option, ignoring validities. By pitting a smaller number of high-validity features against a larger number of low-validity features, we create critical test trials where WADD prefers the option with fewer but more predictive features, while Tallying prefers the option with a greater quantity of less predictive features. We also include tie-breaking trials where Tallying predicts equal preference (guessing) but WADD predicts a distinct preference due to differences in validity weights.

[2] To robustly dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we utilize a 4-feature environment where the highest validity cue is pitted against multiple lower-validity cues. TTB is a non-compensatory heuristic that relies strictly on the single most valid discriminating feature, ignoring all others. In contrast, WADD computes a compensatory weighted sum of all features. By creating trials where one option possesses the single best feature but the other option possesses several slightly less valid features, we engineer a stark divergence: TTB will definitively choose the option with the highest-validity cue, whereas WADD will integrate the multiple lower-validity cues and choose the opposite option. We also include trials where both models agree, and trials where the highest-validity cue is tied, forcing TTB to rely on the second-highest cue while WADD still integrates the remaining cues.

[3] To strongly dissociate the Weighted Additive rule (WADD) from Take The Best (TTB), we use a 5-feature environment where the highest-validity cue is repeatedly pitted against multiple lower-validity cues. TTB is strictly non-compensatory and will always choose the option favored by the single most valid discriminating feature, completely ignoring the remaining features. WADD, however, integrates all features weighted by their validities (and subjective weights), allowing multiple weaker cues to overcome a single strong cue. By presenting trials where one option wins on the best feature but the other wins on several less valid features, we can cleanly separate the two models: TTB predicts choices aligned with the single best cue, while WADD predicts choices aligned with the sum of the remaining cues.

[4] This design perfectly dissociates Take The Best (TTB) from Rank-Weighted Tallying (RWT) by exploiting how each model handles tied validities. TTB uses a stable sort, meaning if two features have identical validities, it will always consult the one that appears first in the array before the second. RWT, however, assigns the same rank (and thus the same exponential weight, alpha^rank) to tied validities. By giving the first two features the same high validity (e.g., 0.9), we can create trials where Option A possesses the first feature but lacks all others, while Option B lacks the first feature but possesses the second feature plus several lower-validity features. TTB will strictly choose Option A because it evaluates the first feature and immediately stops. RWT will strictly choose Option B for any value of alpha > 0, because Option A's score is simply alpha^max_rank, while Option B's score is alpha^max_rank + alpha^(lower_ranks) + ..., making Option B's sum mathematically strictly greater.

[5] To robustly dissociate Rank-Weighted Tallying (RWT) from Take The Best (TTB), we utilize a 5-feature environment with strictly distinct validities. TTB is a strictly non-compensatory heuristic: it evaluates features in descending order of validity and completely ignores all less valid features once a discriminating feature is found. In contrast, RWT weights features exponentially by their validity rank (alpha^rank). Because alpha is a free parameter down to 1.0, RWT can behave in a compensatory manner (reducing to Equal-Weight/Tallying when alpha=1), allowing a larger quantity of lower-validity cues to overcome a single higher-validity cue. By pitting the single most valid discriminating feature against multiple lower-validity features, TTB will always predict the option with the best feature, whereas RWT can predict the opposite option depending on its alpha parameter, providing a clear quantitative dissociation.

[6] To dissociate pure Take The Best (TTB) from a Strategy Selection/Mixture Model (which probabilistically mixes TTB and WADD), we use a 5-feature environment with linearly decreasing validities. In critical trials, Option A is favored by the single highest-validity discriminating cue, but Option B is favored by all remaining lower-validity cues. Pure TTB will consistently choose Option A across these trials. In contrast, WADD strongly favors Option B because the sum of the lower validities outweighs the single best cue. Consequently, the Mixture Model will predict a more equivocal choice probability or a preference for B (depending on the mixing proportion and softmax temperature), yielding a clear quantitative divergence from pure TTB.

[7] The Strategy Selection / Mixture Model posits that individuals probabilistically sample between a non-compensatory rule (Take The Best) and a compensatory rule (Weighted Additive). To cleanly dissociate this from pure TTB, we use a 5-feature environment with a gradual decline in validities. We construct critical trials where the single highest-validity discriminating cue points to Option A, but the sum of the remaining lower-validity cues points strongly to Option B. Pure TTB will deterministically choose Option A (subject only to uniform response noise epsilon). In contrast, WADD will favor Option B, meaning the Mixture Model predicts an intermediate choice probability reflecting both the mixing parameter (p_ttb) and WADD's graded softmax evaluation. We also include agreement trials and trials where the first cue is tied, forcing TTB to rely on the second cue while WADD integrates all features.

[8] To dissociate pure Take The Best (TTB) from Probabilistic Cue Retrieval TTB, we vary the absolute validity of the first discriminating cue while keeping its relative rank highest among discriminating cues. Pure TTB strictly stops at the first discriminating cue and predicts that the choice probability for the favored option will be constant (1 - epsilon) across all trials, regardless of whether the first discriminating cue has a validity of 0.95 or 0.65. In contrast, Probabilistic Cue Retrieval TTB posits that the probability of skipping a cue increases as its validity decreases (skip rate is proportional to 1 - validity). Therefore, it predicts a systematic drop in the choice probability for the TTB-favored option on trials where the highest valid discriminating cue has lower absolute validity. By tying the higher-validity cues and forcing the models to rely on cues with progressively lower validities (0.95 down to 0.65), we create a quantitative dissociation: pure TTB predicts a flat accuracy function across these trial types, while Probabilistic TTB predicts a declining accuracy function.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To sharply dissociate pure Take The Best (TTB) from Probabilistic Cue Retrieval TTB, we manipulate the alignment of lower-validity cues relative to the highest discriminating cue. Pure TTB assumes a strict stopping rule: once the highest discriminating cue is found, all subsequent cues are ignored, meaning the choice probability for the favored option will be identical regardless of whether the remaining cues align with or oppose the first cue. In contrast, Probabilistic Cue Retrieval TTB assumes that cues can be skipped with a probability inversely proportional to their validity. If the highest discriminating cue is skipped, the model evaluates the subsequent cues. Therefore, Probabilistic TTB predicts that the choice probability for the TTB-favored option will be significantly lower when subsequent cues oppose it compared to when they align with it, and this divergence will widen as the absolute validity of the first discriminating cue decreases (which increases its skip rate).",
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
      1,
      1,
      1,
      1
    ],
    [
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
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
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
      1,
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
      1,
      1,
      1
    ],
    [
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
      1
    ],
    [
      0,
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3945,
  "candidates_token_count": 534,
  "total_token_count": 6948
}
```
