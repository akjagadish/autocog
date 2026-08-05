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
**Description:** Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.

**Parameters:**
- mixture_p: [0.0, 1.0]
- beta: [0.1, 3.5]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    mixture_p = float(parameters['mixture_p'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # --- TTB Component ---
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- WADD Component ---
    # WADD uses validities as weights
    scores = stim @ val
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # --- Mixture ---
    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    
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


## COMPETING THEORY
**Description:** Sequential Evidence Accumulation: Decision-making is driven by a sequential sampling process where features are evaluated in order of their subjective validity. As each feature is processed, the validity-weighted difference between the options is added to a running accumulator. If this accumulated evidence reaches a predefined threshold at any point, search is immediately terminated and a choice is made (mimicking non-compensatory heuristics like Take-The-Best). If all features are exhausted without the evidence hitting the boundary, the subject makes a probabilistic choice based on the final accumulated tally (mimicking compensatory strategies like WADD). This single-process model naturally unifies fast-and-frugal heuristics and exhaustive compensatory integration depending on the height of the evidence threshold.

**Parameters:**
- threshold: [0.0, 5.0]
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    threshold = float(parameters['threshold'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    A_val = 0.0
    decided = False
    p_a = 0.5
    
    # Sequentially accumulate evidence
    for j in cue_order:
        A_val += val[j] * (a[j] - b[j])
        if A_val >= threshold:
            p_a = 1.0
            decided = True
            break
        elif A_val <= -threshold:
            p_a = 0.0
            decided = True
            break
            
    # If boundary not reached, make a soft choice based on the final tally
    if not decided:
        z = beta * A_val
        # Numerically stable logistic function for P(A)
        if z > 100:
            p_a = 1.0
        elif z < -100:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    p_b = 1.0 - p_a
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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
[0] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) rule, this design uses a 'compensatory' structure where the most valid cue often points to one option while the sum of the less valid cues points to the other. TTB is a non-compensatory heuristic: it stops at the first discriminating cue (ordered by validity) and ignores all others. In contrast, WADD computes a weighted sum of all features, allowing multiple weaker cues to outvote a single strong cue. By setting the validities such that the highest validity is less than the sum of the lower validities (e.g., 0.95 vs 0.70+0.65+0.60), we create trials where TTB and WADD make diametrically opposed predictions.

[1] To quantitatively dissociate the Weighted Additive (WADD) rule from Take The Best (TTB), this design uses a set of four cues with validities carefully chosen such that the most valid cue is outweighed by combinations of less valid cues. TTB always decides based on the single highest-validity cue that discriminates between the options. In contrast, WADD computes a compensatory weighted sum. By presenting pairs where the highest-validity cue favors one option but the sum of the remaining cues favors the other, we create strong opposite predictions. Additional trials where both models agree serve as internal controls to measure baseline noise and lapse rates independently of strategy differences.

[2] To dissociate Rank-Weighted Additive (Rank-WADD) from cardinal Weighted Additive (WADD) models, we exploit the difference between linear rank-based weights and actual validities. We use 5 features where the highest validity is much larger than the others, while the remaining validities decline gradually (0.95, 0.65, 0.60, 0.55, 0.50). In Rank-WADD, weights decay linearly (5, 4, 3, 2, 1). Thus, combinations of lower-ranked features can easily outweigh the top feature in rank-weight sums, but not in cardinal-validity sums. For instance, comparing [1, 0, 0, 0, 1] to [0, 1, 1, 0, 0]: Rank-WADD prefers Option B (rank sum 7 vs 6), while WADD prefers Option A (cardinal sum 1.45 vs 1.25). We include multiple such dissociative pairs in both directions alongside baseline pairs where models agree.

[3] To quantitatively dissociate the cardinal Weighted Additive (WADD) rule from the Rank-Weighted Additive theory, we exploit the non-linear gaps between feature validities. We use five features with validities [0.98, 0.70, 0.68, 0.66, 0.50]. In Rank-WADD, the weights are strictly linear based on rank (5, 4, 3, 2, 1). Therefore, the second and third features combined (rank sum 4 + 3 = 7) will outweigh the first and fifth features combined (rank sum 5 + 1 = 6). However, in the cardinal WADD model, the first feature is exceptionally strong (0.98), so the sum of the first and fifth features (0.98 + 0.50 = 1.48) outweighs the sum of the second and third (0.70 + 0.68 = 1.38). By including pairs where these combinations are pitted against each other, we create strong opposite predictions between the two models. Additional pairs are included where both models agree, serving as baseline controls.

[4] To quantitatively dissociate Probabilistic Take-The-Best (TTB) from the Weighted Additive (WADD) rule, this design uses five features with a steep drop-off in validities. TTB is a non-compensatory heuristic that makes a decision based entirely on the first discriminating cue (in order of validity), whereas WADD aggregates all features compensatorily. We present pairs of options where the highest-validity cue(s) favor one option, but a larger number of lower-validity cues favor the other option. TTB will consistently choose the option favored by the highest-validity discriminating cue, whereas WADD will choose the other option because the combined weight of the numerous lower-validity cues outweighs the single top cue. Additional trials where the models agree are included as baseline controls to estimate lapse and noise parameters.

[5] This design tests a core qualitative divergence between Probabilistic Take-The-Best (TTB) and Weighted Additive (WADD) models. In Probabilistic TTB, feature evaluation skips over non-discriminating cues without incurring any evaluation error (gamma). Consequently, TTB predicts identical choice probabilities for ANY pair of options that have the same sequence of discriminating cues, regardless of the absolute validity or rank of those cues. For example, TTB predicts the exact same preference strength for comparing cue 1 vs cue 2 as it does for comparing cue 1 vs cue 5, or cue 4 vs cue 5. In contrast, WADD's predictions depend on the cardinal differences between the weighted validities of the features. By presenting an exhaustive set of 1-vs-1 and 1-vs-2 feature matchups at different validity ranks, we create a scenario where TTB predicts a flat, invariant response profile across subsets of trials, whereas WADD predicts a linearly graded hierarchy of preferences.

[6] This design tests the strict additivity of log-odds predicted by a pure compensatory model (WADD) against the non-additive predictions of a Dual-Process Strategy Mixture (TTB + WADD). In pure WADD, the log-odds of choosing Option A over Option B is a linear combination of the feature differences; thus, the impact of changing a lower-validity cue should be independent of whether the highest-validity cue is tied or discriminating. In the mixture model, when the top cue is discriminating, the TTB component is completely locked to that cue, and lower cues only affect the WADD component. When the top cue is tied, the TTB component shifts its focus to the next most valid discriminating cue. This causes a massive violation of log-odds additivity: the lower cues will have a much larger impact on choice probabilities when the top cue is tied than when it is discriminating. By factorially crossing the state of the top cue (tied vs. discriminating) with the states of the lower cues, we can quantitatively dissociate the mixture model from any pure weighted-additive model, even one with free subjective weights.

[7] This design quantitatively dissociates the pure Weighted Additive (WADD) rule from the Dual-Process Strategy Mixture (TTB + WADD) by exploiting asymptotic choice probabilities in the face of overwhelming compensatory evidence. In pure WADD, as the number of lower-validity cues favoring Option B increases against a single higher-validity cue favoring Option A, the preference for Option B can become arbitrarily strong (approaching 100%). However, in the mixture model, the Take-The-Best (TTB) component is non-compensatory and remains completely anchored to the highest-validity discriminating cue (Option A). As a result, the mixture model predicts a 'glass ceiling' or asymptote in choice probability dictated by the mixture parameter (e.g., capping at 1 - mixture_p), no matter how many lower cues favor Option B. By parametrically increasing the number of opposing lower cues across different top-cue tiers (Cue 1 vs rest, Cue 2 vs rest), we can test for the presence of this rigid heuristic asymptote versus WADD's continuous sensitivity.

[8] This design quantitatively dissociates the Sequential Evidence Accumulation model from the Dual-Process Strategy Mixture (TTB + WADD). In the Mixture model, the non-compensatory component (TTB) is exclusively controlled by the single highest-validity discriminating cue. Consequently, lower-validity cues only influence the WADD component, which processes them in a linear, compensatory manner. In contrast, the Sequential Accumulation model adds evidence cue-by-cue. If the threshold is intermediate, the first cue alone might not trigger a decision, but the first AND second cues together might hit the threshold, abruptly terminating search and ignoring all subsequent cues. By presenting trials where Cue 1 favors Option A and Cues 3-5 favor Option B, we manipulate Cue 2 to favor B, be Tied, or favor A. The Mixture model predicts a linear, symmetric shift in choice probability across these three states. The Sequential model predicts a massive non-linear jump: when Cue 2 favors A, the accumulator hits the threshold early, forcing a hard choice for A and blinding the model to the overwhelming evidence for B in Cues 3-5.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design dissociates the Sequential Evidence Accumulation model from the Dual-Process Strategy Mixture model by contrasting cue cancellation with lexicographic priority. In the Sequential model, early cues with similar validities (e.g., Cue 1 and Cue 2) that favor opposite options will largely cancel each other out in the accumulator, allowing the remaining lower-validity cues to dictate the final choice. In contrast, the Mixture model's TTB component exhibits absolute lexicographic priority: it is wholly determined by the first discriminating cue and completely ignores the opposing second cue. By presenting pairs where Cues 1 and 2 oppose each other but Cues 3-5 strongly favor one option, the Sequential model predicts choices aligned with the sum of Cues 3-5, while the Mixture model predicts a substantial proportion of choices anchored strictly to Cue 1, resulting in a massive divergence in predicted choice probabilities.",
  "validities": [
    0.95,
    0.9,
    0.85,
    0.8,
    0.75
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1
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
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4274,
  "candidates_token_count": 535,
  "total_token_count": 7610
}
```
