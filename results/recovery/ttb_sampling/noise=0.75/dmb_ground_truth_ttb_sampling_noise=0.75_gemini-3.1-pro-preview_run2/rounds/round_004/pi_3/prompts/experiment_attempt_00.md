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
**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Dual-Process Hybrid Model (Take-The-Best + Weighted Additive) with soft compensatory noise: Human decision-making is predominantly non-compensatory (Take-The-Best), but subjects sometimes fall back on a compensatory strategy (Weighted Additive). Because WADD operates on sums of validities, its response noise is decoupled and constrained to extremely low values to reflect the highly soft, almost guessing-like nature of human compensatory choice. A high potential lapse rate captures general response noise and further scales down overconfident predictions.

**Parameters:**
- beta_ttb: [0.0, 10.0]
- beta_wadd: [0.0, 0.15]
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Hybrid expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) evaluation
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in ranked_features:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    # Weighted Additive (WADD) evaluation
    wadd_a = np.sum(a * validities)
    wadd_b = np.sum(b * validities)
    wadd_scores = np.array([wadd_a, wadd_b])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Softmax probabilities for TTB
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        p_ttb_vec = np.array([0.5, 0.5])
    else:
        z_ttb = beta_ttb * ttb_scores
        z_ttb = z_ttb - np.max(z_ttb)
        e_ttb = np.exp(z_ttb)
        p_ttb_vec = e_ttb / np.sum(e_ttb)
        
    # Softmax probabilities for WADD
    if wadd_a == wadd_b:
        p_wadd_vec = np.array([0.5, 0.5])
    else:
        z_wadd = beta_wadd * wadd_scores
        z_wadd = z_wadd - np.max(z_wadd)
        e_wadd = np.exp(z_wadd)
        p_wadd_vec = e_wadd / np.sum(e_wadd)
        
    # Mix the two processes
    p_ttb_weight = float(parameters["p_ttb"])
    p_core = p_ttb_weight * p_ttb_vec + (1.0 - p_ttb_weight) * p_wadd_vec
    
    # Apply uniform lapse
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, we use a 5-feature design with a skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, ignoring the validities entirely. WADD weights each feature by its validity, meaning a few highly valid features can outweigh several less valid ones. The trial pairs are designed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option. This mix of congruent, incongruent, and tie trials provides a robust test to identify which strategy subjects are using.

[1] To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, this design uses a 5-feature task with a highly skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, strictly ignoring the magnitude of the validities. In contrast, WADD weights each feature by its validity, allowing a few highly valid features to outweigh a larger number of less valid ones. The trial pairs are strategically constructed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as tie-breaking trials (where Tallying predicts a tie due to an equal number of winning features, while WADD strongly prefers the option with the higher-validity features).

[2] To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) rule, we use a 5-feature design with a set of validities that do not form a non-compensatory environment (i.e., the validity of the best feature is strictly less than the sum of the validities of the remaining features). TTB is a lexicographic strategy that bases its choice entirely on the single most valid discriminating feature, ignoring all other features. In contrast, WADD computes a compensatory weighted sum of all features. The trial pairs are specifically designed such that the option favored by the single most valid feature (which TTB will choose) is opposed by multiple lower-validity features that cumulatively outweigh the best feature (which WADD will choose). By including both congruent trials (where both models agree) and incongruent trials (where the 'best' cue points one way but the 'rest' of the cues point the other), we can clearly distinguish whether subjects are using a one-reason heuristic or a compensatory integration strategy.

[3] To quantitatively dissociate the Weighted Additive (WADD) rule from the Take-The-Best (TTB) heuristic, we employ a 5-feature design where the validities form a compensatory environment (the most valid cue is outweighed by the sum of the remaining cues). TTB is a non-compensatory lexicographic strategy that decides based solely on the single most valid discriminating feature. In contrast, WADD integrates all features weighted by their validities, allowing multiple lower-validity cues to compensate for a disadvantage on the highest-validity cue. The trial pairs contain strong incongruencies where the option favored by the single best cue is opposed by the option favored by a combination of lesser cues. We also include congruent pairs to act as catch trials and ensure subjects do not artificially adapt to a purely incongruent task.

[4] To quantitatively dissociate pure Take-The-Best (TTB) from a Hybrid TTB+Tallying model, we exploit a key property of the pure TTB model: its predicted choice probability depends entirely on the single most valid discriminating feature and is invariant to the number of remaining cues that support or oppose that choice. In contrast, the Hybrid model mixes TTB with a Tallying process, which counts the total number of winning features for each option. By holding the TTB prediction constant (e.g., Option A always wins on the most valid feature) while parametrically varying the Tallying support (from strongly opposing A to strongly supporting A), the pure TTB model predicts a flat, constant choice probability across these conditions, whereas the Hybrid model predicts a graded modulation of choice probability. The trial set includes a full spectrum of Tallying support levels for both Option A and Option B TTB-winners.

[5] This design quantitatively dissociates the pure Take-The-Best (TTB) heuristic from the Hybrid (TTB + Tallying) model by testing for strict preference reversals. Pure TTB is strictly non-compensatory: it predicts that the option favored by the single most valid discriminating feature will ALWAYS be preferred (choice probability > 0.5), regardless of how many lower-validity cues oppose it. In contrast, the Hybrid model mixes TTB with a compensatory Tallying process. If the Tallying process strongly opposes the TTB prediction (e.g., the TTB winner is opposed by 5 or 6 lower-validity cues), the Hybrid model allows the choice probability to cross the 50% threshold, resulting in a preference reversal. By parametrically increasing the number of opposing cues (from 1 to 6) against the single best cue, we can detect whether subjects exhibit this compensatory preference reversal (supporting the Hybrid model) or maintain a constant, non-compensatory preference (supporting pure TTB).

[6] To quantitatively dissociate Take-The-Best (TTB) from Probabilistic Cue Selection (Stochastic TTB), we manipulate the number and validities of subordinate discriminating features while holding the single best discriminating feature constant. Under pure TTB, the decision is strictly based on the most valid discriminating feature; the presence, number, or validities of other lower-ranking discriminating features have no effect on the predicted choice probabilities (the pre-noise score is always 1 for the TTB winner and 0 for the loser). In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). Thus, Stochastic TTB predicts that the choice probability for the option favored by the best cue will decrease as more subordinate cues favor the opposing option, whereas pure TTB predicts a constant choice probability across these conditions. By systematically varying the opposing subordinate cues, we can directly test whether choice probabilities are invariant to subordinate cues (TTB) or sensitive to them (Stochastic TTB).

[7] To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), this design manipulates the number of *supporting* subordinate features while holding the single best discriminating feature and the opposing feature constant. Standard TTB's predicted choice probability is driven entirely by the single most valid discriminating feature; adding more lower-validity features that support the same option does not change the TTB score (it remains 1 for the winner and 0 for the loser), thus predicting a perfectly flat choice probability across these conditions. In contrast, Stochastic TTB samples among all discriminating features. By adding more features that support the TTB winner, the total probability mass for sampling a feature favoring that option increases. Thus, Stochastic TTB predicts a graded increase in choice probability as supporting features are added, whereas standard TTB predicts no change.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Take-The-Best (TTB) from the Dual-Process Hybrid Model (TTB + WADD), we manipulate the compensatory support from lower-validity features while holding the prediction of the single most valid discriminating feature constant. Pure TTB predicts that the choice probability depends entirely on the single best discriminating cue; thus, as long as Option A wins on the highest validity cue, its choice probability should remain constant regardless of the opposing cues. In contrast, the Hybrid model mixes TTB with a Weighted Additive (WADD) process. By systematically parametrising the WADD support for the TTB winner (from strongly supporting to strongly opposing), the Hybrid model predicts a graded modulation of choice probabilities, whereas pure TTB predicts a completely flat response profile across these conditions.",
  "validities": [
    0.9,
    0.85,
    0.8,
    0.75,
    0.7
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
      1,
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
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
      0,
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
  "prompt_token_count": 4231,
  "candidates_token_count": 645,
  "total_token_count": 7141
}
```
