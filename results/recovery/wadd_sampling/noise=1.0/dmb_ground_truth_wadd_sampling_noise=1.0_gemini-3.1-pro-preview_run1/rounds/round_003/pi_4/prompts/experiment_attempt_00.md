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
**Description:** Decision-makers use a Compensatory / Weighted Additive (WADD) strategy, where an overall value for each option is computed by weighting each feature by a subjective transformation of its validity (validity raised to a power) and summing these weighted features. On strict conflict trials, the high weight of the single discriminating cue roughly balances the combined weights of multiple opposing lower-validity cues. This produces a near-zero difference in overall value, naturally resulting in a choice probability near 0.5 without requiring a mixture of distinct heuristics.

**Parameters:**
- gamma: [3.0, 8.0]
- beta: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities
    w = val ** gamma
    
    val_a = np.sum(a * w)
    val_b = np.sum(b * w)
    
    beta = float(parameters["beta"])
    scores = np.array([val_a, val_b])
    
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    p = (1.0 - epsilon) * p + epsilon * 0.5
    return p
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
**Description:** Top-K Tallying with Restricted Scope: Decision-makers simplify complex multi-attribute choices by focusing only on the top K most valid cues, ignoring all lower-validity information. By restricting K to a narrow band (1, 2, or 3) and allowing for a high degree of random guessing (lapse rate), the model produces consistent behavior across individuals while naturally capturing the pervasive ~0.5 choice probabilities on strict conflict trials where the top cues often tie or balance out.

**Parameters:**
- k: {1, 2, 3}
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(parameters["k"])
    
    # Ensure k doesn't exceed the number of available features
    k = min(k, len(val))
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally positive features among top K
    a_score = np.sum(a[top_k_cues])
    b_score = np.sum(b[top_k_cues])
    
    if a_score > b_score:
        p_core = np.array([1.0, 0.0])
    elif b_score > a_score:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with descending validities. In key trials, one option is superior on the single most valid discriminating feature (which TTB relies on exclusively), while the other option is superior on a greater number of less valid features (which Tallying simply counts, ignoring validities). This creates a sharp quantitative dissociation: TTB strongly predicts one choice, whereas Tallying predicts the opposite or a tie.

[1] To cleanly dissociate Tallying from Take The Best (TTB), we use a 5-feature design with descending validities. TTB relies solely on the first discriminating cue (the one with the highest validity), ignoring all other features. Tallying, on the other hand, ignores validities and simply counts the number of features on which each option is superior. We construct trials where one option is superior on the single most valid discriminating feature (so TTB strongly prefers it), but the other option is superior on a larger number of less valid features (so Tallying strongly prefers it). This creates a direct quantitative opposition between the two models on every trial.

[2] The Competing Theory proposes a mixture of Take The Best (TTB) and Tallying, with the probability of using TTB bounded between 0.4 and 0.6. This means that on trials where TTB and Tallying strictly conflict, the Competing Theory predicts choice probabilities close to 50% (indifference). The Advocated Theory, however, posits pure TTB (with softmax noise), predicting strong, deterministic-like preferences for the option favored by the most valid discriminating cue, regardless of how many lower-validity cues point the other way. By presenting trials where one option is favored by the single highest validity cue but the other option is favored by a majority of the remaining cues, we can cleanly dissociate the near-50% prediction of the Competing Theory from the extreme probability prediction of the Advocated Theory.

[3] To demonstrate that decision-makers use a strategy mixture of TTB and Tallying rather than pure TTB, we use a 5-feature design with a mix of trial types. In strict conflict trials (where the most valid cue favors one option while all remaining cues favor the other), pure TTB strongly predicts choosing the TTB-favored option, whereas the Strategy Mixture predicts choice probabilities near 0.5 due to the probabilistic alternation between TTB and Tallying. By including partial conflict trials (where Tallying predicts a tie but TTB favors one option) and agreement trials, we can map the intermediate probabilities predicted by the mixture model (e.g., ~0.75 and ~1.0 respectively) and show they quantitatively diverge from the pure TTB model's consistent, deterministic-like predictions across all these trial types.

[4] To cleanly dissociate the Advocated Theory (WADD with subjective validity transformation) from the Competing Theory (a strategy mixture of TTB and Tallying), we design trials where TTB and Tallying perfectly agree, yet WADD predicts the exact opposite choice. By using a feature set with three highly valid cues and two low validity cues, we can construct options where Product A has the single most valid cue AND a greater total number of positive cues (strongly favored by both TTB and Tallying), while Product B has the second and third most valid cues. Because WADD exponentiates validities, the combined weight of the second and third cues surpasses the first cue even at high exponent values, causing WADD to strongly prefer Product B. This creates a stark qualitative and quantitative divergence: the Mixture model predicts near-certainty for A, while WADD predicts a strong preference for B.

[5] This design systematically dissociates the Strategy Mixture theory (which mixes pure Take The Best and pure Tallying) from the Compensatory WADD theory (which weights features by exponentiated validities). By using three highly valid features and two less valid features, we can construct trials where TTB and Tallying completely agree (e.g., Option A has the highest validity cue AND the most cues overall), leading the Mixture model to predict a strong preference for Option A. However, Option B possesses the second and third most valid cues, whose combined exponentiated weight under WADD exceeds the highest cue plus the two lowest cues, leading WADD to strongly predict Option B. In other trials, TTB and Tallying conflict (Mixture model predicts ~50% choice), but WADD strongly favors one option due to the non-linear weighting of validities. This provides multiple distinct quantitative and qualitative divergence points.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design strictly dissociates the Advocated Theory (WADD with exponentiated validities) from the Competing Theory (Top-K Tallying with restricted scope K in {1, 2, 3}). By using 6 features with relatively close validities, we can construct trials where the Top-K Tallying model consistently predicts Option A (or a tie) regardless of whether K is 1, 2, or 3, because Option A dominates in the top 2 features. However, Option B possesses a larger number of slightly lower-validity features (cues 3, 4, 5, 6). Because WADD evaluates all features and weights them by validities raised to a power, the combined weight of Option B's multiple features easily surpasses the weight of Option A's top features. This creates a stark quantitative dissociation where Top-K Tallying predicts a strong preference for A (or a 50% split), while WADD predicts a strong preference for B.",
  "validities": [
    0.9,
    0.88,
    0.86,
    0.84,
    0.82,
    0.8
  ],
  "trial_a_ratings": [
    [
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
      0,
      0
    ],
    [
      0,
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
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
      1
    ],
    [
      0,
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
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3050,
  "candidates_token_count": 498,
  "total_token_count": 7736
}
```
