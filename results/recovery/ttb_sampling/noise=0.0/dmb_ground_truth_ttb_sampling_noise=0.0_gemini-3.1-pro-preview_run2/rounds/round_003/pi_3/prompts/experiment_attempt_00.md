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
**Description:** Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

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
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Take The Best with Validity-Dependent Confidence: Decision makers use a non-compensatory, lexicographic strategy (Take The Best) to choose between options. However, their execution of this strategy is noisy, and the degree of noise (lapse rate) depends on the validity of the discriminating cue. When options are discriminated by a highly valid cue, confidence is high and the lapse rate is low. When the discriminating cue has low validity, confidence is lower, leading to a higher probability of guessing or lapsing. This maintains the non-compensatory nature of TTB while naturally explaining variations in performance across different trial structures.

**Parameters:**
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(-validities, kind='stable')
    
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    p_core = np.array([0.5, 0.5])
    v_discriminating = None
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            v_discriminating = validities[i]
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            v_discriminating = validities[i]
            break
            
    if v_discriminating is not None:
        # Lapse rate increases as validity of the discriminating cue decreases
        lapse = epsilon + gamma * (1.0 - v_discriminating)
        lapse = max(0.0, min(1.0, lapse))
        return (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    else:
        # Guess if no cues discriminate
        return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design leverages the core difference between Tallying (which counts the number of features where an option is superior, ignoring validities) and WADD (which computes a validity-weighted sum). By using 5 features with a steep validity drop-off, we can construct trials where Option A wins on a larger number of low-validity features (favored by Tallying) while Option B wins on fewer but much higher-validity features (favored by WADD). We also include trials where Tallying predicts a perfect tie (equal number of winning features) but WADD strongly prefers one option, and baseline trials where both models agree.

[1] To strongly dissociate WADD from Tallying, this design uses 5 features with a steep drop-off in validities. We construct trials where one option wins on a larger number of low-validity features (favored by Tallying) while the other option wins on fewer but higher-validity features (favored by WADD). We also include tie-breaking trials where both options win on an equal number of features (Tallying predicts guessing) but WADD strongly prefers the option with the higher-validity features. Finally, baseline trials where both models agree are included to ensure general task engagement.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) rule, this design exploits TTB's non-compensatory, lexicographic nature. We use 5 features with descending validities. In the critical trials, one option is favored by the single highest-validity discriminating feature (which TTB relies on exclusively), while the other option is favored by all remaining lower-validity features. For WADD, the combined weight of these multiple lower-validity features overcomes the single best feature, leading to opposite predictions. We also include trials where the highest validities are tied, forcing TTB to the next cue, while WADD still compensates with the remaining cues. Baseline agreement trials are included to ensure general task engagement.

[3] To cleanly dissociate the Weighted Additive (WADD) rule from the Take The Best (TTB) heuristic, we use a set of 5 features with linearly decreasing validities. The critical trials are designed such that the single most valid discriminating cue favors one option (which TTB will categorically choose), while a combination of several lower-validity cues favors the other option strongly enough to override the single best cue in a compensatory model like WADD. We also include trials where WADD predicts a tie but TTB predicts a strict preference, and baseline trials where both models agree. This provides a robust quantitative dissociation of compensatory vs. non-compensatory feature integration.

[4] To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. While pure TTB relies exclusively on the highest-validity discriminating feature, the Strategy Selection model predicts occasional lapses into Tallying. By keeping the TTB prediction constant (e.g., Option A always favored by the highest validity feature) but varying the number of lower-validity features favoring Option B, we manipulate the Tallying prediction. Pure TTB predicts equal choice probability for Option A across these trials. In contrast, the Strategy Selection model predicts a drop in Option A choice probability specifically on trials where Tallying strongly favors Option B, creating a clear quantitative dissociation.

[5] To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. Pure TTB relies exclusively on the highest-validity discriminating feature and predicts a constant choice probability for the favored option regardless of how many lower-validity features favor the alternative. In contrast, the Strategy Selection model assumes occasional lapses into Tallying. By keeping the TTB prediction constant but varying the Tallying prediction (Agreement, Conflict, or Tie), we can test whether the choice probability for the TTB-favored option drops systematically when Tallying favors the alternative. The trials vary the highest discriminating cue to ensure the effect is robust across different cue validities.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Take The Best (TTB) from TTB with Validity-Dependent Confidence, we systematically vary the validity of the highest discriminating cue across trials. Pure TTB assumes a constant lapse rate and predicts equal choice probabilities for the TTB-favored option regardless of which cue discriminates. In contrast, the Validity-Dependent Confidence model assumes that the lapse rate increases when the discriminating cue has lower validity. By tying the highest validity cues and forcing the decision onto progressively lower-validity cues across different trials, we expect pure TTB to maintain a flat accuracy curve, while the competing theory predicts a monotonically decreasing accuracy curve for the TTB-favored option.",
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
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
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
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
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
  "prompt_token_count": 3100,
  "candidates_token_count": 556,
  "total_token_count": 5707
}
```
