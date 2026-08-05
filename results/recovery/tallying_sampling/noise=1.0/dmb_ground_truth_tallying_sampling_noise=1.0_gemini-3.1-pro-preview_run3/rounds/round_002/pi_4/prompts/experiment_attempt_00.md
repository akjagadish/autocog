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
**Description:** Equal Weights (Tallying)

**Parameters:**
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    
    # Count the number of positive features for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    if tally_a > tally_b:
        scores = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        scores = np.array([0.0, 1.0])
    else:
        # Guess randomly if there is a tie
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary tally score
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Independent lapse
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
**Description:** Probabilistic Cue Selection (Random Cue) posits that decision-makers do not deterministically use the most valid cue (like Take-The-Best) nor do they integrate all cues simultaneously (like WADD). Instead, on each trial, they sample a single cue with a probability proportional to its subjective validity. They then choose the option favored by that sampled cue, guessing uniformly if the sampled cue ties. This single-cue sampling process naturally generates probabilistic choices across trials, producing choice shares near 0.50 for conflict trials where different cues favor different options, without relying on extreme softmax noise.

**Parameters:**
- epsilon: [0.0, 0.5]
- weights: [(0.0, 1.0)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Subjective validities used as sampling weights
    subj_weights = val * w
    sum_w = np.sum(subj_weights)
    
    if sum_w <= 1e-9:
        p_core = np.array([0.5, 0.5])
    else:
        p_cue = subj_weights / sum_w
        a, b = stim[0], stim[1]
        
        p_a = 0.0
        for j in range(n_features):
            if a[j] > b[j]:
                p_a += p_cue[j]
            elif a[j] == b[j]:
                p_a += p_cue[j] * 0.5
                
        p_core = np.array([p_a, 1.0 - p_a])
        
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
[0] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we must exploit their fundamental difference: TTB is a non-compensatory 'one-reason' heuristic, while WADD is fully compensatory. We use five features with linearly decreasing validities. The critical trials present a 'compensatory' conflict: one option is endorsed by the single most valid cue, but the competing option is endorsed by several less valid cues whose combined weight exceeds that of the single best cue. TTB will deterministically choose the option with the highest valid cue, entirely ignoring the others. WADD, conversely, will integrate all cues and choose the option with the greater weighted sum, provided the subjective weights do not overly distort the validities. By varying which specific valid cues are in conflict (e.g., Cue 1 vs Cues 2,3,4,5 or Cues 1,3 vs Cues 2,4,5), we can map out the precise compensatory threshold for the subject, which WADD predicts will be a smooth function of the feature sums, while TTB predicts a step function strictly dependent on the top discriminating cue.

[1] To strongly dissociate the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic, we use a 4-feature environment where the validities are spread such that the most valid cue can be outweighed by a combination of less valid cues. We present a mix of 'conflict' trials where TTB and WADD make opposite predictions (e.g., Option A is favored by the single best cue, while Option B is favored by the sum of the remaining cues) and 'alignment' trials where both models agree. By observing choices across these specific conflicts, we can quantitatively identify whether subjects are integrating all evidence weighted by validity (WADD) or stopping at the first discriminating cue (TTB).

[2] To quantitatively dissociate Probabilistic Cue Selection (Random Cue) from the Weighted Additive rule (WADD), we exploit their different mappings from cue validities to choice probabilities. WADD computes a weighted sum of features for each option and passes the difference through a softmax, predicting a sigmoidal relationship between the score difference and choice probability. In contrast, Random Cue samples a single cue proportional to its validity and chooses the favored option, predicting that choice probabilities scale strictly linearly with the proportion of validities favoring an option (ignoring extreme softmax noise). By carefully designing trials where the difference in weighted sums varies independently of the total validities involved (e.g., comparing trials with few vs. many conflicting cues), we can observe whether choice probabilities follow a linear mixture of validities (Random Cue) or a deterministic integration with softmax noise (WADD).

[3] To quantitatively dissociate the Weighted Additive rule (WADD) from Probabilistic Cue Selection (Random Cue), we exploit the functional form of their choice probabilities. Random Cue's probability of choosing an option is strictly a linear function of the difference in weighted feature sums (because P(A) = 0.5 + 0.5 * ScoreDiff / TotalValidity). In contrast, WADD passes this same score difference through a softmax function, predicting a sigmoidal (S-shaped) relationship. By presenting a series of trials that systematically vary the score difference from zero to large values, we can evaluate whether the choice proportions follow a linear mixture (Random Cue) or a deterministic integration with softmax noise (WADD).

[4] To quantitatively dissociate Probabilistic Cue Selection (Random Cue) from Equal Weights (Tallying), we exploit Tallying's complete insensitivity to cue validities. Tallying simply counts the number of positive features for each option and strictly prefers the one with the higher count, applying softmax noise to the binary outcome. Random Cue, however, samples cues proportionally to their subjective validities (validity * weight) and chooses the option favored by the sampled cue. By presenting trials where one option has fewer positive cues but those cues possess significantly higher validities, Tallying will consistently predict a preference for the option with more positive cues, whereas Random Cue will predict a preference (or at least a much higher choice probability) for the option supported by fewer, more valid cues.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Equal Weights (Tallying) from Probabilistic Cue Selection (Random Cue), we exploit Tallying's complete insensitivity to cue validities. Tallying simply counts the number of positive features for each option and strictly prefers the one with the higher count. Random Cue, however, samples cues proportionally to their subjective validities and chooses the option favored by the sampled cue. By presenting trials where one option has fewer positive cues but those cues possess significantly higher validities, Tallying will consistently predict a preference for the option with more positive cues, whereas Random Cue will predict a preference for the option supported by fewer, more valid cues. We also include trials where the total number of positive cues is tied; Tallying predicts random guessing (0.5 probability), while Random Cue predicts strong deterministic-like preferences based on the validities of the active cues.",
  "validities": [
    0.95,
    0.9,
    0.55,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
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
      1,
      1,
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
      0,
      1,
      1,
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
  "prompt_token_count": 2908,
  "candidates_token_count": 459,
  "total_token_count": 4977
}
```
