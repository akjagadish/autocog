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
**Description:** Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Tallying with Lexicographic Tie-Breaking posits that decision-makers primarily rely on a compensatory equal-weight tallying heuristic, choosing the option with the highest number of positive features. However, when options are tied in their feature tallies, decision-makers do not guess randomly. Instead, they break the tie by comparing the options on the single most valid feature where the options differ (a Take-The-Best mechanism). This hybrid approach perfectly mimics Tallying on unequal feature counts but provides a deterministic, validity-based resolution for ties.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- w_tie: [0.0, 0.05]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Primary mechanism: Equal-weight tallying
    scores = np.sum(stim, axis=1)
    
    # Secondary mechanism: Lexicographic tie-breaking (Take-The-Best)
    if scores[0] == scores[1]:
        validities = np.asarray(parameters["validities"], dtype=float)
        w_tie = float(parameters["w_tie"])
        diff = stim[0] - stim[1]
        valid_diffs = np.where(diff != 0)[0]
        
        if len(valid_diffs) > 0:
            # Find the differing feature with the highest validity
            best_feature = valid_diffs[np.argmax(validities[valid_diffs])]
            if diff[best_feature] > 0:
                scores[0] += w_tie
            else:
                scores[1] += w_tie

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits the single most valid discriminating cue against a coalition of multiple less valid cues. Take The Best (TTB) will invariably choose the option favored by the highest-validity discriminating feature, ignoring all other features. In contrast, the Weighted Additive (WADD) model integrates all features, allowing a larger number of lower-validity cues to compensate for a deficit on the highest-validity cue. By varying which cue is the highest discriminating one (1st, 2nd, or 3rd) and creating opposing coalitions, we achieve a strong quantitative dissociation between the non-compensatory TTB and the compensatory WADD models. Agreement trials are included as a baseline.

[1] This design systematically contrasts the compensatory Weighted Additive (WADD) model with the non-compensatory Take The Best (TTB) heuristic by pitting single high-validity cues against coalitions of multiple lower-validity cues. By shifting which cue is the highest valid discriminating feature (1st, 2nd, or 3rd) and creating opposing coalitions that outweigh it in sum, we force TTB and WADD to make opposite predictions. We also include reversed option assignments to balance the design and baseline trials where the models agree.

[2] This design quantitatively dissociates Tallying (Equal Weight) from the Weighted Additive (WADD) model. Tallying simply counts the number of positive features, whereas WADD weights each feature by its validity. By pitting an option with a greater number of low-validity features against an option with fewer but much higher-validity features, we create strong opposite predictions. For example, an option with 3 low-validity features will be preferred by Tallying, while WADD will prefer the competing option possessing only 2 but highly valid features. Baseline agreement trials and trials where Tallying is indifferent but WADD shows a strong preference are also included.

[3] This design systematically dissociates the compensatory Weighted Additive (WADD) model from the Tallying (Equal Weight) heuristic. By carefully setting validities, we create specific trial types that pull apart the two models based on their core mechanisms: WADD weights features by validity, while Tallying simply counts positive features. The design includes trials where Tallying predicts a strict preference but WADD predicts exact indifference (e.g., one perfect validity cue vs. two lowest validity cues), trials where WADD predicts a strict preference but Tallying is indifferent (equal number of positive features but differing validities), and reversal trials where the two models predict strictly opposite choices.

[4] To effectively dissociate pure Tallying from Soft Threshold Tallying, we use a 5-feature design with a pronounced gap between high and low validities. Pure Tallying simply counts the positive features, remaining completely indifferent to their validities. Soft Threshold Tallying, by contrast, applies a sigmoid threshold mask that heavily down-weights or eliminates low-validity features. The trial set includes pairs where an option with fewer but highly valid features competes against an option with more but lower-validity features (causing opposing predictions), as well as pairs with equal feature counts but differing validities (causing Tallying to be indifferent while Soft Threshold Tallying shows a strong preference).

[5] This design uses a 5-feature environment with two highly valid features and three low-validity features to create a sharp divergence between Tallying and Soft Threshold Tallying. Pure Tallying weights all features equally, choosing the option with the most positive features. Soft Threshold Tallying applies a sigmoid threshold that effectively masks out the low-validity features, causing choices to be driven predominantly by the high-validity features. The trial pairs systematically pit options with fewer high-validity features against options with more low-validity features, leading to strictly opposing predictions. We also include trials where Tallying is indifferent but Soft Threshold Tallying has a strong preference.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Tallying (Equal Weight) from Tallying with Lexicographic Tie-Breaking, we focus on trials where the two options have an equal number of positive features (i.e., their tallies are tied) but differ in which specific features are positive. Pure Tallying predicts exact indifference (50/50 choice) on these tie trials, regardless of the validities of the features. In contrast, Tallying with Lexicographic Tie-Breaking resolves these ties by selecting the option that is favored by the single most valid discriminating feature. By including a mix of tie trials (where the models strongly diverge) and non-tie trials (where both models predict the same choice based on the tally), we can reliably estimate choice parameters and test for the presence of the validity-based tie-breaking mechanism.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2985,
  "candidates_token_count": 520,
  "total_token_count": 4357
}
```
