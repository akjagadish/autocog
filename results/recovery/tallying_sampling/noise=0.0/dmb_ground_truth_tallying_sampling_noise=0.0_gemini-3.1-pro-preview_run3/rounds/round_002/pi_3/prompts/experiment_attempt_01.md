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
**Description:** Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Majority of Confirming Dimensions (MCD): People make decisions by comparing options dimension by dimension rather than computing holistic values. On each feature, the option with the superior value earns a point (a 'confirming dimension'). The option with the most points across all features is chosen. For binary features, this dimension-wise comparison yields the same relative preference as Tallying, but through a fundamentally different cognitive process (pairwise comparisons rather than independent counting).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("MCD expects a (2, n_features) state.")
        
    a, b = stim[0], stim[1]
    
    # MCD: Count how many features strictly favor each option
    score_a = np.sum(a > b)
    score_b = np.sum(b > a)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over MCD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit the compensatory nature of WADD versus the non-compensatory, lexicographic nature of TTB. The experiment uses 5 features with descending validities. In the critical trials, one option possesses the single best discriminating cue (which dictates TTB's choice), while the competing option possesses multiple lower-validity cues whose combined weight exceeds the single best cue (which dictates WADD's choice). By including trials where the 'best' cue is offset by 2, 3, or 4 lower-validity cues, we can robustly measure the degree of compensation subjects exhibit. Control trials where both models agree are also included to ensure basic task engagement.

[1] To cleanly dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, this design uses 4 features with non-descending validities. By scrambling the order of validities (e.g., placing the most valid cue second), we ensure subjects must actively process the validity numbers rather than relying on left-to-right reading order. The trials include critical compensatory cases where the option possessing the single best discriminating cue is pitted against an option possessing multiple lower-validity cues whose combined weight exceeds the best cue. In these cases, TTB and WADD make opposite predictions. We also include trials where the best cue is tied, forcing TTB to look at the second-best cue, which is then outweighed by the remaining cues for WADD, as well as control trials where both models agree.

[2] This design aims to cleanly dissociate Tallying (Equal Weight) from the Weighted Additive (WADD) model. We use 5 features with a steep drop-off in validities (two highly valid cues, three moderately valid cues). Critical trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. In these cases, Tallying predicts the subject will choose the option with the higher count of positive features, whereas WADD predicts the subject will choose the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference, and agreement trials to ensure basic task engagement.

[3] This design specifically dissociates the compensatory Weighted Additive (WADD) model from the Tallying (Equal Weight) heuristic. By utilizing a set of 5 features with a steep drop-off in validities (two highly valid cues, three less valid cues), we create scenarios where the two models make conflicting predictions. Critical trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. In these cases, Tallying predicts the subject will choose the option with the higher count of positive features, whereas WADD predicts the subject will choose the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference based on validity differences, as well as agreement trials to ensure basic task engagement.

[4] While Tallying and Majority of Confirming Dimensions (MCD) produce identical relative score differences for two-option binary-feature choices (since shared features cancel out in both tally differences and MCD score differences), we can test if the presence of shared positive features alters decision noise or reaction times in ways not captured by the basic softmax rule. This design varies the number of shared '1's between options while keeping the number of unique '1's constant, allowing us to observe if subjects process holistic sums (Tallying) differently than pure dimension-wise comparisons (MCD) when cognitive load changes.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "Since Tallying and Majority of Confirming Dimensions (MCD) are mathematically equivalent in their choice probability predictions for any two-option binary-feature task (the score differences are identical), a pure choice-probability dissociation is impossible within the standard softmax framework. However, this design systematically manipulates the baseline number of positive features (shared '1's vs. shared '0's) while keeping the relative difference constant. If Tallying is the true underlying cognitive process, the overall magnitude of the tallies might influence response times or confidence (e.g., via a Weber-Fechner law effect on the sum), whereas MCD, which filters out shared features early as 'ties', should be entirely invariant to the number of shared '1's or '0's.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
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
      1,
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
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
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
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
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
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2762,
  "candidates_token_count": 511,
  "total_token_count": 4673
}
```
