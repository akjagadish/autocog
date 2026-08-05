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
**Description:** Weighted Additive (WADD) model with non-linear validity scaling and extended lapse rate: Decision makers evaluate options by computing a weighted sum of their feature values. The weights are formed by centering the cue validities (val - 0.5) and raising their absolute values to a power gamma (preserving sign), which allows the model to interpolate between equal-weighting, proportional weighting, and non-compensatory behavior without extreme log-odds scaling. To account for the high degree of noise or chance-level behavior observed in specific paradigms, the lapse rate (epsilon) can range up to 1.0, and beta down to 0.0, allowing the model to capture completely random guessing natively.

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Center validities so 0.5 gives 0 weight, avoiding discrete parameter
    base_w = val - 0.5
        
    # Signed exponentiation to handle negative base_w with fractional gamma safely
    w = np.sign(base_w) * (np.abs(base_w) ** gamma)
        
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Random Guessing: The data indicates that subjects in these specific experimental paradigms completely ignore all feature information and validities. Likely due to a lack of trial-by-trial feedback or low motivation, subjects simply choose between the two options with equal probability on every trial, resulting in chance-level performance across all metrics.

**Parameters:**
- dummy: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # Reference the dummy parameter to satisfy the parameter usage constraint
    _ = float(parameters["dummy"])
    
    # Pure random guessing: uniform 50/50 probability regardless of the stimulus
    return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design specifically pits Take The Best (TTB) against Tallying. TTB relies solely on the highest-validity cue that discriminates between the two options, completely ignoring the sheer number of cues favoring each option. In contrast, Tallying counts the total number of features favoring each option, ignoring cue validities. The trials are constructed so that one option always wins on the highest-validity discriminating cue (favored by TTB), while the other option wins on a greater number of lower-validity cues (favored by Tallying). This provides a strong quantitative dissociation between the two models.

[1] This design quantitatively dissociates Tallying from Take The Best (TTB) by exploiting how each model generates choice probabilities. TTB is a 'one-reason' decision rule: once the highest-validity cue discriminates between options, it stops searching. Consequently, TTB's predicted choice probability depends entirely on a fixed binary score difference (1 vs 0) and remains constant across all trials where the first cue discriminates. In contrast, Tallying counts the total number of features favoring each option, and its choice probability scales with the *magnitude* of the difference in tallies. By holding the highest-validity cue constant (always discriminating in favor of one option) while systematically varying the tally difference on the remaining cues (from strongly opposing the first cue to strongly supporting it), Tallying predicts a graded, monotonic change in choice probabilities across trials, whereas TTB predicts a completely flat probability profile.

[2] This design pits Take The Best (TTB) against the Weighted Additive (WADD) model by systematically varying the compensatory evidence from lower-validity cues while keeping the highest-discriminating cue constant. TTB is a non-compensatory, 'one-reason' heuristic: it decides solely based on the highest-validity cue that discriminates between the options, completely ignoring all other cues. Consequently, TTB predicts identical choice probabilities across all trials where the same cue determines the choice. In contrast, WADD computes a weighted sum of all features. By presenting trials where the highest-validity cue favors one option, but an increasing number of lower-validity cues oppose it, WADD predicts a graded change in choice probabilities (even flipping the choice if the gamma parameter allows for compensatory weighting). TTB predicts a completely flat probability profile across these same trials.

[3] This design quantitatively dissociates the Weighted Additive (WADD) model with non-linear validity scaling from Take The Best (TTB) by systematically varying the amount of compensatory evidence on lower-validity cues while holding the highest-validity discriminating cue constant. TTB is a non-compensatory 'one-reason' heuristic: its choice and predicted choice probability are determined entirely by the first discriminating cue, meaning it predicts a completely flat choice probability profile across any set of trials where the same top cue discriminates. In contrast, WADD integrates all features. By creating a gradient of trials where the highest-validity cue favors Option A but the number of lower-validity cues favoring Option B systematically decreases, WADD predicts a graded, monotonic change in choice probabilities, whereas TTB predicts identical probabilities across the gradient. We repeat this logic for subsets of trials where the first cue is tied and the second cue becomes the highest discriminating cue.

[4] This design quantitatively dissociates Random Guessing from the Weighted Additive (WADD) model by presenting a systematic gradient of evidence. By varying the difference in weighted feature values from Option A dominating completely to Option B dominating completely, WADD predicts a monotonic change in choice probabilities tracking the evidence difference. In stark contrast, Random Guessing predicts a completely flat choice probability profile at exactly 0.5 across all trials, regardless of how overwhelming the evidence is for one option over the other.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates the Weighted Additive (WADD) model from Random Guessing by contrasting trials with overwhelming evidence against trials with complex trade-offs. Random Guessing predicts a completely flat 50/50 choice probability across all trials, asserting total insensitivity to feature information. In contrast, WADD predicts that choice probabilities will track the weighted evidence difference. By including 'dominance' trials (where all features favor one option) alongside various trade-off trials (where high-validity cues are pitted against multiple lower-validity cues), we not only test the core divergence between evidence-sensitivity (WADD) and complete insensitivity (Guessing), but also provide the necessary variance to identify WADD's non-linear scaling parameter (gamma) if subjects are indeed evaluating the features.",
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
      1,
      0,
      1,
      0,
      0
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
      1,
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
      0,
      1,
      0,
      1,
      1
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
      0,
      1,
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
  "prompt_token_count": 2689,
  "candidates_token_count": 584,
  "total_token_count": 4262
}
```
