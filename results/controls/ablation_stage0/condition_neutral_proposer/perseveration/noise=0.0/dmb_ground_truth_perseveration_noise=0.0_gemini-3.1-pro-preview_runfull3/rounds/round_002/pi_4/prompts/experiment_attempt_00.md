# experiment_attempt_00

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

**Parameters:**
- bias_a: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## THEORY 2
**Description:** People make decisions by computing the expected value of each option, integrating both the magnitude of the features and their subjectively weighted validities. In a Weighted Additive (WADD) strategy, every feature contributes to an option's total score proportionally to a non-linear transformation of its cue validity, capturing subjective distortion of probabilities or weights. Choice probabilities are generated via a softmax function over these weighted sums, with an additional lapse rate to account for random errors.

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- gamma: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = np.power(val, gamma)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To effectively dissociate Take The Best (TTB) from Tallying, we use a 5-feature design where the cues have strictly descending validities. The trials are constructed such that the single highest-validity discriminating cue always favors one option, while a numerical majority of lower-validity cues favor the alternative option. Because TTB stops at the first discriminating cue, it will systematically choose the option favored by the single best cue. In contrast, Tallying ignores cue validities and simply counts the number of winning features, leading it to systematically choose the opposite option.

[1] To robustly dissociate Take The Best (TTB) from Tallying, this experiment uses a 6-feature design. We systematically vary which cue is the highest-validity discriminating cue (e.g., the 1st, 2nd, or 3rd most valid cue) by introducing ties on the higher-validity cues. In every trial, the first discriminating cue strongly favors one option (which TTB will choose), while a sheer numerical majority of the remaining lower-validity cues favor the opposite option (which Tallying will choose). By shifting the decisive cue down the validity hierarchy across trials, we also ensure that TTB cannot be mimicked by a simple 'always look at cue 1' heuristic, and we test the stopping rule of TTB against the exhaustive compensatory counting of Tallying.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 5-feature design with closely spaced, high validities. In each trial, the options are designed such that the single highest-validity discriminating cue strongly favors one option, which TTB will reliably choose since it ignores all remaining cues. However, a coalition of multiple lower-validity cues favors the alternative option. Because the validities are relatively close to each other, the sum of these lower-validity cues (even after being non-linearly transformed by WADD's gamma parameter, up to gamma=5) will outweigh the single best cue. Thus, WADD will systematically choose the opposite option to TTB across a wide range of parameter values.

[3] To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) strategy, we use 5 closely spaced, high-validity cues. In each trial, the single best discriminating cue strongly favors one option, meaning TTB will invariably select it. However, the alternative option is favored by a coalition of two or three slightly less valid cues. Because the validities are closely spaced (0.82 to 0.90), the sum of the validities for the alternative option will strictly outweigh the single best cue, even if WADD applies a strong non-linear transformation (up to gamma=5). Thus, TTB and WADD yield diametrically opposed predictions across all trials.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To maximally dissociate Theory 1 (Position Bias) from Theory 2 (Weighted Additive), we need to manipulate the stimulus such that the objectively superior option alternates between Option A and Option B. Theory 2's predictions will track the feature values, swinging strongly from preferring A to preferring B depending on which option has higher weighted validities. In stark contrast, Theory 1 ignores the stimulus completely and predicts a constant probability of choosing Option A across all trials. By including extreme cases (e.g., A has all positive features and B has none, and vice versa) as well as closer trade-offs, we ensure that Theory 2 produces highly variable choice probabilities across trials while Theory 1 remains perfectly flat.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0
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
      1
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
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
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
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
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
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
  "prompt_token_count": 2502,
  "candidates_token_count": 438,
  "total_token_count": 3546
}
```
