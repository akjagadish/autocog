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
**Description:** Sequential Evidence Accumulation Theory: Decision-makers sample cues sequentially in order of their validity, accumulating evidence for each option. The accumulation process stops as soon as the absolute difference in evidence between the two options reaches a subject-specific threshold. A low threshold mimics non-compensatory heuristics like Take-The-Best by stopping at the first discriminating cue, while a high threshold mimics compensatory strategies like Weighted Additive (WADD) by integrating all available cues. This natural stopping rule generates variable choice probabilities across different conflict geometries, naturally capturing moderate variance across stimuli without requiring non-linear scaling of validities.

**Parameters:**
- theta: [0.0, 1.25]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    score_a = 0.0
    score_b = 0.0
    
    for j in cue_order:
        score_a += val[j] * a[j]
        score_b += val[j] * b[j]
        
        diff = abs(score_a - score_b)
        # Stop if threshold is reached AND there is a strict difference (to avoid stopping on ties if theta is 0)
        if diff >= theta and diff > 1e-9:
            break
            
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 30.0]
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
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue often points to one option, while the sheer number of winning cues points to the other. TTB will choose the option favored by the single highest-validity discriminating cue, whereas Tallying will choose the option favored by the majority of discriminating cues, regardless of their validities. We also include trials where Tallying predicts a tie but TTB strongly prefers one option.

[1] To robustly dissociate Tallying from Take The Best (TTB), we use 5 features with descending validities. TTB will always choose the option favored by the first discriminating feature (the one with the highest validity). Tallying, however, simply counts the number of features favoring each option, regardless of validity. By pitting a single high-validity feature against multiple lower-validity features, we create clear adversarial trials: TTB strongly prefers the option with the single best feature, while Tallying prefers the option with the majority of winning features. We also include trials where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic choice, further differentiating the two models.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 4-feature design with a steep drop-off in validities after the first cue, but closely spaced validities for the remaining cues (e.g., 0.95, 0.75, 0.65, 0.55). TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, on the other hand, integrates all cues. If WADD's gamma parameter is low, it will behave compensatorily and favor options with multiple lower-validity cues over an option with a single high-validity cue. We construct trials that pit the highest-validity discriminating cue against a coalition of multiple lower-validity cues. For instance, an option favored by only the top cue will be chosen by TTB, whereas WADD (with low-to-moderate gamma) will choose the alternative favored by the three remaining cues. We also include trials where the highest cue is tied, shifting the TTB decision to the second cue, which is then pitted against the third and fourth cues to provide a robust test across different levels of the cue hierarchy.

[3] To robustly dissociate the Weighted Additive (WADD) theory from Take The Best (TTB), we use a 5-feature design with one highly valid cue and several moderately valid cues. TTB relies strictly on the highest-validity discriminating cue, ignoring all others. WADD, however, integrates all cues, allowing a coalition of lower-validity cues to compensate for the lack of the highest-validity cue. We construct a series of adversarial trials where the top discriminating cue points to Option A, but the majority of remaining cues point to Option B. Across these trials, TTB consistently chooses Option A, while WADD (assuming a moderate gamma parameter) will consistently choose Option B due to the overwhelming additive evidence of the lesser cues. By varying which cues tie and which discriminate, we test the compensatory nature of WADD against the non-compensatory lexicographic stopping rule of TTB at different levels of the cue hierarchy.

[4] To dissociate Strategy Mixture Theory from the Generalized WADD theory, we exploit how each model produces intermediate choice probabilities. Strategy Mixture Theory assumes a probabilistic mixture of pure TTB and pure WADD, predicting relatively constant intermediate choice proportions (e.g., matching the mixing parameter alpha) across any trial where TTB and WADD strongly conflict, regardless of the exact margin of the WADD sum. In contrast, Generalized WADD achieves intermediate probabilities either via high noise (low beta) or by precisely tuning the gamma parameter to make the exponentiated sums of the two options nearly equal. By presenting multiple distinct conflict trials with varying differences in raw feature sums, Generalized WADD cannot simultaneously equate the sums for all trials with a single gamma parameter. Consequently, Generalized WADD will predict extreme probabilities on some conflict trials and intermediate on others, whereas Strategy Mixture Theory predicts consistent intermediate probabilities across all such conflict trials.

[5] To quantitatively dissociate Generalized WADD from Strategy Mixture Theory, we manipulate the margin by which the compensatory WADD strategy opposes the non-compensatory TTB strategy. Strategy Mixture Theory assumes choices are a stable probabilistic coin-flip between pure TTB and pure WADD. Therefore, on any trial where TTB strongly favors Option A and WADD strongly favors Option B, Strategy Mixture predicts a relatively constant intermediate choice probability driven by the mixing parameter 'alpha'. In contrast, Generalized WADD integrates all features non-linearly. A single 'gamma' parameter cannot flatten the varying evidence margins across different trials. By presenting a spectrum of conflicts—ranging from the top cue being opposed by all four remaining cues, to being opposed by only two—Generalized WADD is forced to predict a graded shift in probabilities, whereas Strategy Mixture predicts a step-function or constant mixture across these conflicts.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting their fundamentally different mechanisms for scaling evidence. WADD scales validities via exponentiation (gamma), which relies on the ratio between validities. Because the ratio between adjacent validities increases as validities get smaller (e.g., 0.75/0.65 > 0.85/0.75), WADD naturally becomes *more* non-compensatory (TTB-like) at lower levels of the cue hierarchy. In contrast, SEA uses an absolute difference threshold (theta) on unscaled validities. Because the absolute values of the validities decrease down the hierarchy, SEA naturally becomes *less* non-compensatory at lower levels, as single cues are no longer large enough to cross the fixed threshold. By pitting a single discriminating cue against the remaining lower cues at different levels of the hierarchy (e.g., Cue 2 vs Cues 3-5, and Cue 3 vs Cues 4-5), SEA predicts non-compensatory choices at the top of the hierarchy and compensatory choices at the bottom. WADD predicts the exact opposite pattern, making them highly distinguishable.",
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
      0,
      1,
      0,
      0
    ],
    [
      1,
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
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3327,
  "candidates_token_count": 538,
  "total_token_count": 7778
}
```
