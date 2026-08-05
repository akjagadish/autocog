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
**Description:** Evidence Accumulation with Bounded Threshold: People sample cues sequentially in order of validity. Each discriminating cue adds its validity to the evidence for the favored option. If the absolute difference in evidence between the two options reaches a threshold `theta`, sampling stops and a decision is made based on the accumulated evidence (mimicking Take The Best). If all cues are exhausted without the threshold being reached, the decision is based on the final integrated evidence (mimicking Weighted Additive). This allows for early stopping when initial evidence is strong, but continued sampling when evidence is weak or tied.

**Parameters:**
- theta: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        if a[j] > b[j]:
            ev_a += val[j]
        elif b[j] > a[j]:
            ev_b += val[j]
            
        # Early stopping if threshold is reached
        if abs(ev_a - ev_b) >= theta:
            break
            
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## THEORY 2
**Description:** People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

**Parameters:**
- beta: [0.1, 200.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 50.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To distinguish between Take The Best (TTB) and Tallying, we need trials where the highest validity cue points to one option, but the simple majority of features (ignoring validity) points to the other option. TTB relies exclusively on the first discriminating cue in descending order of validity, while Tallying simply counts the number of features won by each option. The selected pairs pit a single high-validity advantage against multiple lower-validity advantages, causing TTB and Tallying to make opposite predictions, or causing one to predict a strong preference while the other predicts a tie.

[1] To maximally distinguish Tallying from Take The Best (TTB), this design pits the highest-validity discriminating cue against the sheer count of winning features. In several trials, the option favored by the single best cue is opposed by an option that wins on multiple lower-validity cues. Tallying will follow the simple majority of features, completely ignoring their validities, while TTB will follow the single highest-validity cue that discriminates, ignoring the number of cues. We also include trials where Tallying predicts a tie (equal number of wins for both options) while TTB predicts a strong preference based on the single highest-validity discriminating cue.

[2] To distinguish the strictly non-compensatory Take The Best (TTB) from the flexible Weighted Additive (WADD) model, we pit the single most valid discriminating cue against a combination of multiple lower-validity cues. TTB always chooses the option favored by the highest-validity discriminating cue, ignoring all others. WADD, depending on its exponent parameter (gamma), can compensate for a disadvantage on the most valid cue if the sum of weighted lower-validity cues is large enough. By systematically varying the number and validity of the opposing cues, we can identify whether subjects use a strict one-reason stopping rule (TTB) or integrate information across all cues (WADD).

[3] To distinguish the fully compensatory but flexible WADD model from the strictly non-compensatory Take The Best (TTB) model, we pit the highest-validity discriminating cue against varying numbers of lower-validity cues. TTB always relies solely on the highest-validity cue that discriminates between the two options. WADD, however, integrates all features and can compensate for a disadvantage on the most valid cue if enough lower-validity cues point the other way (depending on its gamma parameter). By varying whether the top cue is tied, and how many lower-validity cues oppose the first discriminating cue, we generate a profile of choices that can cleanly separate strict one-reason decision making from weighted integration.

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
  "rationale": "To distinguish the Evidence Accumulation with Bounded Threshold model (THEORY 1) from the exponent-scaled Weighted Additive model (THEORY 2), we exploit the threshold mechanism directly. In THEORY 1, if the threshold is reached early, sampling stops and subsequent cues are completely ignored. If the threshold is not reached, it integrates all cues linearly (sum of validities). In THEORY 2, all cues are always integrated, but their weights are non-linearly scaled by an exponent (gamma). We use a set of 4 features with specific validities such that the linear sum of validities is perfectly tied for some pairs (e.g., 0.95 + 0.65 = 1.60 vs 0.85 + 0.75 = 1.60). For these tied pairs, THEORY 1 predicts either a choice for the top cue (if threshold is low) or a perfect tie (if threshold is high). THEORY 2, however, applies an exponent to the validities: if gamma > 1, the option with the more extreme validities (0.95 and 0.65) wins; if gamma < 1, the option with the moderate validities (0.85 and 0.75) wins. By varying the dispersion and sums of the validities across trials, we can cleanly separate the early-stopping/linear-integration dynamics of THEORY 1 from the non-linear full-integration dynamics of THEORY 2.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65
  ],
  "trial_a_ratings": [
    [
      1,
      0,
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
      0
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
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
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
  "prompt_token_count": 2837,
  "candidates_token_count": 557,
  "total_token_count": 5096
}
```
