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
**Description:** People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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


## THEORY 2
**Description:** Two-Stage Confidence-Threshold Strategy Selection: Decision-makers default to the fast and frugal Take-The-Best (TTB) heuristic, evaluating options based solely on the most valid discriminating cue. However, if the validity of this top discriminating cue falls below a subjective confidence threshold, the decision-maker deems the single-cue evidence insufficient and falls back to a compensatory Weighted Additive (WADD) strategy that integrates all available cues.

**Parameters:**
- confidence_threshold: [0.5, 0.8]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    confidence_threshold = float(parameters["confidence_threshold"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    scores = np.zeros(2)
    if np.any(discrim_mask):
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        if max_v >= confidence_threshold:
            # Strategy 1: Take-The-Best (TTB)
            top_idx = np.where((discrim_mask) & (validities == max_v))[0][0]
            if stim[0, top_idx] > stim[1, top_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
        else:
            # Strategy 2: Weighted Additive (WADD) fallback
            wadd_scores = stim @ validities
            if wadd_scores[0] > wadd_scores[1]:
                scores[0] = 1.0
            elif wadd_scores[1] > wadd_scores[0]:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
[0] This experiment is designed to strongly dissociate Tallying (which counts the number of features favoring each option, ignoring validities) from the Weighted Additive rule (WADD, which weighs each feature by its validity). We use 5 features with a steep drop-off in validities between the top two and the bottom three. In several critical trials, Option A is superior on the two most valid features (thus strongly favored by WADD), while Option B is superior on the three least valid features (thus favored by Tallying, as it wins 3 to 2). We also include trials where Tallying predicts indifference (2 wins vs 2 wins) but WADD predicts a clear preference, and baseline trials where both heuristics agree, to ensure overall model identifiability.

[1] To maximally distinguish Weighted Additive (WADD) from Tallying, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. This allows us to construct 'dissociation' trials where one option wins on the two most valid features (favored by WADD) while the other option wins on the three least valid features (favored by Tallying). We also include 'tie' trials where Tallying predicts indifference (equal number of wins) but WADD has a clear preference due to validity weighting, and 'agreement' trials where both heuristics favor the same option. This mix ensures the models are identifiable and their distinct mechanisms (magnitude/validity weighting vs. simple counting) are cleanly separated.

[2] We employ a 5-feature design with a pronounced gap between two high-validity features and three lower-validity features. This structure allows us to construct strong dissociation trials where Tallying favors an option that wins on the three lower-validity features (winning 3 to 2), while the Weighted Additive (WADD) rule favors the option that wins on the two high-validity features because their combined weight exceeds the sum of the bottom three. We also include trials where Tallying predicts a tie (2 wins vs 2 wins) but WADD strongly favors one option, as well as baseline agreement trials to ensure robust parameter estimation for both models.

[3] This experiment uses a 6-feature design to create a robust quantitative dissociation between WADD and Tallying. The validities are structured with two very high values and four much lower values. This allows us to create 'opposition' trials where Option A wins on a small number of high-validity features (favored by WADD), while Option B wins on a larger number of low-validity features (favored by Tallying). Additionally, we include 'tie' trials where Tallying counts an equal number of wins for both options (predicting indifference), but WADD strongly predicts a preference based on the validities. By varying the magnitude of the WADD difference and the Tallying win-differential across trials, we can precisely identify the choice rule and the noise parameters.

[4] To maximally distinguish Take-The-Best (TTB) from the Weighted Additive rule (WADD), we must exploit their fundamental difference: TTB is non-compensatory and relies solely on the highest-validity discriminating cue, whereas WADD is compensatory and integrates all cues weighted by their validities. We use 5 features with linearly decreasing validities. The critical dissociation trials pit the single most valid cue against a combination of several lower-validity cues. In these trials, TTB will deterministically choose the option favored by the top discriminating cue, while WADD will choose the other option because the summed weight of the lower-validity cues exceeds the weight of the single highest-validity cue. We also include trials where the top cue is tied to test whether TTB properly drops down to the second cue while WADD continues to integrate all cues, as well as agreement trials to ensure both models can be fit reliably.

[5] To strongly dissociate the non-compensatory Take-The-Best (TTB) heuristic from the compensatory Weighted Additive (WADD) model, we use a 5-feature design with a clear hierarchy of validities. In critical dissociation trials, one option is favored by the single most valid cue (which TTB relies on exclusively), while the other option is favored by multiple lower-validity cues whose combined weight exceeds the top cue (which WADD integrates). We also include trials where the top cue is tied, forcing TTB to drop down to the second cue, while WADD continues to integrate all cues. Baseline agreement trials are included to stabilize parameter estimation.

[6] To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we exploit how the two models handle lower-validity discriminating cues. TTB is strictly non-compensatory: its choice probabilities depend only on the single highest-validity discriminating cue, meaning the number and strength of lower-validity cues favoring the alternative option have zero effect. In contrast, PCS computes choice probabilities based on the relative weights of all discriminating cues (a Plackett-Luce choice rule). By holding the highest-validity discriminating cue constant while systematically increasing the number of lower-validity cues favoring the alternative option, we create a strong quantitative dissociation: TTB predicts flat choice probabilities across these trials, whereas PCS predicts a decreasing preference for the option favored by the top cue as the competing cues accumulate.

[7] To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we systematically vary the number and validity of lower-validity cues that oppose the highest-validity discriminating cue. TTB is strictly non-compensatory: its choice is determined entirely by the top discriminating cue, meaning any lower-validity cues favoring the alternative option are ignored and have no effect on choice probabilities (beyond fixed lapse/softmax noise). In contrast, PCS uses a Plackett-Luce choice rule over all discriminating cues. Thus, as more lower-validity cues stack up against the top cue, PCS predicts a graded decrease in the probability of choosing the option favored by the top cue. By including trials where the top cue is opposed by one, two, three, or four lower-validity cues, we create a strong quantitative dissociation between the step-function prediction of TTB and the graded-integration prediction of PCS.

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
  "rationale": "To maximally distinguish Theory 1 (pure Take-The-Best) from Theory 2 (Two-Stage Confidence-Threshold Strategy Selection), we must exploit the confidence threshold mechanism in Theory 2. Theory 2 defaults to TTB but falls back to a compensatory Weighted Additive (WADD) rule if the highest discriminating cue's validity is below a subjective confidence threshold (which lies between 0.5 and 0.8). We use 6 features with validities spanning from 0.90 down to 0.50. We design critical trials where the highest discriminating cue has a validity of 0.90, 0.75, 0.65, or 0.60. In these trials, the top discriminating cue favors one option (which TTB will pick), while the sum of the remaining lower-validity cues strongly favors the other option (which WADD will pick). For the 0.90 cue, both theories will confidently use TTB and agree. However, for the 0.75, 0.65, and 0.60 cues, Theory 2 may fall below its confidence threshold and switch to WADD, predicting the opposite choice of Theory 1. By varying the validity of the top discriminating cue across trials, we can precisely identify if and where a subject's strategy shifts from non-compensatory (TTB) to compensatory (WADD).",
  "validities": [
    0.9,
    0.75,
    0.65,
    0.6,
    0.55,
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
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
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
      1,
      1
    ],
    [
      1,
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
      0,
      0,
      0
    ],
    [
      1,
      0,
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
      1
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
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
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
      1,
      1
    ],
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
      0,
      1,
      1
    ],
    [
      0,
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
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1
    ],
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
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
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
  "prompt_token_count": 3675,
  "candidates_token_count": 863,
  "total_token_count": 7270
}
```
