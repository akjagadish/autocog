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
**Description:** Decision makers evaluate options by sequentially processing features in descending order of their validity. They accumulate evidence in the form of log-odds weights for each feature. However, accumulation is not always exhaustive; it stops as soon as the absolute accumulated evidence exceeds a subjective threshold. A low threshold mimics a Take The Best heuristic (stopping at the first discriminating cue), while a high threshold mimics a Weighted Additive strategy (integrating all available cues).

**Parameters:**
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Convert validities to log-odds weights (Bayesian evidence)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    threshold = float(parameters["threshold"])
    
    evidence = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            evidence += weights[j] * diff
            # Stop accumulating if evidence exceeds the subjective threshold
            if abs(evidence) >= threshold:
                break
                
    # The accumulated evidence represents the log-odds favoring option A over B
    scores = np.array([evidence, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Decision makers use a Weighted Additive (WADD) strategy to evaluate options, integrating all available features. Instead of raw validities or linear shifts, they weight each feature by its log-odds, which is the mathematically principled way to linearly accumulate independent evidence (equivalent to Naive Bayes). The total score for each option is the sum of these log-odds weights for the features it possesses. The option with the higher total score is chosen probabilistically via a softmax function over the scores, subject to a baseline lapse rate.

**Parameters:**
- beta: [0.1, 25.0]
- epsilon: [0.0, 0.5]
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
    
    # Calculate log-odds of validities to represent the true Bayesian weight of evidence.
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax choice rule with numerical stability
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
[0] To dissociate Take The Best (TTB) from Tallying, this design uses 5 features with monotonically decreasing validities. The trials are constructed such that the most valid discriminating cue favors one option (which TTB will choose), while a larger number of less valid cues favor the other option (which Tallying will choose). This creates a direct contradiction between the one-reason decision making of TTB and the unweighted feature-counting of Tallying.

[1] To avoid repeating the simple 'most valid vs many less valid' monotonic design, this experiment uses a non-monotonic (scrambled) validity order to prevent simple left-to-right reading strategies. Crucially, it introduces trials where Tallying predicts a perfect tie (50/50 choice) while Take The Best (TTB) predicts a deterministic choice based on the highest-validity cue. It also includes trials where the two heuristics are diametrically opposed, and trials where they agree, providing a robust set of quantitative dissociations. Tallying will show indifference on the tie trials and sensitivity to the count of winning features, whereas TTB will consistently follow the single most valid discriminating cue regardless of the tally.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 5-feature design with scrambled validities to prevent left-to-right reading strategies. The trials are crafted to pit the single most valid discriminating cue (which dictates TTB's choice) against the sum of the remaining validities (which dictates WADD's choice). This includes trials where the models make diametrically opposed predictions, as well as trials where WADD predicts a perfect tie (indifference) while TTB predicts a deterministic choice based on the highest-validity cue. We also include trials where the top cue ties, forcing TTB to drop down to the next best cue, which again contradicts the WADD sum.

[3] This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive model (WADD) using a 5-feature design with non-monotonic validities. The validities are chosen such that the highest validity cue (0.95) is strictly less than the sum of the remaining cues, allowing WADD to override the top cue if enough lower-validity cues align. The trial set includes: (1) TTB choosing based on the top cue while WADD chooses the option with the sum of all other cues; (2) A perfect tie for WADD where TTB is deterministic based on the top cue; (3) Trials where the top cue(s) tie, forcing TTB to drop down its cue hierarchy, yet WADD still prefers the other option based on the remaining sum; and (4) Control trials where both models agree. This provides a rich set of conditions to test the compensatory nature of WADD against the non-compensatory, lexicographic stopping rule of TTB.

[4] This design utilizes 6 features with non-monotonic validities to robustly dissociate Take The Best (TTB) from Tallying (Unit-Weight Linear). Tallying relies on the simple sum of positive features, whereas TTB relies solely on the highest-validity discriminating cue. The trial set includes: (1) Extreme opposition where TTB chooses an option with only 1 positive feature because it is the most valid, while Tallying chooses the other with 5 positive features; (2) Trials where the top cue(s) tie, forcing TTB to drop down its cue hierarchy, yet Tallying strongly prefers the opposite option; (3) Trials where Tallying predicts a perfect tie (equal number of positive features) while TTB makes a deterministic choice based on the top discriminating cue; and (4) Control trials where both models agree. This comprehensive set tests both the non-compensatory nature of TTB and its indifference to the total feature count.

[5] To strongly dissociate Tallying from Take The Best (TTB), this design uses 6 features with non-monotonic validities. The trials are constructed to pit the single-reason lexicographic stopping rule of TTB against the unweighted feature-counting rule of Tallying. The trial set includes: (1) Extreme opposition where TTB chooses an option with only 1 positive feature because it is the most valid, while Tallying chooses the other with 5 positive features; (2) Trials where Tallying predicts a perfect tie (equal number of positive features) while TTB makes a deterministic choice; (3) Trials where the top cue(s) tie, forcing TTB to drop down its hierarchy to make a choice that still directly opposes Tallying's sum; and (4) Reversals of these patterns to ensure balance.

[6] This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive (WADD) model using the log-odds of validities. WADD integrates the Bayesian weight of evidence (log-odds), allowing multiple lower-validity cues to outweigh a single high-validity cue. We use 5 features with non-monotonic validities. The trials pit the highest-validity cue (which TTB relies on exclusively) against combinations of lower-validity cues whose sum of log-odds exceeds the top cue's log-odds (which WADD relies on). The design includes trials where TTB and WADD make opposing choices, trials where they agree, and trials where the top cue ties, forcing TTB to the second-best cue while WADD still integrates all features.

[7] This design quantitatively dissociates the Weighted Additive (WADD) model from Take The Best (TTB) by exploiting WADD's log-odds weighting of evidence. We use 5 features with non-monotonic validities. The validities are chosen such that the single highest-validity cue (which completely determines TTB's choice) has a lower log-odds weight than the sum of several lower-validity cues. The trial set includes direct oppositions where TTB chooses an option based solely on the top cue, while WADD chooses the other option based on the accumulated evidence of the remaining cues. It also includes trials where the top cue (or top two cues) tie, forcing TTB to drop down its hierarchy to make a choice that still directly opposes WADD's compensatory sum.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design tests the advocated theory's threshold-based sequential evidence accumulation against the pure Weighted Additive (WADD) log-odds integration of the competing theory. By systematically varying how many of the top-validity cues favor Option A while the remaining lower-validity cues favor Option B, we can identify whether subjects integrate all information (as WADD predicts) or stop accumulating once a subjective evidence threshold is met (as the advocated theory predicts). The trials include cases where the top cue alone, or the top two cues, are pitted against the sum of all remaining cues, allowing us to pinpoint the presence and approximate level of the stopping threshold.",
  "validities": [
    0.7,
    0.85,
    0.75,
    0.7,
    0.8,
    0.75
  ],
  "trial_a_ratings": [
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
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
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
      0,
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
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3620,
  "candidates_token_count": 458,
  "total_token_count": 7187
}
```
