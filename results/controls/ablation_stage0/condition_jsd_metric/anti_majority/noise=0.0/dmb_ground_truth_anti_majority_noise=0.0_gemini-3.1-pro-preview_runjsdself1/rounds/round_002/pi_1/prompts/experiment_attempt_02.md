# experiment_attempt_02

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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take The Best with Soft Compensatory Check: Decision-makers default to a non-compensatory heuristic (Take The Best) by relying on the most valid discriminating cue. However, instead of a deterministic shift, if the combined evidence (sum of log-odds weights) from cues opposing this initial choice grows, the probability of switching to a fully compensatory integration strategy (Weighted Additive) increases smoothly. This is modeled via a logistic function centered on a subject-specific evidence threshold, allowing for graded, conflict-driven transitions between strategies.

**Parameters:**
- threshold: [0.0, 10.0]
- k: [0.1, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import scipy.special
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds weights for WADD and opposing evidence
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # --- Take The Best (TTB) Phase ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is not None:
        # Calculate opposing evidence: sum of weights where the TTB loser beats the TTB winner
        if winner_ttb == 0:
            opposing_evidence = np.sum(weights * (b > a))
        else:
            opposing_evidence = np.sum(weights * (a > b))
            
        k = float(parameters["k"])
        threshold = float(parameters["threshold"])
        # Soft transition: probability of using WADD is a logistic function of opposing evidence
        p_wadd = float(scipy.special.expit(k * (opposing_evidence - threshold)))
    else:
        # If no cue discriminates, default to WADD (which will tie)
        p_wadd = 1.0
        
    # --- Decision Formulation ---
    # WADD probabilities
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_t = np.exp(z_t)
    p_core_ttb = e_t / np.sum(e_t)
    
    # Mix strategies based on soft threshold
    p_core = p_wadd * p_core_wadd + (1.0 - p_wadd) * p_core_ttb
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] To cleanly dissociate Take The Best (TTB) from Tallying, we use 5 features with distinct validities. In these trials, the highest-validity discriminating cue always favors one option (which TTB will choose), while a larger number of lower-validity cues favor the other option (which Tallying will choose). This creates a strict quantitative dissociation where TTB predicts a choice based on a single high-validity feature, whereas Tallying predicts the opposite choice based on a simple unweighted count of winning features.

[1] To cleanly dissociate Tallying from Take The Best (TTB), we use 5 features with monotonically decreasing validities. The trials are designed such that the highest-validity discriminating cue favors one option (which TTB strictly follows), while a larger number of lower-validity cues favor the alternative option (which Tallying follows). Furthermore, some trials include sets where Tallying predicts a tie (equal number of wins for both options) while TTB makes a deterministic prediction based on the top cue. This creates a strong quantitative dissociation across different trial types.

[2] To cleanly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use 5 features with validities chosen such that the highest validity cue can be out-weighed by the sum of multiple lower-validity cues in terms of log-odds. In critical trials, one option is favored by the single most valid discriminating cue (which TTB strictly follows), while the alternative option is favored by a combination of several less valid cues whose combined log-odds weights exceed the weight of the top cue (which WADD follows). This creates a direct quantitative and qualitative dissociation between the non-compensatory 'one-reason' logic of TTB and the compensatory integration of WADD.

[3] To cleanly dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we use 5 features with validities chosen such that the highest validity cue can be outweighed by the sum of multiple lower-validity cues when their log-odds are combined. In critical trials, one option is favored by the single most valid discriminating cue (which TTB strictly follows), while the alternative option is favored by a combination of several less valid cues whose combined log-odds weights exceed the weight of the top cue (which WADD follows). Control trials are also included where both models agree, ensuring that differences in choice patterns directly reflect the compensatory vs. non-compensatory integration of information.

[4] To dissociate strict Take The Best (TTB) from the Soft Compensatory Check model, we systematically vary the amount of 'opposing evidence' while keeping the prediction of TTB constant. In all critical trials, Option A is favored by the single highest-validity discriminating cue. Across the trials, the number of lower-validity cues favoring Option B increases, thereby increasing the compensatory evidence against Option A. Strict TTB predicts a constant probability of choosing Option A regardless of the lower-validity cues, as it ignores them completely. In contrast, the competing theory predicts that as opposing evidence accumulates, the probability of switching to a Weighted Additive (WADD) strategy increases, leading to a graded decrease in the likelihood of choosing Option A.

[5] To dissociate strict Take The Best (TTB) from the Soft Compensatory Check model, we hold the 'opposing evidence' constant while varying the supporting evidence for the TTB winner. In critical trials, Option A is favored by the highest-validity cue, and Option B is favored by the same fixed set of lower-validity cues (constant opposing evidence). We systematically add lower-validity supporting cues to Option A. Strict TTB ignores these supporting cues (since the top cue already discriminates) and predicts a constant choice probability across these trials. In contrast, the Soft Compensatory Check model, which occasionally switches to a Weighted Additive (WADD) strategy based on the opposing evidence, predicts that adding supporting cues to Option A will increase its WADD score, thereby predicting a graded increase in the probability of choosing Option A.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate strict Take The Best (TTB) from the Soft Compensatory Check model, we hold the 'opposing evidence' constant while varying the absolute validity of the top discriminating cue. In all trials, Option A is favored by the highest-validity discriminating cue, and Option B is favored by a fixed set of lower-validity cues. Across trial sets, we shift the top discriminating cue from the 1st to the 2nd to the 3rd most valid cue (by setting higher cues to tie). Strict TTB predicts identical choice probabilities across all these trials, as it relies purely on the ordinal fact that the top discriminating cue favors Option A. The competing theory, however, computes a constant probability of switching to the Weighted Additive (WADD) strategy (since the opposing evidence is constant); but because the top cue's validity decreases, the WADD score difference shifts from favoring Option A to favoring Option B. Thus, the competing theory predicts a graded decrease in the probability of choosing Option A as the top cue's validity drops, whereas TTB predicts no change.",
  "validities": [
    0.95,
    0.9,
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
    ]
  ],
  "trial_b_ratings": [
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
      1,
      1,
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
      0,
      0,
      0,
      1,
      1,
      0
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
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3994,
  "candidates_token_count": 670,
  "total_token_count": 11965
}
```
