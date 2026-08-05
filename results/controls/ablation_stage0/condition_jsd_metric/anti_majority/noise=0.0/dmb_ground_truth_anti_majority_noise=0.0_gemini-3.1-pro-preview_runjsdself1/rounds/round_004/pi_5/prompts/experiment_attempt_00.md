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
**Description:** Decision-makers use a bounded evidence accumulation process that integrates sequential dependencies. They default to a non-compensatory heuristic (Take The Best) but smoothly transition to a compensatory strategy (Weighted Additive) when opposing evidence exceeds a soft threshold. Furthermore, choices are subject to sequential dependencies: the baseline preference for an option is dynamically adjusted based on the previous trial's choice, capturing choice inertia or auto-correlation. This moving baseline influences the final decision probabilities alongside the strategy-specific evidence.

**Parameters:**
- threshold: [0.0, 10.0]
- k: [0.1, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- inertia: [-5.0, 5.0]
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
        
    # --- Sequential Dependency (Inertia) ---
    inertia = float(parameters["inertia"])
    z_inertia = np.array([0.0, 0.0])
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        z_inertia[last_resp] = inertia
        
    # --- Decision Formulation ---
    # WADD probabilities with inertia
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * scores_wadd + z_inertia
    z_w = z_w - np.max(z_w)
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities with inertia
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * scores_ttb + z_inertia
    z_t = z_t - np.max(z_t)
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


## COMPETING THEORY
**Description:** Probabilistic Evidence Accumulation with Baseline Evidence: Decision-makers sample cues probabilistically proportional to their log-odds validities. Each sampled cue casts a vote for the option it favors, and decision-makers also possess a baseline level of prior evidence for both options. The process continues until a relative evidence threshold is reached, producing compensatory behavior and graded choice probabilities based on the relative accumulated evidence. The inclusion of baseline evidence smooths out extreme probability predictions and naturally handles cases where no cues favor an option.

**Parameters:**
- theta: [0.1, 20.0]
- c: [0.01, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Log-odds weights for proportional sampling probabilities
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # Total sampling weight favoring each option
    v_a = np.sum(weights * (a > b))
    v_b = np.sum(weights * (b > a))
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    c = float(parameters["c"])
    
    # Add baseline constant to smooth out extreme ratios and avoid zero-evidence edge cases
    v_a_eff = v_a + c
    v_b_eff = v_b + c
    
    # Probability of reaching the relative threshold theta first via power-law rule
    log_p_a_unnorm = theta * np.log(v_a_eff)
    log_p_b_unnorm = theta * np.log(v_b_eff)
    max_log = max(log_p_a_unnorm, log_p_b_unnorm)
    exp_a = np.exp(log_p_a_unnorm - max_log)
    exp_b = np.exp(log_p_b_unnorm - max_log)
    p_a = exp_a / (exp_a + exp_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
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

[4] To cleanly dissociate strict Take The Best (TTB) from the Soft Compensatory Check model, we hold the 'opposing evidence' constant while varying the absolute validity of the top discriminating cue. In all trials, Option A is favored by the highest-validity discriminating cue, and Option B is favored by a fixed set of lower-validity cues. Across trial sets, we shift the top discriminating cue from the 1st to the 2nd to the 3rd most valid cue (by setting higher cues to tie). Strict TTB predicts identical choice probabilities across all these trials, as it relies purely on the ordinal fact that the top discriminating cue favors Option A. The competing theory, however, computes a constant probability of switching to the Weighted Additive (WADD) strategy (since the opposing evidence is constant); but because the top cue's validity decreases, the WADD score difference shifts from favoring Option A to favoring Option B. Thus, the competing theory predicts a graded decrease in the probability of choosing Option A as the top cue's validity drops, whereas TTB predicts no change.

[5] To cleanly dissociate the advocated Soft Compensatory Check model from pure Take The Best (TTB), we parametrically increase the 'opposing evidence' (which drives the transition from TTB to WADD) while keeping the WADD score difference strictly constant. We achieve this by adding matched-validity cues to both options simultaneously. Pure TTB ignores all cues beyond the first discriminating one and thus predicts identical choice probabilities across these trial sets. The advocated model, however, predicts a graded shift in choice probability because the increasing opposing evidence smoothly shifts the decision policy from TTB towards WADD, thereby altering the choice probabilities even though the underlying WADD score difference remains unchanged.

[6] To cleanly dissociate the advocated strict Take The Best (TTB) theory from the competing Soft Compensatory Check model, we parametrically increase the amount of 'opposing evidence' (lower-validity cues favoring the TTB loser) while holding the top discriminating cue constant. In this design, Option A is always favored by the highest-validity cue. Across the main trials, the number of lower-validity cues favoring Option B increases, which increases the total log-odds weight of opposing evidence. Strict TTB ignores all cues beyond the first discriminating one and therefore predicts a constant, high probability of choosing Option A across all these trials. The competing theory, however, evaluates the opposing evidence; as it increases, the model's probability of switching from TTB to the Weighted Additive (WADD) strategy increases. Because WADD strongly favors Option B in trials with high opposing evidence, the competing model predicts a graded, sharp decrease in the probability of choosing Option A. This provides a clear, quantitative dissociation: a flat line (TTB) versus a decreasing curve (Competing Model).

[7] To cleanly dissociate the advocated Soft Compensatory Check model from strict Take The Best (TTB), we exploit the sequential dependency (choice inertia) unique to the advocated theory. We construct an unbalanced experimental design where 83% of the unique trial pairs strongly favor Option A under both models (Option A dominates on the highest validity cues). This induces a strong global base-rate of 'A' responses. We randomly intersperse critical trials where Option A and Option B have identical feature vectors (perfect ties). Strict TTB has no mechanism for history dependence and must guess exactly 50/50 on these tied trials. In contrast, the advocated theory updates its baseline preference using the previous trial's choice. Because the preceding trial is highly likely to have resulted in an 'A' choice, the advocated model's inertia mechanism predicts a systematic deviation from 50/50 (typically a strong bias toward Option A) on the perfectly tied critical trials. This provides a pure, quantitative dissociation based on sequential dynamics rather than static feature weights.

[8] To cleanly dissociate the advocated Probabilistic Evidence Accumulation theory from the competing Soft Compensatory Check model, we parametrically add matched pairs of cues favoring Option A and Option B equally, keeping the absolute WADD score difference strictly constant. In all trials, Option A is favored by the highest-validity cue (winning TTB) and also has a higher total WADD score. Across the trial sets, we add lower-validity cues that favor A and B equally. 

For the competing theory, the absolute WADD score difference remains constant, and TTB consistently favors Option A. Thus, it predicts a high, relatively constant probability of choosing A (only slightly mixing between two stable, high-probability strategies as opposing evidence increases). 

In contrast, the advocated theory evaluates the *ratio* of accumulated evidence (plus a baseline constant). As equal amounts of evidence are added to both options, the ratio of total evidence for A vs. B compresses towards 1. Therefore, the advocated theory uniquely predicts a graded decrease in the probability of choosing Option A, moving steadily towards 0.5 as total evidence increases, providing a sharp quantitative dissociation.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate the advocated Soft Compensatory Check model from the competing Probabilistic Evidence Accumulation theory, we construct a series of 'WADD-tied' trials where the total log-odds evidence for Option A and Option B are virtually identical. Across four critical levels of trials, we parametrically increase this tied total evidence (W). For the competing theory, because the evidence for A and B remains balanced (v_a \u2248 v_b = W) in every trial, the ratio of evidence is always ~1, and the model predicts a flat line of exactly 50% choice probability across all levels of W. For the advocated theory, however, Option A is always the Take The Best (TTB) winner (since it wins the single highest validity cue to match the sum of B's multiple lower cues). As W increases, the 'opposing evidence' (the evidence favoring B) parametrically increases. This drives a smooth transition from strict TTB (which strongly favors A) to WADD (which predicts 50% due to the tie). Thus, the advocated theory uniquely predicts a graded decrease in the probability of choosing Option A towards 50% as W increases, providing a striking quantitative dissociation against the competing theory's flat 50% prediction.",
  "validities": [
    0.881,
    0.818,
    0.731,
    0.731,
    0.679,
    0.679,
    0.622,
    0.622,
    0.562,
    0.562
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
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
      0,
      0,
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
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      0,
      1,
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
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
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
      0,
      1,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
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
      0,
      0,
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
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
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
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4596,
  "candidates_token_count": 1154,
  "total_token_count": 23182
}
```
