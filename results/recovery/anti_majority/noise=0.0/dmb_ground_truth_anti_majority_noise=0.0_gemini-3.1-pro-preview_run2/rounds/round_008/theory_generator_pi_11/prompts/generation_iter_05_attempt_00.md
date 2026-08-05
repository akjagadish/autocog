# generation_iter_05_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 2 (= `pi_10`).

Propose a brand-new theory that abandons the global dilution mechanism (dividing by cue count or validity sum). Instead, explore a theory based on heuristic switching (e.g., a mixture of Take-The-Best and Tallying) or rank-based weighting where cues are evaluated based on their relative rank rather than absolute validities. The new theory should naturally account for why adding weak cues has minimal or slightly positive impact rather than causing massive subjective devaluation.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.579)

**Description**
Dual-Process Strategy Selection Theory: Decision-makers probabilistically select between a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive, WADD) on a trial-by-trial basis. The probability of employing the non-compensatory heuristic is a logistic function of the absolute validity of the highest-ranking discriminating cue. When the top discriminating cue is highly valid, subjects are highly likely to rely solely on it (TTB). However, when the top discriminating cue is weak, confidence in the heuristic drops, and subjects fall back to integrating all available information (WADD).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    tau = float(parameters["tau"])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    top_cue = None
    for j in cue_order:
        if a[j] != b[j]:
            top_cue = j
            break
            
    if top_cue is None:
        p_mix = np.array([0.5, 0.5])
    else:
        v_top = val[top_cue]
        
        # Probability of using TTB is a logistic function of the top cue's validity
        p_ttb_use = 1.0 / (1.0 + np.exp(-tau * (v_top - theta)))
        
        # Take-The-Best (TTB) prediction
        winner_ttb = 0 if a[top_cue] > b[top_cue] else 1
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
        # Weighted Additive (WADD) prediction
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
        scores_wadd = np.array([score_a, score_b])
        z_wadd = beta * (scores_wadd - np.max(scores_wadd))
        e_wadd = np.exp(z_wadd)
        p_wadd = e_wadd / np.sum(e_wadd)
        
        # Mix the two strategies based on the top cue validity
        p_mix = p_ttb_use * p_ttb + (1.0 - p_ttb_use) * p_wadd
        
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [0.0, 1.0]
- tau: [1.0, 100.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7581 (var=0.0332) vs this=0.7447 (var=0.0432)
- Experiment 2: real=0.2506 (var=0.0294) vs this=0.2217 (var=0.0293)
- Experiment 3: real=0.7236 (var=0.0302) vs this=0.8004 (var=0.0164)
- Experiment 4: real=0.3975 (var=0.0240) vs this=0.2433 (var=0.0289)
- Experiment 5: real=0.0574 (var=0.0093) vs this=0.0087 (var=0.0002)
- Experiment 6: real=0.0175 (var=0.0095) vs this=0.0012 (var=0.0149)
- Experiment 7: real=0.6875 (var=0.0691) vs this=0.0038 (var=0.0493)
- Experiment 8: real=0.0475 (var=0.1792) vs this=0.0092 (var=0.0047)
- Experiment 9: real=-0.7263 (var=0.0321) vs this=0.0568 (var=0.0274)
- Experiment 10: real=-0.0253 (var=0.2181) vs this=-0.1137 (var=0.0439)
- Experiment 11: real=0.0725 (var=0.0566) vs this=-0.0225 (var=0.0376)
- Experiment 12: real=-0.1700 (var=0.2061) vs this=-0.1537 (var=0.0647)
- Experiment 13: real=-0.6683 (var=0.0841) vs this=0.0508 (var=0.0190)
- Experiment 14: real=0.0100 (var=0.0081) vs this=-0.0025 (var=0.0118)
- Experiment 15: real=-0.0400 (var=0.0118) vs this=-0.0225 (var=0.0153)
- Experiment 16: real=0.0167 (var=0.0107) vs this=0.0250 (var=0.0093)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0037 (var=0.0008)
- Experiment 18: real=0.5825 (var=0.0907) vs this=0.8025 (var=0.0289)


---

### `pi_8` (overall score: 0.523)

**Description**
Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7581 (var=0.0332) vs this=0.7484 (var=0.0105)
- Experiment 2: real=0.2506 (var=0.0294) vs this=0.3008 (var=0.0138)
- Experiment 3: real=0.7236 (var=0.0302) vs this=0.7111 (var=0.0112)
- Experiment 4: real=0.3975 (var=0.0240) vs this=0.3540 (var=0.0095)
- Experiment 5: real=0.0574 (var=0.0093) vs this=0.0296 (var=0.0012)
- Experiment 6: real=0.0175 (var=0.0095) vs this=-0.0200 (var=0.0366)
- Experiment 7: real=0.6875 (var=0.0691) vs this=0.3225 (var=0.0374)
- Experiment 8: real=0.0475 (var=0.1792) vs this=-0.0079 (var=0.0131)
- Experiment 9: real=-0.7263 (var=0.0321) vs this=-0.2421 (var=0.0811)
- Experiment 10: real=-0.0253 (var=0.2181) vs this=-0.1021 (var=0.0423)
- Experiment 11: real=0.0725 (var=0.0566) vs this=0.1388 (var=0.1325)
- Experiment 12: real=-0.1700 (var=0.2061) vs this=-0.0813 (var=0.0382)
- Experiment 13: real=-0.6683 (var=0.0841) vs this=-0.6167 (var=0.0803)
- Experiment 14: real=0.0100 (var=0.0081) vs this=-0.3458 (var=0.0815)
- Experiment 15: real=-0.0400 (var=0.0118) vs this=0.3963 (var=0.0946)
- Experiment 16: real=0.0167 (var=0.0107) vs this=-0.0100 (var=0.0446)
- Experiment 17: real=0.0000 (var=0.0000) vs this=-0.0108 (var=0.0005)
- Experiment 18: real=0.5825 (var=0.0907) vs this=0.7937 (var=0.0170)


---

### `pi_4` (overall score: 0.489)

**Description**
Strategy Mixture Theory: Decision-makers do not uniformly rely on a single decision strategy. Instead, they exhibit variability in strategy selection, probabilistically mixing between a non-compensatory heuristic (Take The Best) and a compensatory strategy (Weighted Additive, WADD). The mixture parameter 'alpha' dictates the probability of using TTB on any given trial, while '1 - alpha' is the probability of using WADD. This allows the model to capture intermediate choice proportions that neither pure strategy can explain on its own, with a bias toward the empirically dominant non-compensatory strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    beta = float(parameters["beta"])
    
    # Take The Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # WADD prediction
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores_wadd = np.array([score_a, score_b])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Strategy Mixture
    alpha = float(parameters["alpha"])
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7581 (var=0.0332) vs this=0.6731 (var=0.0154)
- Experiment 2: real=0.2506 (var=0.0294) vs this=0.2844 (var=0.0129)
- Experiment 3: real=0.7236 (var=0.0302) vs this=0.6902 (var=0.0128)
- Experiment 4: real=0.3975 (var=0.0240) vs this=0.2710 (var=0.0132)
- Experiment 5: real=0.0574 (var=0.0093) vs this=0.0096 (var=0.0001)
- Experiment 6: real=0.0175 (var=0.0095) vs this=-0.0338 (var=0.0268)
- Experiment 7: real=0.6875 (var=0.0691) vs this=-0.1412 (var=0.0218)
- Experiment 8: real=0.0475 (var=0.1792) vs this=0.0071 (var=0.0095)
- Experiment 9: real=-0.7263 (var=0.0321) vs this=0.1189 (var=0.0240)
- Experiment 10: real=-0.0253 (var=0.2181) vs this=0.0084 (var=0.0218)
- Experiment 11: real=0.0725 (var=0.0566) vs this=-0.0675 (var=0.0434)
- Experiment 12: real=-0.1700 (var=0.2061) vs this=-0.0063 (var=0.0216)
- Experiment 13: real=-0.6683 (var=0.0841) vs this=0.1417 (var=0.0210)
- Experiment 14: real=0.0100 (var=0.0081) vs this=0.0592 (var=0.0170)
- Experiment 15: real=-0.0400 (var=0.0118) vs this=-0.0737 (var=0.0145)
- Experiment 16: real=0.0167 (var=0.0107) vs this=0.0058 (var=0.0078)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0040 (var=0.0009)
- Experiment 18: real=0.5825 (var=0.0907) vs this=0.6919 (var=0.0231)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3635 -> ACCEPTED
- iter 2: loss=0.2880 -> ACCEPTED
- iter 3: loss=0.3662 -> REJECTED
- iter 4: loss=0.2927 -> REJECTED
- iter 5: loss=0.3072 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.2880 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1
            
        # Only consider trials where TTB and Tallying make opposite predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            disagree_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if disagree_count == 0:
        return 0.5
    return ttb_match_count / disagree_count

```

**Observed (real) value:** 0.7581 (var=0.0332)
**Previous candidate values (this loop):**
  - iter 1: 0.5284 (var=0.0520) (Δ vs real -0.2297)
  - iter 2: 0.7706 (var=0.0122) (Δ vs real +0.0125)
  - iter 3: 0.8634 (var=0.0061) (Δ vs real +0.1053)
  - iter 4: 0.8006 (var=0.0127) (Δ vs real +0.0425)
  - iter 5 (most recent): 0.7475 (var=0.0108) (Δ vs real -0.0106)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8391 (var=0.0094)
- pi_2: 0.1216 (var=0.0073)
- pi_3: 0.6753 (var=0.0250)
- pi_4: 0.6731 (var=0.0154)
- pi_5: 0.6447 (var=0.1186)
- pi_6: 0.7447 (var=0.0432)
- pi_7: 0.6416 (var=0.0448)
- pi_8: 0.7484 (var=0.0105)
- pi_9: 0.6731 (var=0.0252)
- pi_10: 0.7184 (var=0.0134)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pref)
        
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2506 (var=0.0294)
**Previous candidate values (this loop):**
  - iter 1: 0.4714 (var=0.0439) (Δ vs real +0.2208)
  - iter 2: 0.2322 (var=0.0102) (Δ vs real -0.0183)
  - iter 3: 0.1892 (var=0.0071) (Δ vs real -0.0614)
  - iter 4: 0.2628 (var=0.0134) (Δ vs real +0.0122)
  - iter 5 (most recent): 0.2672 (var=0.0093) (Δ vs real +0.0167)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8581 (var=0.0074)
- pi_1: 0.1211 (var=0.0091)
- pi_3: 0.2033 (var=0.0222)
- pi_4: 0.2844 (var=0.0129)
- pi_5: 0.2444 (var=0.0869)
- pi_6: 0.2217 (var=0.0293)
- pi_7: 0.3081 (var=0.0448)
- pi_8: 0.3008 (var=0.0138)
- pi_9: 0.2414 (var=0.0331)
- pi_10: 0.2583 (var=0.0162)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    # TTB prediction: heavily weights the first discriminating cue
    # Using powers of 10 ensures strict lexicographical ordering (TTB logic)
    weights = np.array([1000, 100, 10, 1])
    ttb_score = np.dot(diff, weights)
    ttb_pred = np.where(ttb_score > 0, 0, 1)
    
    # WADD prediction (with gamma=1, i.e., linear integration)
    wadd_score = np.dot(diff, val)
    wadd_pred = np.where(wadd_score > 0, 0, 1)
    
    # Identify trials where TTB and baseline WADD disagree
    divergent = ttb_pred != wadd_pred
    
    if not np.any(divergent):
        return 0.5
        
    responses = data['response'].values
    # Calculate proportion of choices matching TTB on these critical trials
    ttb_match = (responses[divergent] == ttb_pred[divergent]).mean()
    
    return float(ttb_match)
```

**Observed (real) value:** 0.7236 (var=0.0302)
**Previous candidate values (this loop):**
  - iter 1: 0.5711 (var=0.0459) (Δ vs real -0.1524)
  - iter 2: 0.8302 (var=0.0086) (Δ vs real +0.1067)
  - iter 3: 0.8689 (var=0.0064) (Δ vs real +0.1453)
  - iter 4: 0.7924 (var=0.0101) (Δ vs real +0.0689)
  - iter 5 (most recent): 0.7782 (var=0.0108) (Δ vs real +0.0547)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8418 (var=0.0111)
- pi_3: 0.7036 (var=0.0145)
- pi_2: 0.1582 (var=0.0129)
- pi_4: 0.6902 (var=0.0128)
- pi_5: 0.7213 (var=0.1024)
- pi_6: 0.8004 (var=0.0164)
- pi_7: 0.7058 (var=0.0359)
- pi_8: 0.7111 (var=0.0112)
- pi_9: 0.6898 (var=0.0209)
- pi_10: 0.7000 (var=0.0080)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.3975 (var=0.0240)
**Previous candidate values (this loop):**
  - iter 1: 0.4779 (var=0.0336) (Δ vs real +0.0804)
  - iter 2: 0.2515 (var=0.0100) (Δ vs real -0.1460)
  - iter 3: 0.1558 (var=0.0060) (Δ vs real -0.2417)
  - iter 4: 0.2075 (var=0.0087) (Δ vs real -0.1900)
  - iter 5 (most recent): 0.2854 (var=0.0060) (Δ vs real -0.1121)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3127 (var=0.0167)
- pi_1: 0.1787 (var=0.0150)
- pi_2: 0.8179 (var=0.0096)
- pi_4: 0.2710 (var=0.0132)
- pi_5: 0.3623 (var=0.0867)
- pi_6: 0.2433 (var=0.0289)
- pi_7: 0.3588 (var=0.0302)
- pi_8: 0.3540 (var=0.0095)
- pi_9: 0.3092 (var=0.0287)
- pi_10: 0.3696 (var=0.0059)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trials 1, 2, and 3 are conflict trials where TTB prefers A and WADD prefers B.
    target_trials = ['10000', '01000', '10100']
    df_conflict = data[data['a_str'].isin(target_trials)].copy()
    df_conflict['chose_A'] = (df_conflict['response'] == 0).astype(float)
    
    subj_vars = []
    for subj, subj_df in df_conflict.groupby('subject_id'):
        means = subj_df.groupby('a_str')['chose_A'].mean()
        if len(means) == 3:
            subj_vars.append(means.var(ddof=1))
            
    if not subj_vars:
        return 0.0
        
    return float(np.mean(subj_vars))
```

**Observed (real) value:** 0.0574 (var=0.0093)
**Previous candidate values (this loop):**
  - iter 1: 0.0124 (var=0.0002) (Δ vs real -0.0450)
  - iter 2: 0.0073 (var=0.0001) (Δ vs real -0.0501)
  - iter 3: 0.0047 (var=0.0000) (Δ vs real -0.0527)
  - iter 4: 0.0088 (var=0.0001) (Δ vs real -0.0485)
  - iter 5 (most recent): 0.0077 (var=0.0001) (Δ vs real -0.0497)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0096 (var=0.0001)
- pi_3: 0.0331 (var=0.0015)
- pi_1: 0.0040 (var=0.0000)
- pi_2: 0.0061 (var=0.0000)
- pi_5: 0.0237 (var=0.0046)
- pi_6: 0.0087 (var=0.0002)
- pi_7: 0.0177 (var=0.0003)
- pi_8: 0.0296 (var=0.0012)
- pi_9: 0.0159 (var=0.0004)
- pi_10: 0.0276 (var=0.0005)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    data['chose_A'] = 1 - data['response']
    
    t2_mask = (data['A_str'] == '10001') & (data['B_str'] == '01100')
    t5_mask = (data['A_str'] == '01001') & (data['B_str'] == '00110')
    
    p_a_t2 = data[t2_mask]['chose_A'].mean() if t2_mask.sum() > 0 else 0.5
    p_a_t5 = data[t5_mask]['chose_A'].mean() if t5_mask.sum() > 0 else 0.5
    
    return float(p_a_t2 - p_a_t5)
```

**Observed (real) value:** 0.0175 (var=0.0095)
**Previous candidate values (this loop):**
  - iter 1: -0.0325 (var=0.0225) (Δ vs real -0.0500)
  - iter 2: -0.0687 (var=0.0252) (Δ vs real -0.0862)
  - iter 3: -0.1587 (var=0.0402) (Δ vs real -0.1762)
  - iter 4: -0.0437 (var=0.0416) (Δ vs real -0.0612)
  - iter 5 (most recent): -0.0463 (var=0.0293) (Δ vs real -0.0638)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1713 (var=0.0375)
- pi_4: -0.0338 (var=0.0268)
- pi_1: 0.0162 (var=0.0131)
- pi_2: 0.0150 (var=0.0245)
- pi_5: -0.0250 (var=0.0333)
- pi_6: 0.0012 (var=0.0149)
- pi_7: 0.0725 (var=0.0232)
- pi_8: -0.0200 (var=0.0366)
- pi_9: 0.0463 (var=0.0367)
- pi_10: -0.0350 (var=0.0336)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_seq = data['option_a_ratings'].apply(tuple)
    b_seq = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 0, 0, 0, 0)
    t1_b = (0, 1, 1, 1, 1)
    
    t4_a = (1, 1, 1, 1, 0)
    t4_b = (1, 1, 1, 0, 1)
    
    t1_mask1 = (a_seq == t1_a) & (b_seq == t1_b)
    t1_mask2 = (a_seq == t1_b) & (b_seq == t1_a)
    
    t4_mask1 = (a_seq == t4_a) & (b_seq == t4_b)
    t4_mask2 = (a_seq == t4_b) & (b_seq == t4_a)
    
    p_t1 = 0.0
    n_t1 = t1_mask1.sum() + t1_mask2.sum()
    if n_t1 > 0:
        chose_a_t1 = (t1_mask1 & (data['response'] == 0)) | (t1_mask2 & (data['response'] == 1))
        p_t1 = chose_a_t1.sum() / n_t1
        
    p_t4 = 0.0
    n_t4 = t4_mask1.sum() + t4_mask2.sum()
    if n_t4 > 0:
        chose_a_t4 = (t4_mask1 & (data['response'] == 0)) | (t4_mask2 & (data['response'] == 1))
        p_t4 = chose_a_t4.sum() / n_t4
        
    return float(p_t1 - p_t4)
```

**Observed (real) value:** 0.6875 (var=0.0691)
**Previous candidate values (this loop):**
  - iter 1: -0.2113 (var=0.0408) (Δ vs real -0.8988)
  - iter 2: 0.1150 (var=0.0297) (Δ vs real -0.5725)
  - iter 3: 0.0112 (var=0.0223) (Δ vs real -0.6763)
  - iter 4: 0.1075 (var=0.0289) (Δ vs real -0.5800)
  - iter 5 (most recent): 0.1900 (var=0.0226) (Δ vs real -0.4975)
**Other theories' values on this metric (for reference):**
- pi_5: -0.1338 (var=0.1275)
- pi_3: 0.2250 (var=0.0842)
- pi_1: -0.0088 (var=0.0205)
- pi_2: -0.3850 (var=0.0235)
- pi_4: -0.1412 (var=0.0218)
- pi_6: 0.0038 (var=0.0493)
- pi_7: 0.0450 (var=0.1074)
- pi_8: 0.3225 (var=0.0374)
- pi_9: 0.2762 (var=0.1014)
- pi_10: 0.3413 (var=0.0397)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data['trial_str'] = data.apply(lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), axis=1)
    
    t1 = '110000_001111'
    t2 = '001111_110000'
    t3 = '101000_010111'
    t4 = '010111_101000'
    
    data['chose_a'] = (data['response'] == 0).astype(float)
    subj_means = data.groupby(['subject_id', 'trial_str'])['chose_a'].mean().unstack()
    
    # Preference for the option with the top cue in Trial 1/2 vs Trial 3/4
    if t1 in subj_means.columns and t2 in subj_means.columns:
        pref_1 = (subj_means[t1] + (1.0 - subj_means[t2])) / 2.0
    else:
        return 0.0
        
    if t3 in subj_means.columns and t4 in subj_means.columns:
        pref_3 = (subj_means[t3] + (1.0 - subj_means[t4])) / 2.0
    else:
        return 0.0
        
    diff = pref_1 - pref_3
    
    if isinstance(diff, pd.Series):
        return float(diff.mean())
    return float(diff)
```

**Observed (real) value:** 0.0475 (var=0.1792)
**Previous candidate values (this loop):**
  - iter 1: 0.0075 (var=0.0092) (Δ vs real -0.0400)
  - iter 2: 0.0137 (var=0.0053) (Δ vs real -0.0338)
  - iter 3: -0.0021 (var=0.0048) (Δ vs real -0.0496)
  - iter 4: -0.0054 (var=0.0079) (Δ vs real -0.0529)
  - iter 5 (most recent): 0.0250 (var=0.0069) (Δ vs real -0.0225)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0346 (var=0.0103)
- pi_5: 0.3017 (var=0.1821)
- pi_1: -0.0063 (var=0.0057)
- pi_2: 0.0025 (var=0.0051)
- pi_4: 0.0071 (var=0.0095)
- pi_6: 0.0092 (var=0.0047)
- pi_7: 0.0938 (var=0.0070)
- pi_8: -0.0079 (var=0.0131)
- pi_9: 0.0721 (var=0.0127)
- pi_10: 0.0067 (var=0.0102)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    def get_pA(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        chose_A = 0
        total = 0
        if mask1.sum() > 0:
            chose_A += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            chose_A += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
        return chose_A / total if total > 0 else 0.5

    # Trial 1: Top cue (0.90) vs single opposing cue (0.85)
    pA_t1 = get_pA("10000", "01000")
    
    # Trial 4: Top cue (0.90) vs coalition of 4 opposing cues (0.85 + 0.80 + 0.75 + 0.70)
    pA_t4 = get_pA("10000", "01111")
    
    return float(pA_t1 - pA_t4)

```

**Observed (real) value:** -0.7263 (var=0.0321)
**Previous candidate values (this loop):**
  - iter 1: 0.1558 (var=0.0386) (Δ vs real +0.8821)
  - iter 2: -0.0768 (var=0.0175) (Δ vs real +0.6495)
  - iter 3: -0.0074 (var=0.0114) (Δ vs real +0.7189)
  - iter 4: -0.0653 (var=0.0169) (Δ vs real +0.6611)
  - iter 5 (most recent): -0.0811 (var=0.0225) (Δ vs real +0.6453)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0568 (var=0.0274)
- pi_3: 0.2379 (var=0.0708)
- pi_1: 0.0095 (var=0.0097)
- pi_2: 0.3484 (var=0.0173)
- pi_4: 0.1189 (var=0.0240)
- pi_5: 0.1653 (var=0.0859)
- pi_7: 0.2232 (var=0.0400)
- pi_8: -0.2421 (var=0.0811)
- pi_9: -0.0516 (var=0.0436)
- pi_10: -0.0526 (var=0.0709)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 3 based on option A's ratings
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 0))
    
    if t1_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    # response is 0 for A, 1 for B. So mean() is the proportion of choosing B.
    # We want the difference in proportion of choosing A: P(A | Trial 3) - P(A | Trial 1)
    # Which is (1 - m3) - (1 - m1) = m1 - m3
    m1 = data[t1_mask]['response'].mean()
    m3 = data[t3_mask]['response'].mean()
    
    return float(m1 - m3)
```

**Observed (real) value:** -0.0253 (var=0.2181)
**Previous candidate values (this loop):**
  - iter 1: -0.0011 (var=0.0222) (Δ vs real +0.0242)
  - iter 2: -0.0537 (var=0.0193) (Δ vs real -0.0284)
  - iter 3: -0.0095 (var=0.0110) (Δ vs real +0.0158)
  - iter 4: -0.0621 (var=0.0197) (Δ vs real -0.0368)
  - iter 5 (most recent): -0.0505 (var=0.0155) (Δ vs real -0.0253)
**Other theories' values on this metric (for reference):**
- pi_3: -0.3021 (var=0.0367)
- pi_6: -0.1137 (var=0.0439)
- pi_1: -0.0158 (var=0.0083)
- pi_2: 0.0211 (var=0.0081)
- pi_4: 0.0084 (var=0.0218)
- pi_5: -0.1484 (var=0.0951)
- pi_7: -0.0768 (var=0.0133)
- pi_8: -0.1021 (var=0.0423)
- pi_9: -0.1421 (var=0.0301)
- pi_10: -0.1084 (var=0.0352)

### Experiment 11
**Design**
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    
    # response == 0 means Option A was chosen, so 1 - mean(response) is P(A)
    p_A = 1.0 - data.groupby('a_key')['response'].mean()
    
    # Trials without the shared top cue
    t1 = p_A.get((0, 1, 1, 0, 0), 0.5)
    t3 = p_A.get((0, 1, 0, 0, 0), 0.5)
    t5 = p_A.get((0, 1, 1, 1, 0), 0.5)
    
    # Trials with the shared top cue (Cue 0 = 1 for both options)
    t2 = p_A.get((1, 1, 1, 0, 0), 0.5)
    t4 = p_A.get((1, 1, 0, 0, 0), 0.5)
    t6 = p_A.get((1, 1, 1, 1, 0), 0.5)
    
    # Calculate the regression to chance (0.5) caused by the shared top cue
    diff1 = abs(t1 - 0.5) - abs(t2 - 0.5)
    diff2 = abs(t3 - 0.5) - abs(t4 - 0.5)
    diff3 = abs(t5 - 0.5) - abs(t6 - 0.5)
    
    return float(diff1 + diff2 + diff3)
```

**Observed (real) value:** 0.0725 (var=0.0566)
**Previous candidate values (this loop):**
  - iter 1: -0.0300 (var=0.0427) (Δ vs real -0.1025)
  - iter 2: 0.1600 (var=0.0489) (Δ vs real +0.0875)
  - iter 3: 0.0500 (var=0.0347) (Δ vs real -0.0225)
  - iter 4: 0.0637 (var=0.0790) (Δ vs real -0.0088)
  - iter 5 (most recent): 0.2150 (var=0.0586) (Δ vs real +0.1425)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1062 (var=0.0608)
- pi_3: -0.0700 (var=0.0394)
- pi_1: 0.0025 (var=0.0245)
- pi_2: 0.0100 (var=0.0298)
- pi_4: -0.0675 (var=0.0434)
- pi_5: 0.0125 (var=0.0183)
- pi_6: -0.0225 (var=0.0376)
- pi_8: 0.1388 (var=0.1325)
- pi_9: 0.4700 (var=0.1624)
- pi_10: 0.1950 (var=0.1038)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_t6 = (data['A_str'] == '00100') & (data['B_str'] == '00011')
    mask_t1 = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    
    p_a_t6 = (data[mask_t6]['response'] == 0).mean() if mask_t6.sum() > 0 else 0.5
    p_a_t1 = (data[mask_t1]['response'] == 0).mean() if mask_t1.sum() > 0 else 0.5
    
    return float(p_a_t6 - p_a_t1)
```

**Observed (real) value:** -0.1700 (var=0.2061)
**Previous candidate values (this loop):**
  - iter 1: 0.0100 (var=0.0383) (Δ vs real +0.1800)
  - iter 2: -0.0288 (var=0.0174) (Δ vs real +0.1412)
  - iter 3: -0.0187 (var=0.0210) (Δ vs real +0.1512)
  - iter 4: -0.0075 (var=0.0226) (Δ vs real +0.1625)
  - iter 5 (most recent): -0.0475 (var=0.0192) (Δ vs real +0.1225)
**Other theories' values on this metric (for reference):**
- pi_3: -0.1988 (var=0.0731)
- pi_7: 0.0025 (var=0.0330)
- pi_1: 0.0100 (var=0.0154)
- pi_2: 0.0400 (var=0.0159)
- pi_4: -0.0063 (var=0.0216)
- pi_5: -0.1063 (var=0.1088)
- pi_6: -0.1537 (var=0.0647)
- pi_8: -0.0813 (var=0.0382)
- pi_9: -0.0475 (var=0.0377)
- pi_10: -0.0613 (var=0.0196)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    is_t2 = (a_sums == 4) & (b_sums == 1)
    is_t3 = (a_sums == 1) & (b_sums == 4)
    
    # Probability of choosing A
    p_a_t2 = 1.0 - data.loc[is_t2, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[is_t3, 'response'].mean()
    
    return float(p_a_t2 - p_a_t3)
```

**Observed (real) value:** -0.6683 (var=0.0841)
**Previous candidate values (this loop):**
  - iter 1: 0.4275 (var=0.0646) (Δ vs real +1.0958)
  - iter 2: -0.2992 (var=0.0590) (Δ vs real +0.3692)
  - iter 3: -0.4150 (var=0.0578) (Δ vs real +0.2533)
  - iter 4: -0.2375 (var=0.0504) (Δ vs real +0.4308)
  - iter 5 (most recent): -0.4483 (var=0.0443) (Δ vs real +0.2200)
**Other theories' values on this metric (for reference):**
- pi_8: -0.6167 (var=0.0803)
- pi_3: 0.1117 (var=0.0646)
- pi_1: -0.0108 (var=0.0103)
- pi_2: 0.7508 (var=0.0378)
- pi_4: 0.1417 (var=0.0210)
- pi_5: 0.2633 (var=0.1674)
- pi_6: 0.0508 (var=0.0190)
- pi_7: 0.0867 (var=0.0485)
- pi_9: -0.0517 (var=0.0619)
- pi_10: -0.5883 (var=0.0700)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 0, 0, 0))
    t2_mask = (a_keys == (1, 0, 1, 1, 1)) & (b_keys == (0, 1, 0, 0, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t1)
```

**Observed (real) value:** 0.0100 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.1942 (var=0.0243) (Δ vs real +0.1842)
  - iter 2: -0.1525 (var=0.0221) (Δ vs real -0.1625)
  - iter 3: -0.3600 (var=0.0622) (Δ vs real -0.3700)
  - iter 4: -0.2333 (var=0.0321) (Δ vs real -0.2433)
  - iter 5 (most recent): -0.2667 (var=0.0487) (Δ vs real -0.2767)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0333 (var=0.0132)
- pi_8: -0.3458 (var=0.0815)
- pi_1: 0.0158 (var=0.0097)
- pi_2: 0.3500 (var=0.0331)
- pi_4: 0.0592 (var=0.0170)
- pi_5: 0.0817 (var=0.0290)
- pi_6: -0.0025 (var=0.0118)
- pi_7: 0.0267 (var=0.0263)
- pi_9: -0.0725 (var=0.0503)
- pi_10: -0.5333 (var=0.0687)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract A's features as strings for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A = [1, 0, 0, 0, 0]
    # Trial 2: A = [1, 0, 1, 1, 1]
    # Response is 0 if A, 1 if B. We want P(Choose A), which is 1.0 - response.mean()
    
    t1_data = data[a_str == '10000']
    t2_data = data[a_str == '10111']
    
    if len(t1_data) == 0 or len(t2_data) == 0:
        return 0.0
        
    prob_a_t1 = 1.0 - t1_data['response'].mean()
    prob_a_t2 = 1.0 - t2_data['response'].mean()
    
    return float(prob_a_t1 - prob_a_t2)
```

**Observed (real) value:** -0.0400 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1: -0.1862 (var=0.0305) (Δ vs real -0.1462)
  - iter 2: 0.1675 (var=0.0471) (Δ vs real +0.2075)
  - iter 3: 0.3150 (var=0.0670) (Δ vs real +0.3550)
  - iter 4: 0.1425 (var=0.0414) (Δ vs real +0.1825)
  - iter 5 (most recent): 0.3575 (var=0.0664) (Δ vs real +0.3975)
**Other theories' values on this metric (for reference):**
- pi_8: 0.3963 (var=0.0946)
- pi_9: 0.0625 (var=0.0509)
- pi_1: 0.0113 (var=0.0146)
- pi_2: -0.3688 (var=0.0227)
- pi_3: -0.0138 (var=0.0240)
- pi_4: -0.0737 (var=0.0145)
- pi_5: -0.0637 (var=0.0279)
- pi_6: -0.0225 (var=0.0153)
- pi_7: -0.0550 (var=0.0239)
- pi_10: 0.5263 (var=0.0914)

### Experiment 16
**Design**
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_3_mask = (data['A_str'] == '10011') & (data['B_str'] == '01000')
    trial_1_mask = (data['A_str'] == '00011') & (data['B_str'] == '00010')
    
    p_a_3 = 1.0 - data.loc[trial_3_mask, 'response'].mean() if trial_3_mask.sum() > 0 else 0.5
    p_a_1 = 1.0 - data.loc[trial_1_mask, 'response'].mean() if trial_1_mask.sum() > 0 else 0.5
    
    return float(p_a_3 - p_a_1)
```

**Observed (real) value:** 0.0167 (var=0.0107)
**Previous candidate values (this loop):**
  - iter 1: 0.0267 (var=0.0080) (Δ vs real +0.0100)
  - iter 2: -0.1000 (var=0.0188) (Δ vs real -0.1167)
  - iter 3: -0.0217 (var=0.0168) (Δ vs real -0.0383)
  - iter 4: -0.0625 (var=0.0170) (Δ vs real -0.0792)
  - iter 5 (most recent): -0.1475 (var=0.0193) (Δ vs real -0.1642)
**Other theories' values on this metric (for reference):**
- pi_9: 0.2992 (var=0.0728)
- pi_8: -0.0100 (var=0.0446)
- pi_1: -0.0133 (var=0.0114)
- pi_2: 0.0067 (var=0.0110)
- pi_3: 0.2992 (var=0.0266)
- pi_4: 0.0058 (var=0.0078)
- pi_5: 0.0283 (var=0.0087)
- pi_6: 0.0250 (var=0.0093)
- pi_7: 0.2233 (var=0.0364)
- pi_10: -0.1583 (var=0.0413)

### Experiment 17
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['a_str'] == '10100') & (data['b_str'] == '01010')
    t2_mask = (data['a_str'] == '10001') & (data['b_str'] == '01010')
    
    def subj_metric(df):
        t1_resp = df.loc[t1_mask, 'response']
        t2_resp = df.loc[t2_mask, 'response']
        if len(t1_resp) == 0 or len(t2_resp) == 0:
            return 0.0
            
        p_a_t1 = (t1_resp == 0).mean()
        p_a_t2 = (t2_resp == 0).mean()
        
        # Weight by the subject's signal-to-noise ratio measured via T2.
        weight = max(0.0, p_a_t2 - 0.5)
        return float(weight * (p_a_t1 - p_a_t2))
        
    return float(data.groupby('subject_id').apply(subj_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: -0.0076 (var=0.0005) (Δ vs real -0.0076)
  - iter 2: 0.0009 (var=0.0011) (Δ vs real +0.0009)
  - iter 3: 0.0179 (var=0.0005) (Δ vs real +0.0179)
  - iter 4: 0.0033 (var=0.0005) (Δ vs real +0.0033)
  - iter 5 (most recent): 0.0074 (var=0.0003) (Δ vs real +0.0074)
**Other theories' values on this metric (for reference):**
- pi_8: -0.0108 (var=0.0005)
- pi_10: -0.0003 (var=0.0002)
- pi_1: -0.0052 (var=0.0009)
- pi_2: -0.0029 (var=0.0001)
- pi_3: -0.0008 (var=0.0013)
- pi_4: 0.0040 (var=0.0009)
- pi_5: 0.0017 (var=0.0006)
- pi_6: 0.0037 (var=0.0008)
- pi_7: -0.0062 (var=0.0005)
- pi_9: -0.0099 (var=0.0012)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    
    # Trial 2: A=[0, 1, 0, 0, 0] (sum=0.9), B=[0, 0, 1, 1, 0] (sum=1.0)
    t2 = (a_str == '01000') & (b_str == '00110')
    # Trial 6: A=[0, 1, 0, 0, 0] (sum=0.9), B=[0, 0, 1, 1, 1] (sum=1.5)
    t6 = (a_str == '01000') & (b_str == '00111')
    
    target_trials = t2 | t6
    if not target_trials.any():
        return 0.5
        
    return float((data.loc[target_trials, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5825 (var=0.0907)
**Previous candidate values (this loop):**
  - iter 1: 0.5806 (var=0.0453) (Δ vs real -0.0019)
  - iter 2: 0.8488 (var=0.0123) (Δ vs real +0.2662)
  - iter 3: 0.8656 (var=0.0066) (Δ vs real +0.2831)
  - iter 4: 0.7775 (var=0.0121) (Δ vs real +0.1950)
  - iter 5 (most recent): 0.8206 (var=0.0126) (Δ vs real +0.2381)
**Other theories' values on this metric (for reference):**
- pi_10: 0.8144 (var=0.0167)
- pi_8: 0.7937 (var=0.0170)
- pi_1: 0.8306 (var=0.0102)
- pi_2: 0.1400 (var=0.0094)
- pi_3: 0.6894 (var=0.0189)
- pi_4: 0.6919 (var=0.0231)
- pi_5: 0.7625 (var=0.1062)
- pi_6: 0.8025 (var=0.0289)
- pi_7: 0.8063 (var=0.0314)
- pi_9: 0.8325 (var=0.0159)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Heuristic Switching Theory with Averaging: Decision-makers probabilistically alternate between a non-compensatory strategy (Take-The-Best) and a compensatory Averaging heuristic. Take-The-Best relies entirely on the most valid discriminating cue, while Averaging computes the mean validity of all active features for each option. This mixture avoids a global dilution penalty but naturally penalizes options burdened with many weak cues during the compensatory phase, capturing empirical dilution effects.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    alpha = float(parameters["alpha"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_avg = float(parameters["beta_avg"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Averaging prediction (mean of validities of active features)
    score_a = np.mean(val[a > 0.5]) if np.sum(a > 0.5) > 0 else 0.0
    score_b = np.mean(val[b > 0.5]) if np.sum(b > 0.5) > 0 else 0.0
    scores_avg = np.array([score_a, score_b])
    
    z_avg = beta_avg * (scores_avg - np.max(scores_avg))
    e_avg = np.exp(z_avg)
    p_avg = e_avg / np.sum(e_avg)
    
    # Strategy Mixture
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_avg
    
    # Apply lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- alpha: [0.0, 1.0]
- beta_ttb: [0.1, 20.0]
- beta_avg: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the critic's advice, we replace the Tallying component with an Averaging heuristic. The model now probabilistically mixes between Take-The-Best and Averaging (where the value of an option is the mean validity of its active cues). This captures the dilution effect observed in Exp 9 and 13 (where adding weak cues to an option actively hurts its overall subjective value) without resorting to a global penalty that spans across all strategies.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory implements a heuristic switching mechanism between Take-The-Best (TTB) and Tallying, successfully adhering to the arbiter's prescribed family. However, this pure additive mixture fails fundamentally on Experiments 9 and 13. In Exp 9, adding weak opposing cues to an option (Trial 4) actually makes that option *less* preferred compared to a single opposing cue (Trial 1), resulting in a heavily negative metric (-0.7263). The candidate predicts the opposite (0.1558) because Tallying strictly adds value for each active cue. Exp 13 shows a similarly massive negative value (-0.6683), confirming that adding weak cues to an option actively hurts its subjective value (a dilution effect).
Rationale: To capture the strong empirical dilution effects in Exp 9 and 13 while remaining strictly within the arbiter's prescribed mechanism family (heuristic switching or rank-based weighting, avoiding global dilution), you should modify the compensatory component of the model. Instead of 'Tallying' (an unweighted sum), replace it with an 'Averaging' heuristic. In this Averaging component, the score for an option should be the mean of the validities (or mean of the rank-based weights) of its active features. Probabilistically switching between TTB and Averaging will naturally penalize options burdened with many weak cues (as the weak cues drag down the average), correctly predicting the negative metrics in Exp 9 and 13 without resorting to a global penalty.

**Outcome of this advice:** iter 1 candidate loss=0.3635 -> iter 2 candidate loss=0.2880 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The transition to an Averaging heuristic successfully captured the qualitative dilution effect observed in Experiments 9 and 13, flipping the predicted metric from positive to negative. This confirms that mixing Take-The-Best with an Averaging compensatory strategy is the right structural mechanism. However, the magnitude of the dilution effect is still significantly under-predicted: Experiment 9 shows a real value of -0.7263 but the candidate predicts -0.0768, and Experiment 13 shows a real value of -0.6683 versus the candidate's -0.2992. Additionally, Experiment 7 is heavily under-predicted (0.6875 vs 0.1150). The model needs to be more sensitive to the Averaging differences.
Rationale: To capture the massive magnitude of the dilution effect in Experiments 9 and 13 without abandoning the successful TTB + Averaging mixture, make minor adjustments to the model's sensitivity. Specifically, widen the parameter ranges for the inverse temperatures (`beta_ttb` and `beta_avg`) up to 50.0 or 100.0 to allow the model to produce more deterministic choices when Averaging leans strongly in one direction. Additionally, ensure the scores fed into the Averaging softmax are properly scaled, or consider having the Averaging component compute the mean of the *ranks* of active features rather than raw validities, which might naturally amplify the penalty of adding weak cues.

**Outcome of this advice:** iter 2 candidate loss=0.2880 -> iter 3 candidate loss=0.3662 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate (iter 3) was rejected by the gate. While the transition to rank-averaging was intended to amplify the dilution effect, widening the inverse temperature ranges up to 100.0 caused catastrophic softmax saturation. In Experiment 9, the probabilities of choosing Option A in both Trial 1 and Trial 4 were pushed to 1.0, which wiped out the difference between them and resulted in a metric near 0.0 (instead of the empirically observed -0.7263). The model lost its ability to express the magnitude of the dilution effect because the differences in rank-averages were blown up by the massive temperature.
Rationale: To fix the saturation issue while still capturing the strong dilution effects, build on the accepted base (iter 2). Implement rank-averaging again, but this time normalize the ranks (e.g., divide by the total number of cues so they fall in the range (0, 1]). Additionally, restrict the inverse temperature parameters (`beta_ttb` and `beta_avg`) to a moderate range, such as [0.1, 10.0]. This will prevent the pre-softmax values from exploding, allowing the model to smoothly translate the amplified penalties of weak cues into probability differences without hitting the 1.0 ceiling.

**Outcome of this advice:** iter 3 candidate loss=0.3662 -> iter 4 candidate loss=0.2927 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate (iter 4) was rejected by the gate. While normalized rank-averaging was intended to amplify the dilution effect without causing softmax saturation, it failed to significantly improve the fit over the raw-validity averaging of the accepted base (iter 2). In Experiment 9, the real metric is -0.7263, meaning Option A is preferred much more when Option B has many weak cues compared to when B has only one opposing cue. Iter 4 predicted -0.0653, which is qualitatively correct (negative) but quantitatively far too small. The core issue is that the Take-The-Best (TTB) component of the mixture strongly prefers Option A in both trials (pushing P(A) toward 1.0 in both), which compresses the difference between the trials. If the TTB weight (α) is too high, the dilution effect produced by the Averaging component is washed out.
Rationale: Return to the accepted base from iteration 2 (TTB mixed with raw validity Averaging). To capture the massive magnitude of the dilution effect in Experiments 9 and 13, the Averaging component needs to dominate the mixture on these trials. Restrict the prior range of the mixing weight `alpha` to [0.0, 0.5] so that the model relies more heavily on the Averaging heuristic. Additionally, expand the `beta_avg` range to [1.0, 20.0] to allow the Averaging softmax to express stronger deterministic preferences without the catastrophic saturation we saw at 100.0. This will allow the dilution penalty of the Averaging strategy to shine through and produce the large negative metrics observed empirically.

**Outcome of this advice:** iter 4 candidate loss=0.2927 -> iter 5 candidate loss=0.3072 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate (iter 5) was rejected by the gate. Restricting the mixing weight `alpha` to [0.0, 0.5] did not improve the aggregate fit over the accepted base (iter 2). While it slightly increased the dilution effect in Experiment 13 (-0.4483 vs real -0.6683), it still failed to capture the massive magnitude of the dilution effect in Experiment 9 (-0.0811 vs real -0.7263) and hurt performance on other experiments like Experiment 14. The core issue seems to be that strict averaging (dividing by N) is too rigid; it may over-penalize or under-penalize depending on the trial, and restricting the strategy mixture to force this rigid heuristic to dominate degrades the overall fit.
Rationale: Return to the accepted base from iteration 2. Instead of forcing strict Averaging (dividing the sum of validities by N) and restricting the mixing weight `alpha`, introduce a fractional dilution parameter `gamma` in [0.0, 1.0]. The compensatory score for an option should be the sum of the validities of its active features divided by `(number_of_active_features)^gamma`. This allows the model to smoothly interpolate between Tallying (gamma=0, no dilution) and Averaging (gamma=1, full dilution), letting the data dictate the exact strength of the dilution effect without needing to artificially constrain the strategy mixture. Restore `alpha` to [0.0, 1.0] and keep moderate inverse temperature ranges (e.g., [0.1, 20.0]).

**Outcome of this advice:** iter 5 candidate loss=0.3072 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
```json
{
  "theory": "Heuristic Switching Theory with Fractional Dilution: Decision-makers probabilistically alternate between a non-compensatory strategy (Take-The-Best) and a compensatory heuristic that uses fractional dilution. Instead of strict averaging, the compensatory score divides the sum of validities of active cues by the number of active cues raised to a parameter gamma. This allows the model to flexibly interpolate between pure Tallying (gamma=0) and pure Averaging (gamma=1), capturing empirical dilution effects where adding weak cues can actively hurt an option's subjective value, without forcing a rigid division by N.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    a, b = stim[0], stim[1]\n    \n    alpha = float(parameters[\"alpha\"])\n    beta_ttb = float(parameters[\"beta_ttb\"])\n    beta_avg = float(parameters[\"beta_avg\"])\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Take-The-Best (TTB) prediction\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner_ttb = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner_ttb = 0\n            break\n        elif b[j] > a[j]:\n            winner_ttb = 1\n            break\n            \n    if winner_ttb == 0:\n        scores_ttb = np.array([1.0, 0.0])\n    elif winner_ttb == 1:\n        scores_ttb = np.array([0.0, 1.0])\n    else:\n        scores_ttb = np.array([0.5, 0.5])\n        \n    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))\n    e_ttb = np.exp(z_ttb)\n    p_ttb = e_ttb / np.sum(e_ttb)\n    \n    # Fractional Dilution prediction\n    n_a = np.sum(a > 0.5)\n    score_a = np.sum(val[a > 0.5]) / (n_a ** gamma) if n_a > 0 else 0.0\n    \n    n_b = np.sum(b > 0.5)\n    score_b = np.sum(val[b > 0.5]) / (n_b ** gamma) if n_b > 0 else 0.0\n    \n    scores_avg = np.array([score_a, score_b])\n    \n    z_avg = beta_avg * (scores_avg - np.max(scores_avg))\n    e_avg = np.exp(z_avg)\n    p_avg = e_avg / np.sum(e_avg)\n    \n    # Strategy Mixture\n    p_mix = alpha * p_ttb + (1.0 - alpha) * p_avg\n    \n    # Apply lapse rate\n    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])\n    \n    return p_final\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "beta_ttb": "[0.1, 20.0]",
    "beta_avg": "[0.1, 20.0]",
    "gamma": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we revert to the accepted base from Iteration 2 and replace strict Averaging (dividing by N) with a fractional dilution parameter `gamma` in [0.0, 1.0]. The compensatory component now divides the sum of validities by `(number_of_active_features)^gamma`. This allows the model to smoothly interpolate between Tallying (no dilution) and Averaging (full dilution), letting the data dictate the precise strength of the dilution effect without needing to artificially constrain the strategy mixing weight `alpha`, which has been restored to [0.0, 1.0]."
}
```
```

## Usage

```json
{
  "prompt_token_count": 28582,
  "candidates_token_count": 1149,
  "total_token_count": 31015
}
```
