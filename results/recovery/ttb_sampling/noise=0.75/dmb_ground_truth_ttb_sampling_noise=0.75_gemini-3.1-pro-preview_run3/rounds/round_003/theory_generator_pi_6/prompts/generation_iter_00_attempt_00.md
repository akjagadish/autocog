# generation_iter_00_attempt_00

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a 'Probabilistic Stopping Sequential Search' theory. Instead of a hard, deterministic threshold, the decision to stop searching and make a choice should be stochastic, with the probability of stopping at step t being a logistic or exponential function of the absolute accumulated evidence up to step t. This softens the rigid threshold of TSS (allowing it to better fit the graded compensatory behavior in Experiments 3, 4, and 7) while preserving the sequential accumulation and early-stopping dynamics that were definitively proven necessary by Experiments 1 and 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.912)

**Description**
Take The Best (TTB) is a fast-and-frugal lexicographic heuristic for multi-attribute choice. Instead of integrating all available information (like WADD) or counting unweighted cues (like Tallying), decision-makers search through features in descending order of their validity. The search stops at the first cue that discriminates between the two options, and the option favored by that cue is chosen. If no cue discriminates, the decision-maker guesses. This non-compensatory mechanism allows for rapid choices that often match more complex compensatory rules in environments with dispersed cue validities. Response noise is higher than previously assumed, reflecting softer empirical choice rates.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Lexicographic search: stop at the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores (which are either [1, 0], [0, 1], or [0, 0] if all tied)
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 3.0]
- epsilon: [0.0, 0.8]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6300 (var=0.0262) vs this=0.6967 (var=0.0308)
- Experiment 2: real=0.1822 (var=0.0512) vs this=0.3356 (var=0.1064)
- Experiment 3: real=0.6029 (var=0.0014) vs this=0.6554 (var=0.0109)
- Experiment 4: real=0.4006 (var=0.0048) vs this=0.3253 (var=0.0153)
- Experiment 5: real=-0.0185 (var=0.0133) vs this=0.0076 (var=0.0107)
- Experiment 6: real=0.4147 (var=0.0049) vs this=0.3020 (var=0.0150)
- Experiment 7: real=0.0000 (var=0.0528) vs this=0.0400 (var=0.0687)
- Experiment 8: real=0.0383 (var=0.0204) vs this=-0.0242 (var=0.0198)


---

### `pi_4` (overall score: 0.885)

**Description**
Strategy Mixture (Take-The-Best and Weighted Additive)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    # --- TTB (Take-The-Best) Process ---
    order = np.argsort(validities)[::-1]
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- WADD (Weighted Additive) Process ---
    w = np.asarray(parameters["weights"], dtype=float)
    scores_wadd = stim @ (validities * w)
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Mixture ---
    mix = float(parameters["mixture_ttb"])
    p_mix = mix * p_ttb + (1.0 - mix) * p_wadd
    
    # --- Lapse ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta_ttb: [0.1, 5.0]
- beta_wadd: [0.1, 10.0]
- mixture_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6300 (var=0.0262) vs this=0.6883 (var=0.0344)
- Experiment 2: real=0.1822 (var=0.0512) vs this=0.2489 (var=0.1395)
- Experiment 3: real=0.6029 (var=0.0014) vs this=0.5685 (var=0.0207)
- Experiment 4: real=0.4006 (var=0.0048) vs this=0.4603 (var=0.0301)
- Experiment 5: real=-0.0185 (var=0.0133) vs this=0.1356 (var=0.0222)
- Experiment 6: real=0.4147 (var=0.0049) vs this=0.4440 (var=0.0241)
- Experiment 7: real=0.0000 (var=0.0528) vs this=0.2550 (var=0.1659)
- Experiment 8: real=0.0383 (var=0.0204) vs this=0.1542 (var=0.0469)


---

### `pi_5` (overall score: 0.707)

**Description**
Threshold-based Sequential Search integrates the fast-and-frugal nature of Take-The-Best with the compensatory evidence accumulation of Weighted Additive (WADD) models. Decision-makers search through cues in descending order of their validity, maintaining a running tally of the evidence (weighted by each cue's validity). Instead of stopping at the very first discriminating cue, search terminates only when the absolute accumulated evidence exceeds an internal confidence threshold. If all cues are exhausted without reaching this threshold, the option with the higher accumulated evidence is chosen. This allows for fast, one-reason decisions when a highly valid cue strongly favors one option, while enabling compensatory behavior when early cues provide weak or conflicting evidence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    threshold = float(parameters["threshold"])
    
    # Sequential search with evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Stop search if the confidence threshold is met or exceeded
        if abs(evidence) >= threshold:
            break
            
    # Convert accumulated evidence into discrete choice scores
    scores = np.zeros(2)
    if evidence > 0:
        scores[0] = 1.0
    elif evidence < 0:
        scores[1] = 1.0
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- threshold: [0.0, 1.25]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.25]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6300 (var=0.0262) vs this=0.8383 (var=0.0218)
- Experiment 2: real=0.1822 (var=0.0512) vs this=0.6667 (var=0.0904)
- Experiment 3: real=0.6029 (var=0.0014) vs this=0.6846 (var=0.0369)
- Experiment 4: real=0.4006 (var=0.0048) vs this=0.3278 (var=0.0698)
- Experiment 5: real=-0.0185 (var=0.0133) vs this=0.2520 (var=0.1253)
- Experiment 6: real=0.4147 (var=0.0049) vs this=0.3700 (var=0.0943)
- Experiment 7: real=0.0000 (var=0.0528) vs this=-0.0150 (var=0.0494)
- Experiment 8: real=0.0383 (var=0.0204) vs this=-0.0092 (var=0.0176)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying and WADD strongly conflict
    # Trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    if t1_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen on this trial.
    # Tallying predicts B (response == 1) because B has 3 positive features vs A's 2.
    # WADD predicts A (response == 0) because A's 2 features have higher total validity (1.85 vs 1.65).
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.6300 (var=0.0262)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1383 (var=0.0216)
- pi_2: 0.5967 (var=0.1140)
- pi_3: 0.6967 (var=0.0308)
- pi_4: 0.6883 (var=0.0344)
- pi_5: 0.8383 (var=0.0218)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_t1(x):
        return tuple(x) == (1, 1, 0, 0, 0)
        
    def is_t2(x):
        return tuple(x) == (0, 0, 1, 1, 1)
        
    m1 = data['option_a_ratings'].apply(is_t1)
    m2 = data['option_a_ratings'].apply(is_t2)
    
    r1 = data.loc[m1, 'response'].mean()
    r2 = data.loc[m2, 'response'].mean()
    
    if pd.isna(r1): r1 = 0.5
    if pd.isna(r2): r2 = 0.5
    
    return float(r2 - r1)
```

**Observed (real) value:** 0.1822 (var=0.0512)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0022 (var=0.3435)
- pi_1: -0.6800 (var=0.0606)
- pi_3: 0.3356 (var=0.1064)
- pi_4: 0.2489 (var=0.1395)
- pi_5: 0.6667 (var=0.0904)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    weights = np.array([10000, 1000, 100, 10, 1])
    score = diff @ weights
    
    ttb_choice = (score < 0).astype(int)
    matches = (data['response'] == ttb_choice).mean()
    
    return float(matches)
```

**Observed (real) value:** 0.6029 (var=0.0014)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6554 (var=0.0109)
- pi_2: 0.3979 (var=0.0080)
- pi_1: 0.3181 (var=0.0031)
- pi_4: 0.5685 (var=0.0207)
- pi_5: 0.6846 (var=0.0369)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of positive features for options A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter to trials where option B has more positive features than option A
    # In this specific design, these are the compensatory trials where the 
    # most valid cue favors A but the sheer number of lower-validity cues favors B.
    mask = sum_b > sum_a
    
    if mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times option B (response == 1) was chosen
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4006 (var=0.0048)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6637 (var=0.0226)
- pi_3: 0.3253 (var=0.0153)
- pi_1: 0.8550 (var=0.0127)
- pi_4: 0.4603 (var=0.0301)
- pi_5: 0.3278 (var=0.0698)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    agree_matches = 0
    agree_total = 0
    conflict_matches = 0
    conflict_total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = int(row['response'])
        
        # Find TTB winner
        ttb_winner = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner == -1:
            continue
            
        # Find WADD winner
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        if wadd_a > wadd_b:
            wadd_winner = 0
        elif wadd_b > wadd_a:
            wadd_winner = 1
        else:
            continue
            
        # Check if TTB and WADD agree or conflict
        if ttb_winner == wadd_winner:
            agree_total += 1
            if resp == ttb_winner:
                agree_matches += 1
        else:
            conflict_total += 1
            if resp == ttb_winner:
                conflict_matches += 1
                
    p_agree = agree_matches / agree_total if agree_total > 0 else 0.5
    p_conflict = conflict_matches / conflict_total if conflict_total > 0 else 0.5
    
    return float(p_agree - p_conflict)
```

**Observed (real) value:** -0.0185 (var=0.0133)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0076 (var=0.0107)
- pi_4: 0.1356 (var=0.0222)
- pi_1: 0.5191 (var=0.0255)
- pi_2: 0.3530 (var=0.0317)
- pi_5: 0.2520 (var=0.1253)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # TTB favors A if the first differing cue is positive for A.
    # Using exponentially decreasing weights captures lexicographic order exactly for 5 binary features.
    powers = np.array([10000, 1000, 100, 10, 1])
    diff = a_mat - b_mat
    ttb_favors_a = (diff @ powers) > 0
    
    # WADD favors B if the weighted sum of cues is higher for B.
    wadd_a = a_mat @ validities
    wadd_b = b_mat @ validities
    wadd_favors_b = wadd_b > wadd_a
    
    # Identify compensatory conflict trials
    conflict_mask = ttb_favors_a & wadd_favors_b
    
    if not np.any(conflict_mask):
        return 0.0
        
    # Return the proportion of B choices on these conflict trials
    return float(np.mean(data['response'].values[conflict_mask]))

```

**Observed (real) value:** 0.4147 (var=0.0049)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4440 (var=0.0241)
- pi_3: 0.3020 (var=0.0150)
- pi_1: 0.8737 (var=0.0084)
- pi_2: 0.7617 (var=0.0139)
- pi_5: 0.3700 (var=0.0943)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_key'] = data['option_a_ratings'].apply(tuple)
    
    t3 = (1, 1, 0, 0, 0)
    t4 = (1, 1, 1, 0, 0)
    t7 = (0, 0, 1, 1, 1)
    t8 = (0, 0, 0, 1, 1)
    
    pA_t3 = (data[data['A_key'] == t3]['response'] == 0).mean()
    pA_t4 = (data[data['A_key'] == t4]['response'] == 0).mean()
    
    pB_t7 = (data[data['A_key'] == t7]['response'] == 1).mean()
    pB_t8 = (data[data['A_key'] == t8]['response'] == 1).mean()
    
    pA_t3 = pA_t3 if pd.notna(pA_t3) else 0.5
    pA_t4 = pA_t4 if pd.notna(pA_t4) else 0.5
    pB_t7 = pB_t7 if pd.notna(pB_t7) else 0.5
    pB_t8 = pB_t8 if pd.notna(pB_t8) else 0.5
    
    return float((pA_t4 - pA_t3) + (pB_t8 - pB_t7))
```

**Observed (real) value:** 0.0000 (var=0.0528)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0150 (var=0.0494)
- pi_4: 0.2550 (var=0.1659)
- pi_1: 1.4083 (var=0.1601)
- pi_2: 0.5333 (var=0.4039)
- pi_3: 0.0400 (var=0.0687)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify the option that the Take-The-Best (TTB) heuristic would favor
    # (TTB relies entirely on the first cue since it has the highest validity)
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    b0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_is_A = a0 > b0
    chose_ttb = ((ttb_is_A) & (data['response'] == 0)) | ((~ttb_is_A) & (data['response'] == 1))

    # Convert ratings to strings for exact trial matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))

    # Condition 1: TTB and WADD agree (Trial 6)
    # Both heuristics strongly favor the same option.
    t6 = (a_str == '11100') & (b_str == '00011')

    # Condition 2: TTB and WADD conflict, but TSS stops early (Trials 1 and 4)
    # The first two cues agree, so TSS accumulates 0.95 + 0.80 = 1.75 evidence.
    # Since the max threshold is 1.25, TSS *always* stops early and chooses the TTB option.
    # Strategy Mixture, however, integrates all cues for WADD, so WADD prefers the opposite option.
    t1_t4 = ((a_str == '11000') & (b_str == '00111')) | ((a_str == '00111') & (b_str == '11000'))

    rate_agree = chose_ttb[t6].mean()
    rate_conflict_early = chose_ttb[t1_t4].mean()

    if pd.isna(rate_agree) or pd.isna(rate_conflict_early):
        return 0.0

    # Return the difference in TTB adherence
    return float(rate_agree - rate_conflict_early)
```

**Observed (real) value:** 0.0383 (var=0.0204)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1542 (var=0.0469)
- pi_5: -0.0092 (var=0.0176)
- pi_1: 0.7383 (var=0.0535)
- pi_2: 0.2992 (var=0.1214)
- pi_3: -0.0242 (var=0.0198)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Probabilistic Stopping Sequential Search softens the rigid confidence threshold of standard sequential search models. Decision-makers accumulate evidence by inspecting cues in descending order of validity. Instead of stopping deterministically when a threshold is crossed, the probability of halting search at any step is a logistic function of the absolute accumulated evidence. This creates a stochastic stopping process where weak evidence can occasionally trigger a choice and strong evidence can occasionally be ignored in favor of further search, naturally producing the graded compensatory patterns observed in empirical data while preserving the core early-stopping dynamics of Take-The-Best.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    order = np.argsort(validities)[::-1]\n    \n    a, b = stim[0], stim[1]\n    evidence = 0.0\n    \n    threshold = float(parameters[\"threshold\"])\n    slope = float(parameters[\"slope\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    p_continue = 1.0\n    p_A = 0.0\n    \n    # Sequential search with probabilistic stopping\n    for i, idx in enumerate(order):\n        diff = a[idx] - b[idx]\n        evidence += diff * validities[idx]\n        \n        # Determine stopping probability at this step\n        if i == len(order) - 1:\n            p_stop = 1.0\n        else:\n            # Logistic function for stopping probability\n            z = -slope * (abs(evidence) - threshold)\n            z = np.clip(z, -50, 50)  # Prevent overflow\n            p_stop = 1.0 / (1.0 + np.exp(z))\n            \n        p_stop_here = p_continue * p_stop\n        p_continue *= (1.0 - p_stop)\n        \n        # Softmax choice probability if search stops at this step\n        z_choice = -beta * evidence\n        z_choice = np.clip(z_choice, -50, 50)\n        p_A_given_stop = 1.0 / (1.0 + np.exp(z_choice))\n        \n        p_A += p_stop_here * p_A_given_stop\n        \n    p_B = 1.0 - p_A\n    probs = np.array([p_A, p_B])\n    \n    # Blend in uniform lapse\n    return (1.0 - epsilon) * probs + epsilon * 0.5\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "threshold": "[0.0, 3.0]",
    "slope": "[0.1, 20.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this model instantiates a Probabilistic Stopping Sequential Search. It replaces the strict deterministic threshold of the previous Threshold-based Sequential Search (pi_5) with a stochastic stopping rule. At each step, the probability of terminating search is modeled as a logistic function of the absolute accumulated evidence, controlled by a 'threshold' (inflection point) and a 'slope' (sensitivity). If search stops, the choice is generated via a softmax function over the accumulated evidence. This mechanism inherently captures the graded nature of compensatory decision-making seen in Experiments 3, 4, and 7, while retaining the sequential consideration of cues and early-stopping properties necessary to explain behavior in Experiments 1 and 2."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12062,
  "candidates_token_count": 1012,
  "total_token_count": 15306
}
```
