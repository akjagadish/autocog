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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a brand-new theory based on 'Rank-Weighted Integration'. Instead of using the raw numeric validities (which Theory 2 uses) or applying a strict threshold (Theory 1), this theory posits that decision-makers rely on the *ordinal rank* of the validities. Features are weighted by an exponentially decaying function of their rank. This captures bounded rationality and 'Take-The-Best'-like behavior (the top-ranked feature gets the lion's share of the weight) while still allowing lower-ranked features to contribute in a compensatory manner without being completely zeroed out. This provides a strong contrast to the Tallying+WADD mixture by explaining the data through a single, boundedly-rational integration process rather than a dual-process mixture.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.850)

**Description**
Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4408 (var=0.0085) vs this=0.4198 (var=0.0092)
- Experiment 2: real=0.6317 (var=0.0047) vs this=0.6877 (var=0.0084)
- Experiment 3: real=0.5133 (var=0.0354) vs this=0.5783 (var=0.0312)
- Experiment 4: real=0.5400 (var=0.0176) vs this=0.5792 (var=0.0147)
- Experiment 5: real=0.5550 (var=0.0310) vs this=0.5437 (var=0.0168)
- Experiment 6: real=0.0400 (var=0.2001) vs this=0.1517 (var=0.1130)
- Experiment 7: real=0.0700 (var=0.0369) vs this=0.0933 (var=0.0255)
- Experiment 8: real=0.5525 (var=0.0250) vs this=0.5288 (var=0.0037)
- Experiment 9: real=-0.1038 (var=0.0424) vs this=-0.2475 (var=0.0991)
- Experiment 10: real=0.2850 (var=0.0607) vs this=0.3775 (var=0.0906)


---

### `pi_6` (overall score: 0.629)

**Description**
Decision makers employ a Thresholded Weighted Additive (WADD) strategy. Instead of integrating all available information or relying entirely on a single cue, subjects impose a depth-of-processing limit by ignoring features whose validities fall below a subjective threshold. Features that exceed this threshold are integrated into a compensatory sum weighted by a non-linear transformation of their validities. By allowing the threshold to be arbitrarily low and constraining the non-linear scaling, the model captures bounded rationality while avoiding an over-reliance on a single best cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Thresholded WADD Component: Only integrate features whose validities are >= theta
    mask = val >= theta
    subjective_weights = np.zeros_like(val)
    if np.any(mask):
        subjective_weights[mask] = val[mask] ** gamma
        
    sum_weights = np.sum(subjective_weights)
    if sum_weights > 0:
        score_a_wadd = np.sum(a * subjective_weights) / sum_weights
        score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    else:
        score_a_wadd, score_b_wadd = 0.5, 0.5
        
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Incorporate response noise (lapse rate)
    p_final = (1.0 - epsilon) * p_wadd + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 10.0]
- gamma: [0.0, 3.0]
- theta: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4408 (var=0.0085) vs this=0.4942 (var=0.0165)
- Experiment 2: real=0.6317 (var=0.0047) vs this=0.6469 (var=0.0211)
- Experiment 3: real=0.5133 (var=0.0354) vs this=0.6983 (var=0.0386)
- Experiment 4: real=0.5400 (var=0.0176) vs this=0.6300 (var=0.0182)
- Experiment 5: real=0.5550 (var=0.0310) vs this=0.7075 (var=0.0344)
- Experiment 6: real=0.0400 (var=0.2001) vs this=-0.2867 (var=0.2053)
- Experiment 7: real=0.0700 (var=0.0369) vs this=0.0658 (var=0.0299)
- Experiment 8: real=0.5525 (var=0.0250) vs this=0.5288 (var=0.0312)
- Experiment 9: real=-0.1038 (var=0.0424) vs this=0.0312 (var=0.1337)
- Experiment 10: real=0.2850 (var=0.0607) vs this=0.0837 (var=0.1068)


---

### `pi_5` (overall score: 0.568)

**Description**
Probabilistic Heuristic Toolbox: Subjects make decisions by probabilistically sampling from a repertoire of fast-and-frugal heuristics rather than computing compensatory weighted sums. Specifically, on any given trial, a subject either uses 'Take-The-Best' (TTB) - a lexicographic strategy that bases the choice entirely on the single most valid discriminating feature - or 'Tallying' - an equal-weighting strategy that simply counts the number of winning features for each option. A mixture parameter alpha governs the probability of selecting TTB over Tallying, and an independent lapse rate epsilon accounts for execution noise or random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) Component
    # Sort features by descending validity. Find the first feature that discriminates.
    order = np.argsort(val)[::-1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Component
    # Count strict feature-wise wins for each option.
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Mixture of heuristics
    p_mixed = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Incorporate shared response noise (lapse rate)
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- alpha: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4408 (var=0.0085) vs this=0.5096 (var=0.0367)
- Experiment 2: real=0.6317 (var=0.0047) vs this=0.5315 (var=0.0338)
- Experiment 3: real=0.5133 (var=0.0354) vs this=0.6067 (var=0.0250)
- Experiment 4: real=0.5400 (var=0.0176) vs this=0.6358 (var=0.0240)
- Experiment 5: real=0.5550 (var=0.0310) vs this=0.6212 (var=0.0257)
- Experiment 6: real=0.0400 (var=0.2001) vs this=0.1967 (var=0.1355)
- Experiment 7: real=0.0700 (var=0.0369) vs this=-0.0192 (var=0.0166)
- Experiment 8: real=0.5525 (var=0.0250) vs this=0.3821 (var=0.0162)
- Experiment 9: real=-0.1038 (var=0.0424) vs this=-0.1500 (var=0.0787)
- Experiment 10: real=0.2850 (var=0.0607) vs this=0.2363 (var=0.0747)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    
    ttb_choices = []
    for a, b in zip(a_mat, b_mat):
        choice = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                choice = 0
                break
            elif b[i] > a[i]:
                choice = 1
                break
        ttb_choices.append(choice)
        
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != 0.5
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4408 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8546 (var=0.0116)
- pi_2: 0.2252 (var=0.0040)
- pi_3: 0.4402 (var=0.0132)
- pi_4: 0.4198 (var=0.0092)
- pi_5: 0.5096 (var=0.0367)
- pi_6: 0.4942 (var=0.0165)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying counts the number of features where one option strictly dominates the other
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.zeros(len(data))
    tally_pred[b_wins > a_wins] = 1
    tally_pred[a_wins == b_wins] = -1 # Ignore ties for matching purposes
    
    matches = (data['response'].values == tally_pred)
    valid = (tally_pred != -1)
    
    return float(np.mean(matches[valid])) if np.any(valid) else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0047)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8548 (var=0.0074)
- pi_1: 0.2681 (var=0.0045)
- pi_3: 0.6663 (var=0.0137)
- pi_4: 0.6877 (var=0.0084)
- pi_5: 0.5315 (var=0.0338)
- pi_6: 0.6469 (var=0.0211)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target = (1, 1, 0, 0, 0)
    other = (0, 0, 1, 1, 0)
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    mask = ((a_keys == target) & (b_keys == other)) | ((a_keys == other) & (b_keys == target))
    if not mask.any():
        return 0.5
    rel_data = data[mask]
    chosen_a = rel_data['response'] == 0
    a_is_target = a_keys[mask] == target
    chose_target = (chosen_a & a_is_target) | (~chosen_a & ~a_is_target)
    return float(chose_target.mean())
```

**Observed (real) value:** 0.5133 (var=0.0354)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7067 (var=0.0378)
- pi_2: 0.4883 (var=0.0156)
- pi_1: 0.8567 (var=0.0208)
- pi_4: 0.5783 (var=0.0312)
- pi_5: 0.6067 (var=0.0250)
- pi_6: 0.6983 (var=0.0386)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    mask = ((a_str == '10100') & (b_str == '01010')) | ((a_str == '01100') & (b_str == '00011'))
    
    target_data = data[mask]
    if len(target_data) == 0:
        return 0.5
        
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.5400 (var=0.0176)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5325 (var=0.0103)
- pi_3: 0.6417 (var=0.0197)
- pi_1: 0.8475 (var=0.0132)
- pi_4: 0.5792 (var=0.0147)
- pi_5: 0.6358 (var=0.0240)
- pi_6: 0.6300 (var=0.0182)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['is_t1'] = data.apply(
        lambda row: tuple(row['option_a_ratings']) == (1, 1, 1, 1, 0, 0, 0, 0) and 
                    tuple(row['option_b_ratings']) == (0, 0, 0, 0, 1, 1, 1, 1),
        axis=1
    )
    t1_data = data[data['is_t1']]
    if len(t1_data) == 0:
        return 0.5
    return float((t1_data['response'] == 0).mean())
```

**Observed (real) value:** 0.5550 (var=0.0310)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7887 (var=0.0209)
- pi_4: 0.5437 (var=0.0168)
- pi_1: 0.8512 (var=0.0153)
- pi_2: 0.5288 (var=0.0180)
- pi_5: 0.6212 (var=0.0257)
- pi_6: 0.7075 (var=0.0344)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    # Tallying heavily favors A in these trials
    t7_A = (1, 0, 0, 1, 1, 1)
    t6_A = (0, 0, 1, 1, 1, 0)
    
    # Tallying predicts an exact tie in these trials
    t3_A = (1, 1, 1, 0, 0, 0)
    t4_A = (0, 1, 1, 1, 0, 0)
    
    m7 = data[data['A_tuple'] == t7_A]['response'].mean()
    m6 = data[data['A_tuple'] == t6_A]['response'].mean()
    m3 = data[data['A_tuple'] == t3_A]['response'].mean()
    m4 = data[data['A_tuple'] == t4_A]['response'].mean()
    
    # Response is 0 for A, 1 for B. P(A) = 1 - mean(response)
    p_A_t7 = 1.0 - m7
    p_A_t6 = 1.0 - m6
    p_A_t3 = 1.0 - m3
    p_A_t4 = 1.0 - m4
    
    return float((p_A_t7 + p_A_t6) - (p_A_t3 + p_A_t4))
```

**Observed (real) value:** 0.0400 (var=0.2001)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1517 (var=0.1130)
- pi_3: -0.2383 (var=0.1589)
- pi_1: -0.0050 (var=0.0493)
- pi_2: 0.6633 (var=0.1119)
- pi_5: 0.1967 (var=0.1355)
- pi_6: -0.2867 (var=0.2053)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option lists to tuples to make them hashable for comparison
    a_tups = data['option_a_ratings'].apply(tuple)
    
    # Trial Type 1: The option with the most valid feature (f0) also has the least valid feature (f4).
    # A = [1, 0, 0, 0, 1], B = [0, 1, 1, 1, 0] (and its symmetric counterpart)
    m_t1_a = a_tups == (1, 0, 0, 0, 1)
    m_t1_b = a_tups == (0, 1, 1, 1, 0)
    
    # Trial Type 3: The option with the most valid feature (f0) has NO other features.
    # A = [1, 0, 0, 0, 0], B = [0, 1, 1, 1, 1] (and its symmetric counterpart)
    m_t3_a = a_tups == (1, 0, 0, 0, 0)
    m_t3_b = a_tups == (0, 1, 1, 1, 1)
    
    # Count how often the option possessing the most valid feature (f0=1) was chosen
    t1_chosen = data.loc[m_t1_a, 'response'].eq(0).sum() + data.loc[m_t1_b, 'response'].eq(1).sum()
    t1_total = m_t1_a.sum() + m_t1_b.sum()
    
    t3_chosen = data.loc[m_t3_a, 'response'].eq(0).sum() + data.loc[m_t3_b, 'response'].eq(1).sum()
    t3_total = m_t3_a.sum() + m_t3_b.sum()
    
    p_t1 = t1_chosen / t1_total if t1_total > 0 else 0.0
    p_t3 = t3_chosen / t3_total if t3_total > 0 else 0.0
    
    # Return the difference in preference for the f0=1 option between Trial Type 1 and Trial Type 3
    return float(p_t1 - p_t3)
```

**Observed (real) value:** 0.0700 (var=0.0369)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0192 (var=0.0166)
- pi_4: 0.0933 (var=0.0255)
- pi_1: -0.0092 (var=0.0111)
- pi_2: 0.0258 (var=0.0115)
- pi_3: 0.1267 (var=0.0252)
- pi_6: 0.0658 (var=0.0299)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.9, 0.85, 0.7, 0.65, 0.6])
    
    wadd_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying wins
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            # Tallying is a tie
            ttb_choice = None
            for idx in range(len(val)):
                if a[idx] > b[idx]:
                    ttb_choice = 0
                    break
                elif b[idx] > a[idx]:
                    ttb_choice = 1
                    break
            
            wadd_score_a = np.sum(a * val)
            wadd_score_b = np.sum(b * val)
            wadd_choice = 0 if wadd_score_a > wadd_score_b else 1
            
            if ttb_choice is not None and ttb_choice != wadd_choice:
                response = row['response']
                wadd_matches.append(1 if response == wadd_choice else 0)
                
    if len(wadd_matches) == 0:
        return 0.5
    return float(np.mean(wadd_matches))
```

**Observed (real) value:** 0.5525 (var=0.0250)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5288 (var=0.0037)
- pi_5: 0.3821 (var=0.0162)
- pi_1: 0.1600 (var=0.0121)
- pi_2: 0.5096 (var=0.0035)
- pi_3: 0.5925 (var=0.0082)
- pi_6: 0.5288 (var=0.0312)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    t13_mask = a_str.isin(['100111', '010111'])
    t4_mask = a_str == '110000'
    if t13_mask.sum() == 0 or t4_mask.sum() == 0:
        return 0.0
    return float(data.loc[t13_mask, 'response'].mean() - data.loc[t4_mask, 'response'].mean())
```

**Observed (real) value:** -0.1038 (var=0.0424)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0312 (var=0.1337)
- pi_4: -0.2475 (var=0.0991)
- pi_1: 0.3700 (var=0.0152)
- pi_2: -0.7050 (var=0.0438)
- pi_3: -0.1131 (var=0.1191)
- pi_5: -0.1500 (var=0.0787)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Trial 2: A=[1, 0, 0, 1, 1, 1], B=[1, 0, 0, 0, 0, 0]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1, 1)) & \
            data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0, 0))
            
    # Trial 4: A=[0, 0, 1, 0, 0, 0], B=[0, 0, 0, 1, 1, 1]
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 0, 0)) & \
            data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    
    b_t2 = data[is_t2]['response'].mean() if is_t2.any() else 0.5
    b_t4 = data[is_t4]['response'].mean() if is_t4.any() else 0.5
    
    return float(b_t4 - b_t2)
```

**Observed (real) value:** 0.2850 (var=0.0607)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3775 (var=0.0906)
- pi_6: 0.0837 (var=0.1068)
- pi_1: 0.0200 (var=0.0163)
- pi_2: 0.6750 (var=0.0586)
- pi_3: 0.4175 (var=0.0949)
- pi_5: 0.2363 (var=0.0747)


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
  "theory": "Rank-Weighted Integration: Decision-makers rely on the ordinal rank of cue validities rather than their raw numeric values. Features are integrated into a compensatory sum, but each feature's weight decays exponentially according to its rank (the most valid feature has rank 0, the next rank 1, etc.). This single integration process naturally spans the spectrum from Take-The-Best (steep decay) to Tallying (no decay), capturing boundedly rational decision making without requiring a dual-process mixture.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) state; got shape {stim.shape}.\")\n    \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    decay = float(parameters[\"decay\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Determine the ordinal rank of each feature's validity (0 = most valid)\n    # np.argsort sorts ascending; [::-1] makes it descending\n    order = np.argsort(val)[::-1]\n    ranks = np.zeros_like(val, dtype=float)\n    ranks[order] = np.arange(len(val))\n    \n    # Exponentially decay weights based on rank\n    # If decay is 0.0, 0.0**0 is 1.0 and 0.0**>0 is 0.0 (Take-The-Best)\n    # If decay is 1.0, 1.0**rank is 1.0 for all (Tallying)\n    weights = decay ** ranks\n    \n    sum_weights = np.sum(weights)\n    if sum_weights > 0:\n        score_a = np.sum(a * weights) / sum_weights\n        score_b = np.sum(b * weights) / sum_weights\n    else:\n        score_a, score_b = 0.5, 0.5\n        \n    scores = np.array([score_a, score_b])\n    \n    # Softmax choice rule\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p = e / np.sum(e)\n    \n    # Incorporate lapse rate\n    p_final = (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.01, 10.0]",
    "decay": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "By utilizing the ordinal rank of validities and applying an exponentially decaying weight function, this theory aligns with the arbiter's suggestion to replace strict thresholds and dual-process mixtures with a unified, boundedly rational integration mechanism. The 'decay' parameter elegantly transitions between Take-The-Best (when decay approaches 0) and Tallying (when decay approaches 1), capturing both dominant cue reliance and compensatory weighting."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14025,
  "candidates_token_count": 887,
  "total_token_count": 16039
}
```
