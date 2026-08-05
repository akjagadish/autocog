# generation_iter_01_attempt_00

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

Propose a new 'Integrated Strategy Value Theory' or a 'Multi-attribute Sequential Sampling' framework. This new theory should integrate evidence from non-compensatory (TTB-like), unweighted compensatory (Tallying), and weighted compensatory (WADD) features into a single evidence accumulation process or subjective value, rather than probabilistically sampling one discrete strategy per trial. By allowing a non-compensatory TTB signal to be weighted alongside WADD and Tallying, the model can capture the TTB-aligned choices in Exps 1-4, while the continuous integration process can still predict indifference (P=0.5) in delicately balanced trials (Exps 5, 6, 8) where a strict probabilistic TTB sample would incorrectly force a hard choice.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.752)

**Description**
Probabilistic Strategy Selection Theory: Decision-makers maintain a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive/WADD). On each trial, they probabilistically sample one strategy from this repertoire according to base-rate strategy weights, and execute it purely. This discrete mixture of strategies accounts for intermediate aggregate choice proportions while naturally predicting exact ties when the sampled strategy itself encounters a tie.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB)
    diff = a - b
    order = np.argsort(val)[::-1]
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if diff[idx] > 0:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif diff[idx] < 0:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Tallying
    tal_diff = np.sum(a) - np.sum(b)
    if tal_diff > 0:
        tal_pred = np.array([1.0, 0.0])
    elif tal_diff < 0:
        tal_pred = np.array([0.0, 1.0])
    else:
        tal_pred = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD)
    wadd_diff = np.sum(val * a) - np.sum(val * b)
    if wadd_diff > 0:
        wadd_pred = np.array([1.0, 0.0])
    elif wadd_diff < 0:
        wadd_pred = np.array([0.0, 1.0])
    else:
        wadd_pred = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    w_arr = np.array([w_ttb, w_tal, w_wadd])
    sum_w = np.sum(w_arr)
    if sum_w == 0:
        p_strat = np.array([1/3, 1/3, 1/3])
    else:
        p_strat = w_arr / sum_w
        
    p_core = p_strat[0] * ttb_pred + p_strat[1] * tal_pred + p_strat[2] * wadd_pred
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- w_ttb: [0.0, 1.0]
- w_tal: [0.0, 1.0]
- w_wadd: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4167 (var=0.0087) vs this=0.3848 (var=0.0142)
- Experiment 2: real=0.6044 (var=0.0126) vs this=0.6494 (var=0.0221)
- Experiment 3: real=0.5611 (var=0.0131) vs this=0.7311 (var=0.0098)
- Experiment 4: real=1.0350 (var=0.0694) vs this=1.2200 (var=0.0368)
- Experiment 5: real=0.5792 (var=0.0095) vs this=0.5790 (var=0.0152)
- Experiment 6: real=0.0167 (var=0.0257) vs this=0.2633 (var=0.0436)
- Experiment 7: real=0.4058 (var=0.0101) vs this=0.4204 (var=0.0242)
- Experiment 8: real=0.3950 (var=0.0100) vs this=0.3450 (var=0.0207)


---

### `pi_4` (overall score: 0.725)

**Description**
Hybrid Heuristic Theory: Decision-makers integrate evidence from multiple strategies before making a choice, rather than probabilistically sampling between distinct strategies. Specifically, individuals compute a combined subjective value for each option by taking a weighted average of normalized compensatory (validity-weighted) and non-compensatory (tallying) evidence. A single stochastic decision process then operates on these integrated values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Normalized WADD scores
    sum_val = np.sum(val)
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    # Normalized Tallying scores
    score_tal_a = float(np.sum(a > b)) / n_features
    score_tal_b = float(np.sum(b > a)) / n_features
    
    beta = float(parameters["beta"])
    w_mix = float(parameters["w_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Combined integrated scores
    score_combined_a = w_mix * score_wadd_a + (1.0 - w_mix) * score_tal_a
    score_combined_b = w_mix * score_wadd_b + (1.0 - w_mix) * score_tal_b
    scores = np.array([score_combined_a, score_combined_b])
    
    # Single Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4167 (var=0.0087) vs this=0.2250 (var=0.0140)
- Experiment 2: real=0.6044 (var=0.0126) vs this=0.7922 (var=0.0109)
- Experiment 3: real=0.5611 (var=0.0131) vs this=0.5689 (var=0.0136)
- Experiment 4: real=1.0350 (var=0.0694) vs this=1.0550 (var=0.0349)
- Experiment 5: real=0.5792 (var=0.0095) vs this=0.7350 (var=0.0125)
- Experiment 6: real=0.0167 (var=0.0257) vs this=0.0333 (var=0.0252)
- Experiment 7: real=0.4058 (var=0.0101) vs this=0.2704 (var=0.0135)
- Experiment 8: real=0.3950 (var=0.0100) vs this=0.2050 (var=0.0153)


---

### `pi_3` (overall score: 0.534)

**Description**
Weighted Additive (WADD) Theory with Non-Linear Cue Scaling: People evaluate multi-attribute options by computing an overall value for each option. This value is determined by taking a weighted sum of the option's features, where the weights correspond to the subjective validities of the respective cues scaled by a non-linear parameter. This scaling allows decision-makers to flexibly upweight highly valid cues (approaching a non-compensatory strategy) or downweight them (approaching an equal-weight tallying strategy). Decision-makers then choose probabilistically between the options by comparing these total weighted scores via a softmax function.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for both options
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- gamma: [0.1, 6.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4167 (var=0.0087) vs this=0.4692 (var=0.0436)
- Experiment 2: real=0.6044 (var=0.0126) vs this=0.7206 (var=0.0372)
- Experiment 3: real=0.5611 (var=0.0131) vs this=0.8378 (var=0.0112)
- Experiment 4: real=1.0350 (var=0.0694) vs this=1.4888 (var=0.0953)
- Experiment 5: real=0.5792 (var=0.0095) vs this=0.3456 (var=0.0583)
- Experiment 6: real=0.0167 (var=0.0257) vs this=0.2750 (var=0.0441)
- Experiment 7: real=0.4058 (var=0.0101) vs this=0.4771 (var=0.0514)
- Experiment 8: real=0.3950 (var=0.0100) vs this=0.4763 (var=0.0771)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4193 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4193 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # The validities are monotonically decreasing from left to right.
    # Thus, Take The Best (TTB) will choose based on the first feature
    # (from index 0 to 4) that discriminates between A and B.
    diff = a_ratings - b_ratings
    
    # Find the index of the first discriminating cue (where difference is non-zero)
    first_diff_idx = (diff != 0).argmax(axis=1)
    
    # Extract the difference value at that first discriminating cue
    first_diff_val = np.take_along_axis(diff, first_diff_idx[:, None], axis=1).squeeze()
    
    # If first_diff_val > 0 (A has the feature, B does not), TTB predicts A (response 0).
    # If first_diff_val < 0 (B has the feature, A does not), TTB predicts B (response 1).
    ttb_predictions = (first_diff_val < 0).astype(int)
    
    # Return the proportion of choices that match the TTB prediction.
    return float(np.mean(responses == ttb_predictions))
```

**Observed (real) value:** 0.4167 (var=0.0087)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5933 (var=0.0443) (Δ vs real +0.1767)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8502 (var=0.0111)
- pi_2: 0.1467 (var=0.0087)
- pi_3: 0.4692 (var=0.0436)
- pi_4: 0.2250 (var=0.0140)
- pi_5: 0.3848 (var=0.0142)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    diff = a_sums - b_sums
    mask = diff != 0
    
    if not mask.any():
        return 0.5
        
    tallying_pred = (diff < 0).astype(int)
    accuracy = (data.loc[mask, 'response'] == tallying_pred[mask]).mean()
    
    return float(accuracy)
```

**Observed (real) value:** 0.6044 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4111 (var=0.0595) (Δ vs real -0.1933)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8578 (var=0.0079)
- pi_1: 0.1294 (var=0.0110)
- pi_3: 0.7206 (var=0.0372)
- pi_4: 0.7922 (var=0.0109)
- pi_5: 0.6494 (var=0.0221)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying sees a tie (equal number of features)
    tied_trials = sum_a == sum_b
    
    if tied_trials.sum() == 0:
        return 0.5
        
    # Calculate the proportion of times Option A was chosen in these tied trials
    # (Response 0 means Option A was chosen)
    prob_a_chosen = (data.loc[tied_trials, 'response'] == 0).mean()
    
    return float(prob_a_chosen)
```

**Observed (real) value:** 0.5611 (var=0.0131)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7833 (var=0.0135) (Δ vs real +0.2222)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8378 (var=0.0112)
- pi_2: 0.4872 (var=0.0059)
- pi_1: 0.8583 (var=0.0086)
- pi_4: 0.5689 (var=0.0136)
- pi_5: 0.7311 (var=0.0098)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create a string representation of option A's features to identify trials safely
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 4: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    p_a_t4 = 1.0 - data.loc[data['A_str'] == '11000', 'response'].mean()
    
    # Trial 5: A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 0, 0]
    p_a_t5 = 1.0 - data.loc[data['A_str'] == '10011', 'response'].mean()
    
    # Return the sum of probabilities of choosing A in these two trials
    return float(p_a_t4 + p_a_t5)
```

**Observed (real) value:** 1.0350 (var=0.0694)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.5012 (var=0.0979) (Δ vs real +0.4662)
**Other theories' values on this metric (for reference):**
- pi_2: 0.9800 (var=0.0141)
- pi_3: 1.4888 (var=0.0953)
- pi_1: 1.7412 (var=0.0352)
- pi_4: 1.0550 (var=0.0349)
- pi_5: 1.2200 (var=0.0368)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # a_cue_0 is 1 if option A has the most valid cue, 0 otherwise.
    a_cue_0 = data['option_a_ratings'].apply(lambda x: x[0])
    # response is 0 if A was chosen, 1 if B was chosen.
    # If response == a_cue_0, the subject chose the option that LACKS the most valid cue.
    # For example, if A has it (1) and they chose B (1), they chose the option without it.
    # If A lacks it (0) and they chose A (0), they chose the option without it.
    return float((data['response'] == a_cue_0).mean())
```

**Observed (real) value:** 0.5792 (var=0.0095)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3254 (var=0.0313) (Δ vs real -0.2538)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3456 (var=0.0583)
- pi_4: 0.7350 (var=0.0125)
- pi_1: 0.1410 (var=0.0078)
- pi_2: 0.8802 (var=0.0080)
- pi_5: 0.5790 (var=0.0152)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert rating lists to string representations to identify specific trials
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify the 'linear-tie' trials
    # Trial 1: Cues 1 & 4 vs Cues 2 & 3 (sum of validities = 1.5 for both)
    mask_t1 = (data['A_str'] == '10010') & (data['B_str'] == '01100')
    # Trial 5: Cues 2 & 4 vs Cues 1 & 5 (sum of validities = 1.4 for both)
    mask_t5 = (data['A_str'] == '01010') & (data['B_str'] == '10001')
    
    dev = 0.0
    if mask_t1.sum() > 0:
        p_A_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
        dev += abs(p_A_t1 - 0.5)
        
    if mask_t5.sum() > 0:
        p_A_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
        dev += abs(p_A_t5 - 0.5)
        
    return float(dev)
```

**Observed (real) value:** 0.0167 (var=0.0257)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4600 (var=0.0792) (Δ vs real +0.4433)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0333 (var=0.0252)
- pi_3: 0.2750 (var=0.0441)
- pi_1: 0.6983 (var=0.0516)
- pi_2: 0.0567 (var=0.0222)
- pi_5: 0.2633 (var=0.0436)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    target_mask = a_tuples.isin([(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 0, 1, 0, 0)])
    target_data = data[target_mask]
    if len(target_data) == 0:
        return 0.0
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.4058 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6650 (var=0.0380) (Δ vs real +0.2592)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4204 (var=0.0242)
- pi_4: 0.2704 (var=0.0135)
- pi_1: 0.8154 (var=0.0145)
- pi_2: 0.1304 (var=0.0085)
- pi_3: 0.4771 (var=0.0514)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    
    has_nonzero = np.any(diff != 0, axis=1)
    first_nonzero_idx = np.argmax(diff != 0, axis=1)
    
    ttb_sign = diff[np.arange(len(diff)), first_nonzero_idx]
    ttb_choice = np.where(ttb_sign > 0, 0, 1)
    ttb_choice = np.where(has_nonzero, ttb_choice, -1)
    
    tal_diff = np.sum(a_mat, axis=1) - np.sum(b_mat, axis=1)
    tal_choice = np.where(tal_diff > 0, 0, np.where(tal_diff < 0, 1, -1))
    
    wadd_diff = np.sum(val * a_mat, axis=1) - np.sum(val * b_mat, axis=1)
    wadd_choice = np.where(wadd_diff > 0, 0, np.where(wadd_diff < 0, 1, -1))
    
    mask = (tal_choice == wadd_choice) & (tal_choice != -1) & (ttb_choice != -1) & (ttb_choice != tal_choice)
    
    if not np.any(mask):
        return 0.0
        
    responses = data['response'].values[mask]
    ttb_choices = ttb_choice[mask]
    
    return float(np.mean(responses == ttb_choices))
```

**Observed (real) value:** 0.3950 (var=0.0100)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5804 (var=0.0486) (Δ vs real +0.1854)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2050 (var=0.0153)
- pi_5: 0.3450 (var=0.0207)
- pi_1: 0.8550 (var=0.0153)
- pi_2: 0.1271 (var=0.0106)
- pi_3: 0.4763 (var=0.0771)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Integrated Strategy Value Theory: Decision-makers evaluate options by integrating evidence from multiple heuristics (Take-The-Best, Tallying, and Weighted Additive) into a single continuous subjective value for each option. These integrated values are then compared via a stochastic decision process (softmax) to produce a choice. This allows non-compensatory signals to be smoothly combined with compensatory evidence, naturally accounting for both strong heuristic-aligned preferences and indifference in delicately balanced trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Take-The-Best (TTB) Score
    diff = a - b
    order = np.argsort(val)[::-1]
    score_ttb_a, score_ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            score_ttb_a, score_ttb_b = 1.0, 0.0
            break
        elif diff[idx] < 0:
            score_ttb_a, score_ttb_b = 0.0, 1.0
            break
            
    # Tallying Score
    score_tal_a = np.sum(a) / n_features
    score_tal_b = np.sum(b) / n_features
    
    # Weighted Additive (WADD) Score
    sum_val = np.sum(val)
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights
    w_sum = w_ttb + w_tal + w_wadd
    if w_sum > 0:
        w_ttb /= w_sum
        w_tal /= w_sum
        w_wadd /= w_sum
    else:
        w_ttb = w_tal = w_wadd = 1.0 / 3.0
        
    # Integrated Option Values
    score_a = w_ttb * score_ttb_a + w_tal * score_tal_a + w_wadd * score_wadd_a
    score_b = w_ttb * score_ttb_b + w_tal * score_tal_b + w_wadd * score_wadd_b
    
    scores = np.array([score_a, score_b])
    
    # Single Softmax Decision Process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- w_ttb: [0.0, 1.0]
- w_tal: [0.0, 1.0]
- w_wadd: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Based on the arbiter's feedback, we propose an Integrated Strategy Value Theory. Instead of probabilistically sampling a distinct heuristic on each trial (like pi_5), this theory computes a continuous subjective value for each option by integrating normalized evidence from non-compensatory (TTB), unweighted compensatory (Tallying), and weighted compensatory (WADD) strategies. The integrated values are then transformed into choice probabilities via a softmax function. This allows the model to capture TTB-aligned choices in cases where TTB strongly dominates, while gracefully predicting indifference (P=0.5) in delicately balanced trials where conflicting signals offset each other, overcoming the rigid predictions of discrete strategy sampling.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate model implements the Integrated Strategy Value Theory and was accepted by the gate, but it still shows significant deviations from the empirical data. Notably, in Experiment 6 (linear-tie trials), the observed deviation from 0.5 is near zero (0.0167), but the candidate predicts a massive deviation (0.4600). It also overpredicts TTB-aligned choices in Experiments 1 and 4. The root cause is that the TTB score is implemented as a hard 1.0 or 0.0. When WADD and Tallying are tied or closely matched, this hard TTB signal completely dominates the integrated value, forcing a deterministic choice rather than the observed indifference.
Rationale: To fix the over-determinism on delicately balanced trials without abandoning the Integrated Strategy Value family, the TTB signal needs to be softened so it is commensurate with the compensatory signals. Modify the TTB score so that it returns the normalized validity of the discriminating cue (e.g., `score_ttb_a = val[idx] / np.sum(val)`) rather than a hard 1.0. This scales the TTB evidence by the actual strength of the cue breaking the tie. Additionally, reduce the upper bound of the `beta` parameter (e.g., to 10.0) to prevent the softmax from amplifying small integrated value differences into hard choices.

**Outcome of this advice:** iter 1 candidate loss=0.4193 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Softened Integrated Strategy Value Theory: Decision-makers evaluate options by integrating evidence from multiple heuristics (Take-The-Best, Tallying, and Weighted Additive) into a single continuous subjective value for each option. To prevent non-compensatory heuristics from disproportionately dominating the integrated value, the TTB signal is softened by scaling it according to the normalized validity of the discriminating cue. These integrated values are then compared via a stochastic decision process (softmax) to produce a choice. This allows the model to gracefully capture indifference in delicately balanced trials while still reflecting heuristic-aligned preferences.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    n_features = len(val)\n    sum_val = np.sum(val)\n    \n    # Take-The-Best (TTB) Score (softened)\n    diff = a - b\n    order = np.argsort(val)[::-1]\n    score_ttb_a, score_ttb_b = 0.0, 0.0\n    for idx in order:\n        if diff[idx] > 0:\n            score_ttb_a, score_ttb_b = val[idx] / sum_val, 0.0\n            break\n        elif diff[idx] < 0:\n            score_ttb_a, score_ttb_b = 0.0, val[idx] / sum_val\n            break\n            \n    # Tallying Score\n    score_tal_a = np.sum(a) / n_features\n    score_tal_b = np.sum(b) / n_features\n    \n    # Weighted Additive (WADD) Score\n    score_wadd_a = np.sum(val * a) / sum_val\n    score_wadd_b = np.sum(val * b) / sum_val\n    \n    w_ttb = float(parameters[\"w_ttb\"])\n    w_tal = float(parameters[\"w_tal\"])\n    w_wadd = float(parameters[\"w_wadd\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Normalize weights\n    w_sum = w_ttb + w_tal + w_wadd\n    if w_sum > 0:\n        w_ttb /= w_sum\n        w_tal /= w_sum\n        w_wadd /= w_sum\n    else:\n        w_ttb = w_tal = w_wadd = 1.0 / 3.0\n        \n    # Integrated Option Values\n    score_a = w_ttb * score_ttb_a + w_tal * score_tal_a + w_wadd * score_wadd_a\n    score_b = w_ttb * score_ttb_b + w_tal * score_tal_b + w_wadd * score_wadd_b\n    \n    scores = np.array([score_a, score_b])\n    \n    # Single Softmax Decision Process\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p = e / np.sum(e)\n    \n    # Lapse rate\n    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "w_ttb": "[0.0, 1.0]",
    "w_tal": "[0.0, 1.0]",
    "w_wadd": "[0.0, 1.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we modified the TTB score from a hard [1.0, 0.0] assignment to a softened version that assigns the normalized validity of the discriminating cue (val[idx] / sum_val). This prevents the TTB signal from overly dominating the integrated value when the compensatory signals are closely tied, mitigating the large deviation from 0.5 observed in Experiment 6. Additionally, we reduced the upper bound of the softmax temperature parameter `beta` from 20.0 to 10.0 to prevent the model from amplifying small integrated value differences into deterministic choices."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13881,
  "candidates_token_count": 1174,
  "total_token_count": 16435
}
```
