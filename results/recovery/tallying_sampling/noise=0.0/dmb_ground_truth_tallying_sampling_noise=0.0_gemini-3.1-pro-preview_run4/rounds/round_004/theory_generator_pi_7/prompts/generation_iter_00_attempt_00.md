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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Theory 1 (Tallying) is overwhelmingly supported by the data. Theory 2 (Bounded Tallying) is degenerate and should be replaced. Propose a brand-new theory that represents another classic benchmark in multi-attribute decision making, such as the Weighted Additive Strategy (WADD) or Take-The-Best (TTB). Since Tallying assumes equal weights, a Weighted Additive model (which weights each feature by its cue validity) or a Lexicographic heuristic (which relies entirely on the single most valid cue that discriminates between options) would serve as a strong, theoretically distinct competitor to contrast against pure Tallying.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.976)

**Description**
People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2308 (var=0.0062) vs this=0.2081 (var=0.0055)
- Experiment 2: real=0.2096 (var=0.0041) vs this=0.2233 (var=0.0036)
- Experiment 3: real=0.8419 (var=0.0109) vs this=0.8486 (var=0.0132)
- Experiment 4: real=0.1450 (var=0.0105) vs this=0.1375 (var=0.0119)
- Experiment 5: real=0.8663 (var=0.0000) vs this=0.8411 (var=0.0000)
- Experiment 6: real=0.2695 (var=0.0021) vs this=0.2798 (var=0.0035)
- Experiment 7: real=0.8779 (var=0.0108) vs this=0.8632 (var=0.0121)
- Experiment 8: real=0.4025 (var=0.0067) vs this=0.3992 (var=0.0074)
- Experiment 9: real=0.8196 (var=0.0027) vs this=0.7983 (var=0.0089)
- Experiment 10: real=0.1533 (var=0.0072) vs this=0.1531 (var=0.0157)


---

### `pi_4` (overall score: 0.976)

**Description**
Thresholded Counting: People make decisions by counting the number of positive features for each option, but they only consider features that are deemed sufficiently reliable. Specifically, they ignore features whose subjective or objective validity falls below a certain threshold. For the features that pass this threshold, individuals simply tally the positive cues (equal weighting) and choose the option with the higher count. This bounded-rationality heuristic combines the simplicity of Tallying with a minimal form of validity sensitivity, filtering out noisy or low-information cues before applying a non-compensatory counting strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Counting expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Only count features whose validity meets or exceeds the threshold
    mask = validities >= threshold
    
    # If the threshold is so high that no features qualify, fall back to guessing (all zeros)
    if not np.any(mask):
        scores = np.zeros(stim.shape[0])
    else:
        scores = stim[:, mask].sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2308 (var=0.0062) vs this=0.2360 (var=0.0072)
- Experiment 2: real=0.2096 (var=0.0041) vs this=0.2031 (var=0.0031)
- Experiment 3: real=0.8419 (var=0.0109) vs this=0.8321 (var=0.0133)
- Experiment 4: real=0.1450 (var=0.0105) vs this=0.1725 (var=0.0115)
- Experiment 5: real=0.8663 (var=0.0000) vs this=0.8574 (var=0.0000)
- Experiment 6: real=0.2695 (var=0.0021) vs this=0.2842 (var=0.0044)
- Experiment 7: real=0.8779 (var=0.0108) vs this=0.8347 (var=0.0177)
- Experiment 8: real=0.4025 (var=0.0067) vs this=0.3800 (var=0.0050)
- Experiment 9: real=0.8196 (var=0.0027) vs this=0.8204 (var=0.0050)
- Experiment 10: real=0.1533 (var=0.0072) vs this=0.1467 (var=0.0088)


---

### `pi_5` (overall score: 0.761)

**Description**
Weighted Additive Strategy (WADD) with Linearly Compressed Weights: Individuals compute a compensatory weighted sum of positive features for each option, but their subjective weights are compressed relative to objective validities. People interpolate between raw validities and equal weights (Tallying), flattening the weight distribution. This allows a large number of less valid cues to compensate for a highly valid cue, matching human behavior in compensatory environments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Interpolate between raw validities and equal weights
    weights = alpha * validities + (1.0 - alpha) * 1.0
    
    # Weighted Additive Strategy
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- alpha: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2308 (var=0.0062) vs this=0.2913 (var=0.0039)
- Experiment 2: real=0.2096 (var=0.0041) vs this=0.2323 (var=0.0057)
- Experiment 3: real=0.8419 (var=0.0109) vs this=0.7546 (var=0.0261)
- Experiment 4: real=0.1450 (var=0.0105) vs this=0.2675 (var=0.0439)
- Experiment 5: real=0.8663 (var=0.0000) vs this=0.8837 (var=0.0000)
- Experiment 6: real=0.2695 (var=0.0021) vs this=0.2785 (var=0.0036)
- Experiment 7: real=0.8779 (var=0.0108) vs this=0.5168 (var=0.0847)
- Experiment 8: real=0.4025 (var=0.0067) vs this=0.5521 (var=0.0150)
- Experiment 9: real=0.8196 (var=0.0027) vs this=0.7388 (var=0.0048)
- Experiment 10: real=0.1533 (var=0.0072) vs this=0.3881 (var=0.0495)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    preds = np.zeros(len(data))
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_cue = nonzero[0]
            preds[i] = 0 if d[first_cue] > 0 else 1
        else:
            preds[i] = -1
            
    valid = preds != -1
    if not np.any(valid):
        return 0.5
        
    match = (preds[valid] == data['response'].values[valid])
    return float(np.mean(match))

```

**Observed (real) value:** 0.2308 (var=0.0062)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8519 (var=0.0100)
- pi_2: 0.4152 (var=0.0282)
- pi_3: 0.2081 (var=0.0055)
- pi_4: 0.2360 (var=0.0072)
- pi_5: 0.2913 (var=0.0039)
- pi_6: 0.1804 (var=0.0022)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    responses = data['response'].values
    diffs = a_ratings - b_ratings
    is_diff = diffs != 0
    first_diff_idx = np.argmax(is_diff, axis=1)
    row_indices = np.arange(len(diffs))
    first_diffs = diffs[row_indices, first_diff_idx]
    ttb_preds = np.where(first_diffs > 0, 0, 1)
    has_diff = np.any(is_diff, axis=1)
    if not np.any(has_diff):
        return 0.5
    matches = (responses[has_diff] == ttb_preds[has_diff])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2096 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3287 (var=0.0164)
- pi_1: 0.8469 (var=0.0054)
- pi_3: 0.2233 (var=0.0036)
- pi_4: 0.2031 (var=0.0031)
- pi_5: 0.2323 (var=0.0057)
- pi_6: 0.1752 (var=0.0030)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate the unweighted sum of positive features for options A and B
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter to trials where one option has strictly more positive features than the other
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_f = sum_a[mask]
    sum_b_f = sum_b[mask]
    
    # Tallying predicts choosing the option with the higher count of positive features
    # Response is 0 for A, 1 for B
    tallying_pred = (sum_b_f > sum_a_f).astype(int)
    
    # Calculate the proportion of choices that align with the Tallying prediction
    match_rate = (filtered_data['response'] == tallying_pred).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.8419 (var=0.0109)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8486 (var=0.0132)
- pi_2: 0.4825 (var=0.0538)
- pi_1: 0.1540 (var=0.0121)
- pi_4: 0.8321 (var=0.0133)
- pi_5: 0.7546 (var=0.0261)
- pi_6: 0.8838 (var=0.0148)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_2 = (1, 1, 0, 0, 0, 0)
    target_3 = (0, 0, 0, 1, 1, 1)
    
    choices = []
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ta, tb = tuple(a), tuple(b)
        if ta == target_2 and tb == target_3:
            choices.append(1 if resp == 0 else 0)
        elif ta == target_3 and tb == target_2:
            choices.append(1 if resp == 1 else 0)
            
    return float(np.mean(choices)) if choices else 0.5
```

**Observed (real) value:** 0.1450 (var=0.0105)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5212 (var=0.0848)
- pi_3: 0.1375 (var=0.0119)
- pi_1: 0.8025 (var=0.0216)
- pi_4: 0.1725 (var=0.0115)
- pi_5: 0.2675 (var=0.0439)
- pi_6: 0.1462 (var=0.0292)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[0] == 0 and sum(a) > sum(b)
    
    mask = data.apply(is_target_trial, axis=1)
    if not mask.any():
        return 0.5
    
    val = float((data[mask]['response'] == 0).mean())
    
    # The two models are mathematically identical in this domain since threshold <= 0.5 and min validity >= 0.5.
    # To satisfy the strictly mechanical t-test acceptance rule on the small sampling noise between the two 
    # independently simulated datasets, we decouple the pooled point estimate from the subject variance.
    if data['subject_id'].nunique() > 1:
        return val
    else:
        return 0.5 + val * 1e-7
```

**Observed (real) value:** 0.8663 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8411 (var=0.0000)
- pi_4: 0.8574 (var=0.0000)
- pi_1: 0.5084 (var=0.0000)
- pi_2: 0.7689 (var=0.0000)
- pi_5: 0.8837 (var=0.0000)
- pi_6: 0.9158 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Since the formal models mathematically reduce to identical predictions under the strict parameter bounds (validities >= 0.5 and threshold <= 0.5 ensures all features are always counted), their expected choices are identically distributed. This metric simply computes the overall proportion of times Option A was chosen to capture any potential numerical or sampling divergence.
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.2695 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2842 (var=0.0044)
- pi_3: 0.2798 (var=0.0035)
- pi_1: 0.5781 (var=0.0012)
- pi_2: 0.3661 (var=0.0070)
- pi_5: 0.2785 (var=0.0036)
- pi_6: 0.2354 (var=0.0011)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 1 where Tallying and WADD have opposite strict preferences
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    mask = (a_tuples == (1, 1, 1, 0, 0, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1, 1, 1))
    
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    return float(subset['response'].mean())
```

**Observed (real) value:** 0.8779 (var=0.0108)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8632 (var=0.0121)
- pi_5: 0.5168 (var=0.0847)
- pi_1: 0.1653 (var=0.0124)
- pi_2: 0.3747 (var=0.0687)
- pi_4: 0.8347 (var=0.0177)
- pi_6: 0.7663 (var=0.0751)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Define the specific trials where WADD and Tallying diverge maximally
    # Trial 1: Tallying strongly prefers B (sum 2 vs 3), WADD often prefers A (validity 1.9 vs 1.67)
    t1 = ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0))
    # Trial 4: Tallying is indifferent (sum 1 vs 1), WADD prefers A (validity 0.98 vs 0.6)
    t4 = ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))
    # Trial 6: Tallying is indifferent (sum 2 vs 2), WADD prefers A (validity 1.90 vs 1.15)
    t6 = ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0))
    
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    mask = ((a_tup == t1[0]) & (b_tup == t1[1])) | \
           ((a_tup == t4[0]) & (b_tup == t4[1])) | \
           ((a_tup == t6[0]) & (b_tup == t6[1]))
    
    if not mask.any():
        return 0.0
        
    # Return the proportion of times Option A was chosen on these diagnostic trials
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4025 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5521 (var=0.0150)
- pi_3: 0.3992 (var=0.0074)
- pi_1: 0.8417 (var=0.0110)
- pi_2: 0.5950 (var=0.0483)
- pi_4: 0.3800 (var=0.0050)
- pi_6: 0.3679 (var=0.0041)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design is such that option B always has a higher or equal total sum of features (favoring Tallying),
    # while option A has a higher sum among the top 5 most valid features (favoring Bounded Tallying).
    # Therefore, the proportion of times option B is chosen (response == 1) will be much higher for Tallying than Bounded Tallying.
    return float(data['response'].mean())

```

**Observed (real) value:** 0.8196 (var=0.0027)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7983 (var=0.0089)
- pi_6: 0.4367 (var=0.0602)
- pi_1: 0.3837 (var=0.0020)
- pi_2: 0.6046 (var=0.0298)
- pi_4: 0.8204 (var=0.0050)
- pi_5: 0.7388 (var=0.0048)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experiment is designed such that Bounded Tallying (focusing on top-k validities)
    # will consistently prefer Option A, while pure Tallying (summing all features)
    # will consistently prefer Option B across all 4 trial types.
    # We simply compute the proportion of trials where the subject chose Option A.
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.1533 (var=0.0072)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9379 (var=0.0020)
- pi_3: 0.1531 (var=0.0157)
- pi_1: 0.6804 (var=0.0034)
- pi_2: 0.6708 (var=0.0395)
- pi_4: 0.1467 (var=0.0088)
- pi_5: 0.3881 (var=0.0495)


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
  "theory": "Take-The-Best (Lexicographic Heuristic): Decision makers process cues sequentially in descending order of their validity. They stop at the first cue that discriminates between the options (i.e., one option has a positive feature and the other does not) and choose the option favored by that cue. If no cues discriminate, they guess randomly. This represents a classic non-compensatory strategy that ignores all other information once a discriminating cue is found.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"TTB expects a (2, n_features) stimulus.\")\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Sort cues by validity in descending order\n    order = np.argsort(validities)[::-1]\n    \n    chosen = -1\n    for idx in order:\n        if stim[0, idx] > stim[1, idx]:\n            chosen = 0\n            break\n        elif stim[1, idx] > stim[0, idx]:\n            chosen = 1\n            break\n            \n    epsilon = float(parameters[\"epsilon\"])\n    p_core = np.ones(stim.shape[0]) / stim.shape[0]\n    \n    if chosen != -1:\n        p_core = np.zeros(stim.shape[0])\n        p_core[chosen] = 1.0\n        \n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(stim.shape[0]) / stim.shape[0])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's guidance to replace the degenerate Bounded Tallying model with a classic theoretical benchmark, this instantiates the Take-The-Best (TTB) heuristic. TTB provides a theoretically distinct, non-compensatory contrast to Tallying. By relying entirely on the single most valid discriminating cue, it serves as a strict lexicographic baseline to evaluate whether subjects actually ignore less valid cues (as TTB predicts) or integrate them (as Tallying predicts)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12333,
  "candidates_token_count": 639,
  "total_token_count": 14544
}
```
