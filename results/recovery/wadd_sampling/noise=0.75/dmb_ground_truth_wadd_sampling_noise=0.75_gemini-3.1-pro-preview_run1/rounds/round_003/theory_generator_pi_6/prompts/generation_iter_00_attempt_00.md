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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'High-Lapse Take-The-Best (TTB)' or 'Noisy Lexicographic' theory. This theory should posit that subjects find the task tedious and guess randomly on a large majority of trials (high epsilon, e.g., > 0.6). However, when they do engage, instead of tallying all features (as in Theory 2) or integrating all validities (as in the old Theory 1), they simply look at the single most valid cue that discriminates the options and choose based on that. This maintains the high noise floor required to fit the data while testing a lexicographic rather than compensatory mechanism for the engaged trials.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.967)

**Description**
High-Lapse Tallying Theory: Subjects find the multi-attribute binary feature task cognitively demanding, leading to a very high rate of random guessing. However, when they do engage with the stimulus, they rely on a simple unweighted tally of positive features (Tallying) rather than integrating complex cue validities. This produces a very weak but non-zero behavioral signal that slightly favors options with a higher simple count of positive features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 5.0]
- epsilon: [0.7, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4737 (var=0.0046) vs this=0.4398 (var=0.0056)
- Experiment 2: real=0.5479 (var=0.0020) vs this=0.5650 (var=0.0035)
- Experiment 3: real=0.4800 (var=0.0233) vs this=0.4483 (var=0.0116)
- Experiment 4: real=-0.0025 (var=0.0589) vs this=0.0650 (var=0.0244)
- Experiment 5: real=0.5743 (var=0.0037) vs this=0.5471 (var=0.0035)
- Experiment 6: real=0.1371 (var=0.0006) vs this=0.1269 (var=0.0010)
- Experiment 7: real=0.5300 (var=0.0073) vs this=0.5246 (var=0.0066)
- Experiment 8: real=0.5877 (var=0.0189) vs this=0.5831 (var=0.0206)


---

### `pi_4` (overall score: 0.810)

**Description**
Random Guessing (Zero-Intelligence) Theory: Without trial-by-trial feedback, subjects find the multi-attribute binary feature task too cognitively demanding or unengaging. As a result, they do not attempt to integrate the cue validities or compare the feature values. Instead, they simply guess uniformly at random on every trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # The model assumes pure random guessing, ignoring stimulus and history entirely.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
(none)

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4737 (var=0.0046) vs this=0.4964 (var=0.0023)
- Experiment 2: real=0.5479 (var=0.0020) vs this=0.5008 (var=0.0022)
- Experiment 3: real=0.4800 (var=0.0233) vs this=0.4992 (var=0.0084)
- Experiment 4: real=-0.0025 (var=0.0589) vs this=-0.0212 (var=0.0279)
- Experiment 5: real=0.5743 (var=0.0037) vs this=0.4940 (var=0.0020)
- Experiment 6: real=0.1371 (var=0.0006) vs this=0.1119 (var=0.0012)
- Experiment 7: real=0.5300 (var=0.0073) vs this=0.4742 (var=0.0056)
- Experiment 8: real=0.5877 (var=0.0189) vs this=0.5062 (var=0.0197)


---

### `pi_3` (overall score: 0.225)

**Description**
Generalized Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a compensatory weighted sum of their features. However, instead of using raw cue validities as weights, individuals scale the validities non-linearly using a parameter gamma. This scaling allows the decision process to smoothly interpolate between Tallying (gamma=0, where all cues are weighted equally), standard WADD (gamma=1), and Take The Best (gamma -> infinity, where the most valid cue dominates). The final choice is made probabilistically via a softmax function over the computed option values, mixed with a random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4737 (var=0.0046) vs this=0.4486 (var=0.0430)
- Experiment 2: real=0.5479 (var=0.0020) vs this=0.5290 (var=0.0541)
- Experiment 3: real=0.4800 (var=0.0233) vs this=0.8067 (var=0.0433)
- Experiment 4: real=-0.0025 (var=0.0589) vs this=-0.2850 (var=0.2349)
- Experiment 5: real=0.5743 (var=0.0037) vs this=0.7469 (var=0.0099)
- Experiment 6: real=0.1371 (var=0.0006) vs this=0.3000 (var=0.0064)
- Experiment 7: real=0.5300 (var=0.0073) vs this=0.8029 (var=0.0117)
- Experiment 8: real=0.5877 (var=0.0189) vs this=0.8769 (var=0.0133)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    valid_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB compares cues in order 0 to n_features-1
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_matches += 1
            valid_trials += 1
            
    return float(ttb_matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4737 (var=0.0046)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8528 (var=0.0074)
- pi_2: 0.1516 (var=0.0131)
- pi_3: 0.4486 (var=0.0430)
- pi_4: 0.4964 (var=0.0023)
- pi_5: 0.4398 (var=0.0056)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    tally_a = np.sum(a_ratings > b_ratings, axis=1)
    tally_b = np.sum(b_ratings > a_ratings, axis=1)
    valid = tally_a != tally_b
    if not np.any(valid):
        return 0.5
    pred = np.where(tally_a > tally_b, 0, 1)
    return float((pred[valid] == data['response'].values[valid]).mean())
```

**Observed (real) value:** 0.5479 (var=0.0020)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8823 (var=0.0065)
- pi_1: 0.1715 (var=0.0138)
- pi_3: 0.5290 (var=0.0541)
- pi_4: 0.5008 (var=0.0022)
- pi_5: 0.5650 (var=0.0035)

### Experiment 3
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    return wadd_choices / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0233)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8067 (var=0.0433)
- pi_2: 0.1283 (var=0.0105)
- pi_1: 0.8892 (var=0.0089)
- pi_4: 0.4992 (var=0.0084)
- pi_5: 0.4483 (var=0.0116)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify Trial 3: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Tallying favors A (3 vs 2), meaning response should be closer to 0.
    # WADD with gamma > 1 favors B (due to the 0.95 and 0.85 validities), meaning response closer to 1.
    is_trial_3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 4: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 1]
    # Tallying favors B (2 vs 3), meaning response should be closer to 1.
    # WADD with gamma > 1 favors A (due to the 0.95 validity), meaning response closer to 0.
    is_trial_4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    
    resp_3 = data.loc[is_trial_3, 'response'].mean()
    resp_4 = data.loc[is_trial_4, 'response'].mean()
    
    if pd.isna(resp_3):
        resp_3 = 0.5
    if pd.isna(resp_4):
        resp_4 = 0.5
        
    # Tallying: resp_4 (high) - resp_3 (low) -> Positive
    # WADD: resp_4 (low) - resp_3 (high) -> Negative
    return float(resp_4 - resp_3)
```

**Observed (real) value:** -0.0025 (var=0.0589)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6825 (var=0.0634)
- pi_3: -0.2850 (var=0.2349)
- pi_1: -0.7013 (var=0.0510)
- pi_4: -0.0212 (var=0.0279)
- pi_5: 0.0650 (var=0.0244)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    
    mask = score_a != score_b
    if not np.any(mask):
        return 0.5
        
    expected = (score_b > score_a).astype(int)
    actual = data['response'].values
    
    return float(np.mean(expected[mask] == actual[mask]))
```

**Observed (real) value:** 0.5743 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7469 (var=0.0099)
- pi_4: 0.4940 (var=0.0020)
- pi_1: 0.6543 (var=0.0026)
- pi_2: 0.7102 (var=0.0056)
- pi_5: 0.5471 (var=0.0035)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    keys = [tuple(a) + tuple(b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_key': keys,
        'response': data['response']
    })
    grouped = df.groupby(['subject_id', 'trial_key'])['response'].mean()
    return float((grouped - 0.5).abs().mean())
```

**Observed (real) value:** 0.1371 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1119 (var=0.0012)
- pi_3: 0.3000 (var=0.0064)
- pi_1: 0.3504 (var=0.0079)
- pi_2: 0.2860 (var=0.0046)
- pi_5: 0.1269 (var=0.0010)

### Experiment 7
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    
    mask = (a_sums >= b_sums) & (a_first == 1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.5300 (var=0.0073)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8029 (var=0.0117)
- pi_5: 0.5246 (var=0.0066)
- pi_1: 0.8562 (var=0.0085)
- pi_2: 0.6171 (var=0.0056)
- pi_4: 0.4742 (var=0.0056)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5, where Option A is strictly dominant (all 1s) and Option B has all 0s.
    is_trial_5 = data['option_a_ratings'].apply(lambda x: sum(x) == 5)
    
    if not is_trial_5.any():
        return 0.5
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float((data.loc[is_trial_5, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5877 (var=0.0189)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5831 (var=0.0206)
- pi_3: 0.8769 (var=0.0133)
- pi_1: 0.8108 (var=0.0147)
- pi_2: 0.8631 (var=0.0221)
- pi_4: 0.5062 (var=0.0197)


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
  "theory": "High-Lapse Take-The-Best (TTB) Theory: Subjects generally find the multi-attribute choice task tedious without trial-by-trial feedback, leading to a very high rate of random guessing (high lapse rate). However, on the small fraction of trials where they actively engage, they do not integrate all features or tally them. Instead, they use a fast-and-frugal lexicographic heuristic (Take-The-Best): they inspect cues in descending order of their validity and make a deterministic choice based entirely on the single most valid cue that discriminates between the two options.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Sort cues by validity in descending order\n    order = np.argsort(val)[::-1]\n    \n    ttb_choice = -1\n    for idx in order:\n        if a[idx] > b[idx]:\n            ttb_choice = 0\n            break\n        elif b[idx] > a[idx]:\n            ttb_choice = 1\n            break\n            \n    if ttb_choice == 0:\n        p_core = np.array([1.0, 0.0])\n    elif ttb_choice == 1:\n        p_core = np.array([0.0, 1.0])\n    else:\n        p_core = np.array([0.5, 0.5])\n        \n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Mix the deterministic TTB choice with a high lapse rate (random guessing)\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "epsilon": "[0.6, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this model instantiates a 'High-Lapse Take-The-Best' mechanism. It acknowledges the high noise floor in the experimental data by retaining a large lapse rate (epsilon > 0.6), but replaces the compensatory tallying mechanism of Theory 2 with a purely non-compensatory, lexicographic one. When subjects engage, they simply identify the most valid cue that discriminates the options and choose accordingly, completely ignoring all other cues. This directly tests whether the weak behavioral signal present in the data is driven by a fast-and-frugal lexicographic process rather than unweighted feature summation."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9962,
  "candidates_token_count": 746,
  "total_token_count": 13286
}
```
