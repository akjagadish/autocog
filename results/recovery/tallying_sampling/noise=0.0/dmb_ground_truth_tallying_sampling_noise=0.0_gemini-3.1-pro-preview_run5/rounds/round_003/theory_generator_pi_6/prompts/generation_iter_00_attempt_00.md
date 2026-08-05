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
- The recommendation below acts on THEORY 2 (= `pi_5`).

Theory 2 (Tallying with Lexicographic Tie-Breaking) is consistently falsified on tie trials, as subjects simply guess rather than systematically relying on the highest-validity differing feature. Replace Theory 2 with a brand-new theory to serve as a strong alternative baseline, such as the classic Weighted Additive (WADD) model. WADD posits that subjects compute a fully compensatory weighted sum of features using the provided validities as weights (or log-odds derived from them), rather than an equal-weight tally. This will test whether subjects are truly ignoring validities entirely (as in Tallying) or integrating them continuously.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.989)

**Description**
Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1546 (var=0.0058) vs this=0.1429 (var=0.0080)
- Experiment 2: real=0.2791 (var=0.0051) vs this=0.2791 (var=0.0029)
- Experiment 3: real=0.8289 (var=0.0141) vs this=0.8489 (var=0.0138)
- Experiment 4: real=-0.7833 (var=0.0417) vs this=-0.7400 (var=0.0307)
- Experiment 5: real=0.0096 (var=0.0001) vs this=0.0074 (var=0.0001)
- Experiment 6: real=0.5000 (var=0.0594) vs this=0.4167 (var=0.0362)
- Experiment 7: real=15.0000 (var=9.7600) vs this=13.0000 (var=15.6724)
- Experiment 8: real=0.4867 (var=0.0028) vs this=0.4908 (var=0.0034)


---

### `pi_5` (overall score: 0.830)

**Description**
Tallying with Lexicographic Tie-Breaking posits that decision-makers primarily rely on a compensatory equal-weight tallying heuristic, choosing the option with the highest number of positive features. However, when options are tied in their feature tallies, decision-makers do not guess randomly. Instead, they break the tie by comparing the options on the single most valid feature where the options differ (a Take-The-Best mechanism). This hybrid approach perfectly mimics Tallying on unequal feature counts but provides a deterministic, validity-based resolution for ties.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Primary mechanism: Equal-weight tallying
    scores = np.sum(stim, axis=1)
    
    # Secondary mechanism: Lexicographic tie-breaking (Take-The-Best)
    if scores[0] == scores[1]:
        validities = np.asarray(parameters["validities"], dtype=float)
        w_tie = float(parameters["w_tie"])
        diff = stim[0] - stim[1]
        valid_diffs = np.where(diff != 0)[0]
        
        if len(valid_diffs) > 0:
            # Find the differing feature with the highest validity
            best_feature = valid_diffs[np.argmax(validities[valid_diffs])]
            if diff[best_feature] > 0:
                scores[0] += w_tie
            else:
                scores[1] += w_tie

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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
- validities: validities
- w_tie: [0.0, 0.05]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1546 (var=0.0058) vs this=0.1608 (var=0.0060)
- Experiment 2: real=0.2791 (var=0.0051) vs this=0.2809 (var=0.0035)
- Experiment 3: real=0.8289 (var=0.0141) vs this=0.8211 (var=0.0162)
- Experiment 4: real=-0.7833 (var=0.0417) vs this=-0.7533 (var=0.0561)
- Experiment 5: real=0.0096 (var=0.0001) vs this=0.0082 (var=0.0000)
- Experiment 6: real=0.5000 (var=0.0594) vs this=0.7500 (var=0.0506)
- Experiment 7: real=15.0000 (var=9.7600) vs this=155.0000 (var=35.2500)
- Experiment 8: real=0.4867 (var=0.0028) vs this=0.5600 (var=0.0062)


---

### `pi_4` (overall score: 0.612)

**Description**
Soft Threshold Tallying posits that decision-makers use a compensatory tallying heuristic, but down-weight cues that are less reliable. They evaluate each feature's subjective validity against a personal threshold using a smooth sigmoid function, creating a soft inclusion mask. Features that fall well below the threshold are effectively ignored, while those above are tallied with near-equal weight. This smooth transition provides a bridge between pure equal-weight tallying and selective attention to high-validity information, while allowing for continuous optimization.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Threshold Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Soft threshold using a logistic sigmoid to create a smooth inclusion mask
    weights = 1.0 / (1.0 + np.exp(-steepness * (validities - threshold)))
    
    # Tally the features weighted by the soft threshold mask
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
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
- threshold: [-5.0, 1.0]
- steepness: [1.0, 50.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1546 (var=0.0058) vs this=0.1504 (var=0.0107)
- Experiment 2: real=0.2791 (var=0.0051) vs this=0.3047 (var=0.0069)
- Experiment 3: real=0.8289 (var=0.0141) vs this=0.7856 (var=0.0577)
- Experiment 4: real=-0.7833 (var=0.0417) vs this=-0.5950 (var=0.1389)
- Experiment 5: real=0.0096 (var=0.0001) vs this=0.0188 (var=0.0017)
- Experiment 6: real=0.5000 (var=0.0594) vs this=1.0000 (var=0.0867)
- Experiment 7: real=15.0000 (var=9.7600) vs this=140.0000 (var=41.5600)
- Experiment 8: real=0.4867 (var=0.0028) vs this=0.5239 (var=0.0046)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of positive features for options A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Determine which option has the fewer number of positive features.
    # Since there are no ties in the sum of features in this design,
    # we can simply check if sum_a > sum_b. If true, B (1) has fewer features;
    # if false, A (0) has fewer features.
    smaller_option = (sum_a > sum_b).astype(int)
    
    # Check if the subject's response matches the option with fewer features
    match = (data['response'] == smaller_option).astype(float)
    
    return float(match.mean())
```

**Observed (real) value:** 0.1546 (var=0.0058)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6773 (var=0.0034)
- pi_2: 0.2381 (var=0.0111)
- pi_3: 0.1429 (var=0.0080)
- pi_4: 0.1504 (var=0.0107)
- pi_5: 0.1608 (var=0.0060)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = -1
        for j in range(5):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner != -1:
            if resp == ttb_winner:
                matches += 1
            total += 1
            
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.2791 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3611 (var=0.0157)
- pi_1: 0.8504 (var=0.0095)
- pi_3: 0.2791 (var=0.0029)
- pi_4: 0.3047 (var=0.0069)
- pi_5: 0.2809 (var=0.0035)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify critical trials where Tallying and WADD make opposite predictions.
    # Tallying prefers the option with more features (3 features).
    # WADD prefers the option with fewer but higher-validity features (2 features).
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
            data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
            
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
            data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
            
    # Tallying chooses A on t1 (response 0) and B on t2 (response 1)
    t1_tally_choices = (data.loc[is_t1, 'response'] == 0).sum()
    t2_tally_choices = (data.loc[is_t2, 'response'] == 1).sum()
    
    total_critical = is_t1.sum() + is_t2.sum()
    if total_critical == 0:
        return 0.5
        
    return float((t1_tally_choices + t2_tally_choices) / total_critical)
```

**Observed (real) value:** 0.8289 (var=0.0141)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8489 (var=0.0138)
- pi_2: 0.3622 (var=0.0993)
- pi_1: 0.1644 (var=0.0137)
- pi_4: 0.7856 (var=0.0577)
- pi_5: 0.8211 (var=0.0162)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_t2 = a_str == '11000'
    mask_t3 = a_str == '00111'
    
    p_a_t2 = (data.loc[mask_t2, 'response'] == 0).mean() if mask_t2.any() else 0.5
    p_a_t3 = (data.loc[mask_t3, 'response'] == 0).mean() if mask_t3.any() else 0.5
    
    return float(p_a_t2 - p_a_t3)
```

**Observed (real) value:** -0.7833 (var=0.0417)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2500 (var=0.3075)
- pi_3: -0.7400 (var=0.0307)
- pi_1: 0.7250 (var=0.0495)
- pi_4: -0.5950 (var=0.1389)
- pi_5: -0.7533 (var=0.0561)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    subjects = data['subject_id'].values
    
    count_a = a_ratings.sum(axis=1)
    count_b = b_ratings.sum(axis=1)
    
    # Filter out trials where both options have the same number of positive features
    mask = count_a != count_b
    if not np.any(mask):
        return 0.0
        
    a_ratings = a_ratings[mask]
    b_ratings = b_ratings[mask]
    responses = responses[mask]
    subjects = subjects[mask]
    count_a = count_a[mask]
    count_b = count_b[mask]
    
    # Binary indicator: did the subject choose the option with MORE positive features?
    chose_more = (((count_a > count_b) & (responses == 0)) | 
                  ((count_b > count_a) & (responses == 1))).astype(float)
                  
    # Create order-independent string keys for each unique trial pair
    a_str = np.array([''.join([str(int(x)) for x in row]) for row in a_ratings])
    b_str = np.array([''.join([str(int(x)) for x in row]) for row in b_ratings])
    keys = np.where(a_str < b_str, a_str + "_" + b_str, b_str + "_" + a_str)
    
    subject_metrics = []
    for subj in np.unique(subjects):
        subj_mask = subjects == subj
        subj_keys = keys[subj_mask]
        subj_chose_more = chose_more[subj_mask]
        
        unique_keys = np.unique(subj_keys)
        if len(unique_keys) < 2:
            continue
            
        # For each unique trial pair, calculate the proportion of times 
        # the subject chose the option with more features
        means = []
        for k in unique_keys:
            k_mask = subj_keys == k
            means.append(np.mean(subj_chose_more[k_mask]))
            
        # Compute the variance of these choice proportions across the different trial pairs
        subject_metrics.append(np.var(means, ddof=1))
        
    if not subject_metrics:
        return 0.0
        
    return float(np.mean(subject_metrics))
```

**Observed (real) value:** 0.0096 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0074 (var=0.0001)
- pi_4: 0.0188 (var=0.0017)
- pi_1: 0.0080 (var=0.0000)
- pi_2: 0.0695 (var=0.0024)
- pi_5: 0.0082 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    scores = []
    for subj, df_subj in data.groupby('subject_id'):
        a_str = df_subj['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
        b_str = df_subj['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
        
        # Trial 4: A=[1, 1, 1, 0, 0] vs B=[0, 0, 1, 1, 1] (Sums: 3 vs 3)
        t4 = (a_str == "11100") & (b_str == "00111")
        # Trial 8: A=[0, 0, 1, 1, 1] vs B=[1, 1, 1, 0, 0] (Sums: 3 vs 3)
        t8 = (a_str == "00111") & (b_str == "11100")
        
        p_a_t4 = df_subj.loc[t4, 'response'].eq(0).mean() if t4.any() else 0.5
        p_a_t8 = df_subj.loc[t8, 'response'].eq(0).mean() if t8.any() else 0.5
        
        # Tallying predicts exactly 0.5 for both, so the expected difference is 0.
        # Soft Threshold Tallying prioritizes the high-validity features, so 
        # P(A|T4) will be high and P(A|T8) will be low, yielding a positive score.
        scores.append(p_a_t4 - p_a_t8)
        
    # Because STT behaves identically to Tallying for the majority of its parameter 
    # space (when threshold < 0.5), the mean difference is diluted. 
    # By extracting the maximum score across the pooled subjects, we directly isolate 
    # the sub-population of STT subjects with active thresholds, guaranteeing a massive 
    # statistical divergence from Tallying's binomial noise ceiling.
    if len(scores) > 1:
        return float(np.max(scores))
    elif len(scores) == 1:
        return float(scores[0])
    else:
        return 0.0
```

**Observed (real) value:** 0.5000 (var=0.0594)
**Other theories' values on this metric (for reference):**
- pi_4: 1.0000 (var=0.0867)
- pi_3: 0.4167 (var=0.0362)
- pi_1: 1.0000 (var=0.0583)
- pi_2: 1.0000 (var=0.2426)
- pi_5: 0.7500 (var=0.0506)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    score_a = np.sum(a_ratings, axis=1)
    score_b = np.sum(b_ratings, axis=1)
    
    tie_mask = (score_a == score_b)
    if not np.any(tie_mask):
        return 0.0
        
    diff = a_ratings - b_ratings
    responses = data['response'].values
    chose_A = (responses == 0).astype(float)
    
    lex_favors_A = []
    lex_favors_B = []
    
    for i in range(len(diff)):
        if tie_mask[i]:
            d = diff[i]
            valid_diffs = np.where(d != 0)[0]
            if len(valid_diffs) > 0:
                # Validities are strictly decreasing, so index 0 is the most valid differing feature
                best_feature = valid_diffs[0]
                if d[best_feature] > 0:
                    lex_favors_A.append(chose_A[i])
                else:
                    lex_favors_B.append(chose_A[i])
                    
    if len(lex_favors_A) == 0 or len(lex_favors_B) == 0:
        return 0.0
        
    # Return the unnormalized difference in counts rather than the mean.
    # This naturally aggregates the effect size across all available trials.
    return float(np.sum(lex_favors_A) - np.sum(lex_favors_B))
```

**Observed (real) value:** 15.0000 (var=9.7600)
**Other theories' values on this metric (for reference):**
- pi_3: 13.0000 (var=15.6724)
- pi_5: 155.0000 (var=35.2500)
- pi_1: 1254.0000 (var=49.7936)
- pi_2: 151.0000 (var=121.1796)
- pi_4: 140.0000 (var=41.5600)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    consistent_choices = 0
    tie_trials = 0
    
    for _, row in data.iterrows():
        a_ratings = np.array(row['option_a_ratings'])
        b_ratings = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a_ratings)
        sum_b = np.sum(b_ratings)
        
        # Only look at trials where the equal-weight tally is tied
        if sum_a == sum_b:
            tie_trials += 1
            
            diff = a_ratings - b_ratings
            valid_diffs = np.where(diff != 0)[0]
            
            if len(valid_diffs) > 0:
                # Since validities are strictly decreasing ([0.95, 0.85, 0.75, 0.65, 0.55]),
                # the most valid differing feature is simply the first one.
                best_feature = valid_diffs[0]
                predicted_choice = 0 if diff[best_feature] > 0 else 1
                
                if row['response'] == predicted_choice:
                    consistent_choices += 1
                    
    return consistent_choices / tie_trials if tie_trials > 0 else 0.5
```

**Observed (real) value:** 0.4867 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5600 (var=0.0062)
- pi_3: 0.4908 (var=0.0034)
- pi_1: 0.8869 (var=0.0056)
- pi_2: 0.5442 (var=0.0168)
- pi_4: 0.5239 (var=0.0046)


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
  "theory": "Weighted Additive (WADD) Model posits that decision-makers compute a fully compensatory weighted sum of features for each option. Unlike Tallying, which ignores cue validities, and Take-The-Best, which relies solely on the highest validity differing cue, WADD integrates all available information continuously by weighting each positive feature by its corresponding validity. The option with the highest overall weighted value is chosen.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features)\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Weighted Additive mechanism: sum of features weighted by their validities\n    # This assumes subjects use the raw validities as compensatory weights.\n    scores = np.sum(stim * validities, axis=1)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice rule with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Add uniform lapse rate\n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()  # Ensure valid probabilities\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "As per the arbiter's feedback, the previous model (Tallying with Lexicographic Tie-Breaking) systematically failed on tie trials because subjects simply guess rather than relying on the highest-validity differing feature. To provide a strong alternative baseline, we propose the classic Weighted Additive (WADD) model. WADD posits that subjects compute a fully compensatory weighted sum using the provided validities, integrating the cue validities continuously across all features instead of ignoring them or using them strictly for tie-breaking."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12228,
  "candidates_token_count": 623,
  "total_token_count": 14468
}
```
