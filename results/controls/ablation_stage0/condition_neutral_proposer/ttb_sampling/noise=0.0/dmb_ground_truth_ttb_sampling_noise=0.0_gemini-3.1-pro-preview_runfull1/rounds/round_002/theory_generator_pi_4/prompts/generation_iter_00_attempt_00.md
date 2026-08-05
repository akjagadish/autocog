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
- THEORY 2 = `pi_2_1`
- The recommendation below acts on THEORY 2 (= `pi_2_1`).

WADD is clearly falsified by the data, which shows a strong reliance on non-compensatory heuristics. Propose a new theory that builds upon the success of Take The Best but offers a distinct mechanism. For example, you could propose a 'Probabilistic Take The Best' model where the stopping rule is stochastic (the decision-maker stops at a discriminating cue with a probability proportional to its validity), or a 'Strategy Selection' model where users predominantly use TTB but occasionally fall back to Tallying when the top cues are tied or have low validities.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 1.000)

**Description**
Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1067 (var=0.0120) vs this=0.1250 (var=0.0106)
- Experiment 2: real=0.8649 (var=0.0063) vs this=0.8764 (var=0.0064)
- Experiment 3: real=0.1467 (var=0.0156) vs this=0.1233 (var=0.0056)
- Experiment 4: real=0.1133 (var=0.0069) vs this=0.1208 (var=0.0055)
- Experiment 5: real=0.1780 (var=0.0177) vs this=0.1330 (var=0.0053)
- Experiment 6: real=0.1713 (var=0.0123) vs this=0.1398 (var=0.0066)


---

### `pi_2_1` (overall score: 0.416)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Use the log-odds of objective validities as the subjective weights
    validities = np.asarray(parameters["validities"], dtype=float)
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)  # Prevent division by zero or log(0)
    weights = np.log(validities / (1.0 - validities))
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1067 (var=0.0120) vs this=0.0633 (var=0.0069)
- Experiment 2: real=0.8649 (var=0.0063) vs this=0.9360 (var=0.0056)
- Experiment 3: real=0.1467 (var=0.0156) vs this=0.1823 (var=0.0070)
- Experiment 4: real=0.1133 (var=0.0069) vs this=0.2258 (var=0.0073)
- Experiment 5: real=0.1780 (var=0.0177) vs this=0.8750 (var=0.0104)
- Experiment 6: real=0.1713 (var=0.0123) vs this=0.8498 (var=0.0127)


---

### `pi_2` (overall score: 0.293)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1067 (var=0.0120) vs this=0.4208 (var=0.0840)
- Experiment 2: real=0.8649 (var=0.0063) vs this=0.5004 (var=0.0514)
- Experiment 3: real=0.1467 (var=0.0156) vs this=0.6657 (var=0.0396)
- Experiment 4: real=0.1133 (var=0.0069) vs this=0.7150 (var=0.0325)
- Experiment 5: real=0.1780 (var=0.0177) vs this=0.7440 (var=0.0159)
- Experiment 6: real=0.1713 (var=0.0123) vs this=0.7458 (var=0.0143)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_wadd = a_mat @ validities
    b_wadd = b_mat @ validities
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    wadd_prefers_a = a_wadd > b_wadd
    wadd_prefers_b = b_wadd > a_wadd
    
    disagree = (tally_prefers_a & wadd_prefers_b) | (tally_prefers_b & wadd_prefers_a)
    
    if not np.any(disagree):
        return 0.5
        
    tally_choice = np.where(tally_prefers_a, 0, 1)
    
    match = (data['response'].values[disagree] == tally_choice[disagree])
    return float(np.mean(match))
```

**Observed (real) value:** 0.1067 (var=0.0120)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8575 (var=0.0138)
- pi_2: 0.4208 (var=0.0840)
- pi_2_1: 0.0633 (var=0.0069)
- pi_3: 0.1250 (var=0.0106)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.stack(data['option_a_ratings'].values)
    b_vals = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Tallying and WADD make strictly opposing predictions.
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0] -> Tallying prefers A, WADD prefers B
    is_trial_1 = (a_vals[:, 0] == 0) & (a_vals[:, 2] == 1) & (b_vals[:, 0] == 1) & (b_vals[:, 2] == 0)
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1] -> Tallying prefers B, WADD prefers A
    is_trial_2 = (a_vals[:, 0] == 1) & (a_vals[:, 2] == 0) & (b_vals[:, 0] == 0) & (b_vals[:, 2] == 1)
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    responses = data['response'].values
    wadd_aligned = (is_trial_1 & (responses == 1)) | (is_trial_2 & (responses == 0))
    
    return float(wadd_aligned[mask].mean())
```

**Observed (real) value:** 0.8649 (var=0.0063)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5004 (var=0.0514)
- pi_1: 0.1520 (var=0.0057)
- pi_2_1: 0.9360 (var=0.0056)
- pi_3: 0.8764 (var=0.0064)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Approximate log-odds weights for WADD based on validities [0.88, 0.73, 0.73, 0.62, 0.62, 0.62]
    w = np.array([1.9924, 0.9946, 0.9946, 0.4895, 0.4895, 0.4895])
    
    tally_matches = []
    
    for _, row in data.iterrows():
        A = np.array(row['option_a_ratings'])
        B = np.array(row['option_b_ratings'])
        
        # Tallying counts strict wins
        A_tally = np.sum(A > B)
        B_tally = np.sum(B > A)
        
        # WADD uses weighted sums
        A_wadd = np.sum(A * w)
        B_wadd = np.sum(B * w)
        
        # 0 for A, 1 for B, -1 for tie
        tally_pref = 0 if A_tally > B_tally else (1 if B_tally > A_tally else -1)
        wadd_pref = 0 if A_wadd > B_wadd else (1 if B_wadd > A_wadd else -1)
        
        # We only care about trials where the two models make opposite strict predictions
        if tally_pref != -1 and wadd_pref != -1 and tally_pref != wadd_pref:
            if row['response'] == tally_pref:
                tally_matches.append(1.0)
            else:
                tally_matches.append(0.0)
                
    if len(tally_matches) == 0:
        return 0.5
        
    return float(np.mean(tally_matches))
```

**Observed (real) value:** 0.1467 (var=0.0156)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8460 (var=0.0095)
- pi_2_1: 0.1823 (var=0.0070)
- pi_2: 0.6657 (var=0.0396)
- pi_3: 0.1233 (var=0.0056)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5333])
    weights = np.log(validities / (1.0 - validities))
    
    match_tally = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wadd = np.sum(a * weights)
        b_wadd = np.sum(b * weights)
        
        a_tally = np.sum(a > b)
        b_tally = np.sum(b > a)
        
        wadd_prefers_a = a_wadd > b_wadd + 1e-5
        wadd_prefers_b = b_wadd > a_wadd + 1e-5
        tally_prefers_a = a_tally > b_tally
        tally_prefers_b = b_tally > a_tally
        
        disagree = (wadd_prefers_a and tally_prefers_b) or (wadd_prefers_b and tally_prefers_a)
        
        if disagree:
            chose_a = (row['response'] == 0)
            chose_b = (row['response'] == 1)
            
            if tally_prefers_a and chose_a:
                match_tally.append(1.0)
            elif tally_prefers_b and chose_b:
                match_tally.append(1.0)
            else:
                match_tally.append(0.0)
                
    if len(match_tally) == 0:
        return 0.5
        
    return float(np.mean(match_tally))
```

**Observed (real) value:** 0.1133 (var=0.0069)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.2258 (var=0.0073)
- pi_1: 0.8400 (var=0.0116)
- pi_2: 0.7150 (var=0.0325)
- pi_3: 0.1208 (var=0.0055)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    mask = data['option_a_ratings'].apply(lambda x: sum(x)) == 1
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1780 (var=0.0177)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1330 (var=0.0053)
- pi_2_1: 0.8750 (var=0.0104)
- pi_1: 0.8660 (var=0.0123)
- pi_2: 0.7440 (var=0.0159)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.85, 0.75, 0.7, 0.65, 0.6])
    weights = np.log(validities / (1.0 - validities))
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    score_a = A @ weights
    score_b = B @ weights
    
    wadd_pred = (score_b > score_a).astype(int)
    
    return float((data['response'].values == wadd_pred).mean())
```

**Observed (real) value:** 0.1713 (var=0.0123)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.8498 (var=0.0127)
- pi_3: 0.1398 (var=0.0066)
- pi_1: 0.8446 (var=0.0092)
- pi_2: 0.7458 (var=0.0143)


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
  "theory": "Probabilistic Take The Best (PTTB) posits that decision-makers search through features sequentially in order of their subjective validities. However, unlike deterministic Take The Best, the stopping rule is stochastic: upon finding a feature that discriminates between the two options, the decision-maker stops and bases their choice on that feature with probability gamma. If they do not stop, they continue to the next feature. If all features are evaluated without a decision, they guess randomly. This model captures the primarily non-compensatory nature of human decision-making while allowing for occasional deviations where lower-validity cues might influence the choice.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"PTTB expects a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Sort indices by validity in descending order\n    order = np.argsort(validities)[::-1]\n    \n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    p_choice = np.array([0.0, 0.0])\n    p_reach = 1.0\n    \n    # Search for discriminating cues with stochastic stopping\n    for idx in order:\n        if stim[0, idx] > stim[1, idx]:\n            p_stop = gamma\n            p_choice += p_reach * p_stop * np.array([1.0, 0.0])\n            p_reach *= (1.0 - p_stop)\n        elif stim[1, idx] > stim[0, idx]:\n            p_stop = gamma\n            p_choice += p_reach * p_stop * np.array([0.0, 1.0])\n            p_reach *= (1.0 - p_stop)\n            \n    # If no decision is made after evaluating all cues, guess\n    p_choice += p_reach * np.array([0.5, 0.5])\n        \n    # Apply uniform lapse rate\n    return (1.0 - epsilon) * p_choice + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "gamma": "[0.5, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, WADD models fail because the data strongly indicates the use of non-compensatory heuristics like Take The Best. To build upon TTB while introducing a distinct mechanism, I implemented a 'Probabilistic Take The Best' model. Instead of a deterministic stopping rule, the decision-maker stops at a discriminating cue with probability gamma. This preserves the core sequential and non-compensatory search of TTB, explaining the data similarly well, but provides a more nuanced stochastic mechanism that models occasional consideration of further cues."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10213,
  "candidates_token_count": 811,
  "total_token_count": 12322
}
```
