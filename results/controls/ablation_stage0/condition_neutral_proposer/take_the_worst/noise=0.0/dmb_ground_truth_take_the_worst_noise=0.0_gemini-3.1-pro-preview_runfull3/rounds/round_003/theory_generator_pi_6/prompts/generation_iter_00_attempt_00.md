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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a new theory that acknowledges the systematic tie-breaking behavior observed in the data. Since Theory 1 (Right-to-Left tie-breaking) is highly successful, the new theory should explore a psychologically grounded mechanism for this effect, such as 'Tallying with Recency-Driven Tie-Breaking'. In this theory, subjects use a standard Tallying heuristic, but when counts are tied, they rely on the cues most recently encoded in working memory (the right-most features in a left-to-right reading order) to break the tie, rather than guessing randomly or reverting to the most valid cue.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.992)

**Description**
Tallying with Right-to-Left Position-Based Tie-Breaking

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    # Position-based lexicographic tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in range(len(a) - 1, -1, -1):
            if a[i] > b[i]:
                scores[0] += 1.0
                break
            elif b[i] > a[i]:
                scores[1] += 1.0
                break
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(p), p=p))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1425 (var=0.0068) vs this=0.1392 (var=0.0079)
- Experiment 2: real=0.8556 (var=0.0092) vs this=0.8703 (var=0.0075)
- Experiment 3: real=0.4260 (var=0.0028) vs this=0.4307 (var=0.0025)
- Experiment 4: real=0.1433 (var=0.0085) vs this=0.1511 (var=0.0119)
- Experiment 5: real=0.1183 (var=0.0078) vs this=0.1321 (var=0.0113)
- Experiment 6: real=0.8818 (var=0.0077) vs this=0.8564 (var=0.0092)
- Experiment 7: real=0.7956 (var=0.0173) vs this=0.8306 (var=0.0145)
- Experiment 8: real=0.8433 (var=0.0081) vs this=0.8456 (var=0.0109)


---

### `pi_2` (overall score: 0.583)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1425 (var=0.0068) vs this=0.1579 (var=0.0134)
- Experiment 2: real=0.8556 (var=0.0092) vs this=0.8500 (var=0.0101)
- Experiment 3: real=0.4260 (var=0.0028) vs this=0.4440 (var=0.0050)
- Experiment 4: real=0.1433 (var=0.0085) vs this=0.4956 (var=0.0070)
- Experiment 5: real=0.1183 (var=0.0078) vs this=0.1392 (var=0.0078)
- Experiment 6: real=0.8818 (var=0.0077) vs this=0.8680 (var=0.0081)
- Experiment 7: real=0.7956 (var=0.0173) vs this=0.4961 (var=0.0096)
- Experiment 8: real=0.8433 (var=0.0081) vs this=0.5097 (var=0.0041)


---

### `pi_3` (overall score: 0.321)

**Description**
Weighted Additive Model (WADD): People evaluate options by integrating all available features, weighting each feature by its given validity. Unlike Take The Best, WADD is fully compensatory, meaning multiple lower-validity cues can override a single higher-validity cue. A subjective transformation parameter (gamma) allows the model to scale the validities, seamlessly interpolating between equal-weighting (Tallying, gamma=0) and validity-sensitive compensatory evaluation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Subjective transformation of validities
    gamma = float(parameters["gamma"])
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1425 (var=0.0068) vs this=0.1529 (var=0.0112)
- Experiment 2: real=0.8556 (var=0.0092) vs this=0.8464 (var=0.0123)
- Experiment 3: real=0.4260 (var=0.0028) vs this=0.6570 (var=0.0235)
- Experiment 4: real=0.1433 (var=0.0085) vs this=0.7906 (var=0.0196)
- Experiment 5: real=0.1183 (var=0.0078) vs this=0.1762 (var=0.0122)
- Experiment 6: real=0.8818 (var=0.0077) vs this=0.8613 (var=0.0107)
- Experiment 7: real=0.7956 (var=0.0173) vs this=0.4017 (var=0.0146)
- Experiment 8: real=0.8433 (var=0.0081) vs this=0.3056 (var=0.0116)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    
    diff = a - b
    mask = diff != 0
    first_diff_idx = np.argmax(mask, axis=1)
    has_diff = np.any(mask, axis=1)
    
    row_indices = np.arange(len(data))
    first_diff_val = diff[row_indices, first_diff_idx]
    
    ttb_winner = np.where(first_diff_val == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    tally_winner = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    disagree = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(disagree):
        return 0.5
        
    responses = data['response'].values
    ttb_matches = (responses[disagree] == ttb_winner[disagree])
    
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.1425 (var=0.0068)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8433 (var=0.0153)
- pi_2: 0.1579 (var=0.0134)
- pi_3: 0.1529 (var=0.0112)
- pi_4: 0.8333 (var=0.0124)
- pi_5: 0.1392 (var=0.0079)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = sum_a != sum_b
    if not np.any(mask):
        return 0.5
    tally_choices = (sum_b > sum_a).astype(int)
    responses = np.array(data['response'].values)
    agreements = (responses[mask] == tally_choices[mask]).astype(float)
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.8556 (var=0.0092)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8500 (var=0.0101)
- pi_1: 0.1414 (var=0.0091)
- pi_3: 0.8464 (var=0.0123)
- pi_4: 0.1378 (var=0.0120)
- pi_5: 0.8703 (var=0.0075)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (1, 0, 1, 0, 0)
    t5 = (0, 1, 1, 0, 0)
    t6 = (0, 0, 1, 1, 1)
    t8 = (0, 1, 0, 1, 0)
    
    score = 0.0
    count = 0
    
    m1 = data['A_tuple'] == t1
    if m1.any():
        score += (data.loc[m1, 'response'] == 0).mean()
        count += 1
        
    m2 = data['A_tuple'] == t2
    if m2.any():
        score += (data.loc[m2, 'response'] == 0).mean()
        count += 1
        
    m5 = data['A_tuple'] == t5
    if m5.any():
        score += (data.loc[m5, 'response'] == 1).mean()
        count += 1
        
    m6 = data['A_tuple'] == t6
    if m6.any():
        score += (data.loc[m6, 'response'] == 1).mean()
        count += 1
        
    m8 = data['A_tuple'] == t8
    if m8.any():
        score += (data.loc[m8, 'response'] == 1).mean()
        count += 1
        
    return score / max(1, count)
```

**Observed (real) value:** 0.4260 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6570 (var=0.0235)
- pi_2: 0.4440 (var=0.0050)
- pi_1: 0.8670 (var=0.0072)
- pi_4: 0.8503 (var=0.0088)
- pi_5: 0.4307 (var=0.0025)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.85, 0.5, 0.95, 0.6, 0.9, 0.55])
    wadd_match = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Calculate Tallying score difference
        tally_diff = np.sum(a > b) - np.sum(b > a)
        
        # Focus strictly on Tallying-tie trials
        if tally_diff == 0:
            # Calculate WADD score difference (assuming gamma=1 for direction)
            wadd_diff = np.sum(val * a) - np.sum(val * b)
            
            if wadd_diff > 0:
                # WADD prefers option A (response 0)
                wadd_match.append(1 if row['response'] == 0 else 0)
            elif wadd_diff < 0:
                # WADD prefers option B (response 1)
                wadd_match.append(1 if row['response'] == 1 else 0)
                
    if len(wadd_match) == 0:
        return 0.5
        
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.1433 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4956 (var=0.0070)
- pi_3: 0.7906 (var=0.0196)
- pi_1: 0.8600 (var=0.0101)
- pi_4: 0.8483 (var=0.0153)
- pi_5: 0.1511 (var=0.0119)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.6, 0.9, 0.5, 0.8, 0.7])
    sorted_indices = np.argsort(-validities)
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    ttb_preds = np.full(len(data), 0.5)
    for f_idx in sorted_indices:
        mask = (a_mat[:, f_idx] != b_mat[:, f_idx]) & (ttb_preds == 0.5)
        ttb_preds[mask & (a_mat[:, f_idx] > b_mat[:, f_idx])] = 0
        ttb_preds[mask & (a_mat[:, f_idx] < b_mat[:, f_idx])] = 1
        
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_preds = np.full(len(data), 0.5)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    disagree = (ttb_preds != 0.5) & (tally_preds != 0.5) & (ttb_preds != tally_preds)
    
    if not np.any(disagree):
        return 0.5
        
    match = (responses[disagree] == ttb_preds[disagree])
    return float(np.mean(match))
```

**Observed (real) value:** 0.1183 (var=0.0078)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8400 (var=0.0099)
- pi_2: 0.1392 (var=0.0078)
- pi_1: 0.8321 (var=0.0119)
- pi_3: 0.1762 (var=0.0122)
- pi_5: 0.1321 (var=0.0113)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validity_order = [1, 5, 3, 0, 2, 4]
    
    consistent_count = 0
    conflict_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            continue
        tally_pref = 0 if a_wins > b_wins else 1
        
        ttb_pref = -1
        for idx in validity_order:
            if a[idx] != b[idx]:
                ttb_pref = 0 if a[idx] > b[idx] else 1
                break
                
        if ttb_pref != -1 and tally_pref != ttb_pref:
            conflict_count += 1
            if row['response'] == tally_pref:
                consistent_count += 1
                
    if conflict_count == 0:
        return 0.5
    return float(consistent_count / conflict_count)
```

**Observed (real) value:** 0.8818 (var=0.0077)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8680 (var=0.0081)
- pi_4: 0.1396 (var=0.0082)
- pi_1: 0.1444 (var=0.0086)
- pi_3: 0.8613 (var=0.0107)
- pi_5: 0.8564 (var=0.0092)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tie_count += 1
            # Find right-to-left tie breaker
            pred = None
            for i in range(len(a)-1, -1, -1):
                if a[i] > b[i]:
                    pred = 0
                    break
                elif b[i] > a[i]:
                    pred = 1
                    break
            if pred is not None and row['response'] == pred:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
    return match_count / tie_count
```

**Observed (real) value:** 0.7956 (var=0.0173)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8306 (var=0.0145)
- pi_2: 0.4961 (var=0.0096)
- pi_1: 0.8617 (var=0.0104)
- pi_3: 0.4017 (var=0.0146)
- pi_4: 0.8806 (var=0.0086)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'], dtype=int)
        b = np.array(row['option_b_ratings'], dtype=int)
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            ties += 1
            pred = -1
            for i in range(len(a) - 1, -1, -1):
                if a[i] > b[i]:
                    pred = 0
                    break
                elif b[i] > a[i]:
                    pred = 1
                    break
            
            if pred == row['response']:
                matches += 1
                
    return float(matches / ties) if ties > 0 else 0.5
```

**Observed (real) value:** 0.8433 (var=0.0081)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5097 (var=0.0041)
- pi_5: 0.8456 (var=0.0109)
- pi_1: 0.3817 (var=0.0032)
- pi_3: 0.3056 (var=0.0116)
- pi_4: 0.3831 (var=0.0017)


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
  "theory": "Tallying with Recency-Driven Tie-Breaking: Decision makers primarily evaluate options using a Tallying heuristic, counting the number of features where one option strictly dominates the other. When the tallies are tied, they do not guess randomly. Instead, they rely on the cues most recently encoded into working memory to break the tie. Assuming a standard left-to-right reading order, the right-most features are the most recently processed. The model resolves ties by scanning features from right to left, favoring the option that wins on the first differentiating feature encountered.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    # Primary strategy: Tallying\n    a_wins = np.sum(a > b)\n    b_wins = np.sum(b > a)\n    \n    score_a = float(a_wins)\n    score_b = float(b_wins)\n    \n    # Recency-driven tie-breaking (right-to-left)\n    if a_wins == b_wins:\n        for i in reversed(range(len(a))):\n            if a[i] > b[i]:\n                score_a += 1.0\n                break\n            elif b[i] > a[i]:\n                score_b += 1.0\n                break\n                \n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n    \n    # Softmax choice with epsilon-greedy lapse\n    scores = np.array([score_a, score_b])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(scores)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]"
  },
  "rationale": "The arbiter pointed out that the standard Tallying model fails to capture systematic tie-breaking behavior, often reverting to random guessing when tallies are equal. Inspired by the success of right-to-left tie-breaking, this model introduces 'Tallying with Recency-Driven Tie-Breaking'. It posits that when primary tallying fails to distinguish between options, subjects fall back on the most recently encoded information in working memory. In a left-to-right reading paradigm, the right-most features are the most recent. This provides a psychologically grounded, memory-based explanation for the lexicographic tie-breaking effect without discarding the robust baseline of the Tallying heuristic."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11917,
  "candidates_token_count": 742,
  "total_token_count": 14184
}
```
