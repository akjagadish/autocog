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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a 'Subject-Specific Single-Cue Heuristic' or 'Strong Spatial Bias' theory. The data suggests subjects largely ignore the objective validities provided in the instructions. Instead, they either adopt a strong spatial bias (e.g., always choosing the left or right option) or pick a single, random feature to follow deterministically, ignoring all other features. This explains the 50% aggregate means across most experiments, the extremely high variance (0.25) on specific single trials (where half the population deterministically picks A and the other half B based on their chosen cue/bias), and the failure to align with objective-consensus trials in Experiment 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.753)

**Description**
Heterogeneous Cue Hierarchies: Individuals all employ a lexicographic, non-compensatory decision strategy (Take-The-Best), but they differ in how they construct their cue hierarchies. Rather than perfectly internalizing the objective cue validities provided in instructions, subjects form subjective cue validities by combining the objective validities with idiosyncratic, subjective weights. The degree to which a subject relies on their idiosyncratic weights versus objective validities varies across the population. This results in some subjects strictly following the objective hierarchy, while others use idiosyncratic hierarchies (e.g., heavily overweighting a specific cue). This heterogeneity explains the high between-subject variance and why objectively dominant options in aggregate only receive partial choice shares.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    subjective_weights = np.asarray(parameters["subjective_weights"], dtype=float)
    noise_level = float(parameters["noise_level"])
    epsilon = float(parameters["epsilon"])
    
    # Construct subjective validities by adding individual-specific noise/preferences to objective validities
    # subjective_weights are in [0, 1], so we center them around 0
    subj_val = validities + noise_level * (subjective_weights - 0.5)
    
    # Lexicographic strategy based on subjective hierarchy
    cue_order = np.argsort(-subj_val, kind="stable").tolist()
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- noise_level: [0.0, 10.0]
- subjective_weights: [(0.0, 1.0)] * n_features
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4667 (var=0.0267) vs this=0.5289 (var=0.1492)
- Experiment 2: real=0.3600 (var=0.2304) vs this=0.5267 (var=0.1471)
- Experiment 3: real=0.5200 (var=0.2496) vs this=0.4333 (var=0.1644)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.4667 (var=0.1400)
- Experiment 5: real=0.4800 (var=0.2496) vs this=0.5800 (var=0.1680)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.6000 (var=0.1856)
- Experiment 7: real=0.6080 (var=0.0783) vs this=0.5627 (var=0.1010)
- Experiment 8: real=0.5040 (var=0.0100) vs this=0.8290 (var=0.0113)


---

### `pi_4` (overall score: 0.453)

**Description**
Strategy Mixture Theory: The population of decision-makers is heterogeneous, consisting of distinct subgroups that rely on fundamentally different decision rules. Rather than every individual using a noisy version of the same heuristic, about half the population employs a non-compensatory, frugal strategy (Take The Best), while the other half employs a compensatory, unweighted strategy (Tallying). This individual difference accounts for both the ~50% aggregate choice proportions on critical divergence trials and the high (~0.25) between-subject variance, which is characteristic of a Bernoulli distribution of highly deterministic strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    strategy = int(parameters["strategy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if strategy == 0:
        # Take The Best (Non-compensatory)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner is None:
            scores = np.array([0.5, 0.5])
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
            
    else:
        # Tallying (Compensatory)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- strategy: {0, 1}
- beta: [5.0, 50.0]
- epsilon: [0.0, 0.1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4667 (var=0.0267) vs this=0.5211 (var=0.2280)
- Experiment 2: real=0.3600 (var=0.2304) vs this=0.4633 (var=0.2237)
- Experiment 3: real=0.5200 (var=0.2496) vs this=0.5167 (var=0.2247)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.3933 (var=0.2220)
- Experiment 5: real=0.4800 (var=0.2496) vs this=0.5767 (var=0.2302)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.7567 (var=0.0669)
- Experiment 7: real=0.6080 (var=0.0783) vs this=0.7353 (var=0.0626)
- Experiment 8: real=0.5040 (var=0.0100) vs this=0.9800 (var=0.0006)


---

### `pi_3` (overall score: 0.375)

**Description**
Weighted Additive (WADD) Theory: People evaluate options by computing a weighted sum of their features, where the weights correspond to the cue validities above chance (validity - 0.5). This linear compensatory strategy allows multiple lower-validity cues to outweigh a single high-validity cue without over-privileging the top cue. Choices are made probabilistically using a softmax over the weighted sums, combined with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Use linear evidence above chance instead of log-odds to prevent over-dominance of the top cue
    weights = val - 0.5
    
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4667 (var=0.0267) vs this=0.6144 (var=0.0176)
- Experiment 2: real=0.3600 (var=0.2304) vs this=0.3567 (var=0.0467)
- Experiment 3: real=0.5200 (var=0.2496) vs this=0.3600 (var=0.0471)
- Experiment 4: real=0.5200 (var=0.2496) vs this=0.3333 (var=0.0389)
- Experiment 5: real=0.4800 (var=0.2496) vs this=0.7600 (var=0.0535)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.8333 (var=0.0311)
- Experiment 7: real=0.6080 (var=0.0783) vs this=0.7900 (var=0.0167)
- Experiment 8: real=0.5040 (var=0.0100) vs this=0.7220 (var=0.0103)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB winner
        val_order = [0, 2, 1, 3] # Indices in descending order of validity
        ttb_winner = None
        for j in val_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying winner
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Check strict disagreement
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            total += 1
            if row['response'] == ttb_winner:
                ttb_match += 1
                
    if total == 0:
        return 0.5
    return ttb_match / total
```

**Observed (real) value:** 0.4667 (var=0.0267)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8856 (var=0.0104)
- pi_2: 0.1578 (var=0.0160)
- pi_3: 0.6144 (var=0.0176)
- pi_4: 0.5211 (var=0.2280)
- pi_5: 0.5289 (var=0.1492)

### Experiment 2
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    mask = (a_tup == (0, 1, 1, 1)) & (b_tup == (1, 0, 1, 0))
    if not mask.any():
        return 0.5
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.3600 (var=0.2304)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8467 (var=0.0343)
- pi_1: 0.1467 (var=0.0352)
- pi_3: 0.3567 (var=0.0467)
- pi_4: 0.4633 (var=0.2237)
- pi_5: 0.5267 (var=0.1471)

### Experiment 3
**Design**
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 9, where Option A has ratings [0, 1, 1, 0] and Option B has [1, 0, 0, 0]
    def is_target(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return a == [0, 1, 1, 0] and b == [1, 0, 0, 0]
    
    mask = data.apply(is_target, axis=1)
    target_trials = data[mask]
    
    if len(target_trials) == 0:
        return 0.5
        
    return float(target_trials['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8167 (var=0.0336)
- pi_3: 0.3600 (var=0.0471)
- pi_2: 0.1533 (var=0.0365)
- pi_4: 0.5167 (var=0.2247)
- pi_5: 0.4333 (var=0.1644)

### Experiment 4
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 15 is the only trial where WADD and TTB diverge.
    # A = [0, 1, 1, 0], B = [1, 0, 0, 0]
    # WADD weights are [0.45, 0.27, 0.3, 0.05].
    # WADD score for A = 0.27 + 0.3 = 0.57
    # WADD score for B = 0.45
    # WADD prefers A (response = 0).
    # TTB looks at cue 0 first (validity 0.95), which favors B. 
    # TTB prefers B (response = 1).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0)) & \
           data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3333 (var=0.0389)
- pi_1: 0.8967 (var=0.0188)
- pi_2: 0.1600 (var=0.0266)
- pi_4: 0.3933 (var=0.2220)
- pi_5: 0.4667 (var=0.1400)

### Experiment 5
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask1 = (a_str == '1000') & (b_str == '0101')
    mask2 = (a_str == '0101') & (b_str == '1000')
    
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.5
        
    choices = 0
    if mask1.sum() > 0:
        choices += (data.loc[mask1, 'response'] == 0).sum()
    if mask2.sum() > 0:
        choices += (data.loc[mask2, 'response'] == 1).sum()
        
    return float(choices / total)
```

**Observed (real) value:** 0.4800 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5767 (var=0.2302)
- pi_3: 0.7600 (var=0.0535)
- pi_1: 0.8533 (var=0.0218)
- pi_2: 0.1533 (var=0.0298)
- pi_5: 0.5800 (var=0.1680)

### Experiment 6
**Design**
  A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_match = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0))
    b_match = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0))
    target_trials = data[a_match & b_match]
    if len(target_trials) == 0:
        return 0.5
    return float((target_trials['response'] == 0).mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8333 (var=0.0311)
- pi_4: 0.7567 (var=0.0669)
- pi_1: 0.8133 (var=0.0263)
- pi_2: 0.5567 (var=0.0573)
- pi_5: 0.6000 (var=0.1856)

### Experiment 7
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the sum of positive cues for each option to find Tally-tie trials
    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))
    tally_tie_mask = (a_sums == b_sums)
    
    if not tally_tie_mask.any():
        return 0.5
        
    tie_data = data[tally_tie_mask].copy()
    
    # Determine the objective Take-The-Best (TTB) choice
    # Objective validities are [0.95, 0.74, 0.81, 0.55], so cue order is 0, 2, 1, 3
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for cue in [0, 2, 1, 3]:
            if a[cue] > b[cue]:
                return 0
            elif b[cue] > a[cue]:
                return 1
        return -1
        
    tie_data['ttb_choice'] = tie_data.apply(get_ttb_choice, axis=1)
    
    valid_mask = tie_data['ttb_choice'] != -1
    if not valid_mask.any():
        return 0.5
        
    valid_tie_data = tie_data[valid_mask]
    matches = (valid_tie_data['response'] == valid_tie_data['ttb_choice']).astype(float)
    
    return float(matches.mean())
```

**Observed (real) value:** 0.6080 (var=0.0783)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7353 (var=0.0626)
- pi_5: 0.5627 (var=0.1010)
- pi_1: 0.8507 (var=0.0110)
- pi_2: 0.5020 (var=0.0065)
- pi_3: 0.7900 (var=0.0167)

### Experiment 8
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.8, 0.74, 0.55])
    cue_order = np.argsort(-validities)
    
    def get_ttb_pred(a, b):
        for j in cue_order:
            if a[j] > b[j]: return 0
            if b[j] > a[j]: return 1
        return 0.5
        
    def is_agreement_trial(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        sum_a = sum(a)
        sum_b = sum(b)
        
        if sum_a == sum_b:
            return False
            
        tally_pred = 0 if sum_a > sum_b else 1
        ttb_pred = get_ttb_pred(a, b)
        
        return tally_pred == ttb_pred

    # Calculate predictions and identify agreement trials
    data['ttb_pred'] = data.apply(lambda row: get_ttb_pred(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_agreement'] = data.apply(is_agreement_trial, axis=1)
    
    agreement_data = data[data['is_agreement']].copy()
    
    if len(agreement_data) == 0:
        return 0.0
        
    # Calculate match with the consensus prediction
    agreement_data['match'] = (agreement_data['response'] == agreement_data['ttb_pred']).astype(float)
    
    # Return the mean match proportion
    return float(agreement_data['match'].mean())
```

**Observed (real) value:** 0.5040 (var=0.0100)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8290 (var=0.0113)
- pi_4: 0.9800 (var=0.0006)
- pi_1: 0.8590 (var=0.0100)
- pi_2: 0.8547 (var=0.0095)
- pi_3: 0.7220 (var=0.0103)


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
  "theory": "Subject-Specific Single-Cue Heuristic and Strong Spatial Bias: Subjects largely ignore the objective validities provided in the instructions. Instead of engaging in complex compensatory or non-compensatory integration of multiple cues, individuals adopt highly simplified, deterministic strategies. The population is split: some subjects rely on a strong spatial bias (e.g., always choosing the left or right option), while others adopt a single-cue heuristic, randomly selecting one feature to follow deterministically and ignoring all others. This extreme simplification explains the ~50% aggregate choice proportions on critical divergence trials, the failure to align with objective-consensus trials, and the extremely high between-subject variance, as choices are driven by idiosyncratic, deterministic biases rather than shared objective validities.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be a 2xN array.\")\n        \n    a, b = stim[0], stim[1]\n    \n    strategy_class = int(parameters[\"strategy_class\"])\n    spatial_dir = int(parameters[\"spatial_dir\"])\n    cue_weights = np.asarray(parameters[\"cue_weights\"], dtype=float)\n    epsilon = float(parameters[\"epsilon\"])\n    \n    if strategy_class == 0:\n        # Spatial Bias: deterministically choose Left (0) or Right (1)\n        p_core = np.array([1.0, 0.0]) if spatial_dir == 0 else np.array([0.0, 1.0])\n    else:\n        # Single-Cue Heuristic: follow a single randomly preferred cue\n        k = int(np.argmax(cue_weights))\n        if a[k] > b[k]:\n            p_core = np.array([1.0, 0.0])\n        elif b[k] > a[k]:\n            p_core = np.array([0.0, 1.0])\n        else:\n            # If the chosen cue is tied, guess randomly\n            p_core = np.array([0.5, 0.5])\n            \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "strategy_class": "{0, 1}",
    "spatial_dir": "{0, 1}",
    "cue_weights": "[(0.0, 1.0)] * n_features",
    "epsilon": "[0.0, 0.1]"
  },
  "rationale": "Following the arbiter's feedback, this theory abandons the assumption that subjects internalize the objective cue validities. Instead, it models the population as a mix of individuals who either adopt a strict spatial bias (always picking Option A or Option B) or a single-cue heuristic (picking one random cue to follow deterministically). This naturally accounts for the chance-level (~50%) aggregate means across most experiments, including the consensus trials in Experiment 8 where normative theories predict high agreement. It also inherently produces the extremely high single-trial variance (~0.25) observed in the data, as half the population deterministically picks A and the other half deterministically picks B based on their idiosyncratic bias or chosen cue."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12936,
  "candidates_token_count": 864,
  "total_token_count": 18092
}
```
