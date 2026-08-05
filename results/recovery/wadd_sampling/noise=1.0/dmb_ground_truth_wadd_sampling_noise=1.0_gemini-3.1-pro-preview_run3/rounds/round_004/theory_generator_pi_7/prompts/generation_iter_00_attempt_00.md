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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace Threshold Tallying Theory with a 'Strategy Mixture Theory'. This theory should posit that decision-makers do not use a single complex weighting scheme, but instead probabilistically alternate (or are divided in the population) between two simple heuristics: Take-The-Best (which strictly follows the highest valid cue that discriminates between options) and unweighted Tallying (which simply counts the number of features each option wins on). The model should have a parameter `p_tally` representing the probability of using the Tallying heuristic versus Take-The-Best on any given trial. This elegantly explains the ~50% choice rates in adversarial trials where TTB and Tallying point in opposite directions, without relying on rigid thresholding or arbitrary exponential weights.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.809)

**Description**
Rank-Weighted Additive Theory posits that decision-makers ignore the exact numerical validities provided and instead assign exponentially decaying weights based solely on the ordinal rank of the cues. The lowest-ranked cue is assigned a weight equal to the second-lowest-ranked cue. This ensures that the sum of all lower-ranked cues exactly equals the highest-ranked cue. They then compute a weighted sum for each option and choose probabilistically (via softmax). Because the highest-ranked cue's weight perfectly equals the sum of all lower-ranked cues' weights, this theory naturally predicts the exact ~50% guessing rate observed on '1 vs all' adversarial trials without needing arbitrary thresholds or conflict-based guessing rules.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Get validities and determine ordinal rank (0 is highest rank)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Assign exponentially decaying weights based on rank.
    # To ensure the sum of all lower-ranked cues exactly equals the highest-ranked cue,
    # the lowest-ranked cue gets the same weight as the second-lowest-ranked cue.
    w = np.zeros_like(val)
    n_cues = len(cue_order)
    for i, cue_idx in enumerate(cue_order):
        if i == n_cues - 1 and n_cues > 1:
            w[cue_idx] = 1.0 / (2.0 ** i)
        else:
            w[cue_idx] = 1.0 / (2.0 ** (i + 1))
        
    # Compute weighted sum for each option
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.5685 (var=0.0043)
- Experiment 2: real=0.4981 (var=0.0023) vs this=0.4316 (var=0.0041)
- Experiment 3: real=0.5033 (var=0.0016) vs this=0.5225 (var=0.0021)
- Experiment 4: real=0.5054 (var=0.0018) vs this=0.4740 (var=0.0026)
- Experiment 5: real=-0.0133 (var=0.0243) vs this=0.2300 (var=0.0440)
- Experiment 6: real=0.4916 (var=0.0085) vs this=0.4416 (var=0.0061)
- Experiment 7: real=-0.0319 (var=0.0091) vs this=0.2003 (var=0.0235)
- Experiment 8: real=0.4675 (var=0.0532) vs this=0.8787 (var=0.0732)
- Experiment 9: real=0.5284 (var=0.0096) vs this=0.5011 (var=0.0050)
- Experiment 10: real=0.0400 (var=0.0147) vs this=-0.1974 (var=0.0172)


---

### `pi_5` (overall score: 0.648)

**Description**
Threshold Tallying Theory posits that decision makers evaluate options by counting the number of features where one option strictly dominates the other (unweighted tallying). However, they only make a confident directional choice if the difference in winning features between the two options meets or exceeds a certain cognitive threshold. If the difference in evidence is below this threshold (e.g., a difference of 0 or 1), the decision maker finds the evidence too ambiguous or weak, leading to cognitive overload or uncertainty, and they resort to uniform guessing. This captures the persistent ~50% choice rates observed across many adversarial trials where the feature counts are closely matched.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins for each option
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    diff = abs(a_wins - b_wins)
    threshold = float(parameters["threshold"])
    
    # If the difference is below the threshold, the evidence is deemed too weak -> guessing
    if diff < threshold:
        p_core = np.array([0.5, 0.5])
    else:
        # Otherwise, make a choice based on the tally scores using softmax
        scores = np.array([float(a_wins), float(b_wins)])
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- threshold: [1.5, 3.5]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.4294 (var=0.0045)
- Experiment 2: real=0.4981 (var=0.0023) vs this=0.5400 (var=0.0056)
- Experiment 3: real=0.5033 (var=0.0016) vs this=0.4740 (var=0.0053)
- Experiment 4: real=0.5054 (var=0.0018) vs this=0.5369 (var=0.0040)
- Experiment 5: real=-0.0133 (var=0.0243) vs this=0.0200 (var=0.0429)
- Experiment 6: real=0.4916 (var=0.0085) vs this=0.4963 (var=0.0083)
- Experiment 7: real=-0.0319 (var=0.0091) vs this=-0.3534 (var=0.0167)
- Experiment 8: real=0.4675 (var=0.0532) vs this=0.1562 (var=0.0557)
- Experiment 9: real=0.5284 (var=0.0096) vs this=0.8500 (var=0.0087)
- Experiment 10: real=0.0400 (var=0.0147) vs this=0.3589 (var=0.0156)


---

### `pi_4` (overall score: 0.600)

**Description**
Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
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
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.5360 (var=0.0025)
- Experiment 2: real=0.4981 (var=0.0023) vs this=0.4918 (var=0.0024)
- Experiment 3: real=0.5033 (var=0.0016) vs this=0.4969 (var=0.0038)
- Experiment 4: real=0.5054 (var=0.0018) vs this=0.5098 (var=0.0029)
- Experiment 5: real=-0.0133 (var=0.0243) vs this=0.7133 (var=0.0273)
- Experiment 6: real=0.4916 (var=0.0085) vs this=0.5147 (var=0.0057)
- Experiment 7: real=-0.0319 (var=0.0091) vs this=0.3600 (var=0.0152)
- Experiment 8: real=0.4675 (var=0.0532) vs this=1.1850 (var=0.0461)
- Experiment 9: real=0.5284 (var=0.0096) vs this=0.5168 (var=0.0087)
- Experiment 10: real=0.0400 (var=0.0147) vs this=-0.1611 (var=0.0108)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                winner = 0
                break
            elif b[i] > a[i]:
                winner = 1
                break
        if winner is not None:
            matches.append(1 if resp == winner else 0)
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0012)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8477 (var=0.0132)
- pi_2: 0.2398 (var=0.0060)
- pi_3: 0.6058 (var=0.0018)
- pi_4: 0.5360 (var=0.0025)
- pi_5: 0.4294 (var=0.0045)
- pi_6: 0.5685 (var=0.0043)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4981 (var=0.0023)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8589 (var=0.0080)
- pi_1: 0.1488 (var=0.0095)
- pi_3: 0.4067 (var=0.0021)
- pi_4: 0.4918 (var=0.0024)
- pi_5: 0.5400 (var=0.0056)
- pi_6: 0.4316 (var=0.0041)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_aligned = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            if resp == ttb_pred:
                ttb_aligned += 1
            total += 1
            
    return ttb_aligned / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5033 (var=0.0016)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8465 (var=0.0074)
- pi_3: 0.2437 (var=0.0109)
- pi_2: 0.1525 (var=0.0120)
- pi_4: 0.4969 (var=0.0038)
- pi_5: 0.4740 (var=0.0053)
- pi_6: 0.5225 (var=0.0021)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5054 (var=0.0018)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7238 (var=0.0085)
- pi_1: 0.1356 (var=0.0082)
- pi_2: 0.8200 (var=0.0068)
- pi_4: 0.5098 (var=0.0029)
- pi_5: 0.5369 (var=0.0040)
- pi_6: 0.4740 (var=0.0026)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t5_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    resp_t1 = data.loc[t1_mask, 'response'].mean()
    resp_t5 = data.loc[t5_mask, 'response'].mean()
    
    val_t1 = float(resp_t1) if not pd.isna(resp_t1) else 0.5
    val_t5 = float(resp_t5) if not pd.isna(resp_t5) else 0.5
    
    return val_t5 - val_t1
```

**Observed (real) value:** -0.0133 (var=0.0243)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7133 (var=0.0273)
- pi_3: -0.2583 (var=0.0742)
- pi_1: 0.6767 (var=0.0582)
- pi_2: 0.7150 (var=0.0436)
- pi_5: 0.0200 (var=0.0429)
- pi_6: 0.2300 (var=0.0440)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    mask_t1 = (a_tuples == (0, 1, 1, 1, 0))
    mask_t2 = (a_tuples == (1, 0, 0, 0, 0))
    
    data_t1 = data[mask_t1]
    data_t2 = data[mask_t2]
    
    val1 = (data_t1['response'] == 0).mean() if len(data_t1) > 0 else 0.5
    val2 = (data_t2['response'] == 1).mean() if len(data_t2) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)
```

**Observed (real) value:** 0.4916 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8095 (var=0.0144)
- pi_4: 0.5147 (var=0.0057)
- pi_1: 0.1405 (var=0.0082)
- pi_2: 0.8384 (var=0.0110)
- pi_5: 0.4963 (var=0.0083)
- pi_6: 0.4416 (var=0.0061)

### Experiment 7
**Design**
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_diff_1 = []
    match_diff_4 = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        diff = abs(a_wins - b_wins)
        
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            continue
            
        match = 1 if row['response'] == tally_winner else 0
        
        if diff == 1:
            match_diff_1.append(match)
        elif diff == 4:
            match_diff_4.append(match)
            
    p1 = np.mean(match_diff_1) if match_diff_1 else 0.0
    p4 = np.mean(match_diff_4) if match_diff_4 else 0.0
    
    return float(p1 - p4)
```

**Observed (real) value:** -0.0319 (var=0.0091)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3600 (var=0.0152)
- pi_5: -0.3534 (var=0.0167)
- pi_1: 0.7313 (var=0.0259)
- pi_2: -0.0159 (var=0.0063)
- pi_3: 0.0466 (var=0.0078)
- pi_6: 0.2003 (var=0.0235)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 1) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 6)
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 2) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 1)
    t4_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 3) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 2)
    
    p_b_t1 = np.mean(data[t1_mask]['response'] == 1) if t1_mask.sum() > 0 else 0.5
    p_a_t2 = np.mean(data[t2_mask]['response'] == 0) if t2_mask.sum() > 0 else 0.5
    p_a_t4 = np.mean(data[t4_mask]['response'] == 0) if t4_mask.sum() > 0 else 0.5
    
    return float(p_a_t2 + p_a_t4 - p_b_t1)
```

**Observed (real) value:** 0.4675 (var=0.0532)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1562 (var=0.0557)
- pi_4: 1.1850 (var=0.0461)
- pi_1: 1.5988 (var=0.0663)
- pi_2: 0.7925 (var=0.0424)
- pi_3: 0.9050 (var=0.0347)
- pi_6: 0.8787 (var=0.0732)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Isolate trials 1 and 5 where one option has exactly 1 cue and the other has 5 cues.
    is_trial_1 = (data['option_a_ratings'].apply(sum) == 1) & (data['option_b_ratings'].apply(sum) == 5)
    is_trial_5 = (data['option_a_ratings'].apply(sum) == 5) & (data['option_b_ratings'].apply(sum) == 1)
    
    # We are interested in the proportion of times the subject chose the 5-cue option.
    # In trial 1, the 5-cue option is B (response == 1).
    # In trial 5, the 5-cue option is A (response == 0).
    chose_5_cue_t1 = (is_trial_1 & (data['response'] == 1))
    chose_5_cue_t5 = (is_trial_5 & (data['response'] == 0))
    
    total_target_trials = is_trial_1.sum() + is_trial_5.sum()
    if total_target_trials == 0:
        return 0.5
        
    return float((chose_5_cue_t1.sum() + chose_5_cue_t5.sum()) / total_target_trials)
```

**Observed (real) value:** 0.5284 (var=0.0096)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5011 (var=0.0050)
- pi_5: 0.8500 (var=0.0087)
- pi_1: 0.1795 (var=0.0171)
- pi_2: 0.8789 (var=0.0106)
- pi_3: 0.8432 (var=0.0132)
- pi_4: 0.5168 (var=0.0087)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    
    # Group 1 (Trials 1 & 2): A wins on the highest cue but loses on most others.
    # Identified by A[0] == 1 and A[2] == 0
    g1_mask = (a_ratings[:, 0] == 1) & (a_ratings[:, 2] == 0)
    
    # Group 2 (Trials 3 & 4): A wins on several top cues, closely matched in tally.
    # Identified by A[0] == 1 and A[2] == 1
    g2_mask = (a_ratings[:, 0] == 1) & (a_ratings[:, 2] == 1)
    
    # P(Choose B | Group 1)
    prob_b_g1 = data.loc[g1_mask, 'response'].mean()
    
    # P(Choose A | Group 2)
    prob_a_g2 = 1.0 - data.loc[g2_mask, 'response'].mean()
    
    return float(prob_b_g1 - prob_a_g2)
```

**Observed (real) value:** 0.0400 (var=0.0147)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3589 (var=0.0156)
- pi_6: -0.1974 (var=0.0172)
- pi_1: -0.6758 (var=0.0421)
- pi_2: 0.3979 (var=0.0108)
- pi_3: 0.0558 (var=0.0079)
- pi_4: -0.1611 (var=0.0108)


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
  "theory": "Strategy Mixture Theory posits that decision-makers do not use a single complex weighting scheme, but instead probabilistically alternate (or are divided in the population) between two simple heuristics: Take-The-Best (which strictly follows the highest valid cue that discriminates between options) and unweighted Tallying (which simply counts the number of features each option wins on). The model uses a parameter `p_tally` representing the probability of using the Tallying heuristic versus Take-The-Best on any given trial. This elegantly explains the ~50% choice rates in adversarial trials where TTB and Tallying point in opposite directions.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    \n    # Tallying prediction\n    a_wins = np.sum(a > b)\n    b_wins = np.sum(b > a)\n    tally_scores = np.array([float(a_wins), float(b_wins)])\n    \n    beta = float(parameters[\"beta\"])\n    z_tally = beta * (tally_scores - np.max(tally_scores))\n    e_tally = np.exp(z_tally)\n    p_tally_choice = e_tally / np.sum(e_tally)\n    \n    # Take-The-Best prediction\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    cue_order = np.argsort(-val, kind=\"stable\")\n    \n    ttb_scores = np.array([0.0, 0.0])\n    for j in cue_order:\n        if a[j] > b[j]:\n            ttb_scores = np.array([1.0, 0.0])\n            break\n        elif b[j] > a[j]:\n            ttb_scores = np.array([0.0, 1.0])\n            break\n            \n    z_ttb = beta * (ttb_scores - np.max(ttb_scores))\n    e_ttb = np.exp(z_ttb)\n    p_ttb_choice = e_ttb / np.sum(e_ttb)\n    \n    # Mixture\n    p_tally = float(parameters[\"p_tally\"])\n    p_core = p_tally * p_tally_choice + (1.0 - p_tally) * p_ttb_choice\n    \n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "p_tally": "[0.0, 1.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, we introduce a Strategy Mixture Theory. Instead of relying on rigid thresholds or conflict-induced guessing, this model assumes individuals probabilistically choose between Take-The-Best and unweighted Tallying heuristics on each trial (or equivalently, there is a population-level mixture). The parameter `p_tally` controls the weighting between the two strategies' predictions. In adversarial trials where TTB and Tallying conflict, the mixture naturally produces choice probabilities around 50% on average, capturing the empirical data without arbitrary complexity."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12583,
  "candidates_token_count": 963,
  "total_token_count": 16245
}
```
