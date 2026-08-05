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

Propose a brand-new theory to replace the Strategy Selection Mixture Model. Since pure TTB (Theory 1) explains the data very well but still has some minor deviations, the new theory could be a 'Probabilistic Take The Best' model where the order of feature inspection is not strictly deterministic but sampled proportionally to feature validities. Alternatively, propose a pure Weighted Additive (WADD) model where features are integrated using the log-odds of their validities. The key is to completely remove the 'Tallying' (unit-weight) assumption, as subjects clearly do not just count positive features.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.978)

**Description**
People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


`parameters`:
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2967 (var=0.0025) vs this=0.3267 (var=0.0265)
- Experiment 2: real=0.3600 (var=0.0119) vs this=0.3200 (var=0.0334)
- Experiment 3: real=0.6716 (var=0.0045) vs this=0.6977 (var=0.0191)
- Experiment 4: real=0.3383 (var=0.0054) vs this=0.3031 (var=0.0170)
- Experiment 5: real=-0.0033 (var=0.0403) vs this=0.0050 (var=0.0415)
- Experiment 6: real=0.3371 (var=0.0041) vs this=0.3373 (var=0.0227)
- Experiment 7: real=0.6885 (var=0.0030) vs this=0.6468 (var=0.0132)
- Experiment 8: real=-0.0207 (var=0.0138) vs this=0.0141 (var=0.0112)
- Experiment 9: real=-0.0292 (var=0.0065) vs this=-0.0179 (var=0.0115)
- Experiment 10: real=0.0083 (var=0.0057) vs this=-0.0167 (var=0.0056)


---

### `pi_6` (overall score: 0.855)

**Description**
Strategy Selection Mixture Model: Subjects possess a repertoire of decision strategies and probabilistically switch between them. The primary strategy is the non-compensatory 'Take The Best' (TTB) heuristic, which evaluates features sequentially by validity. However, on a proportion of trials, subjects employ a simple compensatory 'Tallying' strategy, which counts the total number of positive features for each option regardless of validity. This mixture allows the model to capture both the dominant lexicographic behavior and the occasional compensatory deviations observed in human choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(validities)[::-1]
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    ttb_scores = np.array([ttb_a, ttb_b])
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Strategy 2: Tallying (Unit-weight linear model)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    tally_scores = np.array([tally_a, tally_b])
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


`parameters`:
- w_ttb: [0.5, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2967 (var=0.0025) vs this=0.3387 (var=0.0160)
- Experiment 2: real=0.3600 (var=0.0119) vs this=0.3379 (var=0.0248)
- Experiment 3: real=0.6716 (var=0.0045) vs this=0.6343 (var=0.0117)
- Experiment 4: real=0.3383 (var=0.0054) vs this=0.3590 (var=0.0137)
- Experiment 5: real=-0.0033 (var=0.0403) vs this=0.2083 (var=0.0340)
- Experiment 6: real=0.3371 (var=0.0041) vs this=0.3477 (var=0.0194)
- Experiment 7: real=0.6885 (var=0.0030) vs this=0.6567 (var=0.0166)
- Experiment 8: real=-0.0207 (var=0.0138) vs this=0.0776 (var=0.0180)
- Experiment 9: real=-0.0292 (var=0.0065) vs this=0.1646 (var=0.0233)
- Experiment 10: real=0.0083 (var=0.0057) vs this=0.1588 (var=0.0189)


---

### `pi_5` (overall score: 0.653)

**Description**
Exponentially Weighted Additive Model: Subjects integrate all available features in a compensatory manner, but they apply a steep non-linear (exponential) transformation to the feature validities. This causes the most valid features to heavily dominate the decision, effectively mimicking the non-compensatory 'Take The Best' heuristic while remaining mathematically compensatory. The steepness of this transformation dictates how closely the strategy approximates strict lexicographic choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Steep non-linear transformation of validities to weights
    weights = np.exp(gamma * validities)
    
    # Calculate option scores as weighted sums
    scores = stim @ weights
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


`parameters`:
- gamma: [0.5, 20.0]
- beta: [0.05, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.2967 (var=0.0025) vs this=0.2803 (var=0.0196)
- Experiment 2: real=0.3600 (var=0.0119) vs this=0.2589 (var=0.0301)
- Experiment 3: real=0.6716 (var=0.0045) vs this=0.5815 (var=0.0665)
- Experiment 4: real=0.3383 (var=0.0054) vs this=0.4371 (var=0.0330)
- Experiment 5: real=-0.0033 (var=0.0403) vs this=0.1483 (var=0.0912)
- Experiment 6: real=0.3371 (var=0.0041) vs this=0.3406 (var=0.0494)
- Experiment 7: real=0.6885 (var=0.0030) vs this=0.2397 (var=0.0234)
- Experiment 8: real=-0.0207 (var=0.0138) vs this=0.2067 (var=0.0886)
- Experiment 9: real=-0.0292 (var=0.0065) vs this=0.1063 (var=0.0486)
- Experiment 10: real=0.0083 (var=0.0057) vs this=0.1792 (var=0.0962)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_matches.append(1 if row['response'] == 0 else 0)
        elif b_wins > a_wins:
            tally_matches.append(1 if row['response'] == 1 else 0)
    if len(tally_matches) == 0:
        return 0.5
    return float(np.mean(tally_matches))
```

**Observed (real) value:** 0.2967 (var=0.0025)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8713 (var=0.0084)
- pi_2: 0.5197 (var=0.0620)
- pi_3: 0.3267 (var=0.0265)
- pi_4: 0.4307 (var=0.1485)
- pi_5: 0.2803 (var=0.0196)
- pi_6: 0.3387 (var=0.0160)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying and WADD make opposite predictions
    # A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    is_trial_1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                     data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                     
    tally_choices = 0
    total = 0
    
    if is_trial_1.sum() > 0:
        # Tallying favors B (response 1) because it has 3 features vs 2.
        tally_choices += data[is_trial_1]['response'].sum()
        total += is_trial_1.sum()
        
    if is_trial_1_rev.sum() > 0:
        # Tallying favors A (response 0).
        tally_choices += (1 - data[is_trial_1_rev]['response']).sum()
        total += is_trial_1_rev.sum()
        
    if total == 0:
        return 0.5
        
    return float(tally_choices / total)
```

**Observed (real) value:** 0.3600 (var=0.0119)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4853 (var=0.0941)
- pi_1: 0.8737 (var=0.0110)
- pi_3: 0.3200 (var=0.0334)
- pi_4: 0.3484 (var=0.1788)
- pi_5: 0.2589 (var=0.0301)
- pi_6: 0.3379 (var=0.0248)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(5):
            if a[i] > b[i]:
                if r == 0:
                    ttb_matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    ttb_matches += 1
                break
        total += 1
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6716 (var=0.0045)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6977 (var=0.0191)
- pi_2: 0.2653 (var=0.0130)
- pi_1: 0.1585 (var=0.0103)
- pi_4: 0.3592 (var=0.1837)
- pi_5: 0.5815 (var=0.0665)
- pi_6: 0.6343 (var=0.0117)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.65, 0.6, 0.55, 0.5])
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        
        if score_a > score_b:
            wadd_choice = 0
        elif score_b > score_a:
            wadd_choice = 1
        else:
            continue
            
        matches.append(1.0 if row['response'] == wadd_choice else 0.0)
        
    return float(np.mean(matches)) if len(matches) > 0 else 0.0
```

**Observed (real) value:** 0.3383 (var=0.0054)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6625 (var=0.0235)
- pi_3: 0.3031 (var=0.0170)
- pi_1: 0.8710 (var=0.0064)
- pi_4: 0.5729 (var=0.1950)
- pi_5: 0.4371 (var=0.0330)
- pi_6: 0.3590 (var=0.0137)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    data['choose_A'] = (data['response'] == 0).astype(float)
    
    t2_mask = (data['A_str'] == '10000') & (data['B_str'] == '00000')
    t5_mask = (data['A_str'] == '10000') & (data['B_str'] == '01110')
    
    p_a_t2 = data[t2_mask]['choose_A'].mean()
    p_a_t5 = data[t5_mask]['choose_A'].mean()
    
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    if pd.isna(p_a_t5): p_a_t5 = 0.5
    
    return float(p_a_t2 - p_a_t5)
```

**Observed (real) value:** -0.0033 (var=0.0403)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0050 (var=0.0415)
- pi_4: 0.5517 (var=0.2147)
- pi_1: 0.7350 (var=0.0438)
- pi_2: 0.6050 (var=0.1066)
- pi_5: 0.1483 (var=0.0912)
- pi_6: 0.2083 (var=0.0340)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design consists of trials where the first (most valid) feature 
    # always favors Option A, but the subsequent features often favor Option B.
    # TTB will almost exclusively choose Option A (response = 0) because it stops at the first feature.
    # BSFS, depending on its threshold, will often accumulate evidence and choose Option B (response = 1).
    # Therefore, the overall proportion of B choices directly discriminates the two models.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.3371 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5990 (var=0.1059)
- pi_3: 0.3373 (var=0.0227)
- pi_1: 0.8108 (var=0.0060)
- pi_2: 0.6427 (var=0.0210)
- pi_5: 0.3406 (var=0.0494)
- pi_6: 0.3477 (var=0.0194)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    mask = a_sums < b_sums
    if mask.sum() == 0:
        return 0.5
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.6885 (var=0.0030)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6468 (var=0.0132)
- pi_5: 0.2397 (var=0.0234)
- pi_1: 0.1633 (var=0.0089)
- pi_2: 0.2600 (var=0.0204)
- pi_4: 0.3910 (var=0.1966)
- pi_6: 0.6567 (var=0.0166)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    mask_high = b_sum >= 3
    mask_low = b_sum == 1
    
    rate_high = data.loc[mask_high, 'response'].mean()
    rate_low = data.loc[mask_low, 'response'].mean()
    
    if pd.isna(rate_high): rate_high = 0.0
    if pd.isna(rate_low): rate_low = 0.0
    
    return float(rate_high - rate_low)
```

**Observed (real) value:** -0.0207 (var=0.0138)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2067 (var=0.0886)
- pi_3: 0.0141 (var=0.0112)
- pi_1: 0.3644 (var=0.0190)
- pi_2: 0.2946 (var=0.0266)
- pi_4: 0.3309 (var=0.0559)
- pi_6: 0.0776 (var=0.0180)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the difference in the total number of positive features (Tallying evidence)
    tally_diff = data['option_a_ratings'].apply(sum) - data['option_b_ratings'].apply(sum)
    
    # Did the subject choose Option A? (response == 0)
    choose_a = (data['response'] == 0).astype(float)
    
    # Mean probability of choosing A when Tallying favors A
    p_a_pos = choose_a[tally_diff > 0].mean()
    
    # Mean probability of choosing A when Tallying favors B
    p_a_neg = choose_a[tally_diff < 0].mean()
    
    if pd.isna(p_a_pos):
        p_a_pos = 0.0
    if pd.isna(p_a_neg):
        p_a_neg = 0.0
        
    return float(p_a_pos - p_a_neg)
```

**Observed (real) value:** -0.0292 (var=0.0065)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0179 (var=0.0115)
- pi_6: 0.1646 (var=0.0233)
- pi_1: 0.7075 (var=0.0436)
- pi_2: 0.4492 (var=0.0305)
- pi_4: 0.4808 (var=0.2056)
- pi_5: 0.1063 (var=0.0486)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    congruent_mask = sum_a > sum_b
    incongruent_mask = sum_a < sum_b
    
    p_b_congruent = data.loc[congruent_mask, 'response'].mean()
    p_b_incongruent = data.loc[incongruent_mask, 'response'].mean()
    
    if pd.isna(p_b_congruent) or pd.isna(p_b_incongruent):
        return 0.0
        
    return float(p_b_incongruent - p_b_congruent)
```

**Observed (real) value:** 0.0083 (var=0.0057)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1588 (var=0.0189)
- pi_3: -0.0167 (var=0.0056)
- pi_1: 0.7233 (var=0.0197)
- pi_2: 0.6108 (var=0.0410)
- pi_4: 0.5696 (var=0.2088)
- pi_5: 0.1792 (var=0.0962)


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
  "theory": "Naive Bayes Evidence Integration (Log-Odds WADD): Subjects evaluate options by integrating all available features, weighting each feature by its log-odds of validity (the statistically optimal Naive Bayes weight). In many environments, the log-odds of the most valid feature naturally exceeds the sum of the remaining features' log-odds. Consequently, this fully compensatory model organically mimics the non-compensatory 'Take The Best' heuristic without needing a deterministic stopping rule or an ad-hoc mixture with 'Tallying'. A scaling parameter (gamma) captures subjective amplification or dampening of these log-odds.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    a = np.asarray(state[0], dtype=float)\n    b = np.asarray(state[1], dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Clip validities to avoid division by zero or log(0)\n    # Since validities are >= 0.5, log-odds will be strictly positive\n    v = np.clip(validities, 0.50001, 0.99999)\n    log_odds = np.log(v / (1.0 - v))\n    \n    # Apply subjective scaling to the log-odds\n    gamma = float(parameters[\"gamma\"])\n    weights = log_odds ** gamma\n    \n    # Calculate weighted additive scores\n    score_a = np.dot(a, weights)\n    score_b = np.dot(b, weights)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice rule with max-subtraction for numerical stability\n    scores = np.array([score_a, score_b])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Incorporate lapse rate\n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    p = np.asarray(probabilities, dtype=float)\n    p = p / np.sum(p)\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "gamma": "[0.1, 5.0]",
    "beta": "[0.01, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory completely removes the 'Tallying' mixture. Instead, it posits that subjects use a pure Weighted Additive (WADD) strategy where the weights are the log-odds of the feature validities (equivalent to a Naive Bayes classifier). Because the log-odds transformation heavily skews weights toward higher validities, this compensatory mechanism naturally produces the non-compensatory, lexicographic choices typical of 'Take The Best', while still allowing for subtle integrations of lesser features. A 'gamma' parameter is included to allow the model to empirically tune how extremely subjects separate high and low validity evidence."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12218,
  "candidates_token_count": 800,
  "total_token_count": 15999
}
```
