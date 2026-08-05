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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a new theory that retains the core feature-counting mechanism of Tallying (which strongly drives behavior in Exps 3-6) but introduces a different tie-breaking or weighting mechanism to explain the severe drop to 0.145 in Experiment 2. Instead of a top-cue bonus, consider a mechanism where subjects might penalize options that rely too heavily on a single extreme cue, or a 'moderate-cue' bias where a cluster of medium-validity features is preferred over one high-validity feature when tallies are tied. The new theory must preserve the baseline 50/50 guessing on symmetric ties (Exp 1) while capturing the systematic anti-top-cue preference in Exp 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.959)

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
- Experiment 1: real=0.1194 (var=0.0110) vs this=0.1291 (var=0.0066)
- Experiment 2: real=0.7608 (var=0.0062) vs this=0.8690 (var=0.0080)
- Experiment 3: real=0.1562 (var=0.0200) vs this=0.1475 (var=0.0146)
- Experiment 4: real=0.1533 (var=0.0127) vs this=0.1467 (var=0.0111)
- Experiment 5: real=0.4750 (var=0.0075) vs this=0.5131 (var=0.0062)
- Experiment 6: real=0.1450 (var=0.0070) vs this=0.4900 (var=0.0052)


---

### `pi_4` (overall score: 0.525)

**Description**
Validity-Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying), ignoring the magnitudes of the differences and the exact cue validities. However, they are not completely blind to cue validities. Instead of integrating all validities in a compensatory manner (WADD) or strictly following a non-compensatory rule (Take The Best), they give a soft bonus weight to the single most predictive cue. This validity bias acts primarily as a tie-breaker when the tallies are close or equal, slightly skewing choice probabilities toward the option endorsed by the best cue, while still preserving the overall dominance of the Tallying heuristic. This explains why human choices overwhelmingly follow Tallying but show slight deviations toward the most valid cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Bonus for winning the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    score_a = a_wins + w * a_best_win
    score_b = b_wins + w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w: [0.0, 0.9]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1194 (var=0.0110) vs this=0.1534 (var=0.0108)
- Experiment 2: real=0.7608 (var=0.0062) vs this=0.8625 (var=0.0093)
- Experiment 3: real=0.1562 (var=0.0200) vs this=0.1706 (var=0.0123)
- Experiment 4: real=0.1533 (var=0.0127) vs this=0.1933 (var=0.0203)
- Experiment 5: real=0.4750 (var=0.0075) vs this=0.7719 (var=0.0310)
- Experiment 6: real=0.1450 (var=0.0070) vs this=0.8008 (var=0.0186)


---

### `pi_3` (overall score: 0.347)

**Description**
People make decisions by integrating all available cues, weighting each cue proportionally to its validity (Weighted Additive rule, or WADD). Unlike Take The Best, which stops at the first discriminating cue, WADD is a compensatory mechanism where multiple lower-validity cues can collectively outweigh a single high-validity cue. This explains why subjects often deviate from the non-compensatory predictions of TTB when a large number of lower-validity cues point in the opposite direction.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate weighted additive scores for both options
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

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
- Experiment 1: real=0.1194 (var=0.0110) vs this=0.1731 (var=0.0077)
- Experiment 2: real=0.7608 (var=0.0062) vs this=0.8319 (var=0.0117)
- Experiment 3: real=0.1562 (var=0.0200) vs this=0.6056 (var=0.0172)
- Experiment 4: real=0.1533 (var=0.0127) vs this=0.7233 (var=0.0238)
- Experiment 5: real=0.4750 (var=0.0075) vs this=0.8313 (var=0.0121)
- Experiment 6: real=0.1450 (var=0.0070) vs this=0.7633 (var=0.0193)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        # Only consider trials where Tallying does not predict a tie
        if a_wins != b_wins:
            ttb_pred = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_pred = 0
                    break
                elif b[i] > a[i]:
                    ttb_pred = 1
                    break
            
            if ttb_pred is not None:
                if resp == ttb_pred:
                    ttb_matches += 1
                total += 1
                
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1194 (var=0.0110)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8616 (var=0.0098)
- pi_2: 0.1291 (var=0.0066)
- pi_3: 0.1731 (var=0.0077)
- pi_4: 0.1534 (var=0.0108)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        if row['response'] == tally_pred:
            matches += 1
        total += 1
        
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.7608 (var=0.0062)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8690 (var=0.0080)
- pi_1: 0.3217 (var=0.0032)
- pi_3: 0.8319 (var=0.0117)
- pi_4: 0.8625 (var=0.0093)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        # Trial 1: A has top 2 validities (0.9+0.8=1.7), B has bottom 3 (0.6+0.5+0.5=1.6)
        # WADD prefers A (0), Tallying prefers B (1) since B wins 3-2
        if a == (1, 1, 0, 0, 0):
            if resp == 0:
                wadd_match += 1
            count += 1
        # Trial 2: A has bottom 3, B has top 2
        # WADD prefers B (1), Tallying prefers A (0)
        elif a == (0, 0, 1, 1, 1):
            if resp == 1:
                wadd_match += 1
            count += 1
            
    if count == 0:
        return 0.5
    return float(wadd_match / count)
```

**Observed (real) value:** 0.1562 (var=0.0200)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6056 (var=0.0172)
- pi_2: 0.1475 (var=0.0146)
- pi_1: 0.8644 (var=0.0082)
- pi_4: 0.1706 (var=0.0123)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying predicts B (3 wins vs 2) but WADD predicts A (1.8 vs 1.65)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[is_trial_1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1533 (var=0.0127)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1467 (var=0.0111)
- pi_3: 0.7233 (var=0.0238)
- pi_1: 0.8508 (var=0.0143)
- pi_4: 0.1933 (var=0.0203)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    ties = (a_wins == b_wins)
    
    if not np.any(ties):
        return 0.5
        
    responses = data['response'].values
    
    # Did the chosen option win on the most valid cue (index 0)?
    a_wins_cue0 = (a_ratings[:, 0] > b_ratings[:, 0])
    b_wins_cue0 = (b_ratings[:, 0] > a_ratings[:, 0])
    
    chosen_wins_cue0 = np.where(responses == 0, a_wins_cue0, b_wins_cue0)
    
    return float(np.mean(chosen_wins_cue0[ties]))
```

**Observed (real) value:** 0.4750 (var=0.0075)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7719 (var=0.0310)
- pi_2: 0.5131 (var=0.0062)
- pi_1: 0.8431 (var=0.0144)
- pi_3: 0.8313 (var=0.0121)

### Experiment 6
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    alignments = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        resp = row['response']
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where the tally is tied, but the most valid cue (index 0) is not tied.
        if a_wins == b_wins and a[0] != b[0]:
            best_cue_choice = 0 if a[0] > b[0] else 1
            alignments.append(1 if resp == best_cue_choice else 0)
            
    if not alignments:
        return 0.5
    return float(np.mean(alignments))
```

**Observed (real) value:** 0.1450 (var=0.0070)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4900 (var=0.0052)
- pi_4: 0.8008 (var=0.0186)
- pi_1: 0.8542 (var=0.0138)
- pi_3: 0.7633 (var=0.0193)


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
  "theory": "Moderate-Cue Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying). However, rather than giving a bonus to the most valid cue, subjects exhibit an aversion to options that rely too heavily on a single extreme cue at the expense of moderate cues. When tallies are close or tied, this manifests as a penalty to the option endorsed by the most valid cue, reflecting a preference for a broader cluster of supporting evidence over a single strong piece of evidence. This preserves the dominance of Tallying for clear differences but explains the systematic anti-top-cue preference observed in tie-breaking scenarios.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Moderate-Cue Biased Tallying expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    \n    # Primary mechanism: Tallying strict wins\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    \n    # Secondary mechanism: Penalty for relying on the most valid cue\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    best_cue = np.argmax(val)\n    \n    w = float(parameters[\"w\"])\n    a_best_win = float(a[best_cue] > b[best_cue])\n    b_best_win = float(b[best_cue] > a[best_cue])\n    \n    # Subtract the penalty weight from the tally score\n    score_a = a_wins - w * a_best_win\n    score_b = b_wins - w * b_best_win\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "w": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "The arbiter feedback highlights that standard Tallying overestimates the choice probability in Experiment 2 and completely misses the severe anti-top-cue preference in Experiment 6 (where subjects actively choose against the top cue when tallies are tied). To capture this, I introduced a 'Moderate-Cue Bias' via a penalty parameter `w` applied to the option that wins on the most extreme (highest validity) cue. Because `w` is bounded between 0 and 1, Tallying remains the dominant choice rule when one option clearly wins on count (e.g., Exp 1). However, when tallies are tied (as in Exp 6), the penalty forces the model to prefer the option supported by a cluster of moderate cues. Additionally, when the tally difference is small (e.g., 1), the penalty reduces the confidence in the Tallying winner, successfully lowering the predicted match rate in Experiment 2 from ~0.87 down closer to the observed 0.76."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9060,
  "candidates_token_count": 955,
  "total_token_count": 13429
}
```
