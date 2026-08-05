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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Propose a new theory that explains the systematic reversal observed on tally-tie trials. Instead of compressing validities to break ties compensatorily (which favors multiple moderately-high validities over a single highest validity), subjects might be using a strict Lexicographic / Take-The-Best (TTB) tie-breaker when tallies are equal. Alternatively, subjects might be exhibiting a 'feature-drop' heuristic where they ignore the lowest validity features completely when counts are tied. The new theory should combine a primary Tallying/Equal-Weight phase with a Lexicographic tie-breaker phase (e.g., 'Tally-then-TTB'), ensuring that when raw sums are equal, the option with the single highest-validity discriminating cue wins, rather than the option with a higher sum of compressed validities.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.965)

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
- Experiment 1: real=0.1383 (var=0.0087) vs this=0.1297 (var=0.0093)
- Experiment 2: real=0.8286 (var=0.0105) vs this=0.8479 (var=0.0103)
- Experiment 3: real=0.1333 (var=0.0128) vs this=0.1667 (var=0.0219)
- Experiment 4: real=0.1956 (var=0.0240) vs this=0.1511 (var=0.0177)
- Experiment 5: real=0.1283 (var=0.0075) vs this=0.1172 (var=0.0062)
- Experiment 6: real=0.8588 (var=0.0081) vs this=0.7265 (var=0.0054)
- Experiment 7: real=0.3300 (var=0.0032) vs this=0.4969 (var=0.0048)
- Experiment 8: real=0.4571 (var=0.0020) vs this=0.5035 (var=0.0028)
- Experiment 9: real=0.1872 (var=0.0127) vs this=0.5003 (var=0.0046)
- Experiment 10: real=0.1633 (var=0.0099) vs this=0.4853 (var=0.0029)


---

### `pi_6` (overall score: 0.785)

**Description**
Weighted Tallying: People compare options by computing a weighted sum of feature wins, where the weights are a heavily compressed function of the stated cue validities (w_i = validity_i ** gamma, with gamma close to 0). This allows the strategy to behave almost exactly like simple Tallying when one option has a clear numerical advantage in feature wins, but smoothly and compensatorily resolves ties using all available validity information rather than defaulting to a strict lexicographic tie-breaker.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Compress validities to create weights that are close to 1 (Tallying-like) 
    # but still retain ordinal validity information for compensatory tie-breaking.
    weights = val ** gamma
    
    # Calculate weighted feature wins (ignoring ties on individual features)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 0.3]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1383 (var=0.0087) vs this=0.1614 (var=0.0108)
- Experiment 2: real=0.8286 (var=0.0105) vs this=0.8748 (var=0.0081)
- Experiment 3: real=0.1333 (var=0.0128) vs this=0.1425 (var=0.0165)
- Experiment 4: real=0.1956 (var=0.0240) vs this=0.1578 (var=0.0203)
- Experiment 5: real=0.1283 (var=0.0075) vs this=0.1267 (var=0.0058)
- Experiment 6: real=0.8588 (var=0.0081) vs this=0.6652 (var=0.0041)
- Experiment 7: real=0.3300 (var=0.0032) vs this=0.5616 (var=0.0050)
- Experiment 8: real=0.4571 (var=0.0020) vs this=0.5435 (var=0.0053)
- Experiment 9: real=0.1872 (var=0.0127) vs this=0.5550 (var=0.0042)
- Experiment 10: real=0.1633 (var=0.0099) vs this=0.6239 (var=0.0105)


---

### `pi_4` (overall score: 0.517)

**Description**
People use a 'Top-K Tallying' strategy. Instead of relying entirely on a single best cue or integrating all available information, individuals save cognitive effort by tallying feature wins only among the top K most valid features. They ignore the least valid cues entirely. This boundedly rational approach captures the robust, compensatory nature of Tallying while acknowledging cognitive limitations in processing many cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(round(float(parameters["k"]))) 
    
    # Ensure k is at least 1 and at most n_features
    n_features = len(val)
    k = max(1, min(k, n_features))
    
    # Identify the top K most valid cues
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally feature wins only among the top K cues
    a_wins = 0.0
    b_wins = 0.0
    for idx in top_k_cues:
        if a[idx] > b[idx]:
            a_wins += 1.0
        elif b[idx] > a[idx]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- k: [4.0, 8.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1383 (var=0.0087) vs this=0.1517 (var=0.0095)
- Experiment 2: real=0.8286 (var=0.0105) vs this=0.8433 (var=0.0112)
- Experiment 3: real=0.1333 (var=0.0128) vs this=0.1858 (var=0.0265)
- Experiment 4: real=0.1956 (var=0.0240) vs this=0.1933 (var=0.0360)
- Experiment 5: real=0.1283 (var=0.0075) vs this=0.6597 (var=0.0109)
- Experiment 6: real=0.8588 (var=0.0081) vs this=0.2042 (var=0.0058)
- Experiment 7: real=0.3300 (var=0.0032) vs this=0.5050 (var=0.0051)
- Experiment 8: real=0.4571 (var=0.0020) vs this=0.4813 (var=0.0080)
- Experiment 9: real=0.1872 (var=0.0127) vs this=0.5083 (var=0.0050)
- Experiment 10: real=0.1633 (var=0.0099) vs this=0.5286 (var=0.0085)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))
```

**Observed (real) value:** 0.1383 (var=0.0087)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8742 (var=0.0109)
- pi_2: 0.1297 (var=0.0093)
- pi_3: 0.1503 (var=0.0082)
- pi_4: 0.1517 (var=0.0095)
- pi_5: 0.1297 (var=0.0089)
- pi_6: 0.1614 (var=0.0108)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
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
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8286 (var=0.0105)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8479 (var=0.0103)
- pi_1: 0.1536 (var=0.0070)
- pi_3: 0.8264 (var=0.0085)
- pi_4: 0.8433 (var=0.0112)
- pi_5: 0.8626 (var=0.0101)
- pi_6: 0.8748 (var=0.0081)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A='11000', B='00111' -> WADD prefers A (1.9 vs 1.6), Tally prefers B (2 vs 3)
    # Trial 2: A='00111', B='11000' -> WADD prefers B (1.6 vs 1.9), Tally prefers A (3 vs 2)
    
    t1 = df[(df['A_str'] == '11000') & (df['B_str'] == '00111')]
    t2 = df[(df['A_str'] == '00111') & (df['B_str'] == '11000')]
    
    score = 0.0
    n = 0
    
    if len(t1) > 0:
        score += (t1['response'] == 0).sum()
        n += len(t1)
    if len(t2) > 0:
        score += (t2['response'] == 1).sum()
        n += len(t2)
        
    if n == 0:
        return 0.5
    return float(score / n)

```

**Observed (real) value:** 0.1333 (var=0.0128)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7842 (var=0.0284)
- pi_2: 0.1667 (var=0.0219)
- pi_1: 0.8317 (var=0.0133)
- pi_4: 0.1858 (var=0.0265)
- pi_5: 0.1267 (var=0.0124)
- pi_6: 0.1425 (var=0.0165)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    wadd_chosen = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_chosen += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_chosen += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_chosen / total)
```

**Observed (real) value:** 0.1956 (var=0.0240)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1511 (var=0.0177)
- pi_3: 0.7733 (var=0.0291)
- pi_1: 0.8333 (var=0.0283)
- pi_4: 0.1933 (var=0.0360)
- pi_5: 0.1556 (var=0.0257)
- pi_6: 0.1578 (var=0.0203)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of features for option A and option B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Select trials where Option A has fewer total positive features than Option B
    # Standard Tallying will consistently choose Option B on these trials.
    # Top-K Tallying will often choose Option A or tie, because Option A's positive features are concentrated in the top K most valid cues.
    mask = a_sums < b_sums
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(subset['response'] == 0))
```

**Observed (real) value:** 0.1283 (var=0.0075)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6597 (var=0.0109)
- pi_2: 0.1172 (var=0.0062)
- pi_1: 0.6236 (var=0.0025)
- pi_3: 0.1636 (var=0.0089)
- pi_5: 0.1283 (var=0.0080)
- pi_6: 0.1267 (var=0.0058)

### Experiment 6
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Return the proportion of times Option A was chosen.
    # Tallying counts all features, meaning Option A wins on most trials (because it has more 1s overall in these specific pairs).
    # Top-K Tallying ignores the lowest validity features where Option A dominates, so Option B wins on most trials.
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.8588 (var=0.0081)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7265 (var=0.0054)
- pi_4: 0.2042 (var=0.0058)
- pi_1: 0.2633 (var=0.0050)
- pi_3: 0.3983 (var=0.0023)
- pi_5: 0.6302 (var=0.0018)
- pi_6: 0.6652 (var=0.0041)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tie_mask = (a_wins == b_wins)
    
    if not np.any(tie_mask):
        return 0.5
        
    a_tie = a_mat[tie_mask]
    b_tie = b_mat[tie_mask]
    resp_tie = data['response'].values[tie_mask]
    
    diff = a_tie - b_tie
    
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    row_indices = np.arange(len(a_tie))
    first_diff_sign = diff[row_indices, first_diff_idx]
    
    favored = np.where(first_diff_sign == 1, 0, 1)
    consistent = (resp_tie == favored)
    
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3300 (var=0.0032)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8491 (var=0.0189)
- pi_2: 0.4969 (var=0.0048)
- pi_1: 0.8653 (var=0.0119)
- pi_3: 0.6944 (var=0.0091)
- pi_4: 0.5050 (var=0.0051)
- pi_6: 0.5616 (var=0.0050)

### Experiment 8
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_matrix = np.array(data['option_a_ratings'].tolist())
    b_matrix = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_matrix > b_matrix, axis=1)
    b_wins = np.sum(b_matrix > a_matrix, axis=1)
    
    ties = (a_wins == b_wins)
    
    if not np.any(ties):
        return 0.5
        
    a_ties = a_matrix[ties]
    b_ties = b_matrix[ties]
    responses_ties = data['response'].values[ties]
    
    diffs = a_ties - b_ties
    
    matches = 0
    valid_trials = 0
    
    for i in range(len(diffs)):
        row_diff = diffs[i]
        nonzero = np.nonzero(row_diff)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choice = 0 if row_diff[first_diff_idx] > 0 else 1
            if responses_ties[i] == ttb_choice:
                matches += 1
            valid_trials += 1
            
    if valid_trials == 0:
        return 0.5
        
    return float(matches / valid_trials)
```

**Observed (real) value:** 0.4571 (var=0.0020)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5035 (var=0.0028)
- pi_5: 0.8479 (var=0.0158)
- pi_1: 0.8603 (var=0.0108)
- pi_3: 0.6692 (var=0.0077)
- pi_4: 0.4813 (var=0.0080)
- pi_6: 0.5435 (var=0.0053)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wt_favored = {
        (1, 0, 0, 0, 1): 1,
        (0, 1, 1, 0, 0): 0,
        (1, 1, 0, 0, 0): 0,
        (0, 0, 1, 1, 0): 1,
        (1, 0, 0, 0, 0): 0,
        (0, 1, 0, 0, 0): 0
    }
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a_tuple = tuple(row['option_a_ratings'])
        if a_tuple in wt_favored:
            if row['response'] == wt_favored[a_tuple]:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
    return match_count / total_count
```

**Observed (real) value:** 0.1872 (var=0.0127)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5550 (var=0.0042)
- pi_2: 0.5003 (var=0.0046)
- pi_1: 0.6128 (var=0.0025)
- pi_3: 0.6919 (var=0.0108)
- pi_4: 0.5083 (var=0.0050)
- pi_5: 0.6019 (var=0.0039)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.6, 0.5])
    weights = val ** 0.1
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    a_wins = A > B
    b_wins = B > A
    
    tally_a = a_wins.sum(axis=1)
    tally_b = b_wins.sum(axis=1)
    
    tie_mask = (tally_a == tally_b) & (tally_a > 0)
    
    if not tie_mask.any():
        return 0.5
        
    wt_a = (a_wins[tie_mask] * weights).sum(axis=1)
    wt_b = (b_wins[tie_mask] * weights).sum(axis=1)
    
    expected = np.where(wt_b > wt_a, 1, 0)
    valid = wt_a != wt_b
    
    if not valid.any():
        return 0.5
        
    actual = data['response'].values[tie_mask][valid]
    expected = expected[valid]
    
    return float(np.mean(actual == expected))
```

**Observed (real) value:** 0.1633 (var=0.0099)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4853 (var=0.0029)
- pi_6: 0.6239 (var=0.0105)
- pi_1: 0.7231 (var=0.0072)
- pi_3: 0.7675 (var=0.0134)
- pi_4: 0.5286 (var=0.0085)
- pi_5: 0.6981 (var=0.0079)


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
  "theory": "Tally-then-TTB: Decision makers primarily use a Tallying (Equal-Weight) heuristic, counting the number of features on which each option is superior. However, when this raw count results in a tie, rather than guessing randomly or using a compensatory weighted sum, they employ a strict Lexicographic (Take-The-Best) tie-breaker. They search through the cues in order of validity and choose the option favored by the single highest-validity discriminating cue.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be of shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    \n    # Primary phase: Tallying feature wins\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    \n    if a_wins == b_wins:\n        # Secondary phase: Lexicographic (TTB) tie-breaker\n        val = np.asarray(parameters[\"validities\"], dtype=float)\n        order = np.argsort(-val, kind=\"stable\")\n        \n        ttb_a = 0.0\n        ttb_b = 0.0\n        for idx in order:\n            if a[idx] > b[idx]:\n                ttb_a = 1.0\n                break\n            elif b[idx] > a[idx]:\n                ttb_b = 1.0\n                break\n                \n        scores = np.array([a_wins + ttb_a, b_wins + ttb_b])\n    else:\n        scores = np.array([a_wins, b_wins])\n        \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=float)\n    probabilities /= np.sum(probabilities)\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Based on the arbiter's feedback, compressing validities to resolve tally ties compensatorily fails to capture the mechanistic nature of the tie-breaking process. Instead, this theory instantiates a 'Tally-then-TTB' strategy: it first compares the raw sum of feature wins (Tallying), but upon encountering a tie, it switches to a non-compensatory Lexicographic (Take-The-Best) rule. The option with the single highest-validity discriminating cue receives a point, breaking the tie decisively without allowing multiple lower-validity cues to override the most important one. This correctly models the systematic reversal observed on tally-tie trials."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13981,
  "candidates_token_count": 854,
  "total_token_count": 17289
}
```
