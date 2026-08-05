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

Propose a brand-new theory that replaces the Downstream Tallying Mixture. Since the empirical data strongly supports a strict non-compensatory (lexicographic) stopping rule without downstream evaluation, the new theory should maintain TTB as the core mechanism but explore a different source for the observed variance in Experiments 3-5. For example, consider a 'Take-The-Best with Cue-Reading Errors' theory, where decision-makers attempt to execute pure TTB but have a non-zero probability of misreading or misremembering individual cue values, or a 'Probabilistic Cue Validity' model where the subjective order of cues varies slightly across trials due to memory noise, rather than mixing with a compensatory tallying strategy.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.919)

**Description**
Take The Best (TTB) heuristic posits a lexicographic decision rule where individuals search through features in descending order of validity. They stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. If no feature discriminates, they guess randomly. To account for empirical levels of noise, the choice is mixed with a lapse rate (epsilon) that can span up to 1.0 (pure guessing).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3683 (var=0.0161) vs this=0.2442 (var=0.0244)
- Experiment 2: real=0.3600 (var=0.0410) vs this=0.5000 (var=0.1061)
- Experiment 3: real=0.3228 (var=0.0034) vs this=0.2447 (var=0.0248)
- Experiment 4: real=0.6746 (var=0.0031) vs this=0.7031 (var=0.0172)
- Experiment 5: real=-0.0052 (var=0.0086) vs this=-0.0067 (var=0.0122)
- Experiment 6: real=-0.0017 (var=0.0044) vs this=0.0087 (var=0.0058)
- Experiment 7: real=0.0517 (var=0.0088) vs this=-0.0040 (var=0.0090)
- Experiment 8: real=0.0237 (var=0.0234) vs this=-0.0119 (var=0.0097)


---

### `pi_5` (overall score: 0.870)

**Description**
Take-The-Best with Downstream Tallying Mixture: Decision makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in descending order of validity. However, to account for corroboration and tie-breaking effects, individuals occasionally mix this strategy with a 'downstream tally'—an equal-weight count of only the remaining, unexamined cues that have lower validity than the discriminating cue. This prevents the discriminating cue from double-contributing to the tally, reducing over-prediction of compensatory WADD-like behavior in environments with highly valid cues, while preserving compensatory variance in environments where downstream cues strongly oppose the best cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    # TTB Prediction
    p_ttb = np.array([0.5, 0.5])
    discrim_idx = len(order)
    for i, idx in enumerate(order):
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_idx = i
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_idx = i
            break
            
    # Downstream Tallying Prediction (only unexamined cues)
    if discrim_idx < len(order) - 1:
        remaining_indices = order[discrim_idx + 1:]
        tally_a = np.sum(a[remaining_indices])
        tally_b = np.sum(b[remaining_indices])
    else:
        tally_a = 0
        tally_b = 0
        
    if tally_a > tally_b:
        p_tally = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Mix TTB and Downstream Tallying
    w_tally = float(parameters["w_tally"])
    p_mix = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- epsilon: [0.0, 0.5]
- w_tally: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3683 (var=0.0161) vs this=0.3258 (var=0.0212)
- Experiment 2: real=0.3600 (var=0.0410) vs this=0.4067 (var=0.0996)
- Experiment 3: real=0.3228 (var=0.0034) vs this=0.2869 (var=0.0139)
- Experiment 4: real=0.6746 (var=0.0031) vs this=0.7081 (var=0.0156)
- Experiment 5: real=-0.0052 (var=0.0086) vs this=0.1674 (var=0.0193)
- Experiment 6: real=-0.0017 (var=0.0044) vs this=0.0904 (var=0.0092)
- Experiment 7: real=0.0517 (var=0.0088) vs this=0.1928 (var=0.0210)
- Experiment 8: real=0.0237 (var=0.0234) vs this=0.1807 (var=0.0176)


---

### `pi_4` (overall score: 0.682)

**Description**
Strategy Selection Theory (TTB + WADD): Individuals probabilistically mix between a non-compensatory lexicographic heuristic (Take The Best) and a compensatory heuristic (Weighted Additive, WADD). To account for varying degrees of confidence even when using a lexicographic rule, the TTB component makes probabilistic predictions rather than purely deterministic ones. Depending on individual differences or trial-by-trial strategy selection, a decision-maker relies on the single best discriminating cue a certain fraction of the time, and otherwise considers the validity-weighted sum of all feature differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction with confidence delta
    order = np.argsort(validities)[::-1]
    delta = float(parameters["delta"])
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0 - delta, delta])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([delta, 1.0 - delta])
            break
            
    # WADD prediction
    scores = stim @ validities
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # Mix strategies
    w_ttb = float(parameters["w_ttb"])
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # Add lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.2]
- w_ttb: [0.1, 0.85]
- delta: [0.0, 0.15]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3683 (var=0.0161) vs this=0.2825 (var=0.0143)
- Experiment 2: real=0.3600 (var=0.0410) vs this=0.4967 (var=0.0514)
- Experiment 3: real=0.3228 (var=0.0034) vs this=0.4494 (var=0.0233)
- Experiment 4: real=0.6746 (var=0.0031) vs this=0.6254 (var=0.0123)
- Experiment 5: real=-0.0052 (var=0.0086) vs this=0.3161 (var=0.0316)
- Experiment 6: real=-0.0017 (var=0.0044) vs this=0.3096 (var=0.0417)
- Experiment 7: real=0.0517 (var=0.0088) vs this=0.3475 (var=0.0406)
- Experiment 8: real=0.0237 (var=0.0234) vs this=0.3022 (var=0.0351)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_consistent = 0
    total_relevant = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: Tallying prefers A (count 3 vs 2), WADD prefers B (weight 1.8 vs 1.7)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if row['response'] == 0:
                tally_consistent += 1
            total_relevant += 1
        # Trial 2: Tallying prefers B (count 3 vs 2), WADD prefers A (weight 1.8 vs 1.7)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if row['response'] == 1:
                tally_consistent += 1
            total_relevant += 1
            
    if total_relevant == 0:
        return 0.5
    return float(tally_consistent / total_relevant)
```

**Observed (real) value:** 0.3683 (var=0.0161)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8408 (var=0.0182)
- pi_2: 0.5383 (var=0.0797)
- pi_3: 0.2442 (var=0.0244)
- pi_4: 0.2825 (var=0.0143)
- pi_5: 0.3258 (var=0.0212)

### Experiment 2
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    
    m1 = data.loc[is_t1, 'response'].mean()
    m4 = data.loc[is_t4, 'response'].mean()
    
    if pd.isna(m1):
        m1 = 0.5
    if pd.isna(m4):
        m4 = 0.5
        
    return float(m1 - m4)
```

**Observed (real) value:** 0.3600 (var=0.0410)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0733 (var=0.4268)
- pi_1: -0.6600 (var=0.0611)
- pi_3: 0.5000 (var=0.1061)
- pi_4: 0.4967 (var=0.0514)
- pi_5: 0.4067 (var=0.0996)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    mask = b_sum > a_sum
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.3228 (var=0.0034)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2447 (var=0.0248)
- pi_2: 0.6783 (var=0.0163)
- pi_1: 0.8519 (var=0.0086)
- pi_4: 0.4494 (var=0.0233)
- pi_5: 0.2869 (var=0.0139)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        preds.append(pred)
    matches = (data['response'] == preds).mean()
    return float(matches)
```

**Observed (real) value:** 0.6746 (var=0.0031)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4290 (var=0.0142)
- pi_3: 0.7031 (var=0.0172)
- pi_1: 0.3683 (var=0.0021)
- pi_4: 0.6254 (var=0.0123)
- pi_5: 0.7081 (var=0.0156)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    # Extract ratings into 2D arrays
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Calculate WADD scores
    score_a = a_ratings @ validities
    score_b = b_ratings @ validities
    wadd_diffs = score_a - score_b
    
    # Response == 0 means Option A was chosen
    chose_a = (data['response'] == 0).astype(float).values
    
    if np.std(wadd_diffs) == 0 or np.std(chose_a) == 0:
        return 0.0
        
    return float(np.corrcoef(wadd_diffs, chose_a)[0, 1])
```

**Observed (real) value:** -0.0052 (var=0.0086)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0067 (var=0.0122)
- pi_4: 0.3161 (var=0.0316)
- pi_1: 0.6756 (var=0.0234)
- pi_2: 0.4721 (var=0.0325)
- pi_5: 0.1674 (var=0.0193)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    def wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum((a - b) * validities)
        
    diffs = data.apply(wadd_diff, axis=1)
    
    choose_a = 1.0 - data['response']
    
    wadd_a_mask = diffs > 0
    wadd_b_mask = diffs < 0
    
    if wadd_a_mask.sum() == 0 or wadd_b_mask.sum() == 0:
        return 0.0
        
    p_a_given_wadd_a = choose_a[wadd_a_mask].mean()
    p_a_given_wadd_b = choose_a[wadd_b_mask].mean()
    
    return float(p_a_given_wadd_a - p_a_given_wadd_b)
```

**Observed (real) value:** -0.0017 (var=0.0044)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3096 (var=0.0417)
- pi_3: 0.0087 (var=0.0058)
- pi_1: 0.6367 (var=0.0343)
- pi_2: 0.4592 (var=0.0421)
- pi_5: 0.0904 (var=0.0092)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Find the first discriminating cue (TTB stopping point)
    diff = a_mat != b_mat
    discrim_idx = np.argmax(diff, axis=1)
    
    # TTB choice: 0 if A is better on the discriminating cue, else 1
    ttb_choice = (a_mat[np.arange(len(a_mat)), discrim_idx] < b_mat[np.arange(len(b_mat)), discrim_idx]).astype(int)
    
    # Downstream Tally choice: sum only cues after the discriminating one
    cols = np.arange(a_mat.shape[1])
    mask = cols > discrim_idx[:, None]
    
    tally_a = np.sum(a_mat * mask, axis=1)
    tally_b = np.sum(b_mat * mask, axis=1)
    
    tally_choice = np.zeros(len(a_mat), dtype=int) - 1
    tally_choice[tally_a > tally_b] = 0
    tally_choice[tally_b > tally_a] = 1
    
    # Identify trials where Downstream Tally agrees or disagrees with TTB
    agree_mask = (tally_choice == ttb_choice) & (tally_choice != -1)
    disagree_mask = (tally_choice != ttb_choice) & (tally_choice != -1)
    
    match = (resp == ttb_choice)
    
    if np.sum(agree_mask) == 0 or np.sum(disagree_mask) == 0:
        return 0.0
        
    # Return the difference in TTB-adherence between agree and disagree trials
    return float(np.mean(match[agree_mask]) - np.mean(match[disagree_mask]))
```

**Observed (real) value:** 0.0517 (var=0.0088)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0040 (var=0.0090)
- pi_5: 0.1928 (var=0.0210)
- pi_1: 0.6919 (var=0.0304)
- pi_2: 0.5322 (var=0.0465)
- pi_4: 0.3475 (var=0.0406)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    high_tally_A = {
        ('11110', '00001'),
        ('11111', '00000'),
        ('11111', '11000')
    }
    
    high_tally_B = {
        ('10000', '01111'),
        ('11000', '00111'),
        ('11100', '11011')
    }
    
    def get_trial_type(row):
        t = (row['A_str'], row['B_str'])
        if t in high_tally_A:
            return 'A'
        elif t in high_tally_B:
            return 'B'
        return 'Other'
        
    data['trial_type'] = data.apply(get_trial_type, axis=1)
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    mean_A = data[data['trial_type'] == 'A']['chose_A'].mean()
    mean_B = data[data['trial_type'] == 'B']['chose_A'].mean()
    
    if pd.isna(mean_A):
        mean_A = 0.5
    if pd.isna(mean_B):
        mean_B = 0.5
        
    return float(mean_A - mean_B)
```

**Observed (real) value:** 0.0237 (var=0.0234)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1807 (var=0.0176)
- pi_3: -0.0119 (var=0.0097)
- pi_1: 0.7348 (var=0.0269)
- pi_2: 0.5067 (var=0.0603)
- pi_4: 0.3022 (var=0.0351)


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
  "theory": "Probabilistic Cue Validity (Take-The-Best with Memory Noise): Decision-makers fundamentally use a non-compensatory lexicographic strategy (Take-The-Best), but the subjective validities of cues vary slightly from trial to trial due to memory noise or retrieval fluctuations. Instead of mixing TTB with a compensatory tallying strategy, this theory posits that variations in cue order allow options with more positive cues (which would be favored by tallying) to occasionally win, because having more positive cues increases the statistical likelihood that one of those cues will temporarily fluctuate to the top of the subjective validity hierarchy. This preserves the core non-compensatory mechanism while elegantly explaining variance in environments with multiple opposing cues.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    sigma = float(parameters[\"sigma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    n_samples = 200\n    if sigma > 0:\n        noise = np.random.normal(0, sigma, size=(n_samples, len(validities)))\n        subj_vals = validities + noise\n    else:\n        subj_vals = np.tile(validities, (n_samples, 1))\n        \n    # Sort cues by subjective validities descending\n    orders = np.argsort(subj_vals, axis=1)[:, ::-1]\n    \n    p_a = 0.0\n    for i in range(n_samples):\n        order = orders[i]\n        for idx in order:\n            if a[idx] > b[idx]:\n                p_a += 1.0\n                break\n            elif b[idx] > a[idx]:\n                break\n        else:\n            p_a += 0.5\n            \n    p_a /= n_samples\n    p_core = np.array([p_a, 1.0 - p_a])\n    \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "sigma": "[0.0, 0.5]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this theory discards the Downstream Tallying Mixture and replaces it with Probabilistic Cue Validity. The empirical data strongly favors pure TTB over compensatory mixtures in Experiments 5-8, but pure TTB struggles with the variance in Experiments 1-4. By introducing memory noise (sigma) to the cue validities, the subjective order of cues fluctuates across trials. Options with a higher tally of positive cues naturally have a statistically higher chance of having one of their positive cues jump to the top of the hierarchy. This produces tally-like behavior probabilistically without ever calculating a compensatory sum, aligning perfectly with the observed data while maintaining a strict non-compensatory stopping rule on every individual trial."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11517,
  "candidates_token_count": 812,
  "total_token_count": 14441
}
```
