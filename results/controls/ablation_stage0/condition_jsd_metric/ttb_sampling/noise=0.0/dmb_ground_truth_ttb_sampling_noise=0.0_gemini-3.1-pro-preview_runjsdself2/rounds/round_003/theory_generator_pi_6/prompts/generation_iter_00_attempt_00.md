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

Propose a new theory to replace the Weighted Additive model. Since Take-The-Best (THEORY 1) is highly successful, the new theory should be a strong competing heuristic from the fast-and-frugal heuristics literature, such as the 'Tallying' (Equal Weights) heuristic. Tallying ignores cue validities entirely and simply counts the number of positive features for each option, choosing the one with the higher count (with ties broken randomly or via a secondary mechanism). Alternatively, consider a 'Lexicographic Semi-Order' model, which operates like TTB but assumes a discrimination threshold (features only discriminate if the difference in their values exceeds a certain epsilon, though here features are binary so this might reduce to a probabilistic TTB). Tallying is a standard baseline that contrasts with both TTB (which uses validities but ignores most cues) and WADD (which uses validities and integrates all cues).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.978)

**Description**
Take-The-Best (TTB) proposes that individuals use a non-compensatory lexicographic heuristic for decision making. They search through features in descending order of their validity and base their choice entirely on the first feature that discriminates between the two options, ignoring all remaining features. If no feature discriminates, they guess. Response noise is incorporated via a softmax function over the resulting binary scores and a lapse rate for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.001, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1100 (var=0.0022) vs this=0.1011 (var=0.0025)
- Experiment 2: real=0.0623 (var=0.0021) vs this=0.0744 (var=0.0030)
- Experiment 3: real=0.0045 (var=0.0002) vs this=0.0004 (var=0.0002)
- Experiment 4: real=0.1288 (var=0.0025) vs this=0.1342 (var=0.0035)
- Experiment 5: real=0.0024 (var=0.0003) vs this=0.0010 (var=0.0002)
- Experiment 6: real=0.0774 (var=0.0008) vs this=0.0633 (var=0.0013)
- Experiment 7: real=0.0017 (var=0.0001) vs this=0.0011 (var=0.0002)
- Experiment 8: real=0.2047 (var=0.0037) vs this=0.1799 (var=0.0045)


---

### `pi_4` (overall score: 0.435)

**Description**
Two-Stage Heuristic (TTB with Rare Tallying Fallback)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Identify discriminating cues
    disc_cues = []
    for idx in order:
        if a[idx] != b[idx]:
            disc_cues.append(idx)
            
    delta_v = float(parameters["delta_v"])
    
    # Determine whether to fall back to tallying
    use_tally = False
    if len(disc_cues) >= 2:
        v1 = validities[disc_cues[0]]
        v2 = validities[disc_cues[1]]
        if (v1 - v2) < delta_v:
            use_tally = True
    elif len(disc_cues) == 0:
        use_tally = True
        
    scores = np.array([0.0, 0.0])
    if use_tally:
        # Tallying: count strict feature-wise wins
        scores[0] = float(np.sum(a > b))
        scores[1] = float(np.sum(b > a))
    else:
        # TTB: use the single best discriminating cue
        if len(disc_cues) > 0:
            best_cue = disc_cues[0]
            if a[best_cue] > b[best_cue]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- delta_v: [0.0, 0.1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1100 (var=0.0022) vs this=0.1145 (var=0.0020)
- Experiment 2: real=0.0623 (var=0.0021) vs this=0.0092 (var=0.0022)
- Experiment 3: real=0.0045 (var=0.0002) vs this=0.0029 (var=0.0002)
- Experiment 4: real=0.1288 (var=0.0025) vs this=0.1271 (var=0.0041)
- Experiment 5: real=0.0024 (var=0.0003) vs this=0.0718 (var=0.0053)
- Experiment 6: real=0.0774 (var=0.0008) vs this=0.0041 (var=0.0009)
- Experiment 7: real=0.0017 (var=0.0001) vs this=0.0303 (var=0.0019)
- Experiment 8: real=0.2047 (var=0.0037) vs this=0.1238 (var=0.0050)


---

### `pi_5` (overall score: 0.359)

**Description**
Weighted Additive (WADD) model with Log-Odds Weights: Decision makers integrate all available information by computing a fully compensatory weighted sum of the features for each option. To properly scale the importance of each cue, objective validities (probabilities) are transformed into log-odds. This ensures that non-predictive cues (validity = 0.5) receive a weight of zero and do not distort the evaluation. The option with the higher weighted sum is favored, with choice probabilities generated via a softmax function to account for decision noise, alongside a uniform lapse rate for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds for proper weighting
    # Clip to avoid log(0) or division by zero
    v_clipped = np.clip(validities, 1e-5, 1.0 - 1e-5)
    weights = np.log(v_clipped / (1.0 - v_clipped))
    
    # WADD: compute the weighted sum of features for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.001, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1100 (var=0.0022) vs this=0.1057 (var=0.0017)
- Experiment 2: real=0.0623 (var=0.0021) vs this=0.0812 (var=0.0033)
- Experiment 3: real=0.0045 (var=0.0002) vs this=0.0520 (var=0.0005)
- Experiment 4: real=0.1288 (var=0.0025) vs this=0.0727 (var=0.0013)
- Experiment 5: real=0.0024 (var=0.0003) vs this=0.1134 (var=0.0020)
- Experiment 6: real=0.0774 (var=0.0008) vs this=0.0238 (var=0.0004)
- Experiment 7: real=0.0017 (var=0.0001) vs this=0.1259 (var=0.0023)
- Experiment 8: real=0.2047 (var=0.0037) vs this=0.0006 (var=0.0001)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|0': 0.5018703241895262, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))|1': 0.5100200400801603, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.13202933985330073, '((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.13440514469453377, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.1417437895762299, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.13316095669036845, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|0': 0.8727193744569939, '((0, 0, 0, 1, 1), (1, 1, 1, 0, 0))|1': 0.8713405238828967, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8623737373737373, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8715277777777778, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.507400828892836, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.4845630559916274, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.4828744123572868, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.4870237437879624, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|0': 0.523680649526387, '((0, 1, 0, 1, 0), (1, 0, 1, 0, 0))|1': 0.49858623939679547}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.1100 (var=0.0022)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0005 (var=0.0001)
- pi_2: 0.0221 (var=0.0020)
- pi_3: 0.1011 (var=0.0025)
- pi_4: 0.1145 (var=0.0020)
- pi_5: 0.1057 (var=0.0017)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.5449591280653951, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.6626633698339809, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.39959839357429716, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.45427728613569324, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.38930517711171664, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.4015918958031838, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.49880260006842286, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.43356139719121356, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.5448098001289491, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.5173210161662818}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0623 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0008 (var=0.0019)
- pi_1: 0.0354 (var=0.0007)
- pi_3: 0.0744 (var=0.0030)
- pi_4: 0.0092 (var=0.0022)
- pi_5: 0.0812 (var=0.0033)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.16174466151749206, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.18656182987848463, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.16737698566447112, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.20804710500490678, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.8371098760153912, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.7929240374609782, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.15610766045548655, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.20337552742616033, '((1, 0, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.15672178289000718, '((1, 0, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.2114914425427873, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.16711486746062235, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.18856569709127383, '((1, 1, 0, 0, 0), (0, 0, 0, 1, 1))|0': 0.18994413407821228, '((1, 1, 0, 0, 0), (0, 0, 0, 1, 1))|1': 0.17244846656611362, '((0, 0, 0, 1, 0), (1, 0, 0, 0, 0))|0': 0.8334047109207708, '((0, 0, 0, 1, 0), (1, 0, 0, 0, 0))|1': 0.8031620553359684}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0045 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0004 (var=0.0002)
- pi_2: 0.1149 (var=0.0036)
- pi_1: 0.1978 (var=0.0028)
- pi_4: 0.0029 (var=0.0002)
- pi_5: 0.0520 (var=0.0005)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.8224527150701647, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8326309452137267, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.6560088202866593, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.5923852183650616, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.6092219020172911, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.6123324396782842, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.3986083499005964, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.3822418136020151, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.6947314049586777, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.6442307692307693, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|0': 0.17918088737201365, '((0, 1, 1, 1, 1), (1, 0, 0, 0, 0))|1': 0.1498371335504886, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.4097051597051597, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.3433062880324544, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.336104513064133, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.33611691022964507}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.1288 (var=0.0025)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0018 (var=0.0015)
- pi_3: 0.1342 (var=0.0035)
- pi_1: 0.0195 (var=0.0008)
- pi_4: 0.1271 (var=0.0041)
- pi_5: 0.0727 (var=0.0013)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.1551562792438705, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.2154011847065159, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|0': 0.834345186470078, '((0, 1, 1, 1, 0), (1, 0, 0, 0, 1))|1': 0.8064142194744977, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.16761862917398945, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.18693353474320243, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.17256433823529413, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.18720565149136578}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0024 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0010 (var=0.0002)
- pi_4: 0.0718 (var=0.0053)
- pi_1: 0.2081 (var=0.0042)
- pi_2: 0.1351 (var=0.0068)
- pi_5: 0.1134 (var=0.0020)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.6228886590830748, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.819935691318328, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|0': 0.15285379202501956, '((1, 0, 1, 1, 1), (0, 1, 0, 0, 0))|1': 0.12571610439210693, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|0': 0.39766081871345027, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 1))|1': 0.6946693657219973, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.3609512850019179, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.7132234076947948, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.5852514919011083, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8163064833005894}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0774 (var=0.0008)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0041 (var=0.0009)
- pi_3: 0.0633 (var=0.0013)
- pi_1: 0.0271 (var=0.0006)
- pi_2: 0.0133 (var=0.0006)
- pi_5: 0.0238 (var=0.0004)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.15884353741496599, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.2318181818181818, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|0': 0.15529411764705883, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))|1': 0.2416, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|0': 0.1573109243697479, '((1, 0, 0, 0, 1), (0, 1, 1, 1, 0))|1': 0.2512, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.15580926796507724, '((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.25241157556270094, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.15680973301791146, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.30109204368174725, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.16162361623616237, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.24067796610169492, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|0': 0.1705662936588674, '((0, 1, 0, 1, 0), (0, 0, 1, 0, 1))|1': 0.250384024577573, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.16967871485943775, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.23529411764705882}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.0017 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0011 (var=0.0002)
- pi_5: 0.1259 (var=0.0023)
- pi_1: 0.1816 (var=0.0028)
- pi_2: 0.1163 (var=0.0032)
- pi_4: 0.0303 (var=0.0019)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|0': 0.8635346756152126, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 0))|1': 0.8649845064187693, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.8753918495297806, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.8646245059288538, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 1))|0': 0.8346456692913385, '((1, 0, 0, 0, 0), (0, 1, 1, 0, 1))|1': 0.8620268620268621, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|0': 0.2277542372881356, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 0))|1': 0.17996987951807228, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8466367713004485, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.8631790744466801, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.7530054644808744, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.7981378026070763, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.2550335570469799, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.2139689578713969, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.8418688230008985, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.856453558504222}
def metric(data):
    import numpy as np
    def jsd2(p, q):
        # Jensen-Shannon divergence (nats) between Bernoulli(p), Bernoulli(q)
        v = 0.0
        for x, y in ((1.0 - p, 1.0 - q), (p, q)):
            m = 0.5 * (x + y)
            if x > 0:
                v += 0.5 * x * np.log(x / m)
            if y > 0:
                v += 0.5 * y * np.log(y / m)
        return float(v)
    sums, counts = {}, {}
    for _, subj in data.groupby('subject_id', sort=False):
        a = list(subj['option_a_ratings'])
        b = list(subj['option_b_ratings'])
        r = subj['response'].to_numpy(dtype=int)
        for t in range(1, len(r)):
            key = str((tuple(a[t]), tuple(b[t]))) + '|' + str(int(r[t-1]))
            sums[key] = sums.get(key, 0) + int(r[t])
            counts[key] = counts.get(key, 0) + 1
    num = den = 0.0
    for k, n in counts.items():
        num += n * jsd2(sums[k] / n, P_REF.get(k, 0.5))
        den += n
    return float(num / den) if den else 0.0

```

**Observed (real) value:** 0.2047 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0006 (var=0.0001)
- pi_3: 0.1799 (var=0.0045)
- pi_1: 0.0172 (var=0.0002)
- pi_2: 0.0282 (var=0.0019)
- pi_4: 0.1238 (var=0.0050)


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
  "theory": "Tallying (Equal Weights) heuristic: Decision makers ignore cue validities entirely and simply count the number of positive features for each option. They choose the option with the higher tally (i.e., the greatest number of positive cues). This represents a fast-and-frugal compensatory strategy where all cues are weighted equally, minimizing cognitive effort while still integrating all available features. Response noise is incorporated via a softmax function over the tallies, alongside a uniform lapse rate for random errors.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    \n    # Tallying: count the number of positive features for each option\n    scores = np.array([np.sum(a), np.sum(b)], dtype=float)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n    \n    # Blend with uniform lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probs = np.asarray(probabilities, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.001, 10.0]",
    "epsilon": "[0.0, 0.5]"
  },
  "rationale": "Following the arbiter's suggestion, the Weighted Additive model was replaced with the Tallying heuristic. Tallying represents a major alternative in the fast-and-frugal heuristics literature. Instead of weighting cues by their validities (like WADD) or relying on a single best cue (like TTB), Tallying integrates all cues but weights them equally. This is operationalized by simply summing the positive features for each option and applying a softmax choice rule over the totals. It contrasts mechanistically with both TTB and WADD, offering a robust baseline that handles environments where validities are either unknown, ignored, or too computationally costly to integrate."
}
```
```

## Usage

```json
{
  "prompt_token_count": 17850,
  "candidates_token_count": 642,
  "total_token_count": 19559
}
```
