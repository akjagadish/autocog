# generation_iter_02_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a completely new theory that uses a different mechanism from TTB. For example, a noisy Weighted Additive (WADD) model where subjects integrate all cues weighted by their validities but with late-stage decision noise, or a two-stage heuristic (e.g., Tallying first, and using TTB only to break ties). This will test if a compensatory or hybrid approach can rival the strict non-compensatory TTB model.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.992)

**Description**
People use a non-compensatory, lexicographic heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of their subjective validity and stop at the first feature that discriminates between the two options (i.e., one option has a positive rating and the other does not). The choice is based entirely on this single discriminating cue, completely ignoring all remaining features, regardless of how many lower-validity cues might favor the alternative.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    # Deterministic choice based on the first discriminating cue
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # If all features tie, guess randomly
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse rate for noise
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1311 (var=0.0202) vs this=0.1222 (var=0.0114)
- Experiment 2: real=-0.6650 (var=0.0405) vs this=-0.7075 (var=0.0449)
- Experiment 3: real=0.1600 (var=0.0219) vs this=0.1425 (var=0.0219)
- Experiment 4: real=0.1333 (var=0.0092) vs this=0.1267 (var=0.0090)
- Experiment 5: real=0.8375 (var=0.0070) vs this=0.8883 (var=0.0058)
- Experiment 6: real=0.1500 (var=0.0087) vs this=0.1217 (var=0.0073)
- Experiment 7: real=0.1229 (var=0.0065) vs this=0.1260 (var=0.0054)
- Experiment 8: real=0.0033 (var=0.0153) vs this=0.0033 (var=0.0180)


---

### `pi_4` (overall score: 0.591)

**Description**
Probabilistic Take-The-Best (PTTB) with Softmax Smoothing: Subjects use a non-compensatory lexicographic search to choose between options, but their subjective representation of cue validities is subject to slight trial-by-trial noise. The cue hierarchy is determined by sorting the noisy validities. This causes occasional inversions in the search order (especially for cues with similar validities). Instead of a uniform lapse rate, the resulting deterministic choices are smoothed via a softmax function with a temperature parameter, allowing for a consistent baseline deviation from strict TTB without over-penalizing high-confidence predictions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"PTTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    temperature = float(parameters["temperature"])
    
    n_samples = 100
    p_core = np.zeros(2)
    
    for _ in range(n_samples):
        # Add Gaussian noise to the validities to simulate subjective trial-by-trial hierarchy
        noisy_v = validities + np.random.normal(0, sigma, size=len(validities))
        order = np.argsort(noisy_v)[::-1]
        
        chosen = -1
        for idx in order:
            if a[idx] > b[idx]:
                chosen = 0
                break
            elif b[idx] > a[idx]:
                chosen = 1
                break
                
        if chosen == 0:
            p_core[0] += 1.0
        elif chosen == 1:
            p_core[1] += 1.0
        else:
            p_core += 0.5
            
    p_core /= n_samples
    
    # Apply softmax with temperature to the core probabilities for smoother deviation
    z = p_core / temperature
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- sigma: [0.0, 0.1]
- temperature: [0.1, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1311 (var=0.0202) vs this=0.1367 (var=0.0146)
- Experiment 2: real=-0.6650 (var=0.0405) vs this=-0.7492 (var=0.0451)
- Experiment 3: real=0.1600 (var=0.0219) vs this=0.1725 (var=0.0174)
- Experiment 4: real=0.1333 (var=0.0092) vs this=0.1358 (var=0.0136)
- Experiment 5: real=0.8375 (var=0.0070) vs this=0.7529 (var=0.0208)
- Experiment 6: real=0.1500 (var=0.0087) vs this=0.1875 (var=0.0156)
- Experiment 7: real=0.1229 (var=0.0065) vs this=0.3148 (var=0.0217)
- Experiment 8: real=0.0033 (var=0.0153) vs this=0.2050 (var=0.0648)


---

### `pi_2_1` (overall score: 0.299)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson 1993).
    # Stimulus is the pair of option feature vectors for the current trial:
    # array-like of shape (2, n_features), row 0 = option A, row 1 = option B.
    # Each option's score is the dot product of its feature vector with the
    # subjective validity weights (which are given directly by the experiment).
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted sum per option (dot product with validities directly)
    scores = stim @ validities

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.05]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1311 (var=0.0202) vs this=0.1233 (var=0.0232)
- Experiment 2: real=-0.6650 (var=0.0405) vs this=-0.6142 (var=0.0387)
- Experiment 3: real=0.1600 (var=0.0219) vs this=0.1650 (var=0.0384)
- Experiment 4: real=0.1333 (var=0.0092) vs this=0.2058 (var=0.0284)
- Experiment 5: real=0.8375 (var=0.0070) vs this=0.0154 (var=0.0008)
- Experiment 6: real=0.1500 (var=0.0087) vs this=0.9683 (var=0.0036)
- Experiment 7: real=0.1229 (var=0.0065) vs this=0.8765 (var=0.0006)
- Experiment 8: real=0.0033 (var=0.0153) vs this=0.0350 (var=0.0111)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2905 -> ACCEPTED
- iter 2: loss=0.2854 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.2854 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers A (0), WADD prefers B (1)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        # Trial 2: Tallying prefers B (1), WADD prefers A (0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.1311 (var=0.0202)
**Previous candidate values (this loop):**
  - iter 1: 0.1389 (var=0.0145) (Δ vs real +0.0078)
  - iter 2 (most recent): 0.1967 (var=0.0269) (Δ vs real +0.0656)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8544 (var=0.0163)
- pi_2: 0.4400 (var=0.0801)
- pi_2_1: 0.1233 (var=0.0232)
- pi_3: 0.1222 (var=0.0114)
- pi_4: 0.1367 (var=0.0146)

### Experiment 2
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: Tallying prefers A (3 vs 2 wins), WADD prefers B (higher validity sum)
    t1_a = (0, 0, 0, 1, 1, 1)
    
    # Trial 5: Tallying prefers B (3 vs 2 wins), WADD prefers A (higher validity sum)
    t5_a = (1, 1, 0, 0, 0, 0)
    
    t1_mask = a_tuples == t1_a
    t5_mask = a_tuples == t5_a
    
    t1_resp = data.loc[t1_mask, 'response']
    t5_resp = data.loc[t5_mask, 'response']
    
    if len(t1_resp) == 0 or len(t5_resp) == 0:
        return 0.0
        
    p_a_t1 = (t1_resp == 0).mean()
    p_a_t5 = (t5_resp == 0).mean()
    
    # Tallying: P(A|T1) is high, P(A|T5) is low -> Positive difference
    # WADD: P(A|T1) is low, P(A|T5) is high -> Negative difference
    return float(p_a_t1 - p_a_t5)
```

**Observed (real) value:** -0.6650 (var=0.0405)
**Previous candidate values (this loop):**
  - iter 1: -0.7325 (var=0.0370) (Δ vs real -0.0675)
  - iter 2 (most recent): -0.8883 (var=0.0105) (Δ vs real -0.2233)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1150 (var=0.2399)
- pi_1: 0.6958 (var=0.0454)
- pi_2_1: -0.6142 (var=0.0387)
- pi_3: -0.7075 (var=0.0449)
- pi_4: -0.7492 (var=0.0451)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 pits an option A with 3 low-validity features against an option B with 2 high-validity features.
    # Tallying prefers A (3 wins vs 2 wins), whereas WADD prefers B (score 1.7 vs 1.8).
    is_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if is_target.sum() == 0:
        return 0.5
    return float((data.loc[is_target, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1600 (var=0.0219)
**Previous candidate values (this loop):**
  - iter 1: 0.1425 (var=0.0194) (Δ vs real -0.0175)
  - iter 2 (most recent): 0.0425 (var=0.0048) (Δ vs real -0.1175)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8300 (var=0.0267)
- pi_2_1: 0.1650 (var=0.0384)
- pi_2: 0.4075 (var=0.1143)
- pi_3: 0.1425 (var=0.0219)
- pi_4: 0.1725 (var=0.0174)

### Experiment 4
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify critical trials (Trial 1 and Trial 2) where A is [0, 0, 0, 1, 1, 1]
    # and B has the first feature as 1 (either [1, 1, 0, 0, 0, 0] or [1, 0, 1, 0, 0, 0]).
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_B_starts_1 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    mask = is_A_target & is_B_starts_1
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1333 (var=0.0092)
**Previous candidate values (this loop):**
  - iter 1: 0.1283 (var=0.0096) (Δ vs real -0.0050)
  - iter 2 (most recent): 0.2117 (var=0.0286) (Δ vs real +0.0783)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.2058 (var=0.0284)
- pi_1: 0.8600 (var=0.0114)
- pi_2: 0.4858 (var=0.0768)
- pi_3: 0.1267 (var=0.0090)
- pi_4: 0.1358 (var=0.0136)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_choices = 0
    total = 0
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) > 0:
            first_idx = non_zero[0]
            if diff[first_idx] > 0 and np.sum(b) > np.sum(a):
                total += 1
                if resp == 0:
                    ttb_choices += 1
            elif diff[first_idx] < 0 and np.sum(a) > np.sum(b):
                total += 1
                if resp == 1:
                    ttb_choices += 1
                    
    return float(ttb_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8375 (var=0.0070)
**Previous candidate values (this loop):**
  - iter 1: 0.4779 (var=0.0438) (Δ vs real -0.3596)
  - iter 2 (most recent): 0.5346 (var=0.0180) (Δ vs real -0.3029)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8883 (var=0.0058)
- pi_2_1: 0.0154 (var=0.0008)
- pi_1: 0.1521 (var=0.0097)
- pi_2: 0.2225 (var=0.0247)
- pi_4: 0.7529 (var=0.0208)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1500 (var=0.0087)
**Previous candidate values (this loop):**
  - iter 1: 0.3342 (var=0.0357) (Δ vs real +0.1842)
  - iter 2 (most recent): 0.3150 (var=0.0171) (Δ vs real +0.1650)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.9683 (var=0.0036)
- pi_3: 0.1217 (var=0.0073)
- pi_1: 0.8271 (var=0.0125)
- pi_2: 0.6958 (var=0.0269)
- pi_4: 0.1875 (var=0.0156)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Strict TTB always predicts option A (response 0) for all trials in this design
    # because the first discriminating cue always favors A.
    # Therefore, any response of 1 (choosing B) is due to noise.
    # PTTB will have a much higher rate of choosing B due to validity swaps
    # (especially when the gap between validities is small) and softmax smoothing.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.1229 (var=0.0065)
**Previous candidate values (this loop):**
  - iter 1: 0.5790 (var=0.0189) (Δ vs real +0.4560)
  - iter 2 (most recent): 0.4925 (var=0.0123) (Δ vs real +0.3696)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1260 (var=0.0054)
- pi_4: 0.3148 (var=0.0217)
- pi_1: 0.8090 (var=0.0106)
- pi_2: 0.7154 (var=0.0221)
- pi_2_1: 0.8765 (var=0.0006)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1))
    
    # Response is 0 for A, 1 for B. 
    # mean(response) is the rate of choosing option B.
    mean_t1 = data[t1_mask]['response'].mean()
    mean_t2 = data[t2_mask]['response'].mean()
    
    return float(mean_t1 - mean_t2)
```

**Observed (real) value:** 0.0033 (var=0.0153)
**Previous candidate values (this loop):**
  - iter 1: 0.5333 (var=0.0597) (Δ vs real +0.5300)
  - iter 2 (most recent): 0.1600 (var=0.0850) (Δ vs real +0.1567)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2050 (var=0.0648)
- pi_3: 0.0033 (var=0.0180)
- pi_1: 0.0033 (var=0.0089)
- pi_2: 0.2083 (var=0.0887)
- pi_2_1: 0.0350 (var=0.0111)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Power-law Weighted Additive Model: Subjects use a fully compensatory integration process where they calculate a weighted sum of all features, but they subjectively scale the stated cue validities using a power-law transformation. By exponentiating the validities, differences between high and low validity cues are amplified. This allows a purely compensatory, additive mechanism to rival and approximate strict non-compensatory behavior (like Take-The-Best) when the exponent is sufficiently large, while still integrating all available information.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Model expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Power-law transformation of validities to amplify importance of top cues
    weights = validities ** gamma
    
    # Compensatory integration (Weighted Additive)
    scores = stim @ weights

    # Softmax decision rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Blended with a uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gamma: [1.0, 100.0]
- beta: [1.0, 50.0]
- epsilon: [0.0, 0.2]
- validities: validities

`rationale`: Following the critic's advice, we increased the upper bound of the gamma parameter to 100.0 and decreased the upper bound of the epsilon parameter to 0.2. This allows the power-law transformation to reach a much steeper, truly non-compensatory regime, helping the additive mechanism more closely mimic strict lexicographic behavior on critical trials while preventing excessive uniform lapse rates from washing out the amplified differences.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The Power-law Weighted Additive model successfully reproduces the qualitative patterns in Experiments 1-4, demonstrating that a compensatory architecture can capture some non-compensatory-like choices. However, in Experiments 5 through 8, the model still behaves too compensatorily. For instance, in Experiment 7, the model incorrectly chooses option B 58% of the time (real data = 12.3%), and in Experiment 8, it shows a large sensitivity to lower-validity cues (metric difference of 0.533) where humans show almost none (0.003). This indicates that the power-law transformation is not steep enough to fully mimic strict non-compensatory behavior.
Rationale: The candidate was accepted and the Power-law WADD is a great implementation of the arbiter's recommendation. However, it still falls short of capturing the strict non-compensatory behavior in Experiments 5-8. To fix this while staying within the prescribed mechanism, increase the upper bound of the `gamma` parameter significantly (e.g., [1.0, 100.0]) so the model can reach a truly non-compensatory regime when fitted. Additionally, reduce the upper bound of the `epsilon` parameter (e.g., [0.0, 0.2]) to prevent high uniform lapse rates from washing out the amplified cue differences.

**Outcome of this advice:** iter 1 candidate loss=0.2905 -> iter 2 candidate loss=0.2854 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The increase in the upper bound of `gamma` successfully improved the model's overall fit (the candidate was ACCEPTED). However, the model still falls short of capturing the highly non-compensatory behavior observed in Experiments 5, 7, and 8. In Experiment 7, for instance, the model still chooses option B 49% of the time (down from 58%, but still far from the human rate of 12.3%). The likely mathematical culprit is the power-law transformation `validities ** gamma`. Because validities are typically fractions (e.g., between 0.5 and 1.0), raising them to a large power shrinks all weights close to zero. This makes the absolute differences in `scores` extremely small, requiring the `beta` parameter to be impossibly large to produce deterministic softmax probabilities.
Rationale: To allow the Weighted Additive mechanism to truly approximate strict non-compensatory behavior without suffering from vanishing weights, change the weight computation to use a softmax function over the validities: `weights = np.exp(gamma * validities) / np.sum(np.exp(gamma * validities))`. This ensures the weights always sum to 1, and when `gamma` is large, the cue with the highest validity will receive a weight near 1.0 while the rest go to 0.0, perfectly mimicking TTB without shrinking the overall score magnitudes. Keep `gamma` in a range like [0.0, 50.0] and retain the current `beta` and `epsilon` bounds. This keeps the model firmly within the prescribed WADD family while fixing the scaling issue.

**Outcome of this advice:** iter 2 candidate loss=0.2854 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

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
  "theory": "Softmax-Weighted Additive Model: Subjects use a fully compensatory integration process where they calculate a weighted sum of all features, but they subjectively scale the stated cue validities using a softmax transformation. By exponentiating and normalizing the validities with a scaling parameter, differences between high and low validity cues are amplified without shrinking the overall magnitudes of the weights. This allows a purely compensatory, additive mechanism to rival and approximate strict non-compensatory behavior (like Take-The-Best) when the scaling parameter is sufficiently large, while still integrating all available information.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Model expects a (2, n_features) stimulus; got shape {stim.shape}.\")\n\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax transformation of validities to amplify importance of top cues\n    # without shrinking the overall magnitudes of the weights to zero.\n    z_v = gamma * validities\n    z_v = z_v - np.max(z_v)\n    e_v = np.exp(z_v)\n    weights = e_v / np.sum(e_v)\n    \n    # Compensatory integration (Weighted Additive)\n    scores = stim @ weights\n\n    # Softmax decision rule\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n\n    # Blended with a uniform lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "gamma": "[0.0, 50.0]",
    "beta": "[1.0, 50.0]",
    "epsilon": "[0.0, 0.2]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we replaced the power-law transformation of validities with a softmax transformation. Because validities are fractions, raising them to a large power caused the resulting weights to vanish, which shrank the absolute differences in option scores to nearly zero and prevented the softmax decision rule from acting deterministically. By using a softmax transformation over the validities, the weights sum to 1, preserving the magnitude of the scores while still allowing the highest-validity cue to dominate the weights when gamma is large. This allows the Weighted Additive mechanism to more successfully approximate strict non-compensatory behavior."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13942,
  "candidates_token_count": 752,
  "total_token_count": 15033
}
```
