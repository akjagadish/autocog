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

Propose a brand-new theory that models inter- or intra-subject strategy selection between Tallying and Take-The-Best (TTB). Since pure Tallying with high noise captures the data well but leaves slight systematic gaps, a hybrid theory where decision-makers probabilistically sample between a pure Tallying heuristic and a Lexicographic (TTB) heuristic on a trial-by-trial basis might better capture the small deviations from pure Tallying without needing the complex subjective weighting of WADD.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.996)

**Description**
Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.01, 1.0]
- epsilon: [0.3, 0.8]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4249 (var=0.0029) vs this=0.4247 (var=0.0036)
- Experiment 2: real=0.4027 (var=0.0071) vs this=0.4300 (var=0.0059)
- Experiment 3: real=-0.0383 (var=0.0218) vs this=0.0183 (var=0.0170)
- Experiment 4: real=0.4012 (var=0.0049) vs this=0.4512 (var=0.0076)
- Experiment 5: real=0.4294 (var=0.0047) vs this=0.4267 (var=0.0064)
- Experiment 6: real=0.4069 (var=0.0041) vs this=0.3997 (var=0.0059)
- Experiment 7: real=0.4333 (var=0.0072) vs this=0.4575 (var=0.0125)
- Experiment 8: real=0.4888 (var=0.0062) vs this=0.5081 (var=0.0059)


---

### `pi_5` (overall score: 0.750)

**Description**
Weighted Additive (WADD) Decision Theory with Flexible Sigmoid Subjective Validity Transformation

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    
    # Transform raw probabilities into subjective weights using a sigmoid function
    w = 1.0 / (1.0 + np.exp(-gamma * (validities - delta)))
    
    # Calculate the overall score for each option by multiplying cue values by subjective weights
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice based on the weighted scores
    z = beta * scores
    z = z - np.max(z)  # for numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate for uniform guessing
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.1, 5.0]
- delta: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4249 (var=0.0029) vs this=0.3791 (var=0.0085)
- Experiment 2: real=0.4027 (var=0.0071) vs this=0.3173 (var=0.0229)
- Experiment 3: real=-0.0383 (var=0.0218) vs this=0.1333 (var=0.0460)
- Experiment 4: real=0.4012 (var=0.0049) vs this=0.4788 (var=0.0426)
- Experiment 5: real=0.4294 (var=0.0047) vs this=0.3328 (var=0.0179)
- Experiment 6: real=0.4069 (var=0.0041) vs this=0.2622 (var=0.0181)
- Experiment 7: real=0.4333 (var=0.0072) vs this=0.3125 (var=0.0237)
- Experiment 8: real=0.4888 (var=0.0062) vs this=0.6525 (var=0.0217)


---

### `pi_2` (overall score: 0.635)

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
- Experiment 1: real=0.4249 (var=0.0029) vs this=0.3453 (var=0.0144)
- Experiment 2: real=0.4027 (var=0.0071) vs this=0.2787 (var=0.0354)
- Experiment 3: real=-0.0383 (var=0.0218) vs this=0.2392 (var=0.1175)
- Experiment 4: real=0.4012 (var=0.0049) vs this=0.6331 (var=0.0834)
- Experiment 5: real=0.4294 (var=0.0047) vs this=0.3086 (var=0.0191)
- Experiment 6: real=0.4069 (var=0.0041) vs this=0.2975 (var=0.0181)
- Experiment 7: real=0.4333 (var=0.0072) vs this=0.4525 (var=0.0455)
- Experiment 8: real=0.4888 (var=0.0062) vs this=0.6913 (var=0.0480)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are already sorted by validity in descending order.
        # Find the first discriminating cue.
        ttb_choice = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            if resp == ttb_choice:
                ttb_consistent += 1
            total += 1
            
    return ttb_consistent / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4249 (var=0.0029)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8504 (var=0.0082)
- pi_2: 0.3453 (var=0.0144)
- pi_3: 0.4247 (var=0.0036)
- pi_4: 0.7380 (var=0.0192)
- pi_5: 0.3791 (var=0.0085)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = [0.9, 0.8, 0.7, 0.6]
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice
        ttb = -1
        for i in range(4):
            if a[i] > b[i]: 
                ttb = 0
                break
            elif b[i] > a[i]: 
                ttb = 1
                break
                
        # Expected WADD choice (with equal subjective weights)
        score_a = sum(a[i] * validities[i] for i in range(4))
        score_b = sum(b[i] * validities[i] for i in range(4))
        wadd = 0 if score_a > score_b else (1 if score_b > score_a else -1)
        
        # Only consider trials where TTB and WADD diverge
        if ttb != -1 and wadd != -1 and ttb != wadd:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.4027 (var=0.0071)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2787 (var=0.0354)
- pi_1: 0.8543 (var=0.0110)
- pi_3: 0.4300 (var=0.0059)
- pi_4: 0.7050 (var=0.0209)
- pi_5: 0.3173 (var=0.0229)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    def get_mask(a_target, b_target):
        m1 = (a_str == a_target) & (b_str == b_target)
        m2 = (a_str == b_target) & (b_str == a_target)
        return m1, m2
    
    t1_m1, t1_m2 = get_mask('00111', '11000')
    t5_m1, t5_m2 = get_mask('00110', '00001')
    t7_m1, t7_m2 = get_mask('01110', '10001')
    t8_m1, t8_m2 = get_mask('00111', '10100')
    
    def get_p_a(m1, m2):
        mask = m1 | m2
        if not mask.any():
            return 0.5
        chose_a = np.where(m1, data['response'] == 0, data['response'] == 1)
        return float(chose_a[mask].mean())
    
    p_a_t1 = get_p_a(t1_m1, t1_m2)
    p_a_t5 = get_p_a(t5_m1, t5_m2)
    p_a_t7 = get_p_a(t7_m1, t7_m2)
    p_a_t8 = get_p_a(t8_m1, t8_m2)
    
    group1_p = (p_a_t5 + p_a_t7) / 2.0
    group2_p = (p_a_t1 + p_a_t8) / 2.0
    
    return float(group1_p - group2_p)
```

**Observed (real) value:** -0.0383 (var=0.0218)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0183 (var=0.0170)
- pi_2: 0.2392 (var=0.1175)
- pi_1: 0.3675 (var=0.0181)
- pi_4: 0.2325 (var=0.0240)
- pi_5: 0.1333 (var=0.0460)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t6_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    t1_wadd_match = (data['response'] == 0) & t1_mask
    t6_wadd_match = (data['response'] == 1) & t6_mask
    
    relevant_trials = t1_mask | t6_mask
    if not relevant_trials.any():
        return 0.5
        
    return float((t1_wadd_match.sum() + t6_wadd_match.sum()) / relevant_trials.sum())
```

**Observed (real) value:** 0.4012 (var=0.0049)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6331 (var=0.0834)
- pi_3: 0.4512 (var=0.0076)
- pi_1: 0.8544 (var=0.0104)
- pi_4: 0.7106 (var=0.0257)
- pi_5: 0.4788 (var=0.0426)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_mat.sum(axis=1)
    sum_b = b_mat.sum(axis=1)
    
    tally_choice = np.zeros(len(data)) - 1
    tally_choice[sum_a > sum_b] = 0
    tally_choice[sum_b > sum_a] = 1
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    has_diff = np.any(non_zero, axis=1)
    
    ttb_choice = np.zeros(len(data)) - 1
    row_indices = np.arange(len(data))
    first_diff_vals = diff[row_indices, first_diff_idx]
    
    ttb_choice[(has_diff) & (first_diff_vals > 0)] = 0
    ttb_choice[(has_diff) & (first_diff_vals < 0)] = 1
    
    conflict = (tally_choice != -1) & (ttb_choice != -1) & (tally_choice != ttb_choice)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[conflict] == ttb_choice[conflict])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4294 (var=0.0047)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4267 (var=0.0064)
- pi_4: 0.7261 (var=0.0194)
- pi_1: 0.8586 (var=0.0086)
- pi_2: 0.3086 (var=0.0191)
- pi_5: 0.3328 (var=0.0179)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return np.nan
        
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.4069 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7066 (var=0.0179)
- pi_3: 0.3997 (var=0.0059)
- pi_1: 0.8631 (var=0.0104)
- pi_2: 0.2975 (var=0.0181)
- pi_5: 0.2622 (var=0.0181)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Convert lists to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Identify Trial 1 and Trial 2 where Option A has the highest-validity feature 
    # but fewer total features than Option B.
    t1_a = (1, 1, 0, 0, 0)
    t2_a = (1, 0, 0, 0, 0)
    
    target_trials = data[a_tuples.isin([t1_a, t2_a])]
    
    if len(target_trials) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return (target_trials['response'] == 0).mean()

```

**Observed (real) value:** 0.4333 (var=0.0072)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4575 (var=0.0125)
- pi_5: 0.3125 (var=0.0237)
- pi_1: 0.8700 (var=0.0112)
- pi_2: 0.4525 (var=0.0455)
- pi_4: 0.7433 (var=0.0302)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t3_a = (1, 1, 1, 0, 0, 0)
    t3_b = (0, 0, 0, 1, 1, 1)
    
    t4_a = (1, 0, 0, 0, 0, 0)
    t4_b = (0, 0, 0, 0, 0, 1)
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t3_a and b == t3_b:
            total += 1
            if resp == 0: match_count += 1
        elif a == t3_b and b == t3_a:
            total += 1
            if resp == 1: match_count += 1
            
        elif a == t4_a and b == t4_b:
            total += 1
            if resp == 0: match_count += 1
        elif a == t4_b and b == t4_a:
            total += 1
            if resp == 1: match_count += 1
            
    if total == 0:
        return 0.5
        
    return float(match_count / total)
```

**Observed (real) value:** 0.4888 (var=0.0062)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6525 (var=0.0217)
- pi_3: 0.5081 (var=0.0059)
- pi_1: 0.8762 (var=0.0132)
- pi_2: 0.6913 (var=0.0480)
- pi_4: 0.7619 (var=0.0258)


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
  "theory": "Decision-makers do not rely on a single strategy; instead, they probabilistically sample between a compensatory Equal-Weight (Tallying) heuristic and a non-compensatory Lexicographic (Take-The-Best) heuristic on a trial-by-trial basis. While Tallying accounts for the majority of choices by simply counting the number of positive features, occasional reliance on Take-The-Best captures systematic deviations where subjects prioritize the most valid discriminating cue. This hybrid strategy avoids the cognitive complexity of computing subjective weights (as in WADD) while providing a more nuanced fit than pure Tallying with uniform noise.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Tallying: sum the unweighted feature values for each option\n    scores_tally = stim.sum(axis=1)\n    beta = float(parameters[\"beta\"])\n    z = beta * (scores_tally - scores_tally.max())\n    e = np.exp(z)\n    p_tally = e / e.sum()\n    \n    # Take-The-Best (TTB): find the first discriminating cue ordered by validity\n    order = np.argsort(validities)[::-1]\n    diff = stim[0, order] - stim[1, order]\n    non_zero = np.where(diff != 0)[0]\n    \n    p_ttb = np.array([0.5, 0.5])\n    if len(non_zero) > 0:\n        first_diff = diff[non_zero[0]]\n        if first_diff > 0:\n            p_ttb = np.array([1.0, 0.0])\n        else:\n            p_ttb = np.array([0.0, 1.0])\n            \n    # Mixture of Tallying and TTB\n    w_tally = float(parameters[\"w_tally\"])\n    p_mix = w_tally * p_tally + (1.0 - w_tally) * p_ttb\n    \n    # Global lapse rate\n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = len(p_mix)\n    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.01, 5.0]",
    "w_tally": "[0.0, 1.0]",
    "epsilon": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this model instantiates a hybrid theory where subjects dynamically switch between Tallying and Take-The-Best (TTB) on a trial-by-trial basis. Pure Tallying modeled the data well but left slight systematic gaps because it uniformly penalized all deviations. By explicitly incorporating TTB as an alternative heuristic (with probability 1 - w_tally), the model can capture choices where the subject relies strictly on the most valid discriminating feature, without invoking the complex cardinal weighting of WADD. A global lapse rate (epsilon) accounts for general choice stochasticity."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11370,
  "candidates_token_count": 877,
  "total_token_count": 14494
}
```
