# generation_iter_01_attempt_00

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
- THEORY 2 = `pi_2_1`
- The recommendation below acts on THEORY 2 (= `pi_2_1`).

Propose a new non-compensatory heuristic to serve as a stronger competitor to Take-The-Best. A good candidate would be a Probabilistic Take-The-Best model (where the feature search order is stochastic but weighted by validity) or an Elimination by Aspects (EBA) model. This will test whether the strict deterministic ordering of TTB is strictly necessary, or if a more flexible non-compensatory process can explain the minor deviations in the data.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 1.000)

**Description**
Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8844 (var=0.0081) vs this=0.8728 (var=0.0081)
- Experiment 2: real=0.8533 (var=0.0160) vs this=0.8750 (var=0.0083)
- Experiment 3: real=0.1500 (var=0.0150) vs this=0.1625 (var=0.0258)
- Experiment 4: real=0.9033 (var=0.0065) vs this=0.8964 (var=0.0042)
- Experiment 5: real=0.8671 (var=0.0115) vs this=0.8735 (var=0.0058)
- Experiment 6: real=0.1378 (var=0.0101) vs this=0.1311 (var=0.0080)


---

### `pi_2_1` (overall score: 0.730)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson 1993).
    # Stimulus is the pair of option feature vectors for the current trial:
    # array-like of shape (2, n_features), row 0 = option A, row 1 = option B.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective weights are a non-linear scaling of objective validities.
    # gamma > 1 increases the relative weight of highly valid features.
    subjective_weights = validities ** gamma
    
    # Weighted sum per option (dot product with subjective validities).
    scores = stim @ subjective_weights
    
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
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.1]
- gamma: [0.1, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8844 (var=0.0081) vs this=0.9467 (var=0.0101)
- Experiment 2: real=0.8533 (var=0.0160) vs this=0.9433 (var=0.0380)
- Experiment 3: real=0.1500 (var=0.0150) vs this=0.0575 (var=0.0264)
- Experiment 4: real=0.9033 (var=0.0065) vs this=0.7872 (var=0.0401)
- Experiment 5: real=0.8671 (var=0.0115) vs this=0.6110 (var=0.0358)
- Experiment 6: real=0.1378 (var=0.0101) vs this=0.4178 (var=0.1099)


---

### `pi_2` (overall score: 0.358)

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
- Experiment 1: real=0.8844 (var=0.0081) vs this=0.5661 (var=0.0650)
- Experiment 2: real=0.8533 (var=0.0160) vs this=0.4992 (var=0.0988)
- Experiment 3: real=0.1500 (var=0.0150) vs this=0.4025 (var=0.0939)
- Experiment 4: real=0.9033 (var=0.0065) vs this=0.5028 (var=0.0365)
- Experiment 5: real=0.8671 (var=0.0115) vs this=0.4213 (var=0.0137)
- Experiment 6: real=0.1378 (var=0.0101) vs this=0.7097 (var=0.0234)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2645 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2645 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    a_wadd = np.sum(a_ratings * validities, axis=1)
    b_wadd = np.sum(b_ratings * validities, axis=1)
    
    conflict_mask = (a_wins > b_wins) & (b_wadd > a_wadd)
    
    if np.sum(conflict_mask) == 0:
        return 0.5
        
    return float(np.mean(data['response'].values[conflict_mask]))
```

**Observed (real) value:** 0.8844 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8100 (var=0.0200) (Δ vs real -0.0744)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1633 (var=0.0132)
- pi_2: 0.5661 (var=0.0650)
- pi_2_1: 0.9467 (var=0.0101)
- pi_3: 0.8728 (var=0.0081)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    is_a_heavy = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_b_heavy = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    is_a_many = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_b_many = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    trial_1 = is_a_heavy & is_b_many
    trial_2 = is_a_many & is_b_heavy
    
    critical_trials = trial_1 | trial_2
    
    if not critical_trials.any():
        return 0.5
        
    heavy_chosen = (trial_1 & (data['response'] == 0)) | (trial_2 & (data['response'] == 1))
    
    return float(heavy_chosen[critical_trials].mean())
```

**Observed (real) value:** 0.8533 (var=0.0160)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7775 (var=0.0250) (Δ vs real -0.0758)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4992 (var=0.0988)
- pi_1: 0.1292 (var=0.0084)
- pi_2_1: 0.9433 (var=0.0380)
- pi_3: 0.8750 (var=0.0083)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    v = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_v = np.sum(a_ratings * v, axis=1)
    b_v = np.sum(b_ratings * v, axis=1)
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    wadd_prefers_a = a_v > b_v
    wadd_prefers_b = b_v > a_v
    
    conflict_a = tally_prefers_a & wadd_prefers_b
    conflict_b = tally_prefers_b & wadd_prefers_a
    
    conflict_mask = conflict_a | conflict_b
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    tally_aligned = np.zeros(len(data), dtype=bool)
    tally_aligned[conflict_a & (responses == 0)] = True
    tally_aligned[conflict_b & (responses == 1)] = True
    
    return float(np.mean(tally_aligned[conflict_mask]))
```

**Observed (real) value:** 0.1500 (var=0.0150)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2050 (var=0.0336) (Δ vs real +0.0550)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8350 (var=0.0303)
- pi_2_1: 0.0575 (var=0.0264)
- pi_2: 0.4025 (var=0.0939)
- pi_3: 0.1625 (var=0.0258)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def f0_chosen(row):
        a0 = row['option_a_ratings'][0]
        b0 = row['option_b_ratings'][0]
        if a0 == b0:
            return None
        return a0 if row['response'] == 0 else b0
        
    f0_vals = data.apply(f0_chosen, axis=1).dropna()
    if len(f0_vals) == 0:
        return 0.5
    return float(f0_vals.mean())
```

**Observed (real) value:** 0.9033 (var=0.0065)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6817 (var=0.0153) (Δ vs real -0.2217)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.7872 (var=0.0401)
- pi_1: 0.4092 (var=0.0018)
- pi_2: 0.5028 (var=0.0365)
- pi_3: 0.8964 (var=0.0042)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred is not None:
            matches.append(1 if r == pred else 0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.8671 (var=0.0115)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5758 (var=0.0119) (Δ vs real -0.2913)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8735 (var=0.0058)
- pi_2_1: 0.6110 (var=0.0358)
- pi_1: 0.3229 (var=0.0026)
- pi_2: 0.4213 (var=0.0137)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB winner: based on the first feature that discriminates
    diff = a_ratings - b_ratings
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    has_diff = np.any(diff != 0, axis=1)
    
    # Tally winner: based on simple sum of features
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    tally_winner = np.where(sum_a > sum_b, 0, np.where(sum_b > sum_a, 1, -1))
    
    # Identify conflict trials where TTB and Tallying disagree
    conflict = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(conflict):
        return 0.0
        
    responses = data['response'].values
    tally_chosen = (responses[conflict] == tally_winner[conflict]).sum()
    
    return float(tally_chosen / np.sum(conflict))
```

**Observed (real) value:** 0.1378 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3947 (var=0.0305) (Δ vs real +0.2569)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.4178 (var=0.1099)
- pi_3: 0.1311 (var=0.0080)
- pi_1: 0.8639 (var=0.0060)
- pi_2: 0.7097 (var=0.0234)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Probabilistic Take-The-Best (PTTB): Decision-makers search through features sequentially but stochastically, with the probability of checking a feature proportional to a non-linear function of its validity (v^gamma). The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the decision-maker guesses. Mathematically, this sequential sampling without replacement evaluates to a Luce choice rule over the sum of weights of the discriminating features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Feature selection probabilities are proportional to v^gamma
    weights = validities ** gamma
    
    # Identify discriminating features for each option
    d_a = (a > b).astype(float)
    d_b = (b > a).astype(float)
    
    # Sum of weights of features favoring A and B
    w_a = np.sum(d_a * weights)
    w_b = np.sum(d_b * weights)
    
    # The probability that the first discriminating feature found favors A
    # is exactly w_a / (w_a + w_b) under this sequential search process.
    if w_a + w_b == 0:
        p_a = 0.5
    else:
        p_a = w_a / (w_a + w_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- gamma: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, we implemented a Probabilistic Take-The-Best (PTTB) model. In this model, features are searched sequentially but stochastically, weighted by a non-linear scaling of their validities (v^gamma). The search stops at the first feature that discriminates. Mathematically, the probability that the first drawn discriminating feature favors option A is exactly the sum of weights of features favoring A divided by the total weight of all discriminating features. This yields a Luce choice rule over discriminating features. Unlike WADD (which uses a softmax over score differences and is thus insensitive to the total magnitude of discriminating weights), PTTB's choice probabilities depend on the ratio of discriminating evidence, maintaining the non-compensatory stopping-rule spirit of TTB while naturally explaining stochastic deviations in search order.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Probabilistic Take-The-Best (PTTB) model successfully improved the aggregate loss and was accepted as the new baseline. However, a detailed look at Experiments 4, 5, and 6 shows that it systematically underestimates the subjects' reliance on the most valid discriminating feature. For instance, in Exp 5, humans choose the option favored by the highest-validity discriminating feature 86.7% of the time, whereas the candidate predicts only 57.6%. This happens because the power-law weighting `v^gamma` with gamma bounded at 20 does not create a steep enough drop-off between validities (e.g., 0.95^20 vs 0.90^20 is only about a 3:1 ratio). Consequently, the model behaves too much like a compensatory model when multiple lower-validity features oppose the highest-validity one.
Rationale: The PTTB mechanism is structurally sound and within the arbiter's prescribed family, but it needs to be able to approximate deterministic TTB more closely to capture the high empirical match rates (0.85-0.90) for the most valid feature. Please keep the PTTB logic but make the feature weighting much steeper. You can do this by either drastically increasing the upper bound of `gamma` (e.g., up to 100) or by changing the weight function to an exponential form like `exp(gamma * validities)` to ensure the highest validity feature can dominate the denominator when necessary.

**Outcome of this advice:** iter 1 candidate loss=0.2645 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Probabilistic Take-The-Best (PTTB) with exponential weighting: Decision-makers search through features sequentially but stochastically. The probability of checking a feature is proportional to an exponential function of its validity (exp(gamma * validity)), allowing high-validity features to strongly dominate the search order. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the decision-maker guesses. This yields a Luce choice rule over the exponentially weighted discriminating features.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"PTTB expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Identify discriminating features for each option\n    d_a = (a > b).astype(float)\n    d_b = (b > a).astype(float)\n    d_any = d_a + d_b\n    \n    if np.sum(d_any) == 0:\n        p_a = 0.5\n    else:\n        # To prevent numerical overflow, subtract max validity among discriminating features\n        max_v = np.max(validities[d_any > 0])\n        weights = np.exp(gamma * (validities - max_v))\n        \n        # Sum of weights of features favoring A and B\n        w_a = np.sum(d_a * weights)\n        w_b = np.sum(d_b * weights)\n        \n        # The probability that the first discriminating feature found favors A\n        p_a = w_a / (w_a + w_b)\n        \n    p_core = np.array([p_a, 1.0 - p_a])\n    \n    # Incorporate uniform lapse rate\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return np.random.choice(len(probs), p=probs)\n",
  "parameters": {
    "gamma": "[0.1, 100.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we modified the weighting function in the Probabilistic Take-The-Best (PTTB) model from a power-law (v^gamma) to an exponential form (exp(gamma * v)) and increased the upper bound of gamma to 100.0. To ensure numerical stability, we subtract the maximum validity among discriminating features before exponentiating. This allows the highest-validity discriminating feature to much more strongly dominate the stochastic search order, closely approximating deterministic TTB when gamma is large, and thereby capturing the high empirical reliance on the most valid feature observed in Experiments 4, 5, and 6."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12139,
  "candidates_token_count": 824,
  "total_token_count": 14060
}
```
