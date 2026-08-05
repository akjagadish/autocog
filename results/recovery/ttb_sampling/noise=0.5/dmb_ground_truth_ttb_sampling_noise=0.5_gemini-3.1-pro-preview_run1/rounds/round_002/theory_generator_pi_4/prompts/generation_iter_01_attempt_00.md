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

Propose a new theory such as 'Rank-Dependent Weighting' or a 'Strategy Selection / Mixture' model. A Rank-Dependent Weighting model would assign subjective weights to features based solely on their validity rank (e.g., exponentially decaying weights) rather than their raw validity values. This would allow the model to behave non-compensatorily (like TTB) when the decay is steep, but still integrate multiple cues when the top cues tie or when the decay is moderate, better capturing the intermediate variance seen in Experiments 3 and 4 without completely failing on the strict dissociation designs of Experiments 1 and 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.920)

**Description**
Take The Best (TTB) is a lexicographic, non-compensatory heuristic. Decision-makers evaluate options by comparing them sequentially on features, starting with the cue that has the highest subjective validity. The first feature that discriminates between the options determines the choice. If all features tie, the decision-maker guesses. Response noise is modeled as an independent lapse rate that occasionally results in a random choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    p_core = np.array([0.5, 0.5])  # Default to guessing if all features tie
    
    # Sequential comparison
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic TTB choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3500 (var=0.0088) vs this=0.2437 (var=0.0293)
- Experiment 2: real=0.3283 (var=0.0114) vs this=0.2250 (var=0.0228)
- Experiment 3: real=0.7133 (var=0.0245) vs this=0.7650 (var=0.0277)
- Experiment 4: real=0.6695 (var=0.0161) vs this=0.7800 (var=0.0231)
- Experiment 5: real=0.6896 (var=0.0029) vs this=0.7556 (var=0.0186)
- Experiment 6: real=0.3234 (var=0.0041) vs this=0.2371 (var=0.0204)


---

### `pi_2` (overall score: 0.589)

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
- Experiment 1: real=0.3500 (var=0.0088) vs this=0.4875 (var=0.0782)
- Experiment 2: real=0.3283 (var=0.0114) vs this=0.4275 (var=0.0930)
- Experiment 3: real=0.7133 (var=0.0245) vs this=0.5517 (var=0.1150)
- Experiment 4: real=0.6695 (var=0.0161) vs this=0.6084 (var=0.0902)
- Experiment 5: real=0.6896 (var=0.0029) vs this=0.3477 (var=0.0199)
- Experiment 6: real=0.3234 (var=0.0041) vs this=0.7467 (var=0.0148)


---

### `pi_2_1` (overall score: 0.446)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # Paper-faithful Weighted Additive rule (WADD)
    # Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. 
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    # Use the experiment-provided validities directly as the weights.
    validities = np.asarray(parameters["validities"], dtype=float)

    # Weighted sum per option (dot product with validities).
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3500 (var=0.0088) vs this=0.3844 (var=0.0199)
- Experiment 2: real=0.3283 (var=0.0114) vs this=0.2450 (var=0.0191)
- Experiment 3: real=0.7133 (var=0.0245) vs this=0.6417 (var=0.0184)
- Experiment 4: real=0.6695 (var=0.0161) vs this=0.7337 (var=0.0171)
- Experiment 5: real=0.6896 (var=0.0029) vs this=0.2069 (var=0.0085)
- Experiment 6: real=0.3234 (var=0.0041) vs this=0.8596 (var=0.0118)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1831 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1831 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    tally_matches = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        wadd_a = np.dot(a, v)
        wadd_b = np.dot(b, v)
        
        tally_pref = None
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
            
        wadd_pref = None
        if wadd_a > wadd_b:
            wadd_pref = 0
        elif wadd_b > wadd_a:
            wadd_pref = 1
            
        if tally_pref is not None and wadd_pref is not None and tally_pref != wadd_pref:
            disagreement_trials += 1
            if row['response'] == tally_pref:
                tally_matches += 1
                
    if disagreement_trials == 0:
        return 0.5
        
    return float(tally_matches / disagreement_trials)
```

**Observed (real) value:** 0.3500 (var=0.0088)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2125 (var=0.0591) (Δ vs real -0.1375)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8569 (var=0.0086)
- pi_2: 0.4875 (var=0.0782)
- pi_2_1: 0.3844 (var=0.0199)
- pi_3: 0.2437 (var=0.0293)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def tally_match(row):
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        if sum_a > sum_b:
            return 1.0 if row['response'] == 0 else 0.0
        elif sum_b > sum_a:
            return 1.0 if row['response'] == 1 else 0.0
        else:
            return np.nan
            
    matches = data.apply(tally_match, axis=1)
    return float(matches.dropna().mean())
```

**Observed (real) value:** 0.3283 (var=0.0114)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2367 (var=0.0618) (Δ vs real -0.0917)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4275 (var=0.0930)
- pi_1: 0.8558 (var=0.0129)
- pi_2_1: 0.2450 (var=0.0191)
- pi_3: 0.2250 (var=0.0228)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the specific trial where Tallying and WADD have opposite strict preferences.
    # Trial 1: Option A has features [1, 1, 0, 0, 0] and Option B has [0, 0, 1, 1, 1].
    # Tallying prefers B (3 wins vs 2 wins).
    # WADD prefers A (0.95 + 0.85 = 1.8 vs 0.65 + 0.55 + 0.5 = 1.7).
    is_target = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))) & \
                (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)))
    
    target_data = data[is_target]
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.7133 (var=0.0245)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7433 (var=0.0647) (Δ vs real +0.0300)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1650 (var=0.0196)
- pi_2_1: 0.6417 (var=0.0184)
- pi_2: 0.5517 (var=0.1150)
- pi_3: 0.7650 (var=0.0277)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_target_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return 1.0 if row['response'] == 0 else 0.0
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return 1.0 if row['response'] == 1 else 0.0
        return np.nan

    choices = data.apply(get_target_choice, axis=1)
    val = np.nanmean(choices)
    if np.isnan(val):
        return 0.5
    return float(val)
```

**Observed (real) value:** 0.6695 (var=0.0161)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7800 (var=0.0406) (Δ vs real +0.1105)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.7337 (var=0.0171)
- pi_1: 0.1768 (var=0.0115)
- pi_2: 0.6084 (var=0.0902)
- pi_3: 0.7800 (var=0.0231)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are pre-sorted by validity in descending order based on the design.
        # TTB evaluates features sequentially.
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred == resp:
            ttb_match += 1
        total += 1
        
    return float(ttb_match / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6896 (var=0.0029)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5444 (var=0.0602) (Δ vs real -0.1452)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7556 (var=0.0186)
- pi_2_1: 0.2069 (var=0.0085)
- pi_1: 0.1975 (var=0.0079)
- pi_2: 0.3477 (var=0.0199)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    w = np.array([0.95, 0.75, 0.65, 0.55, 0.5])
    
    def wadd_choice(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, w)
        score_b = np.dot(b, w)
        return 0 if score_a > score_b else 1
        
    predicted = data.apply(wadd_choice, axis=1)
    return np.mean(data['response'] == predicted)

```

**Observed (real) value:** 0.3234 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4836 (var=0.0716) (Δ vs real +0.1602)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.8596 (var=0.0118)
- pi_3: 0.2371 (var=0.0204)
- pi_1: 0.8878 (var=0.0063)
- pi_2: 0.7467 (var=0.0148)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Rank-Dependent Weighting posits that decision-makers assign subjective weights to features based on their validity rank rather than their raw validity values. The most valid feature receives the highest weight, and subsequent features receive exponentially decaying weights controlled by a decay parameter. This mechanism allows the model to smoothly interpolate between non-compensatory, lexicographic decision-making (like Take The Best) when the decay is steep, and compensatory, equal-weight integration (like Tallying) when the decay is gradual. The weighted sums for each option are then compared using a softmax choice rule with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Determine the rank of each feature (0 for highest validity)
    order = np.argsort(-validities)
    ranks = np.empty_like(validities)
    ranks[order] = np.arange(len(validities))
    
    # Calculate rank-dependent weights with exponential decay
    lambda_decay = float(parameters["lambda_decay"])
    weights = lambda_decay ** ranks
    
    # Calculate scores for each option
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- lambda_decay: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, this Rank-Dependent Weighting model ignores the raw validity magnitudes and instead uses their rank order to assign exponentially decaying weights. By varying the decay parameter (`lambda_decay`), the model can exhibit non-compensatory behavior (matching Take The Best when decay is steep) or compensatory behavior (when decay is moderate to high). This flexibility allows it to capture the intermediate variance observed in Experiments 3 and 4, while still being able to reproduce the strict dissociations found in Experiments 1 and 2, which purely magnitude-based WADD or purely lexicographic TTB struggle to balance simultaneously.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Rank-Dependent Weighting model with exponential decay successfully reduced the aggregate loss and was ACCEPTED by the gate. It captures the intermediate variance in Experiments 3 and 4 quite well. However, it still struggles slightly with Experiment 5 (TTB match: real=0.69 vs cand=0.54) and Experiment 6 (WADD match: real=0.32 vs cand=0.48). The exponential decay (lambda_decay ** rank) might be too aggressive, forcing the model into either extreme lexicographic behavior or completely uniform weights, leaving a sparse middle ground.
Rationale: While the Rank-Dependent Weighting family is definitely the right direction (as shown by the accepted loss), the exponential decay function is likely too rigid. Try swapping the exponential decay for a power-law decay: `weights = 1.0 / ((ranks + 1.0) ** gamma)` where `gamma` is a parameter (e.g., [0.0, 5.0]). Power-law decay provides a softer drop-off than exponential decay, which should allow the model to better capture the nuanced partial-compensatory behavior seen in Experiments 5 and 6 while maintaining the good fits on Experiments 3 and 4.

**Outcome of this advice:** iter 1 candidate loss=0.1831 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Rank-Dependent Weighting posits that decision-makers assign subjective weights to features based on their validity rank rather than their raw validity values. The most valid feature receives the highest weight, and subsequent features receive weights that decay according to a power law. This mechanism allows the model to smoothly interpolate between non-compensatory, lexicographic decision-making (like Take The Best) when the decay is steep, and compensatory, equal-weight integration (like Tallying) when the decay is gradual. A power-law decay provides a softer drop-off than an exponential one, better capturing nuanced partial-compensatory behavior.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Determine the rank of each feature (0 for highest validity)\n    order = np.argsort(-validities)\n    ranks = np.empty_like(validities)\n    ranks[order] = np.arange(len(validities))\n    \n    # Calculate rank-dependent weights with power-law decay\n    gamma = float(parameters[\"gamma\"])\n    weights = 1.0 / ((ranks + 1.0) ** gamma)\n    \n    # Calculate scores for each option\n    scores = stim @ weights\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n    \n    # Blend with uniform lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()  # Ensure valid probabilities.\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "gamma": "[0.0, 5.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, the exponential decay was replaced with a power-law decay for the rank-dependent weights. Exponential decay can be too aggressive, pushing the model toward either purely lexicographic (TTB-like) or completely uniform behavior. The power-law decay (controlled by gamma) provides a softer drop-off, which should allow the model to better capture the nuanced partial-compensatory behavior seen in Experiments 5 and 6 while maintaining the strong fits on Experiments 3 and 4."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11195,
  "candidates_token_count": 759,
  "total_token_count": 12641
}
```
