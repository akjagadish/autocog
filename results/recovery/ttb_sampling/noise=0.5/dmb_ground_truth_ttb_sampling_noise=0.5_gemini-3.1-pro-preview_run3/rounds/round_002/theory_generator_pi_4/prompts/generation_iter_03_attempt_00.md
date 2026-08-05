# generation_iter_03_attempt_00

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

Propose a 'Strategy Mixture' theory. Instead of assuming all subjects uniformly apply a single rule, this theory should posit that decision-makers draw from a repertoire of strategies. Specifically, on any given trial (or for any given subject), there is a probability 'alpha' of using the Take-The-Best (TTB) heuristic and a probability '1 - alpha' of using a compensatory rule (like WADD or Tallying). This mixture approach will naturally capture the intermediate choice proportions observed across all experiments, blending the distinct predictions of the two extremes.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.823)

**Description**
Take-The-Best (TTB) heuristic: People make binary choices by evaluating features sequentially in descending order of their validity. The first feature that discriminates between the two options (i.e., one option has a higher value than the other) entirely determines the choice, and all remaining features are ignored. This strictly non-compensatory strategy allows decision makers to heavily weight highly predictive cues without needing to compute complex compensatory trade-offs.

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
    scores = np.array([0.0, 0.0])
    
    # Evaluate features one by one in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3400 (var=0.0108) vs this=0.1317 (var=0.0092)
- Experiment 2: real=0.3567 (var=0.0242) vs this=0.1450 (var=0.0127)
- Experiment 3: real=0.3256 (var=0.0090) vs this=0.1500 (var=0.0142)
- Experiment 4: real=0.6717 (var=0.0180) vs this=0.8350 (var=0.0162)
- Experiment 5: real=0.6817 (var=0.0051) vs this=0.8515 (var=0.0097)
- Experiment 6: real=0.6725 (var=0.0056) vs this=0.8658 (var=0.0090)


---

### `pi_2_1` (overall score: 0.787)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Apply subjective distortion to validities
    subjective_weights = validities ** gamma
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ subjective_weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3400 (var=0.0108) vs this=0.2625 (var=0.0762)
- Experiment 2: real=0.3567 (var=0.0242) vs this=0.2350 (var=0.0560)
- Experiment 3: real=0.3256 (var=0.0090) vs this=0.1983 (var=0.0373)
- Experiment 4: real=0.6717 (var=0.0180) vs this=0.7767 (var=0.0555)
- Experiment 5: real=0.6817 (var=0.0051) vs this=0.5006 (var=0.0593)
- Experiment 6: real=0.6725 (var=0.0056) vs this=0.3035 (var=0.0229)


---

### `pi_2` (overall score: 0.632)

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
- Experiment 1: real=0.3400 (var=0.0108) vs this=0.4967 (var=0.0864)
- Experiment 2: real=0.3567 (var=0.0242) vs this=0.5483 (var=0.0811)
- Experiment 3: real=0.3256 (var=0.0090) vs this=0.4339 (var=0.0636)
- Experiment 4: real=0.6717 (var=0.0180) vs this=0.5292 (var=0.1147)
- Experiment 5: real=0.6817 (var=0.0051) vs this=0.3688 (var=0.0305)
- Experiment 6: real=0.6725 (var=0.0056) vs this=0.2457 (var=0.0155)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1369 -> ACCEPTED
- iter 2: loss=0.1755 -> REJECTED
- iter 3: loss=0.1882 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1369 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_choices = 0
    total_relevant = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_choices += 1
            total_relevant += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_choices += 1
            total_relevant += 1
    if total_relevant == 0:
        return 0.5
    return tallying_choices / total_relevant
```

**Observed (real) value:** 0.3400 (var=0.0108)
**Previous candidate values (this loop):**
  - iter 1: 0.1925 (var=0.0187) (Δ vs real -0.1475)
  - iter 2: 0.4600 (var=0.0380) (Δ vs real +0.1200)
  - iter 3 (most recent): 0.2017 (var=0.0629) (Δ vs real -0.1383)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8517 (var=0.0133)
- pi_2: 0.4967 (var=0.0864)
- pi_2_1: 0.2625 (var=0.0762)
- pi_3: 0.1317 (var=0.0092)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify the specific trial where WADD and Tallying strongly disagree.
    # Trial: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    # WADD prefers A (validity sum 1.8 > 1.7) while Tallying prefers B (tally 3 > 2).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.3567 (var=0.0242)
**Previous candidate values (this loop):**
  - iter 1: 0.2217 (var=0.0338) (Δ vs real -0.1350)
  - iter 2: 0.4700 (var=0.0535) (Δ vs real +0.1133)
  - iter 3 (most recent): 0.2150 (var=0.0570) (Δ vs real -0.1417)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5483 (var=0.0811)
- pi_1: 0.8683 (var=0.0120)
- pi_2_1: 0.2350 (var=0.0560)
- pi_3: 0.1450 (var=0.0127)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_mismatch = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 1, 1, 1, 0) and b == (1, 1, 0, 0, 0, 0):
            tally_choices += (resp == 0)
            total_mismatch += 1
        elif a == (0, 0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0, 0):
            tally_choices += (resp == 0)
            total_mismatch += 1
        elif a == (1, 1, 0, 0, 0, 0) and b == (0, 0, 1, 1, 1, 0):
            tally_choices += (resp == 1)
            total_mismatch += 1
            
    return float(tally_choices / total_mismatch) if total_mismatch > 0 else 0.5

```

**Observed (real) value:** 0.3256 (var=0.0090)
**Previous candidate values (this loop):**
  - iter 1: 0.2361 (var=0.0429) (Δ vs real -0.0894)
  - iter 2: 0.4206 (var=0.0503) (Δ vs real +0.0950)
  - iter 3 (most recent): 0.1822 (var=0.0544) (Δ vs real -0.1433)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8644 (var=0.0110)
- pi_2_1: 0.1983 (var=0.0373)
- pi_2: 0.4339 (var=0.0636)
- pi_3: 0.1500 (var=0.0142)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_is_11000 = np.all(a_ratings == [1, 1, 0, 0, 0], axis=1)
    b_is_00111 = np.all(b_ratings == [0, 0, 1, 1, 1], axis=1)
    
    a_is_00111 = np.all(a_ratings == [0, 0, 1, 1, 1], axis=1)
    b_is_11000 = np.all(b_ratings == [1, 1, 0, 0, 0], axis=1)
    
    trial_type_1 = a_is_11000 & b_is_00111
    trial_type_2 = a_is_00111 & b_is_11000
    
    target_trials = trial_type_1 | trial_type_2
    
    if not np.any(target_trials):
        return 0.5
        
    responses = data['response'].values
    
    wadd_chosen = np.zeros_like(responses, dtype=bool)
    wadd_chosen[trial_type_1 & (responses == 0)] = True
    wadd_chosen[trial_type_2 & (responses == 1)] = True
    
    return float(np.mean(wadd_chosen[target_trials]))
```

**Observed (real) value:** 0.6717 (var=0.0180)
**Previous candidate values (this loop):**
  - iter 1: 0.8067 (var=0.0294) (Δ vs real +0.1350)
  - iter 2: 0.6067 (var=0.0430) (Δ vs real -0.0650)
  - iter 3 (most recent): 0.8258 (var=0.0487) (Δ vs real +0.1542)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.7767 (var=0.0555)
- pi_1: 0.1400 (var=0.0145)
- pi_2: 0.5292 (var=0.1147)
- pi_3: 0.8350 (var=0.0162)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Validities are strictly decreasing from feature 0 to 4.
    # TTB evaluates features in order 0, 1, 2, 3, 4.
    diff = a_mat - b_mat
    
    # Weight features exponentially to find the first differing feature's sign
    weights = np.array([10000, 1000, 100, 10, 1])
    scores = diff.dot(weights)
    
    # Positive score means A dominates on the most valid discriminating feature (TTB chooses A -> 0)
    # Negative score means B dominates (TTB chooses B -> 1)
    ttb_choices = np.where(scores > 0, 0, 1)
    
    valid = scores != 0
    if not np.any(valid):
        return 0.5
        
    match = (ttb_choices[valid] == resp[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6817 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: 0.6765 (var=0.0271) (Δ vs real -0.0052)
  - iter 2: 0.5033 (var=0.0307) (Δ vs real -0.1783)
  - iter 3 (most recent): 0.6448 (var=0.0425) (Δ vs real -0.0369)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8515 (var=0.0097)
- pi_2_1: 0.5006 (var=0.0593)
- pi_1: 0.2342 (var=0.0039)
- pi_2: 0.3688 (var=0.0305)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where TTB prefers B (first differing feature favors B)
    # and WADD tends to prefer A (sum of features favors A)
    def is_compensatory_B(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # TTB prefers B?
        ttb_b = False
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_b = (b[i] > a[i])
                break
        # Sum prefers A?
        sum_a = sum(a) > sum(b)
        return ttb_b and sum_a
        
    mask = data.apply(is_compensatory_B, axis=1)
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6725 (var=0.0056)
**Previous candidate values (this loop):**
  - iter 1: 0.5982 (var=0.0313) (Δ vs real -0.0743)
  - iter 2: 0.5170 (var=0.0700) (Δ vs real -0.1555)
  - iter 3 (most recent): 0.4845 (var=0.0490) (Δ vs real -0.1880)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.3035 (var=0.0229)
- pi_3: 0.8658 (var=0.0090)
- pi_1: 0.1395 (var=0.0083)
- pi_2: 0.2457 (var=0.0155)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by their subjective validities (validities^gamma), naturally subsuming Tallying when gamma=0 and WADD when gamma>0. This allows the model to capture intermediate choice proportions observed across experiments by blending the distinct predictions of both strategy extremes.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's feedback, this model instantiates a Strategy Mixture approach. Neither TTB nor WADD alone can capture the intermediate choice rates observed across the experiments (e.g., TTB overpredicts in Exp 5 and 6, while WADD underpredicts; Tallying overpredicts Exp 1 while TTB underpredicts). By introducing a mixture parameter `alpha`, the model interpolates between a non-compensatory one-reason decision heuristic (TTB) and a compensatory rule. Furthermore, introducing `gamma` to distort validities enables the compensatory arm to smoothly transition between Tallying (gamma=0, where all features are weighted equally) and WADD (gamma>0). This flexibility allows the model to organically fit the intermediate choice proportions across all experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The Strategy Mixture theory successfully captures the intermediate nature of the subjects' choices, representing a massive improvement over pure WADD or pure TTB. The candidate model was accepted by the gate. However, looking at the fine-grained fits: it underpredicts Tallying-consistent choices in Experiments 1-3 (Candidate ~0.20-0.23 vs Observed ~0.33-0.35) and overpredicts WADD/TTB-consistent choices in Experiment 4 (Candidate ~0.80 vs Observed ~0.67). Since TTB and WADD agree in Exp 4 (both prefer the target option), the only way to pull the prediction down to 0.67 is if the non-TTB arm produces Tallying-like choices (which prefer the opposite).
Rationale: To fix the underprediction of Tallying in Exps 1-3 and the overprediction of WADD/TTB in Exp 4, the compensatory arm needs to behave more like Tallying and less like strict WADD. Shrink the upper bound of the `gamma` parameter range (e.g., from [0.0, 5.0] to [0.0, 1.0] or even [0.0, 0.5]). This will force the subjective validities to be flatter, ensuring the compensatory arm acts as a Tallying heuristic. This should seamlessly align the model with the ~1/3 Tallying to ~2/3 TTB ratio observed across the data.

**Outcome of this advice:** iter 1 candidate loss=0.1369 -> iter 2 candidate loss=0.1755 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the gate because the aggregate loss worsened. The previous advice to restrict the upper bound of `gamma` to 1.0 caused the model to overshoot drastically: it became too heavily biased toward Tallying. This overcorrection led to overpredicting Tallying-consistent choices in Experiments 1-3 (e.g., Exp 1 went from 0.19 to 0.46, overshooting the observed 0.34) and severely degraded the fits for Experiments 5 and 6, where the model lost its strong TTB/WADD alignment. The base model from Iteration 1 remains our best attempt.
Rationale: Since shrinking `gamma` to [0.0, 1.0] was rejected for overshooting toward Tallying, we should allow a middle ground. On top of the Iteration 1 base, adjust the `gamma` parameter range to something intermediate, like [0.0, 3.0] or [0.0, 2.5]. This will give the optimization enough room to find a subjective validity weighting that balances WADD and Tallying without forcing it entirely into the Tallying regime. Additionally, restrict `epsilon` to a tighter range (e.g., [0.0, 0.2]) to ensure the uniform lapse term isn't washing out the delicate mixture dynamics.

**Outcome of this advice:** iter 2 candidate loss=0.1755 -> iter 3 candidate loss=0.1882 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was REJECTED by the gate because the aggregate loss worsened. Adjusting the gamma range to [0.0, 3.0] failed to find the right balance between Tallying and WADD. The root cause of the model's rigidity appears to be a scaling mismatch: the TTB scores (which are strictly 0.0 or 1.0) and the compensatory WADD scores (which are sums of exponentiated validities) are on completely different numerical scales, yet they are multiplied by the exact same inverse temperature parameter `beta`. This shared temperature forces a compromise where one strategy is either too deterministic or too noisy, preventing the mixture from accurately matching the ~1/3 Tallying to ~2/3 TTB ratio observed in the empirical data.
Rationale: To fix the scaling mismatch, decouple the temperatures for the two strategies. Replace the single `beta` parameter with `beta_ttb` and `beta_wadd` (both with ranges like [0.1, 20.0]). This minor structural adjustment will allow the optimization to independently calibrate the determinism of the heuristic and compensatory arms. Restore the `gamma` range to a broad [0.0, 5.0] so the model has full flexibility to find the optimal subjective validity weighting. Keep `alpha` in [0.0, 1.0] and `epsilon` in [0.0, 0.2].

**Outcome of this advice:** iter 3 candidate loss=0.1882 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Strategy Mixture Theory (Decoupled Temperature): Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. Because the underlying scores for TTB (0 or 1) and WADD (sum of subjective validities) are on vastly different numerical scales, each strategy arm employs its own independent temperature parameter (beta_ttb and beta_wadd). This allows the determinism of each strategy to be calibrated independently, allowing the mixture to accurately match the intermediate choice rates observed across experiments.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.\")\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    alpha = float(parameters[\"alpha\"])\n    beta_ttb = float(parameters[\"beta_ttb\"])\n    beta_wadd = float(parameters[\"beta_wadd\"])\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # --- Take-The-Best (TTB) Prediction ---\n    order = np.argsort(validities)[::-1]\n    a, b = stim[0], stim[1]\n    ttb_scores = np.array([0.0, 0.0])\n    for idx in order:\n        if a[idx] > b[idx]:\n            ttb_scores[0] = 1.0\n            break\n        elif b[idx] > a[idx]:\n            ttb_scores[1] = 1.0\n            break\n            \n    z_ttb = beta_ttb * (ttb_scores - ttb_scores.max())\n    e_ttb = np.exp(z_ttb)\n    p_ttb = e_ttb / e_ttb.sum()\n    \n    # --- Compensatory (WADD/Tallying) Prediction ---\n    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD\n    subjective_weights = validities ** gamma\n    wadd_scores = stim @ subjective_weights\n    \n    z_wadd = beta_wadd * (wadd_scores - wadd_scores.max())\n    e_wadd = np.exp(z_wadd)\n    p_wadd = e_wadd / e_wadd.sum()\n    \n    # --- Strategy Mixture ---\n    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd\n    \n    # --- Uniform Lapse ---\n    n_opts = p_core.shape[0]\n    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "beta_ttb": "[0.1, 20.0]",
    "beta_wadd": "[0.1, 20.0]",
    "gamma": "[0.0, 5.0]",
    "epsilon": "[0.0, 0.2]",
    "validities": "validities"
  },
  "rationale": "Following the critic's diagnosis, the previous model suffered from a scaling mismatch: TTB scores are strictly 0.0 or 1.0, whereas WADD scores are sums of exponentiated validities. Using a single `beta` forced a compromise that prevented the model from accurately matching empirical choice proportions. By splitting the temperature into `beta_ttb` and `beta_wadd`, the model can independently calibrate the determinism of the heuristic and compensatory arms. We restore `gamma` to [0.0, 5.0] to allow full flexibility in subjective validity weighting and restrict `epsilon` to [0.0, 0.2] to prevent uniform noise from washing out the mixture dynamics."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13215,
  "candidates_token_count": 1127,
  "total_token_count": 15326
}
```
