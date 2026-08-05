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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'Mixture of Strategies' (or Probabilistic Strategy Selection) theory, where on any given trial, a subject stochastically chooses to either employ a lexicographic heuristic (like TTB) or a compensatory strategy (like Tallying or WADD). Alternatively, propose a 'Non-linear Cue Weighting' theory where cue validities are transformed via a steep but continuous decay function, allowing the model to act mostly lexicographic but still permitting compensation when a large number of lower-validity cues stack up. This would replace the rigid TTB theory and better capture the intermediate, low-variance subject averages.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.722)

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
- Experiment 1: real=0.4113 (var=0.0093) vs this=0.4050 (var=0.0960)
- Experiment 2: real=0.5900 (var=0.0027) vs this=0.5321 (var=0.0463)
- Experiment 3: real=0.6004 (var=0.0022) vs this=0.3115 (var=0.0131)
- Experiment 4: real=0.4131 (var=0.0041) vs this=0.7538 (var=0.0204)


---

### `pi_3` (overall score: 0.532)

**Description**
Take-The-Best (TTB) models decision-making as a non-compensatory, lexicographic process. People search through available features in descending order of their validity (importance) and stop at the first feature that discriminates between the two options. The option with the higher value on this decisive feature is chosen. If no features discriminate, the decision reduces to a guess. Unlike Weighted Additive (WADD) models, TTB does not allow deficits on high-validity cues to be compensated by advantages on lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order (stable sort for ties)
    order = np.argsort(-validities, kind='stable')
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Lexicographic search: stop at the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores (scores are either [1,0], [0,1], or [0,0])
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4113 (var=0.0093) vs this=0.1544 (var=0.0114)
- Experiment 2: real=0.5900 (var=0.0027) vs this=0.8462 (var=0.0136)
- Experiment 3: real=0.6004 (var=0.0022) vs this=0.8496 (var=0.0075)
- Experiment 4: real=0.4131 (var=0.0041) vs this=0.1300 (var=0.0100)


---

### `pi_1` (overall score: 0.000)

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
- Experiment 1: real=0.4113 (var=0.0093) vs this=0.8669 (var=0.0105)
- Experiment 2: real=0.5900 (var=0.0027) vs this=0.2685 (var=0.0055)
- Experiment 3: real=0.6004 (var=0.0022) vs this=0.2550 (var=0.0045)
- Experiment 4: real=0.4131 (var=0.0041) vs this=0.8506 (var=0.0097)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1878 -> ACCEPTED
- iter 2: loss=0.2823 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1878 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tallying_consistent = 0
    total_dissociation_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Identify dissociation trials: one option has more features, but lacks the first (highest validity) feature
        if sum_a > sum_b and a[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 0:
                tallying_consistent += 1
        elif sum_b > sum_a and b[0] == 0:
            total_dissociation_trials += 1
            if row['response'] == 1:
                tallying_consistent += 1
                
    if total_dissociation_trials == 0:
        return 0.5
        
    return float(tallying_consistent / total_dissociation_trials)
```

**Observed (real) value:** 0.4113 (var=0.0093)
**Previous candidate values (this loop):**
  - iter 1: 0.1456 (var=0.0142) (Δ vs real -0.2656)
  - iter 2 (most recent): 0.1519 (var=0.0133) (Δ vs real -0.2594)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8669 (var=0.0105)
- pi_2: 0.4050 (var=0.0960)
- pi_3: 0.1544 (var=0.0114)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_f1 = np.array([a[0] for a in data['option_a_ratings']])
    b_f1 = np.array([b[0] for b in data['option_b_ratings']])
    resp = data['response'].values
    chosen_f1 = np.where(resp == 0, a_f1, b_f1)
    return float(np.mean(chosen_f1))
```

**Observed (real) value:** 0.5900 (var=0.0027)
**Previous candidate values (this loop):**
  - iter 1: 0.7833 (var=0.0135) (Δ vs real +0.1933)
  - iter 2 (most recent): 0.7802 (var=0.0149) (Δ vs real +0.1902)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5321 (var=0.0463)
- pi_1: 0.2685 (var=0.0055)
- pi_3: 0.8462 (var=0.0136)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return row['response'] == 0
            elif b[i] > a[i]:
                return row['response'] == 1
        return False

    return float(data.apply(ttb_match, axis=1).mean())
```

**Observed (real) value:** 0.6004 (var=0.0022)
**Previous candidate values (this loop):**
  - iter 1: 0.5773 (var=0.0446) (Δ vs real -0.0231)
  - iter 2 (most recent): 0.4996 (var=0.0347) (Δ vs real -0.1008)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8496 (var=0.0075)
- pi_2: 0.3115 (var=0.0131)
- pi_1: 0.2550 (var=0.0045)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Select trials where Option B has more positive features than Option A
    # In the experimental design (Trials 1, 2, 3, 5), Option A is endorsed by the 
    # single highest-validity discriminating cue, but Option B has a higher sheer 
    # number of lower-validity cues.
    mask = sum_b > sum_a
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4131 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.4062 (var=0.0663) (Δ vs real -0.0069)
  - iter 2 (most recent): 0.6059 (var=0.0639) (Δ vs real +0.1928)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7538 (var=0.0204)
- pi_3: 0.1300 (var=0.0100)
- pi_1: 0.8506 (var=0.0097)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Non-linear Cue Weighting (Rank-Decay) Theory

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending to determine their rank
    # Rank 0 is the highest validity cue
    order = np.argsort(-validities, kind='stable')
    ranks = np.zeros_like(order)
    ranks[order] = np.arange(len(validities))
    
    # Apply exponential decay to cue weights based on rank.
    # If gamma is small (e.g., 0.1), weights decay rapidly, emulating the lexicographic 
    # non-compensatory nature of Take-The-Best. If gamma is 1.0, weights are simply 
    # proportional to validities, emulating Weighted Additive (WADD).
    weights = validities * (gamma ** ranks)
    
    # Calculate options' scores as the weighted sum of their features
    scores = stim @ weights
    
    # Softmax choice with lapse rate
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- gamma: [0.01, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, this theory implements 'Non-linear Cue Weighting'. Instead of treating strategy selection as a rigid binary choice or probabilistic mixture between TTB and WADD, this model unifies them into a single continuous framework. By ranking cues by validity and applying an exponential decay parameter (`gamma`) to their weights, the model smoothly interpolates between strategies. When `gamma` is small, the weights decay so rapidly that no combination of lower-validity cues can outweigh a higher-validity cue, perfectly emulating the lexicographic, tie-breaking behavior of TTB. When `gamma` approaches 1.0, the model behaves like WADD. Intermediate values of `gamma` allow the model to be mostly lexicographic while still permitting compensation when a sufficiently large number of lower-validity cues stack up against the primary cue, effectively capturing the intermediate empirical averages observed across the experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory implements the Non-linear Cue Weighting (Rank-Decay) mechanism and was accepted by the gate, marking a significant improvement. It perfectly captures the intermediate behaviors in Experiments 3 and 4 (matches the ~0.41 and ~0.58 empirical means). However, in Experiments 1 and 2, the model still behaves too lexicographically: in Experiment 1, its tallying consistency on dissociation trials is only 0.14 (vs real 0.41), and in Experiment 2, it chooses the option endorsed by the highest-validity cue 78% of the time (vs real 59%).
Rationale: The Rank-Decay mechanism is working well overall but currently permits too much extreme lexicographic behavior, causing it to under-predict compensatory choices in Experiments 1 and 2. Keep the mechanism exactly as is, but tighten the lower bound of the `gamma` parameter range (e.g., change `[0.01, 1.0]` to `[0.4, 1.0]`). This will prevent the model from assigning near-zero weights to lower-ranked cues, forcing a more compensatory balance that should pull Exp 1's tallying consistency up and Exp 2's highest-cue reliance down, while retaining the good fits on Exps 3 and 4.

**Outcome of this advice:** iter 1 candidate loss=0.1878 -> iter 2 candidate loss=0.2823 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate was REJECTED by the gate. My previous advice to restrict the `gamma` lower bound to 0.4 backfired: while it was intended to increase compensatory behavior in Experiments 1 and 2, it barely moved those metrics and instead ruined the previously excellent fits for Experiments 3 and 4 (e.g., Exp 4 metric jumped from 0.40 to 0.60, overshooting the real value of 0.41). The Rank-Decay model struggles to simultaneously capture the highly compensatory behavior in Exp 1/2 and the more lexicographic behavior in Exp 3/4 with a single set of continuous weights.
Rationale: Since tweaking the gamma bounds of the Rank-Decay model was rejected by the gate for degrading Exp 3 and 4 fits, we should pivot to the other mechanism explicitly authorized in the arbiter's recommendation: the 'Mixture of Strategies' (Probabilistic Strategy Selection) theory. Implement a model where, on each trial, the subject stochastically chooses to use a pure lexicographic strategy (Take-The-Best) with probability `p_lex`, and a compensatory strategy (like Tallying or WADD) with probability `1 - p_lex`. This mixture approach may naturally capture the intermediate pooled metrics across all four experiments better than a single compromised weighting scheme. Keep beta and epsilon parameters for the final choice.

**Outcome of this advice:** iter 2 candidate loss=0.2823 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Probabilistic Strategy Selection (Mixture of TTB and Tallying)",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    p_lex = float(parameters[\"p_lex\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    a, b = stim[0], stim[1]\n    \n    # Strategy 1: Take-The-Best (Lexicographic)\n    order = np.argsort(-validities, kind='stable')\n    scores_ttb = np.zeros(2)\n    for idx in order:\n        if a[idx] > b[idx]:\n            scores_ttb[0] = 1.0\n            break\n        elif b[idx] > a[idx]:\n            scores_ttb[1] = 1.0\n            break\n            \n    # Strategy 2: Tallying (Compensatory)\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    scores_tal = np.array([a_wins, b_wins])\n    \n    # Softmax for TTB\n    z_ttb = beta * (scores_ttb - scores_ttb.max())\n    e_ttb = np.exp(z_ttb)\n    p_ttb = e_ttb / e_ttb.sum()\n    \n    # Softmax for Tallying\n    z_tal = beta * (scores_tal - scores_tal.max())\n    e_tal = np.exp(z_tal)\n    p_tal = e_tal / e_tal.sum()\n    \n    # Mixture of strategies\n    p_core = p_lex * p_ttb + (1.0 - p_lex) * p_tal\n    \n    # Uniform lapse blended into the mixture\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "p_lex": "[0.0, 1.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we implement a 'Mixture of Strategies' theory. The previous rank-decay model struggled because adjusting a single set of continuous weights caused it to overshoot on some experiments while trying to fit others. By instead modeling behavior as a probabilistic mixture between a pure lexicographic strategy (Take-The-Best) and a pure compensatory strategy (Tallying) on each trial, the model can natively capture the intermediate pooled metrics across all four experiments by balancing the `p_lex` parameter."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9820,
  "candidates_token_count": 860,
  "total_token_count": 12228
}
```
