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

Replace Theory 2 with a brand-new theory that competes with Pure Tallying without relying on cue validities. A strong candidate would be 'Stochastic Cue Sampling' or 'Random Subset Tallying'. In this theory, decision-makers are bounded by working memory and cannot process all features simultaneously. Instead of applying a post-tallying softmax noise, they randomly sample a subset of available cues on each trial (e.g., exactly k cues) and perform pure tallying on that subset. This provides a mechanistic, cognitive explanation for the observed choice noise while preserving the validity-agnostic nature of the Equal Weight heuristic that so perfectly matches the data.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 1.000)

**Description**
People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3500 (var=0.0124) vs this=0.3367 (var=0.0130)
- Experiment 2: real=0.1074 (var=0.0059) vs this=0.1324 (var=0.0111)
- Experiment 3: real=0.8213 (var=0.0136) vs this=0.8387 (var=0.0187)
- Experiment 4: real=0.1383 (var=0.0112) vs this=0.1467 (var=0.0120)
- Experiment 5: real=0.5133 (var=0.0104) vs this=0.5058 (var=0.0098)
- Experiment 6: real=0.5144 (var=0.0066) vs this=0.4917 (var=0.0053)


---

### `pi_4` (overall score: 0.628)

**Description**
Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
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
- theta: [-1.0, 1.0]
- k: [1.0, 20.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3500 (var=0.0124) vs this=0.3817 (var=0.0453)
- Experiment 2: real=0.1074 (var=0.0059) vs this=0.2116 (var=0.0309)
- Experiment 3: real=0.8213 (var=0.0136) vs this=0.6025 (var=0.1203)
- Experiment 4: real=0.1383 (var=0.0112) vs this=0.3342 (var=0.1021)
- Experiment 5: real=0.5133 (var=0.0104) vs this=0.6250 (var=0.0323)
- Experiment 6: real=0.5144 (var=0.0066) vs this=0.6483 (var=0.0285)


---

### `pi_2` (overall score: 0.565)

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
- Experiment 1: real=0.3500 (var=0.0124) vs this=0.2608 (var=0.0530)
- Experiment 2: real=0.1074 (var=0.0059) vs this=0.3596 (var=0.0246)
- Experiment 3: real=0.8213 (var=0.0136) vs this=0.4719 (var=0.1069)
- Experiment 4: real=0.1383 (var=0.0112) vs this=0.5417 (var=0.0901)
- Experiment 5: real=0.5133 (var=0.0104) vs this=0.6300 (var=0.0373)
- Experiment 6: real=0.5144 (var=0.0066) vs this=0.6172 (var=0.0474)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2296 -> ACCEPTED
- iter 2: loss=0.1120 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1120 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    is_comp = []
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is None:
            match = 0.5
        else:
            match = 1.0 if resp == ttb_choice else 0.0
            
        ttb_match.append(match)
        # Trials where the sum of features differ are compensatory in this design
        is_comp.append(sum(a) != sum(b))
        
    ttb_match = np.array(ttb_match)
    is_comp = np.array(is_comp)
    
    if is_comp.sum() == 0 or (~is_comp).sum() == 0:
        return 0.0
        
    comp_mean = ttb_match[is_comp].mean()
    non_comp_mean = ttb_match[~is_comp].mean()
    
    return float(non_comp_mean - comp_mean)
```

**Observed (real) value:** 0.3500 (var=0.0124)
**Previous candidate values (this loop):**
  - iter 1: 0.2042 (var=0.0113) (Δ vs real -0.1458)
  - iter 2 (most recent): 0.3129 (var=0.0136) (Δ vs real -0.0371)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0108 (var=0.0080)
- pi_2: 0.2608 (var=0.0530)
- pi_3: 0.3367 (var=0.0130)
- pi_4: 0.3817 (var=0.0453)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5
        
    ttb_preds = data.apply(ttb_choice, axis=1)
    match = (data['response'] == ttb_preds).mean()
    return float(match)
```

**Observed (real) value:** 0.1074 (var=0.0059)
**Previous candidate values (this loop):**
  - iter 1: 0.3048 (var=0.0116) (Δ vs real +0.1975)
  - iter 2 (most recent): 0.1617 (var=0.0060) (Δ vs real +0.0543)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3596 (var=0.0246)
- pi_1: 0.8385 (var=0.0129)
- pi_3: 0.1324 (var=0.0111)
- pi_4: 0.2116 (var=0.0309)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Convert rating lists to tuples for exact matching
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: A has the 2 most valid cues, B has the 3 less valid cues
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    # Trial 2: B has the 2 most valid cues, A has the 3 less valid cues
    t2_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    # For these trials, Tallying strictly prefers the option with 3 positive features.
    # WADD typically prefers the option with the 2 most valid cues.
    # We calculate the proportion of times the subject chose the 3-feature option.
    t1_tally_choices = (data.loc[t1_mask, 'response'] == 1).sum()
    t2_tally_choices = (data.loc[t2_mask, 'response'] == 0).sum()
    
    total_tally_choices = t1_tally_choices + t2_tally_choices
    total_critical_trials = t1_mask.sum() + t2_mask.sum()
    
    if total_critical_trials == 0:
        return 0.5
        
    return float(total_tally_choices / total_critical_trials)
```

**Observed (real) value:** 0.8213 (var=0.0136)
**Previous candidate values (this loop):**
  - iter 1: 0.6462 (var=0.0167) (Δ vs real -0.1750)
  - iter 2 (most recent): 0.7481 (var=0.0159) (Δ vs real -0.0731)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8387 (var=0.0187)
- pi_2: 0.4719 (var=0.1069)
- pi_1: 0.1531 (var=0.0109)
- pi_4: 0.6025 (var=0.1203)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify critical dissociation trials where WADD and Tallying make opposite predictions
    a_is_11000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    b_is_00111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    t1 = a_is_11000 & b_is_00111
    
    a_is_00111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    b_is_11000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    t7 = a_is_00111 & b_is_11000
    
    # WADD prefers the option with fewer, but higher-validity features ([1, 1, 0, 0, 0])
    wadd_t1 = t1 & (data['response'] == 0)
    wadd_t7 = t7 & (data['response'] == 1)
    
    wadd_choices = wadd_t1.sum() + wadd_t7.sum()
    total_trials = t1.sum() + t7.sum()
    
    if total_trials == 0:
        return 0.5
    return float(wadd_choices / total_trials)
```

**Observed (real) value:** 0.1383 (var=0.0112)
**Previous candidate values (this loop):**
  - iter 1: 0.3692 (var=0.0178) (Δ vs real +0.2308)
  - iter 2 (most recent): 0.2742 (var=0.0157) (Δ vs real +0.1358)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5417 (var=0.0901)
- pi_3: 0.1467 (var=0.0120)
- pi_1: 0.8350 (var=0.0156)
- pi_4: 0.3342 (var=0.1021)

### Experiment 5
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    top3_a = a_ratings[:, :3].sum(axis=1)
    top3_b = b_ratings[:, :3].sum(axis=1)
    
    # Isolate trials where total features are tied but top-3 features differ
    valid_mask = (sum_a == sum_b) & (top3_a != top3_b)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Did the subject choose the option with more top-3 features?
    chose_high_val = np.zeros(len(data), dtype=bool)
    chose_high_val[(top3_a > top3_b) & (responses == 0)] = True
    chose_high_val[(top3_a < top3_b) & (responses == 1)] = True
    
    return float(np.mean(chose_high_val[valid_mask]))
```

**Observed (real) value:** 0.5133 (var=0.0104)
**Previous candidate values (this loop):**
  - iter 1: 0.4900 (var=0.0126) (Δ vs real -0.0233)
  - iter 2 (most recent): 0.4983 (var=0.0085) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5058 (var=0.0098)
- pi_4: 0.6250 (var=0.0323)
- pi_1: 0.8658 (var=0.0121)
- pi_2: 0.6300 (var=0.0373)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Pure Tallying relies only on the unweighted sum of features.
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Identify "tie" trials where both options have the exact same number of total positive features.
    # In the provided design, these are trials 5, 6, 7, and 9.
    tie_mask = (sum_a == sum_b)
    
    if not np.any(tie_mask):
        return 0.5
        
    responses = data['response'].values
    
    # In all of these tie trials, Option A was adversarially designed to contain strictly more 
    # high-validity features than Option B. 
    # We calculate the proportion of times the subject chose Option A on these trials.
    p_a = np.mean(responses[tie_mask] == 0)
    
    return float(p_a)
```

**Observed (real) value:** 0.5144 (var=0.0066)
**Previous candidate values (this loop):**
  - iter 1: 0.5017 (var=0.0061) (Δ vs real -0.0128)
  - iter 2 (most recent): 0.4828 (var=0.0060) (Δ vs real -0.0317)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6483 (var=0.0285)
- pi_3: 0.4917 (var=0.0053)
- pi_1: 0.6867 (var=0.0048)
- pi_2: 0.6172 (var=0.0474)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Random Subset Tallying: Decision-makers use an equal-weight heuristic but are bounded by working memory, preventing them from processing all features simultaneously. Instead of calculating a complete tally and applying post-decision softmax noise, they stochastically sample a subset of the available features on each trial (each feature included independently with some probability) and perform pure tallying strictly on that subset. This provides a mechanistic, cognitive origin for choice variability while preserving the validity-agnostic, compensatory nature of the Tallying heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    import itertools
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    n_features = stim.shape[1]
    p = float(parameters["sample_prob"])
    epsilon = float(parameters["epsilon"])
    
    prob_A = 0.0
    
    # Iterate over all possible subsets of features (2^n_features)
    for seq in itertools.product([0, 1], repeat=n_features):
        mask = np.array(seq)
        # Probability of sampling this specific subset
        subset_prob = np.prod(np.where(mask == 1, p, 1.0 - p))
        
        if subset_prob == 0:
            continue
            
        score_A = np.sum(stim[0] * mask)
        score_B = np.sum(stim[1] * mask)
        
        # Pure tallying on the sampled subset
        if score_A > score_B:
            prob_A += subset_prob
        elif score_A == score_B:
            prob_A += 0.5 * subset_prob
            
    prob_B = 1.0 - prob_A
    p_core = np.array([prob_A, prob_B])
    
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- sample_prob: [0.5, 1.0]
- epsilon: [0.0, 0.2]

`rationale`: Following the critic's advice, we restrict the parameter ranges for Random Subset Tallying. By setting `sample_prob` to [0.5, 1.0] and limiting the lapse rate `epsilon` to [0.0, 0.2], we reduce the excessive guessing that previously caused the model to underestimate the strength of the Tallying preference on non-tie trials. This allows the model to match the high >80% tallying match rates seen in Experiments 3 and 4 while preserving the correct 50% tie-breaking behavior in Experiments 5 and 6.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The Random Subset Tallying model successfully captures the core validity-agnostic nature of human decisions, perfectly matching the ~0.50 guessing rates on tie trials in Experiments 5 and 6. However, it systematically underestimates the strength of the Tallying preference on non-tie trials. In Experiment 3, subjects choose the tallying-favored option 82% of the time, but the model only predicts 64%. Similarly in Experiment 4, the WADD-favored choice is made only 13% of the time, but the model predicts 36% (closer to 50%). This indicates that the simulated subjects are guessing too often, likely because the parameter ranges for `sample_prob` allow for too few features to be sampled, or `epsilon` allows for too much lapse rate.
Rationale: The mechanism is an excellent fit for the arbiter's recommendation and correctly captures the qualitative phenomena. To improve the quantitative fit, restrict the parameter ranges to reduce excessive guessing. Change `sample_prob` to [0.5, 1.0] (ensuring a reasonable number of features are processed) and `epsilon` to [0.0, 0.2] (limiting the uniform lapse rate). This will allow the model to reach the high >80% tallying match rates seen in Experiments 3 and 4 while maintaining the 50% tie-breaking behavior.

**Outcome of this advice:** iter 1 candidate loss=0.2296 -> iter 2 candidate loss=0.1120 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The Random Subset Tallying model continues to perform very well, and the previous parameter restrictions significantly improved the quantitative fit (loss decreased from 0.2296 to 0.1120). The model perfectly captures the ~0.50 tie-breaking behavior in Experiments 5 and 6, showing it correctly ignores cue validities. However, there is still some room for improvement on the non-tie trials. In Experiment 3, the model predicts a 74.8% tallying preference (real = 82.1%). In Experiment 4, it predicts a 27.4% WADD-favored choice rate (real = 13.8%). The model is still slightly underestimating the strength of the pure tallying effect when one option has more features.
Rationale: The current mechanism is highly successful, but the parameter ranges can be tightened further to reduce the remaining excess noise on non-tie trials. Restrict `sample_prob` to [0.7, 1.0] and `epsilon` to [0.0, 0.1]. This will further increase the probability of sampling enough features to reliably detect the tallying winner, driving the Experiment 3 prediction closer to 0.82 and the Experiment 4 prediction closer to 0.14, while keeping the structural 50% tie-breaking behavior intact.

**Outcome of this advice:** iter 2 candidate loss=0.1120 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Random Subset Tallying: Decision-makers use an equal-weight heuristic but are bounded by working memory, preventing them from processing all features simultaneously. Instead of calculating a complete tally and applying post-decision softmax noise, they stochastically sample a subset of the available features on each trial (each feature included independently with some probability) and perform pure tallying strictly on that subset. This provides a mechanistic, cognitive origin for choice variability while preserving the validity-agnostic, compensatory nature of the Tallying heuristic.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    import itertools\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    n_features = stim.shape[1]\n    p = float(parameters[\"sample_prob\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    prob_A = 0.0\n    \n    # Iterate over all possible subsets of features (2^n_features)\n    for seq in itertools.product([0, 1], repeat=n_features):\n        mask = np.array(seq)\n        # Probability of sampling this specific subset\n        subset_prob = np.prod(np.where(mask == 1, p, 1.0 - p))\n        \n        if subset_prob == 0:\n            continue\n            \n        score_A = np.sum(stim[0] * mask)\n        score_B = np.sum(stim[1] * mask)\n        \n        # Pure tallying on the sampled subset\n        if score_A > score_B:\n            prob_A += subset_prob\n        elif score_A == score_B:\n            prob_A += 0.5 * subset_prob\n            \n    prob_B = 1.0 - prob_A\n    p_core = np.array([prob_A, prob_B])\n    \n    # Blend with uniform lapse distribution\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()  # Ensure valid probabilities\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "sample_prob": "[0.7, 1.0]",
    "epsilon": "[0.0, 0.1]"
  },
  "rationale": "Following the critic's advice, we further tighten the parameter ranges for Random Subset Tallying. By restricting `sample_prob` to [0.7, 1.0] and `epsilon` to [0.0, 0.1], we further reduce the excess noise on non-tie trials. This increases the probability of sampling enough features to reliably detect the tallying winner, driving the Experiment 3 prediction closer to 0.82 and the Experiment 4 prediction closer to 0.14, while preserving the structural 50% tie-breaking behavior seen in Experiments 5 and 6."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12440,
  "candidates_token_count": 791,
  "total_token_count": 13491
}
```
