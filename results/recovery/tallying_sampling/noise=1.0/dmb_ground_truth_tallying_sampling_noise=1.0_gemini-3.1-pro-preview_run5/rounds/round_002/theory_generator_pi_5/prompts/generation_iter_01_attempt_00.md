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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a 'First-Feature Heuristic' (or 'Left-to-Right Reading Bias') theory. Instead of integrating all features (like WADD) or guessing entirely at random (like THEORY 1), subjects might minimize cognitive effort by only inspecting the very first feature in the array (the leftmost one). If one option has a 1 and the other a 0 on this first feature, they choose the option with the 1. If they tie on the first feature, the subject guesses. Because the first feature's validity and values vary across trials and don't necessarily align with the overall WADD score or TTB predictions, this heuristic would produce aggregate metric values near 0.5 for the current metrics, making it a strong and mechanistically distinct competitor to pure Random Guessing.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.991)

**Description**
When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- bias: [0.4, 0.6]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4883 (var=0.0032) vs this=0.4985 (var=0.0023)
- Experiment 2: real=0.4963 (var=0.0022) vs this=0.4925 (var=0.0018)
- Experiment 3: real=0.4833 (var=0.0088) vs this=0.4767 (var=0.0081)
- Experiment 4: real=0.4500 (var=0.0206) vs this=0.4750 (var=0.0231)
- Experiment 5: real=0.5050 (var=0.0055) vs this=0.4708 (var=0.0099)
- Experiment 6: real=0.5015 (var=0.0036) vs this=0.5078 (var=0.0034)


---

### `pi_2` (overall score: 0.471)

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
- Experiment 1: real=0.4883 (var=0.0032) vs this=0.4183 (var=0.0236)
- Experiment 2: real=0.4963 (var=0.0022) vs this=0.4096 (var=0.0238)
- Experiment 3: real=0.4833 (var=0.0088) vs this=0.4708 (var=0.1009)
- Experiment 4: real=0.4500 (var=0.0206) vs this=0.4667 (var=0.0733)
- Experiment 5: real=0.5050 (var=0.0055) vs this=0.8708 (var=0.0124)
- Experiment 6: real=0.5015 (var=0.0036) vs this=0.6715 (var=0.0171)


---

### `pi_3` (overall score: 0.443)

**Description**
People compare two options by tallying the number of positive features for each option, treating all cues as equally important and completely ignoring their provided validities. This Tallying (or Equal-Weight) heuristic is a simple compensatory strategy that is less cognitively demanding than the Weighted Additive (WADD) rule but avoids relying on just a single feature like Take-The-Best (TTB). The option with the higher tally is chosen. When the tallies are tied, the decision-maker has no basis for preference and guesses uniformly. Response noise is incorporated through a softmax function over the tally scores with an inverse temperature parameter (beta), plus an independent lapse rate (epsilon) that occasionally replaces the choice with a uniform random guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: simply count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4883 (var=0.0032) vs this=0.3554 (var=0.0111)
- Experiment 2: real=0.4963 (var=0.0022) vs this=0.3056 (var=0.0127)
- Experiment 3: real=0.4833 (var=0.0088) vs this=0.7875 (var=0.0299)
- Experiment 4: real=0.4500 (var=0.0206) vs this=0.2683 (var=0.0392)
- Experiment 5: real=0.5050 (var=0.0055) vs this=0.7367 (var=0.0243)
- Experiment 6: real=0.5015 (var=0.0036) vs this=0.5968 (var=0.0058)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4801 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4801 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Calculate the difference between features
    diff = a_mat - b_mat
    
    # Find the index of the first discriminating cue (first non-zero difference)
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    
    ttb_preds = np.zeros(len(diff))
    for i in range(len(diff)):
        idx = first_diff_idx[i]
        if diff[i, idx] > 0:
            ttb_preds[i] = 0
        elif diff[i, idx] < 0:
            ttb_preds[i] = 1
        else:
            ttb_preds[i] = 0.5
            
    # Return the proportion of choices that align with the Take-The-Best prediction
    return float(np.mean(resp == ttb_preds))
```

**Observed (real) value:** 0.4883 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7006 (var=0.0124) (Δ vs real +0.2123)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8669 (var=0.0066)
- pi_2: 0.4183 (var=0.0236)
- pi_3: 0.3554 (var=0.0111)
- pi_4: 0.4985 (var=0.0023)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    valid_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        if ttb_pred != -1:
            if row['response'] == ttb_pred:
                matches += 1
            valid_trials += 1
    return matches / valid_trials if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4963 (var=0.0022)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6527 (var=0.0138) (Δ vs real +0.1565)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4096 (var=0.0238)
- pi_1: 0.8650 (var=0.0066)
- pi_3: 0.3056 (var=0.0127)
- pi_4: 0.4925 (var=0.0018)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has 3 low-validity cues, B has 2 high-validity cues
    is_trial_1 = (a_str == '00111') & (b_str == '11000')
    # Trial 2: A has 2 high-validity cues, B has 3 low-validity cues
    is_trial_2 = (a_str == '11000') & (b_str == '00111')
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    # Tallying prefers the option with 3 positive features
    tally_correct_1 = is_trial_1 & (data['response'] == 0)
    tally_correct_2 = is_trial_2 & (data['response'] == 1)
    
    return float((tally_correct_1 | tally_correct_2).sum() / mask.sum())
```

**Observed (real) value:** 0.4833 (var=0.0088)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2092 (var=0.0219) (Δ vs real -0.2742)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7875 (var=0.0299)
- pi_2: 0.4708 (var=0.1009)
- pi_1: 0.1333 (var=0.0089)
- pi_4: 0.4767 (var=0.0081)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 directly pits the two most valid features (Option A) 
    # against the three least valid features (Option B).
    # Tallying strictly prefers Option B (tally 3 vs 2).
    # WADD, on average, prefers Option A because the combined validity 
    # of the top 2 features outweighs the bottom 3.
    t1_mask = data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)
    if not t1_mask.any():
        return 0.5
    
    # Return the proportion of times Option A was chosen
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4500 (var=0.0206)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7617 (var=0.0378) (Δ vs real +0.3117)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4667 (var=0.0733)
- pi_3: 0.2683 (var=0.0392)
- pi_1: 0.8383 (var=0.0179)
- pi_4: 0.4750 (var=0.0231)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of features for options A and B
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Identify extreme trials where one option completely dominates the other
    diff = sum_a - sum_b
    extreme_mask = diff.abs() == 5
    
    if extreme_mask.sum() == 0:
        return 0.5
        
    # Determine the 'correct' choice (0 for A, 1 for B)
    correct_responses = (diff[extreme_mask] < 0).astype(int)
    actual_responses = data.loc[extreme_mask, 'response']
    
    # Calculate the proportion of choices that align with the dominating option
    return float((actual_responses == correct_responses).mean())
```

**Observed (real) value:** 0.5050 (var=0.0055)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7442 (var=0.0334) (Δ vs real +0.2392)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4708 (var=0.0099)
- pi_2: 0.8708 (var=0.0124)
- pi_1: 0.8658 (var=0.0109)
- pi_3: 0.7367 (var=0.0243)

### Experiment 6
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.8, 0.6, 0.5])
    
    # Vectorized computation of WADD scores for Option A and B
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    score_A = A.dot(validities)
    score_B = B.dot(validities)
    diff = score_A - score_B
    
    # Filter out trials where options have identical WADD scores
    mask = np.abs(diff) > 1e-5
    if not np.any(mask):
        return 0.5
        
    # Predicted response: 0 (Option A) if score_A > score_B, else 1 (Option B)
    predicted = (diff[mask] < 0).astype(int)
    actual = data['response'].values[mask]
    
    # Return the proportion of choices consistent with the WADD rule
    return float(np.mean(predicted == actual))
```

**Observed (real) value:** 0.5015 (var=0.0036)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6378 (var=0.0095) (Δ vs real +0.1363)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6715 (var=0.0171)
- pi_4: 0.5078 (var=0.0034)
- pi_1: 0.7205 (var=0.0034)
- pi_3: 0.5968 (var=0.0058)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
First-Feature Heuristic (Left-to-Right Reading Bias): Subjects minimize cognitive effort by only inspecting the very first (leftmost) feature in the array. If one option has a 1 and the other a 0 on this first feature, they choose the option with the 1. If both options tie on the first feature, they guess randomly. This heuristic produces aggregate metric values near 0.5 on complex multi-attribute tasks because the first feature's value varies across trials and does not consistently align with overall normative scores like WADD or TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a_first = stim[0, 0]
    b_first = stim[1, 0]
    
    epsilon = float(parameters["epsilon"])
    
    if a_first > b_first:
        p_core = np.array([1.0, 0.0])
    elif b_first > a_first:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend deterministic first-feature rule with a random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- epsilon: [0.0, 1.0]

`rationale`: Following the arbiter's recommendation, this model implements the 'First-Feature Heuristic'. Instead of guessing entirely at random or integrating all features, subjects might simply look at the leftmost feature due to reading habits and minimize cognitive effort. If it discriminates, they use it; if it ties, they guess. This provides a mechanistically distinct alternative to pure Random Guessing that still yields aggregate metrics near 0.5 across the varied experimental designs, explaining the observed low performance on complex rationality metrics without assuming subjects completely ignore the stimulus.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The First-Feature Heuristic was proposed under the assumption that the first feature's value would not consistently align with the metrics across experiments, yielding aggregate values near 0.5. However, in the specific stimulus designs used (e.g., Experiment 5's extreme trials, or Experiment 4's focal trial), the first feature actually discriminates strongly and aligns with the metric's target choice. This causes the model's predictions to deviate significantly from 0.5 (e.g., 0.76 in Exp 4, 0.74 in Exp 5, 0.20 in Exp 3), whereas the human data is tightly clustered around 0.5 across all experiments.
Rationale: While the mechanism perfectly follows the arbiter's recommendation, the assumption that the first feature is uncorrelated with the metrics failed on these specific experimental designs. To bring the simulated values down to the ~0.5 observed in the human data while keeping the First-Feature mechanism intact, the model needs a much higher base rate of guessing. Constrain the `epsilon` (lapse rate) parameter range to heavily favor guessing (e.g., `[0.8, 1.0]`). This will preserve the prescribed mechanism but properly calibrate its effect size to match the empirical data.

**Outcome of this advice:** iter 1 candidate loss=0.4801 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "First-Feature Heuristic (Left-to-Right Reading Bias) with High Lapse: Subjects minimize cognitive effort by inspecting only the first (leftmost) feature in the array. However, because this task is demanding and feedback is absent, subjects rely heavily on random guessing on the vast majority of trials, only occasionally applying the first-feature rule. This results in behavior that is near-random but retains a faint trace of the leftmost feature's influence.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.\")\n    \n    a_first = stim[0, 0]\n    b_first = stim[1, 0]\n    \n    epsilon = float(parameters[\"epsilon\"])\n    \n    if a_first > b_first:\n        p_core = np.array([1.0, 0.0])\n    elif b_first > a_first:\n        p_core = np.array([0.0, 1.0])\n    else:\n        p_core = np.array([0.5, 0.5])\n        \n    # Blend deterministic first-feature rule with a random lapse rate\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "epsilon": "[0.8, 1.0]"
  },
  "rationale": "Following the critic's diagnosis, the First-Feature Heuristic mechanism was producing metric values too far from the ~0.5 observed in humans due to structural correlations in the experimental designs. To fix this while retaining the arbiter's prescribed mechanism, I constrained the lapse rate (epsilon) to [0.8, 1.0]. This ensures the model guesses randomly on most trials, diluting the extreme predictions of the first-feature rule and bringing the aggregate metrics down to the empirical ~0.5 baseline."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11015,
  "candidates_token_count": 589,
  "total_token_count": 11916
}
```
