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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a 'Validity-Biased Tallying' or 'Mixture of Tallying and WADD' theory. Instead of a strict lexicographic rule like NV-TTB, this new theory should posit that decision-makers use a compensatory approach but with compressed or noisy weights. For example, a model where subjects calculate a weighted sum of cues, but the weights are heavily compressed toward equality (shrinking validities toward a uniform value), or a population-level mixture model where a majority of subjects use pure Tallying and a minority use a Weighted Additive (WADD) rule. This would capture the dominant tallying behavior while accounting for the slight pull of cue validities seen in the observed data.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.933)

**Description**
Tallying (Equal-Weight) Theory: People compare two options by simply counting the number of positive cues (features) for each option and choosing the one with the higher count. This theory posits that decision-makers ignore the varying validities of the cues, treating all features as equally important. It is a compensatory strategy because multiple cues can outweigh a single cue, but it is more frugal than a Weighted Additive (WADD) rule because it avoids multiplying by or storing cardinal validities. Response noise is modeled via a softmax function over the tally scores, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: count the number of positive cues for each option
    # Since cues are binary (0 or 1), we can just sum them.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.1, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6919 (var=0.0034) vs this=0.7638 (var=0.0082)
- Experiment 2: real=0.3892 (var=0.0034) vs this=0.3602 (var=0.0033)
- Experiment 3: real=0.6783 (var=0.0060) vs this=0.7333 (var=0.0207)
- Experiment 4: real=0.3467 (var=0.0161) vs this=0.2508 (var=0.0216)
- Experiment 5: real=0.7113 (var=0.0051) vs this=0.7943 (var=0.0137)
- Experiment 6: real=0.2792 (var=0.0051) vs this=0.2604 (var=0.0207)


---

### `pi_4` (overall score: 0.790)

**Description**
Noisy-Validity Take-The-Best (NV-TTB) Model: Decision makers use a strict lexicographic search (Take-The-Best), consulting cues in descending order of their validity and stopping at the first cue that discriminates between options. However, subjects have noisy internal representations of cue validities. On each choice, Gaussian noise is added to the objective validities, and cues are sorted based on these noisy values. This probabilistic cue ordering allows the model to capture aggregate deviations from pure TTB (and approach Tallying-like behavior when noise is high) while preserving the non-compensatory, one-reason decision mechanism at the single-trial level.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    epsilon = float(parameters["epsilon"])
    
    n_features = stim.shape[1]
    
    # Marginalize over noisy validities via sampling to produce choice probabilities
    n_samples = 200
    noise = np.random.normal(0, sigma + 1e-9, size=(n_samples, n_features))
    noisy_validities = validities + noise
    
    # Sort cues for each sample (descending order of noisy validity)
    cue_orders = np.argsort(-noisy_validities, axis=1)
    
    a, b = stim[0], stim[1]
    wins = np.zeros(2)
    
    for i in range(n_samples):
        winner = None
        for j in cue_orders[i]:
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            wins += 0.5
        else:
            wins[winner] += 1.0
            
    p_core = wins / n_samples
    
    # Incorporate lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- sigma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6919 (var=0.0034) vs this=0.6048 (var=0.0038)
- Experiment 2: real=0.3892 (var=0.0034) vs this=0.4408 (var=0.0037)
- Experiment 3: real=0.6783 (var=0.0060) vs this=0.4612 (var=0.0235)
- Experiment 4: real=0.3467 (var=0.0161) vs this=0.5117 (var=0.0112)
- Experiment 5: real=0.7113 (var=0.0051) vs this=0.5950 (var=0.0171)
- Experiment 6: real=0.2792 (var=0.0051) vs this=0.4246 (var=0.0098)


---

### `pi_2` (overall score: 0.784)

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
- Experiment 1: real=0.6919 (var=0.0034) vs this=0.6681 (var=0.0277)
- Experiment 2: real=0.3892 (var=0.0034) vs this=0.4569 (var=0.0152)
- Experiment 3: real=0.6783 (var=0.0060) vs this=0.4296 (var=0.0575)
- Experiment 4: real=0.3467 (var=0.0161) vs this=0.5575 (var=0.0876)
- Experiment 5: real=0.7113 (var=0.0051) vs this=0.7053 (var=0.0223)
- Experiment 6: real=0.2792 (var=0.0051) vs this=0.3287 (var=0.0335)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_cues = data['option_b_ratings'].apply(sum)
    a_cues = data['option_a_ratings'].apply(sum)
    mask = b_cues > a_cues
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6919 (var=0.0034)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1636 (var=0.0114)
- pi_2: 0.6681 (var=0.0277)
- pi_3: 0.7638 (var=0.0082)
- pi_4: 0.6048 (var=0.0038)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3892 (var=0.0034)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4569 (var=0.0152)
- pi_1: 0.8435 (var=0.0127)
- pi_3: 0.3602 (var=0.0033)
- pi_4: 0.4408 (var=0.0037)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.55, 0.5, 0.5])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    wadd_a = np.dot(a_ratings, validities)
    wadd_b = np.dot(b_ratings, validities)
    
    tally_pred = (sum_b > sum_a).astype(int)
    wadd_pred = (wadd_b > wadd_a).astype(int)
    
    critical_mask = (sum_a != sum_b) & (wadd_a != wadd_b) & (tally_pred != wadd_pred)
    
    if not np.any(critical_mask):
        return 0.5
        
    responses = data['response'].values[critical_mask]
    tally_preds = tally_pred[critical_mask]
    
    return float(np.mean(responses == tally_preds))

```

**Observed (real) value:** 0.6783 (var=0.0060)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7333 (var=0.0207)
- pi_2: 0.4296 (var=0.0575)
- pi_1: 0.1713 (var=0.0169)
- pi_4: 0.4612 (var=0.0235)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_consistent += (1 if resp == 0 else 0)
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_consistent += (1 if resp == 1 else 0)
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_consistent / total
```

**Observed (real) value:** 0.3467 (var=0.0161)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5575 (var=0.0876)
- pi_3: 0.2508 (var=0.0216)
- pi_1: 0.8392 (var=0.0186)
- pi_4: 0.5117 (var=0.0112)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    valid_trials = sum_a != sum_b
    if not valid_trials.any():
        return 0.5
        
    tally_choice = (sum_b > sum_a).astype(int)
    match = (data['response'] == tally_choice)
    
    return float(match[valid_trials].mean())
```

**Observed (real) value:** 0.7113 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7943 (var=0.0137)
- pi_4: 0.5950 (var=0.0171)
- pi_1: 0.1290 (var=0.0059)
- pi_2: 0.7053 (var=0.0223)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_pred = np.where(sum_a > sum_b, 0, np.where(sum_b > sum_a, 1, -1))
    ttb_pred = np.where(a_ratings[:, 0] > b_ratings[:, 0], 0, np.where(b_ratings[:, 0] > a_ratings[:, 0], 1, -1))
    discordant = (tally_pred != -1) & (ttb_pred != -1) & (tally_pred != ttb_pred)
    if not np.any(discordant):
        return 0.5
    responses = data['response'].values
    ttb_matches = (responses[discordant] == ttb_pred[discordant]).sum()
    return float(ttb_matches / np.sum(discordant))
```

**Observed (real) value:** 0.2792 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4246 (var=0.0098)
- pi_3: 0.2604 (var=0.0207)
- pi_1: 0.8592 (var=0.0112)
- pi_2: 0.3287 (var=0.0335)


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
  "theory": "Validity-Biased Tallying: Decision-makers evaluate options using a compensatory strategy where they compute a weighted sum of the features. However, instead of using the objective cue validities as weights (as in pure WADD) or completely ignoring them (as in pure Tallying), they use subjective weights that are heavily compressed toward equality. This means the weight of each cue is a mixture of a uniform value and its actual validity, capturing the dominant tallying behavior while allowing for a slight, noisy pull from the objective cue validities.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Expects a (2, n_features) stimulus.\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    alpha = float(parameters[\"alpha\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Interpolate between uniform weights (Tallying) and objective validities (WADD)\n    w = (1.0 - alpha) * 1.0 + alpha * validities\n    \n    # Calculate weighted sum of features for each option\n    scores = stim @ w\n    \n    # Softmax choice rule\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Incorporate uniform lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "beta": "[0.0, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "The arbiter suggested a 'Validity-Biased Tallying' theory where decision-makers use a compensatory approach but with weights compressed heavily toward equality. This model implements this by introducing an `alpha` parameter that interpolates between pure Tallying (equal weights of 1.0) and Weighted Additive (WADD, weights equal to validities). By allowing `alpha` to vary, the model can capture the dominant tallying behavior while accounting for the slight pull of cue validities seen in the observed data, overcoming the rigidity of strict lexicographic or pure equal-weight models."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9342,
  "candidates_token_count": 696,
  "total_token_count": 11271
}
```
