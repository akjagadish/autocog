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

Replace the Two-Stage Heuristic with a purely compensatory model, such as the Weighted Additive (WADD) model. In WADD, decision-makers compute an overall value for each option by taking the sum of the validities (or log-odds) for all features where the option has a positive cue. Choice probabilities are then derived from the difference in these overall values. This will provide a strong, classical compensatory alternative to the strictly non-compensatory Take-The-Best model.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 1.000)

**Description**
Take-The-Best (TTB) heuristic: Individuals make decisions by sequentially searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the choice is based entirely on that single feature, ignoring all others. If no feature discriminates, a random guess is made.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            scores = np.array([0.0, 1.0])
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1872 (var=0.0074) vs this=0.1725 (var=0.0055)
- Experiment 2: real=0.0567 (var=0.0012) vs this=0.0569 (var=0.0021)
- Experiment 3: real=0.0015 (var=0.0001) vs this=0.0003 (var=0.0001)
- Experiment 4: real=0.1506 (var=0.0048) vs this=0.1473 (var=0.0034)
- Experiment 5: real=0.0024 (var=0.0002) vs this=0.0015 (var=0.0002)
- Experiment 6: real=0.0363 (var=0.0007) vs this=0.0311 (var=0.0008)


---

### `pi_4` (overall score: 0.639)

**Description**
Two-Stage Heuristic (TTB with Tallying Fallback): Decision-makers initially employ a non-compensatory Take-The-Best (TTB) strategy, searching sequentially through features in descending order of validity. However, they only trust features whose validity exceeds a certain subjective threshold. If a discriminating feature is found above this threshold, the choice is based entirely on it. If no such feature exists (either because all high-validity features are tied or none meet the threshold), the decision-maker abandons the sequential search and falls back to a compensatory Tallying strategy, weighing all features equally and choosing the option with the most winning features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = None
    # Stage 1: Take-The-Best for features above the validity threshold
    for idx in order:
        if validities[idx] >= threshold:
            if a[idx] > b[idx]:
                scores = np.array([1.0, 0.0])
                break
            elif b[idx] > a[idx]:
                scores = np.array([0.0, 1.0])
                break
        else:
            # Stop searching if we hit features below the threshold
            break
            
    # Stage 2: Fallback to Tallying if no high-validity cue discriminated
    if scores is None:
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.5, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1872 (var=0.0074) vs this=0.1578 (var=0.0067)
- Experiment 2: real=0.0567 (var=0.0012) vs this=0.0559 (var=0.0016)
- Experiment 3: real=0.0015 (var=0.0001) vs this=0.0055 (var=0.0035)
- Experiment 4: real=0.1506 (var=0.0048) vs this=0.0664 (var=0.0046)
- Experiment 5: real=0.0024 (var=0.0002) vs this=0.0849 (var=0.0070)
- Experiment 6: real=0.0363 (var=0.0007) vs this=0.0006 (var=0.0014)


---

### `pi_2` (overall score: 0.158)

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
- Experiment 1: real=0.1872 (var=0.0074) vs this=0.0401 (var=0.0034)
- Experiment 2: real=0.0567 (var=0.0012) vs this=0.0006 (var=0.0016)
- Experiment 3: real=0.0015 (var=0.0001) vs this=0.1054 (var=0.0057)
- Experiment 4: real=0.1506 (var=0.0048) vs this=0.0013 (var=0.0012)
- Experiment 5: real=0.0024 (var=0.0002) vs this=0.1548 (var=0.0081)
- Experiment 6: real=0.0363 (var=0.0007) vs this=0.0634 (var=0.0023)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|0': 0.5089498806682577, '((1, 0, 0, 1, 0), (0, 1, 1, 0, 0))|1': 0.5178571428571429, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.17070484581497797, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.14742152466367714, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|0': 0.15504358655043587, '((0, 1, 0, 1, 1), (1, 0, 1, 0, 0))|1': 0.15195586760280844, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|0': 0.48748293126991354, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 0))|1': 0.4925160370634355, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|0': 0.4931506849315068, '((1, 0, 0, 0, 1), (0, 1, 1, 0, 0))|1': 0.4925986842105263, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.8503787878787878, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.8474462365591398, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|0': 0.5023847376788553, '((0, 1, 1, 0, 0), (1, 0, 0, 0, 1))|1': 0.4997081144191477, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.8523111612175873, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.848302300109529}
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

**Observed (real) value:** 0.1872 (var=0.0074)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0005 (var=0.0001)
- pi_2: 0.0401 (var=0.0034)
- pi_3: 0.1725 (var=0.0055)
- pi_4: 0.1578 (var=0.0067)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.6349115801483172, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.6096372495939361, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|0': 0.3445139283683911, '((1, 0, 1, 1, 0), (0, 1, 0, 0, 1))|1': 0.3454644215100489, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|0': 0.4446428571428571, '((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))|1': 0.5481481481481482, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|0': 0.5600208768267223, '((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))|1': 0.42755344418052255, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|0': 0.6502320185614849, '((0, 1, 1, 0, 0), (1, 0, 0, 1, 1))|1': 0.5591684434968017, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|0': 0.3051813471502591, '((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))|1': 0.49101796407185627, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.44503862150920975, '((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.4282733437663015, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|0': 0.6614420062695925, '((0, 0, 1, 1, 1), (1, 1, 0, 0, 0))|1': 0.5}
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

**Observed (real) value:** 0.0567 (var=0.0012)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0006 (var=0.0016)
- pi_1: 0.0386 (var=0.0011)
- pi_3: 0.0569 (var=0.0021)
- pi_4: 0.0559 (var=0.0016)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 1), (0, 1, 1, 0))|0': 0.14050822122571002, '((1, 0, 0, 1), (0, 1, 1, 0))|1': 0.16446955430006277, '((0, 1, 0, 0), (0, 0, 1, 1))|0': 0.17253774263120059, '((0, 1, 0, 0), (0, 0, 1, 1))|1': 0.15436849253055682, '((0, 1, 1, 0), (1, 0, 0, 1))|0': 0.835639180962922, '((0, 1, 1, 0), (1, 0, 0, 1))|1': 0.8326826547685443, '((0, 1, 1, 1), (1, 0, 0, 0))|0': 0.85, '((0, 1, 1, 1), (1, 0, 0, 0))|1': 0.8391089108910891, '((0, 0, 1, 1), (0, 1, 0, 0))|0': 0.8409255242227043, '((0, 0, 1, 1), (0, 1, 0, 0))|1': 0.8592692828146143, '((1, 0, 0, 0), (0, 1, 1, 1))|0': 0.14977755808205634, '((1, 0, 0, 0), (0, 1, 1, 1))|1': 0.14774889029803423, '((0, 1, 0, 1), (1, 0, 1, 0))|0': 0.8511904761904762, '((0, 1, 0, 1), (1, 0, 1, 0))|1': 0.8495762711864406, '((1, 0, 1, 0), (0, 1, 0, 1))|0': 0.15014005602240896, '((1, 0, 1, 0), (0, 1, 0, 1))|1': 0.15973597359735975}
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

**Observed (real) value:** 0.0015 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0003 (var=0.0001)
- pi_2: 0.1054 (var=0.0057)
- pi_1: 0.1825 (var=0.0018)
- pi_4: 0.0055 (var=0.0035)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|0': 0.45472703062583225, '((1, 0, 1, 0, 0), (0, 1, 0, 1, 1))|1': 0.629297075423294, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|0': 0.7783191230207065, '((1, 0, 0, 0, 0), (0, 1, 1, 1, 1))|1': 0.8400689995071464, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|0': 0.5674967234600262, '((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))|1': 0.6655486344034499, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|0': 0.7061978545887961, '((1, 1, 0, 0, 0), (1, 0, 1, 1, 1))|1': 0.7849328692192938, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|0': 0.7192262602579133, '((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))|1': 0.7954431647471207}
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

**Observed (real) value:** 0.1506 (var=0.0048)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0013 (var=0.0012)
- pi_3: 0.1473 (var=0.0034)
- pi_1: 0.0166 (var=0.0005)
- pi_4: 0.0664 (var=0.0046)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
P_REF = {'((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|0': 0.1345585822120118, '((1, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1))|1': 0.1735985533453888, '((0, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|0': 0.13043478260869565, '((0, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1))|1': 0.20147874306839186, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|0': 0.13251454427925016, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|1': 0.16798418972332016, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|0': 0.12548387096774194, '((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))|1': 0.194, '((1, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|0': 0.14308733483725428, '((1, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|1': 0.22334004024144868, '((1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 1, 1))|0': 0.14262295081967213, '((1, 0, 1, 0, 0, 0), (1, 0, 0, 1, 1, 1))|1': 0.22181818181818183, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13987252124645894, '((0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.21638655462184875, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|0': 0.13222079589216945, '((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))|1': 0.19421487603305784}
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

**Observed (real) value:** 0.0024 (var=0.0002)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0015 (var=0.0002)
- pi_4: 0.0849 (var=0.0070)
- pi_1: 0.2264 (var=0.0043)
- pi_2: 0.1548 (var=0.0081)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
P_REF = {'((0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 0, 0))|0': 0.5708358032009484, '((0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 0, 0))|1': 0.6063774176685833, '((1, 1, 1, 0, 0, 0), (1, 1, 0, 1, 1, 1))|0': 0.4607361963190184, '((1, 1, 1, 0, 0, 0), (1, 1, 0, 1, 1, 1))|1': 0.4025380710659898, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|0': 0.6022727272727273, '((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 1, 1))|1': 0.5747716894977168, '((0, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0))|0': 0.8103298611111112, '((0, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0))|1': 0.7932098765432098, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|0': 0.20738820479585224, '((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 1))|1': 0.1973748176956733, '((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0))|0': 0.14415862808145766, '((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0))|1': 0.14705882352941177, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|0': 0.8484848484848485, '((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1))|1': 0.8550505050505051, '((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|0': 0.5021186440677966, '((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0))|1': 0.6478937728937729}
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

**Observed (real) value:** 0.0363 (var=0.0007)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0006 (var=0.0014)
- pi_3: 0.0311 (var=0.0008)
- pi_1: 0.1112 (var=0.0021)
- pi_2: 0.0634 (var=0.0023)


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
  "theory": "Weighted Additive (WADD) Model with Log-Odds: Decision-makers evaluate options using a purely compensatory strategy. They compute an overall value for each option by summing the log-odds of the validities for all features where the option has a positive cue. A choice is then made probabilistically based on the difference in these overall values, using a softmax rule combined with a uniform lapse rate to account for decision noise.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Transform validities into log-odds weights\n    v = np.clip(validities, 1e-5, 1.0 - 1e-5)\n    w = np.log(v / (1.0 - v))\n    \n    # Compute the overall value (weighted sum) for each option\n    scores = stim @ w\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax over scores with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's recommendation, this proposes a strong, classical compensatory alternative to the non-compensatory Take-The-Best model. By implementing a Weighted Additive (WADD) model that directly derives its weights from the log-odds of the provided validities, we avoid the overfitting associated with learning free weights for each feature. Decision-makers sum these log-odds weights for all positive cues and make a probabilistic choice based on the resulting values."
}
```
```

## Usage

```json
{
  "prompt_token_count": 15341,
  "candidates_token_count": 644,
  "total_token_count": 17499
}
```
