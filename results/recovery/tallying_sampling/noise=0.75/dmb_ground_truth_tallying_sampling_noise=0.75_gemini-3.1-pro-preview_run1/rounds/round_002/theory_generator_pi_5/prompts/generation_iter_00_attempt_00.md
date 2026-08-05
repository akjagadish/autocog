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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a 'Strategy Mixture' or 'Dual-Process' theory. The new theory should posit that decisions are generated by a mixture of strategies: with some probability (or for some proportion of the population), a subject uses a non-compensatory heuristic like Take-The-Best, and otherwise, they use a compensatory rule like WADD or Tallying (Equal Weights). This mixture approach will naturally predict the intermediate metric values observed across all experiments, blending the flat sensitivity of TTB with the graded sensitivity of WADD.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.840)

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
- Experiment 1: real=0.4071 (var=0.0029) vs this=0.3210 (var=0.0381)
- Experiment 2: real=0.6008 (var=0.0065) vs this=0.6663 (var=0.0281)
- Experiment 3: real=0.4467 (var=0.0059) vs this=0.4662 (var=0.0377)
- Experiment 4: real=0.5000 (var=0.0194) vs this=0.4875 (var=0.0942)
- Experiment 5: real=0.4429 (var=0.0044) vs this=0.3958 (var=0.0239)
- Experiment 6: real=0.0907 (var=0.0122) vs this=0.2465 (var=0.0374)


---

### `pi_4` (overall score: 0.784)

**Description**
Probabilistic Take-The-Best (TTB with evaluation error): Subjects evaluate features sequentially in order of validity, making a decision based on the first feature that discriminates between the options. However, feature evaluation is subject to noise. When a feature actually discriminates, there is a probability 'gamma' (evaluation error) that the subject fails to notice or utilize this discrimination, causing them to mistakenly proceed to the next feature in the hierarchy. If all features are exhausted without a decision, the subject guesses. A baseline lapse rate 'epsilon' accounts for completely random choices independent of the cue evaluation process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    gamma = float(parameters['gamma'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    for j in cue_order:
        if a[j] > b[j]:
            p_a += p_reach * (1.0 - gamma)
            p_reach *= gamma
        elif b[j] > a[j]:
            p_b += p_reach * (1.0 - gamma)
            p_reach *= gamma
            
    # If no feature discriminates (or all discriminating features were missed), guess uniformly
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gamma: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4071 (var=0.0029) vs this=0.5329 (var=0.0133)
- Experiment 2: real=0.6008 (var=0.0065) vs this=0.4592 (var=0.0108)
- Experiment 3: real=0.4467 (var=0.0059) vs this=0.4775 (var=0.0047)
- Experiment 4: real=0.5000 (var=0.0194) vs this=0.4238 (var=0.0246)
- Experiment 5: real=0.4429 (var=0.0044) vs this=0.5746 (var=0.0122)
- Experiment 6: real=0.0907 (var=0.0122) vs this=0.0162 (var=0.0172)


---

### `pi_3` (overall score: 0.132)

**Description**
Rank-Weighted Additive Theory: People evaluate options by computing a weighted sum of features, but instead of using cardinal validities, they assign weights based on the ordinal rank of the cues' validities. Specifically, weights decay linearly with rank (e.g., n_features - rank), providing a compensatory mechanism where the top cue has higher importance but not so much that it completely dominates lower-ranked cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Rank-Weighted Additive: weight cues by linear rank decay (n_features - rank)
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    cue_order = np.argsort(-val, kind="stable")
    weights = np.zeros(n_features)
    for rank, idx in enumerate(cue_order):
        weights[idx] = float(n_features - rank)
        
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4071 (var=0.0029) vs this=0.4281 (var=0.0019)
- Experiment 2: real=0.6008 (var=0.0065) vs this=0.6779 (var=0.0068)
- Experiment 3: real=0.4467 (var=0.0059) vs this=0.8458 (var=0.0130)
- Experiment 4: real=0.5000 (var=0.0194) vs this=0.8600 (var=0.0162)
- Experiment 5: real=0.4429 (var=0.0044) vs this=0.5575 (var=0.0014)
- Experiment 6: real=0.0907 (var=0.0122) vs this=0.4505 (var=0.0181)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None and row['response'] == ttb_choice:
            ttb_matches += 1
            
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4071 (var=0.0029)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8638 (var=0.0118)
- pi_2: 0.3210 (var=0.0381)
- pi_3: 0.4281 (var=0.0019)
- pi_4: 0.5329 (var=0.0133)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_resp = data.loc[mask, 'response']
    a_higher = sum_a[mask] > sum_b[mask]
    
    chose_higher = (a_higher & (filtered_resp == 0)) | (~a_higher & (filtered_resp == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.6008 (var=0.0065)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6663 (var=0.0281)
- pi_1: 0.1458 (var=0.0146)
- pi_3: 0.6779 (var=0.0068)
- pi_4: 0.4592 (var=0.0108)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total_count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        if a == (1, 0, 0, 0, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 0, 1, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 1, 0, 0):
            match_count += (resp == 0)
            total_count += 1
    if total_count == 0:
        return 0.5
    return match_count / total_count
```

**Observed (real) value:** 0.4467 (var=0.0059)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8458 (var=0.0130)
- pi_2: 0.4662 (var=0.0377)
- pi_1: 0.3825 (var=0.0049)
- pi_4: 0.4775 (var=0.0047)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # We want to identify trials with the critical pair:
    # [1, 0, 0, 0, 1] vs [0, 1, 1, 0, 0]
    
    def is_target(a, b):
        return (tuple(a) == (1, 0, 0, 0, 1) and tuple(b) == (0, 1, 1, 0, 0))
    
    def is_target_swapped(a, b):
        return (tuple(a) == (0, 1, 1, 0, 0) and tuple(b) == (1, 0, 0, 0, 1))

    target_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if is_target(a, b):
            # Option B is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 1 else 0)
        elif is_target_swapped(a, b):
            # Option A is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 0 else 0)
            
    if not target_chosen:
        return 0.5
    
    return float(np.mean(target_chosen))
```

**Observed (real) value:** 0.5000 (var=0.0194)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4875 (var=0.0942)
- pi_3: 0.8600 (var=0.0162)
- pi_1: 0.1400 (var=0.0118)
- pi_4: 0.4238 (var=0.0246)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0
        
    ttb_choices = data.apply(get_ttb_choice, axis=1)
    return float((data['response'] == ttb_choices).mean())
```

**Observed (real) value:** 0.4429 (var=0.0044)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5746 (var=0.0122)
- pi_2: 0.3958 (var=0.0239)
- pi_1: 0.8410 (var=0.0115)
- pi_3: 0.5575 (var=0.0014)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    is_1v1 = (sum_a == 1) & (sum_b == 1)
    is_1v2 = (sum_a == 1) & (sum_b == 2)
    
    chose_a = (data['response'] == 0).values.astype(float)
    
    p_1v1 = np.mean(chose_a[is_1v1]) if np.sum(is_1v1) > 0 else 0.5
    p_1v2 = np.mean(chose_a[is_1v2]) if np.sum(is_1v2) > 0 else 0.5
    
    return float(p_1v1 - p_1v2)
```

**Observed (real) value:** 0.0907 (var=0.0122)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2465 (var=0.0374)
- pi_4: 0.0162 (var=0.0172)
- pi_1: -0.0252 (var=0.0073)
- pi_3: 0.4505 (var=0.0181)


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
  "theory": "Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    val = np.asarray(parameters['validities'], dtype=float)\n    \n    mixture_p = float(parameters['mixture_p'])\n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n    \n    # --- TTB Component ---\n    cue_order = np.argsort(-val, kind='stable')\n    a, b = stim[0], stim[1]\n    \n    p_ttb = np.array([0.5, 0.5])\n    for j in cue_order:\n        if a[j] > b[j]:\n            p_ttb = np.array([1.0, 0.0])\n            break\n        elif b[j] > a[j]:\n            p_ttb = np.array([0.0, 1.0])\n            break\n            \n    # --- WADD Component ---\n    # WADD uses validities as weights\n    scores = stim @ val\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_wadd = e / e.sum()\n    \n    # --- Mixture ---\n    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd\n    \n    # --- Lapse ---\n    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5\n    \n    return p_final",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "mixture_p": "[0.0, 1.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Based on the arbiter's feedback, this theory implements a Dual-Process or Strategy Mixture model. The previous models (like pure TTB with evaluation error or pure WADD) either over-predicted or under-predicted the extent of compensatory behavior across different experiments. By positing that behavior is a mixture of pure Take-The-Best (non-compensatory, flat sensitivity) and Weighted Additive (compensatory, graded sensitivity), the model can naturally interpolate between these extremes. This allows it to capture the intermediate metric values observed in the human data, providing a robust, experiment-invariant explanation of binary feature decision-making."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9279,
  "candidates_token_count": 794,
  "total_token_count": 12033
}
```
