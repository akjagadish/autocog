# improvement_iter_00_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a NEW model instantiation of an EXISTING theory, given arbiter feedback on the previous instantiation. The theory's prose claim is fixed — you are NOT redefining the theory. You are regenerating only the runnable bits: the `predict` function, the `policy` function, and the `parameters` ranges. The newly proposed model should display human-like behavior when simulated on experiments in the Decision Making (Binary Features) domain.

The goal of the model improvement process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
If your model fails to compile or behaves badly, you may receive feedback and have to propose another instantiation. Iterate until accepted.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new model instance that ignores the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

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

## ROUND THEORIES
The arbiter compared the two theories below this round. Your job is to regenerate ONLY the runnable bits (`predict`, `policy`, `parameters`) of the one tagged **TO REVISE**, keeping its description verbatim. The other theory is shown for context — it is NOT being changed.

### THEORY 1 — `pi_5` **TO REVISE**

**Description (PRESERVE VERBATIM — do not rewrite the prose):**
Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

**Previous model instance — REGENERATE `predict`, `policy`, and `parameters` ranges:**

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities

---

### THEORY 2 — `pi_6` (other theory — context only, NOT revised)

**Description:**
Weak Weighted Additive with Extreme Noise: In complex multi-cue decision environments, subjects attempt to integrate all available information by weighting each cue according to its provided validity (a compensatory Weighted Additive strategy). However, the cognitive demands of multiplying and summing multiple cues lead to severe cognitive overload and disengagement. Consequently, subjects exhibit an overwhelmingly high lapse rate, effectively guessing on the vast majority of trials. The underlying Weighted Additive evaluation only weakly shines through the noise, explaining why choice behavior hovers very close to random guessing (0.5) across conflict metrics while maintaining a slight systematic preference for higher-validity cue combinations.

**Model instance (shown for context):**

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted Additive (WADD): sum of cues weighted by their validities
    scores = stim @ validities
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities


## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Preserve the Single-Cue Focus theory description, but modify the parameter ranges in the model. The `epsilon` (lapse rate) parameter range should be shifted to be extremely high (e.g., [0.90, 0.99] or even [0.95, 1.0]) to strongly heavily mask the single-cue heuristic. This will bring the predicted metrics much closer to the near-random observed values (e.g., pulling the Exp 1 prediction from -0.41 down to the observed -0.07, and pushing the proportion metrics in other experiments closer to 0.5).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.906)

**Description**
Equal Weights with High Lapse: Subjects often find the integration of complex cue validities too cognitively demanding or disengaging in these conflict paradigms. As a result, they ignore the provided validities entirely and fall back on a simple 'Tallying' (Equal Weights) heuristic, where they just count the number of positive features for each option. Furthermore, due to the high cognitive friction or confusion, subjects exhibit a very high baseline guessing rate (lapse), meaning that on the vast majority of trials they simply guess randomly. This explains why behavior across various conflict and agreement metrics hovers so closely to 0.5 or 0 difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: equal weights for all features (counting positive cues)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallied scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # High uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.8, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5025 (var=0.0026) vs this=0.4506 (var=0.0030)
- Experiment 2: real=0.4996 (var=0.0028) vs this=0.5415 (var=0.0036)
- Experiment 3: real=0.4947 (var=0.0048) vs this=0.4503 (var=0.0051)
- Experiment 4: real=-0.0111 (var=0.0415) vs this=-0.0378 (var=0.0365)
- Experiment 5: real=0.4996 (var=0.0028) vs this=0.4892 (var=0.0028)
- Experiment 6: real=0.5350 (var=0.0082) vs this=0.4662 (var=0.0191)
- Experiment 7: real=0.5004 (var=0.0029) vs this=0.4752 (var=0.0030)
- Experiment 8: real=0.4925 (var=0.0050) vs this=0.5306 (var=0.0065)
- Experiment 9: real=-0.0745 (var=0.4396) vs this=0.3078 (var=0.5855)
- Experiment 10: real=0.5130 (var=0.0042) vs this=0.5326 (var=0.0061)


---

### `pi_3` (overall score: 0.836)

**Description**
People are heterogeneous in their decision-making strategies, with some choices driven by a non-compensatory heuristic (Take The Best) and others by a compensatory strategy (Weighted Additive). The population consists of individuals who employ a mixture of these strategies, governed by a subjective mixture weight. By blending a frugal, single-reason strategy with a fully compensatory evaluation, the model captures both the variance and the balanced aggregate behavior observed across decision-making experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    n_features = stim.shape[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # --- TTB (Take The Best) ---
    cue_order = np.argsort(-validities, kind="stable").tolist()
    a, b = stim[0], stim[1]
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    beta = float(parameters["beta"])
    
    if winner_ttb is None:
        p_ttb = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * scores_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
        
    # --- WADD (Weighted Additive) ---
    scores_wadd = stim @ (validities * w)
    z_wadd = beta * scores_wadd
    e_wadd = np.exp(z_wadd - np.max(z_wadd))
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- MIXTURE ---
    wadd_prob = float(parameters["wadd_prob"])
    p_core = wadd_prob * p_wadd + (1.0 - wadd_prob) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features
- wadd_prob: [0.4, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5025 (var=0.0026) vs this=0.4948 (var=0.0127)
- Experiment 2: real=0.4996 (var=0.0028) vs this=0.5142 (var=0.0127)
- Experiment 3: real=0.4947 (var=0.0048) vs this=0.4603 (var=0.0125)
- Experiment 4: real=-0.0111 (var=0.0415) vs this=0.2022 (var=0.0418)
- Experiment 5: real=0.4996 (var=0.0028) vs this=0.6104 (var=0.0132)
- Experiment 6: real=0.5350 (var=0.0082) vs this=0.6225 (var=0.0523)
- Experiment 7: real=0.5004 (var=0.0029) vs this=0.5277 (var=0.0149)
- Experiment 8: real=0.4925 (var=0.0050) vs this=0.5625 (var=0.0215)
- Experiment 9: real=-0.0745 (var=0.4396) vs this=-0.0897 (var=1.7146)
- Experiment 10: real=0.5130 (var=0.0042) vs this=0.5028 (var=0.0187)


---

### `pi_2` (overall score: 0.539)

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
- Experiment 1: real=0.5025 (var=0.0026) vs this=0.3358 (var=0.0293)
- Experiment 2: real=0.4996 (var=0.0028) vs this=0.7156 (var=0.0163)
- Experiment 3: real=0.4947 (var=0.0048) vs this=0.2377 (var=0.0211)
- Experiment 4: real=-0.0111 (var=0.0415) vs this=0.0344 (var=0.0342)
- Experiment 5: real=0.4996 (var=0.0028) vs this=0.5387 (var=0.0218)
- Experiment 6: real=0.5350 (var=0.0082) vs this=0.5700 (var=0.0928)
- Experiment 7: real=0.5004 (var=0.0029) vs this=0.4117 (var=0.0301)
- Experiment 8: real=0.4925 (var=0.0050) vs this=0.7306 (var=0.0427)
- Experiment 9: real=-0.0745 (var=0.4396) vs this=1.5937 (var=1.8780)
- Experiment 10: real=0.5130 (var=0.0042) vs this=0.7372 (var=0.0223)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    diff = A - B
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(A.shape[1]):
            if diff[i, j] == 1:
                ttb_preds[i] = 0
                break
            elif diff[i, j] == -1:
                ttb_preds[i] = 1
                break
                
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5025 (var=0.0026)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8521 (var=0.0072)
- pi_2: 0.3358 (var=0.0293)
- pi_3: 0.4948 (var=0.0127)
- pi_4: 0.4506 (var=0.0030)
- pi_5: 0.5258 (var=0.0030)
- pi_6: 0.4808 (var=0.0028)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7156 (var=0.0163)
- pi_1: 0.1435 (var=0.0097)
- pi_3: 0.5142 (var=0.0127)
- pi_4: 0.5415 (var=0.0036)
- pi_5: 0.4927 (var=0.0023)
- pi_6: 0.5098 (var=0.0027)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_chosen = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_f0 = a[0]
        b_f0 = b[0]
        
        # Focus on conflict trials where the option with the best cue (f0) 
        # has very few other positive cues, while the other option has many.
        if a_f0 == 1 and b_f0 == 0:
            if sum(a) <= 2 and sum(b) >= 4:
                ttb_chosen.append(1 if resp == 0 else 0)
        elif b_f0 == 1 and a_f0 == 0:
            if sum(b) <= 2 and sum(a) >= 4:
                ttb_chosen.append(1 if resp == 1 else 0)
                
    if not ttb_chosen:
        return 0.5
    return float(np.mean(ttb_chosen))
```

**Observed (real) value:** 0.4947 (var=0.0048)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4603 (var=0.0125)
- pi_2: 0.2377 (var=0.0211)
- pi_1: 0.8807 (var=0.0103)
- pi_4: 0.4503 (var=0.0051)
- pi_5: 0.5133 (var=0.0036)
- pi_6: 0.4747 (var=0.0056)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Safely convert list of ratings to string for easy matching
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Agreement trials: WADD and TTB both strongly favor the '11111' option
    t5_ab = data[(data['a_str'] == '11111') & (data['b_str'] == '00000')]
    t5_ba = data[(data['a_str'] == '00000') & (data['b_str'] == '11111')]
    
    agree_wadd = 0
    agree_total = 0
    if len(t5_ab) > 0:
        agree_wadd += (t5_ab['response'] == 0).sum()
        agree_total += len(t5_ab)
    if len(t5_ba) > 0:
        agree_wadd += (t5_ba['response'] == 1).sum()
        agree_total += len(t5_ba)
    p_agree = agree_wadd / agree_total if agree_total > 0 else 0.5
    
    # Conflict trials: WADD strongly favors '01111' but TTB favors '10000'
    t1_ab = data[(data['a_str'] == '01111') & (data['b_str'] == '10000')]
    t1_ba = data[(data['a_str'] == '10000') & (data['b_str'] == '01111')]
    
    conflict_wadd = 0
    conflict_total = 0
    if len(t1_ab) > 0:
        conflict_wadd += (t1_ab['response'] == 0).sum()
        conflict_total += len(t1_ab)
    if len(t1_ba) > 0:
        conflict_wadd += (t1_ba['response'] == 1).sum()
        conflict_total += len(t1_ba)
        
    p_conflict = conflict_wadd / conflict_total if conflict_total > 0 else 0.5
    
    return float(p_agree - p_conflict)
```

**Observed (real) value:** -0.0111 (var=0.0415)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0344 (var=0.0342)
- pi_3: 0.2022 (var=0.0418)
- pi_1: 0.6822 (var=0.0612)
- pi_4: -0.0378 (var=0.0365)
- pi_5: 0.0589 (var=0.0404)
- pi_6: 0.0544 (var=0.0356)

### Experiment 5
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract the first cue (highest validity) for options A and B
    a_cue1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Determine if the chosen option had a 1 on the most valid cue
    # response == 0 means A was chosen, response == 1 means B was chosen
    chosen_cue1 = np.where(data['response'] == 0, a_cue1, b_cue1)
    
    # Return the proportion of trials where the choice aligned with the most valid cue
    return float(np.mean(chosen_cue1))
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6104 (var=0.0132)
- pi_4: 0.4892 (var=0.0028)
- pi_1: 0.8494 (var=0.0122)
- pi_2: 0.5387 (var=0.0218)
- pi_5: 0.5306 (var=0.0022)
- pi_6: 0.5069 (var=0.0027)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_trial_1(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1)
        
    mask = data.apply(is_trial_1, axis=1)
    if not mask.any():
        return 0.5
        
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.5350 (var=0.0082)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4662 (var=0.0191)
- pi_3: 0.6225 (var=0.0523)
- pi_1: 0.8213 (var=0.0191)
- pi_2: 0.5700 (var=0.0928)
- pi_5: 0.5238 (var=0.0164)
- pi_6: 0.5025 (var=0.0192)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Cue 0 is the highest validity cue (0.95)
    a_cue0 = a_ratings[:, 0]
    b_cue0 = b_ratings[:, 0]
    
    # Determine which option has the highest validity cue
    expected = np.where(a_cue0 > b_cue0, 0, np.where(b_cue0 > a_cue0, 1, -1))
    
    valid = expected != -1
    if not np.any(valid):
        return 0.5
        
    # Calculate the proportion of choices aligning with the highest validity cue
    match = (data['response'].values[valid] == expected[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5004 (var=0.0029)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5175 (var=0.0019)
- pi_4: 0.4752 (var=0.0030)
- pi_1: 0.8646 (var=0.0072)
- pi_2: 0.4117 (var=0.0301)
- pi_3: 0.5277 (var=0.0149)
- pi_6: 0.4838 (var=0.0024)

### Experiment 8
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    best_a = data['option_a_ratings'].apply(lambda x: x[0])
    best_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    tally_favors_a = tally_a > tally_b
    tally_favors_b = tally_a < tally_b
    best_favors_a = best_a > best_b
    best_favors_b = best_a < best_b
    
    conflict = (tally_favors_a & best_favors_b) | (tally_favors_b & best_favors_a)
    
    if conflict.sum() == 0:
        return 0.5
        
    aligned_tally = (tally_favors_a & (data['response'] == 0)) | (tally_favors_b & (data['response'] == 1))
    
    return float(aligned_tally[conflict].mean())
```

**Observed (real) value:** 0.4925 (var=0.0050)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5306 (var=0.0065)
- pi_5: 0.4637 (var=0.0069)
- pi_1: 0.1700 (var=0.0143)
- pi_2: 0.7306 (var=0.0427)
- pi_3: 0.5625 (var=0.0215)
- pi_6: 0.5356 (var=0.0080)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def trial_score(row):
        A = tuple(row['option_a_ratings'])
        B = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Weights proportional to the difference in predictions (WADD - Single-Cue)
        pairs = {
            ((1, 0, 0, 0, 0), (0, 1, 1, 1, 1)): 14,
            ((1, 1, 1, 0, 0), (1, 0, 0, 1, 1)): -1,
            ((0, 1, 0, 0, 0), (0, 0, 1, 1, 1)): 5,
            ((1, 1, 0, 0, 0), (0, 0, 1, 0, 0)): 3,
            ((1, 0, 0, 1, 0), (0, 1, 1, 0, 0)): 6
        }
        
        if (A, B) in pairs:
            w = pairs[(A, B)]
            return w if resp == 1 else -w
        elif (B, A) in pairs:
            w = pairs[(B, A)]
            return -w if resp == 1 else w
        else:
            return 0.0

    scores = data.apply(trial_score, axis=1)
    subject_means = scores.groupby(data['subject_id']).mean()
    return float(subject_means.mean())
```

**Observed (real) value:** -0.0745 (var=0.4396)
**Other theories' values on this metric (for reference):**
- pi_5: -0.4164 (var=0.5920)
- pi_6: 0.1444 (var=0.4164)
- pi_1: -3.6888 (var=1.1385)
- pi_2: 1.5937 (var=1.8780)
- pi_3: -0.0897 (var=1.7146)
- pi_4: 0.3078 (var=0.5855)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract the first cue (highest validity) for both options
    a_cue0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Filter for conflict trials where the options differ on the highest-validity cue
    conflict_mask = a_cue0 != b_cue0
    conflict_data = data[conflict_mask]
    
    if len(conflict_data) == 0:
        return 0.5
        
    # The WADD model predicts subjects will choose the option that LACKS the highest validity cue
    # because the sum of the secondary cues outweighs it on these specific conflict trials.
    # If B lacks the first cue (b_cue0 == 0), WADD predicts 1 (Option B).
    # If A lacks the first cue (a_cue0 == 0), WADD predicts 0 (Option A).
    target_responses = (conflict_data['option_b_ratings'].apply(lambda x: x[0]) == 0).astype(int)
    
    # Calculate the proportion of choices that align with the WADD prediction
    return float((conflict_data['response'] == target_responses).mean())
```

**Observed (real) value:** 0.5130 (var=0.0042)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5249 (var=0.0035)
- pi_5: 0.4789 (var=0.0036)
- pi_1: 0.1389 (var=0.0111)
- pi_2: 0.7372 (var=0.0223)
- pi_3: 0.5028 (var=0.0187)
- pi_4: 0.5326 (var=0.0061)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a new model instance for the theory above from scratch. Keep the theory's prose claim implicit but unchanged — your only job is to regenerate `predict`, `policy`, and the `parameters` ranges so that the implementation actually displays the behavior the theory claims, while faithfully implementing the mechanism family the arbiter prescribed.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this model design; used to explain why this model is better then the previous ones.


## Response

```json
```json
{
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Identify the single highest-validity cue\n    best_cue = np.argmax(validities)\n    \n    # Evaluate options based only on this single cue\n    scores = stim[:, best_cue]\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax over the single-cue scores\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Overwhelmingly high uniform lapse blended in\n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.95, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, the lapse rate (epsilon) parameter range has been shifted to [0.95, 1.0]. This extremely high noise level strongly masks the underlying single-cue heuristic, ensuring that the model's predictions align with the near-random behavior observed across the experiments, bringing metric evaluations closer to 0.5 (or 0 for differences)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14014,
  "candidates_token_count": 459,
  "total_token_count": 15620
}
```
