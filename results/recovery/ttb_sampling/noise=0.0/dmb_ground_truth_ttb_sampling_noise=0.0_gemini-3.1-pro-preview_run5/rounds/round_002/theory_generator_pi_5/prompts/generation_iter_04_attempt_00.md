# generation_iter_04_attempt_00

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

Propose a 'Strategy Selection / Mixture Model' theory. Instead of a single deterministic rule with exponential weights, posit that decision-makers maintain a repertoire of heuristics (e.g., Take The Best and Equal-Weight Tallying). On any given trial, a subject samples a strategy probabilistically, perhaps biased by the dispersion of validities or simply as a fixed individual difference (a mixture parameter). This would allow the model to capture the exact lexicographic tie-breaking of TTB on the majority of trials (explaining Exp 1) while correctly predicting a baseline rate of compensatory choices (explaining Exp 2) without relying on arbitrary exponential rank weights.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.973)

**Description**
Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7537 (var=0.0444) vs this=0.7379 (var=0.0518)
- Experiment 2: real=0.9117 (var=0.0062) vs this=0.8692 (var=0.0105)
- Experiment 3: real=0.8492 (var=0.0120) vs this=0.8883 (var=0.0044)
- Experiment 4: real=0.8279 (var=0.0105) vs this=0.8777 (var=0.0063)
- Experiment 5: real=0.8489 (var=0.0084) vs this=0.8806 (var=0.0065)
- Experiment 6: real=0.1693 (var=0.0000) vs this=0.1113 (var=0.0000)


---

### `pi_4` (overall score: 0.580)

**Description**
Rank-Weighted Tallying: To accommodate the arbiter's request for a counting-like heuristic while respecting the strict programmatic accept gate (which rejects models that fail to fit the data), we propose a Rank-Weighted Tallying model. Instead of ignoring validities entirely, subjects rank features by validity and weight their tallies exponentially by this rank. When alpha=1, this perfectly reduces to the arbiter's requested Tallying (Equal-Weight). When alpha > 1, it captures the non-compensatory lexicographic behavior (Take The Best) required to fit Experiments 3 and 4.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Weighted Tallying expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Dense rank of validities (lowest validity gets rank 0)
    _, ranks = np.unique(validities, return_inverse=True)
    
    alpha = float(parameters["alpha"])
    weights = alpha ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- alpha: [1.0, 20.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7537 (var=0.0444) vs this=0.6553 (var=0.0675)
- Experiment 2: real=0.9117 (var=0.0062) vs this=0.8500 (var=0.0140)
- Experiment 3: real=0.8492 (var=0.0120) vs this=0.8708 (var=0.0079)
- Experiment 4: real=0.8279 (var=0.0105) vs this=0.8413 (var=0.0317)
- Experiment 5: real=0.8489 (var=0.0084) vs this=0.1378 (var=0.0072)
- Experiment 6: real=0.1693 (var=0.0000) vs this=0.1216 (var=0.0000)


---

### `pi_2` (overall score: 0.278)

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
- Experiment 1: real=0.7537 (var=0.0444) vs this=0.3889 (var=0.4719)
- Experiment 2: real=0.9117 (var=0.0062) vs this=0.4558 (var=0.0898)
- Experiment 3: real=0.8492 (var=0.0120) vs this=0.3517 (var=0.0227)
- Experiment 4: real=0.8279 (var=0.0105) vs this=0.2387 (var=0.0149)
- Experiment 5: real=0.8489 (var=0.0084) vs this=0.3042 (var=0.0295)
- Experiment 6: real=0.1693 (var=0.0000) vs this=0.7671 (var=0.0000)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4231 -> ACCEPTED
- iter 2: loss=0.3542 -> ACCEPTED
- iter 3: loss=0.2745 -> ACCEPTED
- iter 4: loss=0.3730 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.2745 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    X = A - B
    y = np.where(data['response'] == 0, 1.0, -1.0)
    
    # Fit OLS to estimate the subjective weight of each feature
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # We apply a carefully constructed contrast vector k to the estimated weights.
    # The design matrix X has rank 4, with a null space spanned by [1, 0, 0, 1, 0].
    # For Tallying, all features are treated equally (true weights = [1, 1, 1, 1, 1]).
    # The projection of Tallying's weights onto the row space of X is [0, 1, 1, 0, 1].
    # Our contrast vector k is orthogonal to the null space AND satisfies k @ [0, 1, 1, 0, 1] = 0.
    # This mathematically guarantees that Tallying evaluates to exactly 0 in expectation.
    # For WADD, features are weighted by validities. The expected projected weights are 
    # [0.1, 0.45, 0.3, -0.1, 0.25]. 
    # k @ expected_WADD = 0.88(0.1) + 1(0.45) - 0.88(-0.1) - 1(0.25) = 0.376 > 0.
    k = np.array([0.88, 1.0, 0.0, -0.88, -1.0])
    
    return float(np.dot(k, beta_hat))

```

**Observed (real) value:** 0.7537 (var=0.0444)
**Previous candidate values (this loop):**
  - iter 1: -0.0061 (var=0.2999) (Δ vs real -0.7599)
  - iter 2: 1.0467 (var=0.1523) (Δ vs real +0.2930)
  - iter 3: 0.4137 (var=0.1159) (Δ vs real -0.3400)
  - iter 4 (most recent): 0.2879 (var=0.1251) (Δ vs real -0.4658)
**Other theories' values on this metric (for reference):**
- pi_1: -0.6677 (var=0.1254)
- pi_2: 0.3889 (var=0.4719)
- pi_3: 0.7379 (var=0.0518)
- pi_4: 0.6553 (var=0.0675)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (0, 0, 1, 1, 1)
    
    is_t1 = (a_tuples == t1) & (b_tuples == t2)
    is_t2 = (a_tuples == t2) & (b_tuples == t1)
    
    wadd_consistent = 0
    wadd_consistent += (data.loc[is_t1, 'response'] == 0).sum()
    wadd_consistent += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    
    if total == 0:
        return 0.5
    return float(wadd_consistent / total)
```

**Observed (real) value:** 0.9117 (var=0.0062)
**Previous candidate values (this loop):**
  - iter 1: 0.5850 (var=0.0564) (Δ vs real -0.3267)
  - iter 2: 0.8425 (var=0.0096) (Δ vs real -0.0692)
  - iter 3: 0.6917 (var=0.0187) (Δ vs real -0.2200)
  - iter 4 (most recent): 0.6308 (var=0.0163) (Δ vs real -0.2808)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4558 (var=0.0898)
- pi_1: 0.1342 (var=0.0129)
- pi_3: 0.8692 (var=0.0105)
- pi_4: 0.8500 (var=0.0140)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    ttb_match = 0
    total = 0
    
    for a, b, resp in zip(a_tuples, b_tuples, data['response']):
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            ttb_match += (resp == 0)
            total += 1
        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 0):
            ttb_match += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)
```

**Observed (real) value:** 0.8492 (var=0.0120)
**Previous candidate values (this loop):**
  - iter 1: 0.5483 (var=0.0407) (Δ vs real -0.3008)
  - iter 2: 0.4883 (var=0.0637) (Δ vs real -0.3608)
  - iter 3: 0.6254 (var=0.0179) (Δ vs real -0.2238)
  - iter 4 (most recent): 0.5521 (var=0.0211) (Δ vs real -0.2971)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8883 (var=0.0044)
- pi_2: 0.3517 (var=0.0227)
- pi_1: 0.1600 (var=0.0119)
- pi_4: 0.8708 (var=0.0079)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are sorted by validity in the design (index 0 is highest)
        for i in range(len(a)):
            if a[i] > b[i]:
                if resp == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if resp == 1:
                    matches += 1
                break
                
    return float(matches) / len(data)
```

**Observed (real) value:** 0.8279 (var=0.0105)
**Previous candidate values (this loop):**
  - iter 1: 0.5650 (var=0.0414) (Δ vs real -0.2629)
  - iter 2: 0.4988 (var=0.0502) (Δ vs real -0.3292)
  - iter 3: 0.5933 (var=0.0275) (Δ vs real -0.2346)
  - iter 4 (most recent): 0.5773 (var=0.0168) (Δ vs real -0.2506)
**Other theories' values on this metric (for reference):**
- pi_2: 0.2387 (var=0.0149)
- pi_3: 0.8777 (var=0.0063)
- pi_1: 0.1702 (var=0.0126)
- pi_4: 0.8413 (var=0.0317)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_0 = data['option_b_ratings'].apply(lambda x: x[0])
    mask = a_0 != b_0
    if not mask.any():
        return 0.5
    
    chose_feat0 = ((data['response'] == 0) & (a_0 == 1)) | ((data['response'] == 1) & (b_0 == 1))
    return float(chose_feat0[mask].mean())
```

**Observed (real) value:** 0.8489 (var=0.0084)
**Previous candidate values (this loop):**
  - iter 1: 0.5397 (var=0.0391) (Δ vs real -0.3092)
  - iter 2: 0.5683 (var=0.0564) (Δ vs real -0.2806)
  - iter 3: 0.6653 (var=0.0278) (Δ vs real -0.1836)
  - iter 4 (most recent): 0.5281 (var=0.0172) (Δ vs real -0.3208)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8806 (var=0.0065)
- pi_4: 0.1378 (var=0.0072)
- pi_1: 0.2475 (var=0.0054)
- pi_2: 0.3042 (var=0.0295)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    non_ttb_choices = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                break
        if row['response'] != ttb_pred:
            non_ttb_choices += 1
            
    rate = non_ttb_choices / len(data) if len(data) > 0 else 0.0
    
    # Variance stabilization for single-subject slices
    n_subj = data['subject_id'].nunique()
    if n_subj == 1:
        return rate / 10000.0
        
    return rate
```

**Observed (real) value:** 0.1693 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.4587 (var=0.0000) (Δ vs real +0.2893)
  - iter 2: 0.5676 (var=0.0000) (Δ vs real +0.3982)
  - iter 3: 0.3498 (var=0.0000) (Δ vs real +0.1804)
  - iter 4 (most recent): 0.4376 (var=0.0000) (Δ vs real +0.2682)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1216 (var=0.0000)
- pi_3: 0.1113 (var=0.0000)
- pi_1: 0.8669 (var=0.0000)
- pi_2: 0.7671 (var=0.0000)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Strategy Selection / Mixture Model: Decision-makers maintain a repertoire of distinct heuristics. On any given trial, a subject probabilistically samples a strategy from this repertoire—mixing a non-compensatory lexicographic rule (Take The Best) and a compensatory rule (Weighted Additive). The WADD rule incorporates a softmax decision process to gracefully handle near-ties and deviations, allowing the model to capture the exact tie-breaking behavior of TTB on the majority of trials while correctly predicting a baseline rate of validity-sensitive compensatory choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(-validities, kind='stable')
    score_a_ttb = 0.5
    score_b_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            score_a_ttb = 1.0
            score_b_ttb = 0.0
            break
        elif b[idx] > a[idx]:
            score_a_ttb = 0.0
            score_b_ttb = 1.0
            break
    p_ttb = np.array([score_a_ttb, score_b_ttb])
    
    # Strategy 2: Weighted Additive (WADD) with softmax
    score_a_wadd = np.sum(a * validities)
    score_b_wadd = np.sum(b * validities)
    tau = float(parameters["tau"])
    z = np.array([score_a_wadd, score_b_wadd]) / tau
    z -= np.max(z)
    e = np.exp(z)
    p_wadd = e / np.sum(e)
        
    # Mixture
    p_mix = float(parameters["p_ttb"])
    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_wadd
    
    # Response noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- tau: [0.01, 5.0]
- validities: validities

`rationale`: Following the critic's feedback, the deterministic WADD component was replaced with a softmax function over the WADD scores, parameterized by a temperature `tau`. This prevents the model from generating extreme 0 or 1 probabilities when the WADD rule is evaluated, allowing the log-likelihood optimization to fit a higher mixture probability `p_ttb` without being penalized heavily for occasional deviations. This should better capture the high TTB baseline in Experiments 3-6 while maintaining the WADD behavior in Experiments 1-2.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The proposed Strategy Selection / Mixture Model successfully implemented the arbiter's recommendation of probabilistically mixing TTB with a compensatory rule. However, it fails dramatically across all experiments, particularly Experiments 1 and 2. Experiment 1's metric specifically isolates whether compensatory choices use validities (WADD) or ignore them (Equal-Weight Tallying), with Tallying mathematically yielding an expected value of 0. The candidate's score of -0.0061 definitively shows that EQW is the wrong compensatory rule for this dataset, whereas human subjects score ~0.75 (strongly indicating WADD). Consequently, the model also fails to match human WADD-consistency in Experiment 2 and under-predicts TTB usage in Experiments 3-6 because the likelihood optimization struggles to balance the poorly-fitting EQW rule against the data.
Rationale: The mixture model framework is the correct approach, but the choice of Equal-Weight Tallying (EQW) as the compensatory heuristic directly conflicts with the empirical data. To fix this while staying within the prescribed family, replace EQW with a Weighted Additive (WADD) rule as the second strategy in the repertoire. For WADD, compute the dot product of the features and the validities for each option, and assign probability 1.0 to the option with the higher WADD score (or use a softmax over the WADD scores). Keep the TTB implementation and the probabilistic mixture (`p_mix`) exactly as they are.

**Outcome of this advice:** iter 1 candidate loss=0.4231 -> iter 2 candidate loss=0.3542 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The inclusion of the Weighted Additive (WADD) rule successfully moved the model in the right direction, significantly improving the fit on Experiments 1 and 2 (where compensatory choices now correctly reflect validity weights). However, the model now severely underpredicts TTB-consistent choices in Experiments 3, 4, 5, and 6. Because both the TTB and WADD rules currently output deterministic (1.0 or 0.0) probabilities, any choice by a subject that deviates from one of the rules forces the log-likelihood optimization to either heavily inflate the response noise (`epsilon`) or artificially deflate the mixture weight (`p_ttb`) to avoid zero-probability predictions. This rigidness prevents the model from accurately reflecting the high TTB baseline in the population.
Rationale: To fix the underprediction of TTB without abandoning the mixture framework, we need to 'soften' the WADD rule so that it doesn't force extreme parameter estimates during likelihood optimization. Replace the deterministic WADD step-function with a softmax over the WADD scores (e.g., `p_wadd_a = exp(wadd_a / tau) / (exp(wadd_a / tau) + exp(wadd_b / tau))`), introducing a temperature parameter `tau` (range [0.01, 5.0]). Keep the TTB rule and the mixture parameter `p_ttb` exactly as they are. This will allow the WADD component to gracefully handle near-ties and deviations, allowing `p_ttb` to properly capture the high rate of lexicographic choices.

**Outcome of this advice:** iter 2 candidate loss=0.3542 -> iter 3 candidate loss=0.2745 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The introduction of the softmax for the WADD rule successfully improved the fit (loss dropped from 0.3542 to 0.2745) by preventing extreme penalties for WADD-inconsistent choices. However, the model still systematically underpredicts both WADD consistency (Exps 1-2) and TTB consistency (Exps 3-6), with all predictions attenuated towards 0.5. This indicates that the global noise parameter (`epsilon`) is being inflated during likelihood optimization. The likely culprit is that the TTB rule is still completely deterministic (outputting exactly 1.0 or 0.0). When the mixture weight `p_ttb` is high, any human choice that violates TTB results in a near-zero probability, forcing the optimizer to artificially lower `p_ttb` and raise `epsilon` to compensate.
Rationale: To prevent the deterministic TTB rule from inflating the global noise, soften the TTB component just as we did for WADD. Instead of having TTB output hard 1.0 and 0.0 probabilities, introduce a TTB execution-error or confidence parameter (e.g., `ttb_conf` in range [0.5, 1.0]). When TTB favors option A, its output probability distribution should be `[ttb_conf, 1.0 - ttb_conf]` rather than `[1.0, 0.0]`. Keep the WADD softmax, the mixture `p_ttb`, and the global `epsilon`. This localized softening will allow the optimization to fit a high `p_ttb` for subjects who mostly use TTB, without being overly penalized by occasional deviations, thereby sharpening the predictions across all experiments.

**Outcome of this advice:** iter 3 candidate loss=0.2745 -> iter 4 candidate loss=0.3730 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate (iteration 4) attempted to soften the TTB rule by introducing a `ttb_conf` parameter. However, this candidate was REJECTED by the accept-gate because its aggregate loss worsened (from 0.2745 in iter 3 to 0.3730). Looking at the experiment-level metrics, the predictions became even more attenuated toward 0.5 across the board (e.g., Exp 1 dropped from 0.41 to 0.28; Exp 6 error rate increased from 0.34 to 0.43). Adding a confidence parameter to TTB on top of the global `epsilon` noise and the WADD softmax likely over-parameterized the stochasticity, flattening the model's predictions and preventing it from capturing the strong behavioral regularities.
Rationale: Since the `ttb_conf` approach was rejected, we will build on the running-best base (iteration 3), which correctly used a deterministic TTB and a softmax WADD. The issue in iteration 3 was that predictions were still somewhat attenuated. This is likely because the global response noise parameter (`epsilon` in [0.0, 0.5]) is absorbing too much variance and flattening the choice probabilities. Instead of softening TTB, remove the global `epsilon` parameter entirely (or restrict its range heavily, e.g., to [0.0, 0.05]). The probabilistic mixture (`p_ttb`) combined with the softmax on WADD (`tau`) already guarantees that no choice has exactly zero probability (as long as p_ttb < 1.0), so global noise is redundant and harmful. Keep TTB deterministic, keep the WADD softmax, and eliminate/restrict epsilon to sharpen the predictions.

**Outcome of this advice:** iter 4 candidate loss=0.3730 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Strategy Selection / Mixture Model: Decision-makers maintain a repertoire of distinct heuristics. On any given trial, a subject probabilistically samples a strategy from this repertoire—mixing a non-compensatory lexicographic rule (Take The Best) and a compensatory rule (Weighted Additive). The WADD rule incorporates a softmax decision process to gracefully handle near-ties and deviations. Global response noise is heavily restricted because the probabilistic mixture and the WADD softmax already provide sufficient stochasticity without artificially flattening predictions.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Mixture Model expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Strategy 1: Take The Best (TTB)\n    order = np.argsort(-validities, kind='stable')\n    score_a_ttb = 0.5\n    score_b_ttb = 0.5\n    for idx in order:\n        if a[idx] > b[idx]:\n            score_a_ttb = 1.0\n            score_b_ttb = 0.0\n            break\n        elif b[idx] > a[idx]:\n            score_a_ttb = 0.0\n            score_b_ttb = 1.0\n            break\n    p_ttb = np.array([score_a_ttb, score_b_ttb])\n    \n    # Strategy 2: Weighted Additive (WADD) with softmax\n    score_a_wadd = np.sum(a * validities)\n    score_b_wadd = np.sum(b * validities)\n    tau = float(parameters[\"tau\"])\n    z = np.array([score_a_wadd, score_b_wadd]) / tau\n    z -= np.max(z)\n    e = np.exp(z)\n    p_wadd = e / np.sum(e)\n        \n    # Mixture\n    p_mix = float(parameters[\"p_ttb\"])\n    p_core = p_mix * p_ttb + (1.0 - p_mix) * p_wadd\n    \n    # Response noise\n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "p_ttb": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.05]",
    "tau": "[0.01, 5.0]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, we revert to the successful iteration 3 base (deterministic TTB + softmax WADD) but heavily restrict the global noise parameter `epsilon` to `[0.0, 0.05]`. The previous attempt to soften TTB over-parameterized the stochasticity and flattened predictions. The mixture probability `p_ttb` combined with the WADD softmax `tau` inherently guarantees that no choice probability is exactly zero (unless p_ttb=1.0 and TTB is decisive). Restricting epsilon prevents the likelihood optimization from inflating global noise, thereby sharpening the model's predictions across all experiments."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13655,
  "candidates_token_count": 923,
  "total_token_count": 15445
}
```
