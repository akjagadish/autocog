# generation_iter_00_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Heuristic Decision Making domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
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
- n_features: Number of expert ratings per option (= len(validities)).
- validities: List of per-expert validities (length n_features), each in [0.5, 1.0], stored descending (highest first) and fixed for every trial of the experiment instance. Subjects are told these values up front. A theory should reference them as `validities: "validities"` whenever it needs the actual subjective weights subjects saw — re-sampling them per run from a uniform prior gives degenerate (often reversed-sign) behavior.
- rating_max: Upper bound (inclusive) of each rating value (>= 1). Ratings are integers in [0, rating_max].

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
  - `"option_a_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
  - `"option_b_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a 'Strategy Selection' or 'Hybrid' theory to replace Tallying. This new theory should posit that decision-makers rely on a mixture of heuristics depending on the cognitive load or individual differences (e.g., a mixture model where a parameter 'p_wadd' dictates the probability of using a compensatory WADD strategy versus a non-compensatory Tallying strategy on any given trial). Alternatively, propose a 'Threshold' model where subjects default to Tallying unless the cardinal difference on a high-validity feature exceeds a certain threshold, at which point they switch to a compensatory evaluation.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.393)

**Description**
Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.01, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3308 (var=0.0607) vs this=0.1652 (var=0.0103)
- Experiment 2: real=0.7270 (var=0.0429) vs this=0.8744 (var=0.0071)
- Experiment 3: real=0.6446 (var=0.0854) vs this=0.8840 (var=0.0047)
- Experiment 4: real=n/a (var=n/a) vs this=0.8067 (var=0.0051)


---

### `pi_2` (overall score: 0.147)

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
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3308 (var=0.0607) vs this=0.0000 (var=0.0000)
- Experiment 2: real=0.7270 (var=0.0429) vs this=1.0000 (var=0.0000)
- Experiment 3: real=0.6446 (var=0.0854) vs this=0.0000 (var=0.0000)
- Experiment 4: real=n/a (var=n/a) vs this=0.0000 (var=0.0000)


---

### `pi_1` (overall score: 0.000)

**Description**
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3308 (var=0.0607) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.7270 (var=0.0429) vs this=0.1250 (var=0.0000)
- Experiment 3: real=0.6446 (var=0.0854) vs this=0.6000 (var=0.0000)
- Experiment 4: real=n/a (var=n/a) vs this=0.5000 (var=0.0000)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[4, 1, 1, 1]  B=[3, 2, 2, 2]
  A=[2, 3, 3, 3]  B=[3, 1, 1, 1]
  A=[5, 5, 1, 1]  B=[5, 4, 2, 2]
  A=[3, 2, 4, 4]  B=[3, 3, 1, 1]
  A=[4, 2, 1, 1]  B=[3, 3, 2, 2]
  A=[1, 4, 4, 4]  B=[2, 2, 2, 2]
  A=[5, 4, 1, 1]  B=[5, 3, 2, 2]
  A=[2, 2, 4, 4]  B=[2, 3, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = None
        # Features are implicitly ordered by validity (highest to lowest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_match += 1
            total += 1
    return ttb_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.3308 (var=0.0607)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.1652 (var=0.0103)

### Experiment 2
**Design**
  A=[5, 2, 2, 2, 2]  B=[2, 5, 5, 5, 2]
  A=[2, 5, 5, 5, 2]  B=[5, 2, 2, 2, 2]
  A=[5, 5, 2, 2, 2]  B=[5, 2, 5, 5, 5]
  A=[4, 4, 4, 2, 2]  B=[4, 4, 2, 5, 5]
  A=[3, 2, 5, 5, 5]  B=[3, 5, 2, 2, 2]
  A=[4, 3, 3, 3, 3]  B=[3, 4, 4, 4, 4]
  A=[1, 5, 1, 1, 1]  B=[1, 1, 5, 5, 5]
  A=[5, 1, 5, 5, 5]  B=[1, 5, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def tally_predicts(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(x > y for x, y in zip(a, b))
        b_wins = sum(y > x for x, y in zip(a, b))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        return -1
    predictions = data.apply(tally_predicts, axis=1)
    return float(np.mean(data['response'] == predictions))
```

**Observed (real) value:** 0.7270 (var=0.0429)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.0000)
- pi_1: 0.1250 (var=0.0000)
- pi_3: 0.8744 (var=0.0071)

### Experiment 3
**Design**
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 10]
  A=[5, 5, 0, 0]  B=[4, 4, 10, 10]
  A=[0, 0, 10, 10]  B=[2, 2, 0, 0]
  A=[10, 10, 0, 0]  B=[9, 9, 1, 1]
  A=[1, 1, 1, 1]  B=[4, 0, 0, 0]
  A=[0, 2, 2, 2]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # trial 1: A=[10, 0, 0, 0]  B=[0, 1, 1, 1] -> WADD prefers A (0), Tallying prefers B
        if a[0] == 10 and a[1] == 0 and b[0] == 0:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 2: A=[0, 8, 0, 0]  B=[1, 0, 1, 1] -> WADD prefers A (0), Tallying prefers B
        elif a[0] == 0 and a[1] == 8 and b[0] == 1:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 3: A=[1, 1, 1, 0]  B=[0, 0, 0, 10] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 0 and b[3] == 10:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 7: A=[1, 1, 1, 1]  B=[4, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 1 and b[0] == 4:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 8: A=[0, 2, 2, 2]  B=[10, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 0 and a[1] == 2 and b[0] == 10:
            wadd_consistent += (resp == 1)
            total += 1
            
    return wadd_consistent / total if total > 0 else 0.5
```

**Observed (real) value:** 0.6446 (var=0.0854)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8840 (var=0.0047)
- pi_2: 0.0000 (var=0.0000)
- pi_1: 0.6000 (var=0.0000)

### Experiment 4
**Design**
  A=[5, 5, 5, 5, 5]  B=[10, 4, 4, 4, 4]
  A=[9, 9, 4, 4, 4]  B=[5, 5, 5, 5, 5]
  A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  A=[8, 8, 2, 2, 5]  B=[2, 2, 8, 8, 5]
  A=[3, 3, 3, 3, 3]  B=[2, 2, 2, 2, 10]
  A=[1, 9, 9, 9, 9]  B=[10, 8, 8, 8, 8]
  A=[7, 2, 7, 2, 7]  B=[6, 8, 6, 8, 6]
  A=[5, 6, 5, 6, 5]  B=[6, 5, 6, 5, 6]
  A=[4, 8, 4, 8, 4]  B=[5, 2, 5, 2, 5]
  A=[8, 8, 8, 0, 0]  B=[9, 9, 9, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    responses = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_diff = np.sum(a > b) - np.sum(b > a)
        wadd_diff = np.sum(a * v) - np.sum(b * v)
        
        if tally_diff > 0 and wadd_diff < 0:
            responses.append(row['response'])
        elif tally_diff < 0 and wadd_diff > 0:
            responses.append(1 - row['response'])
            
    if len(responses) == 0:
        return 0.5
    return float(np.mean(responses))
```

**Observed (real) value:** n/a (var=n/a)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.8067 (var=0.0051)
- pi_1: 0.5000 (var=0.0000)


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
  "theory": "Strategy Selection (WADD-Tallying Mixture): Decision-makers do not rigidly adhere to a single heuristic. Instead, they dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy based on individual differences and cognitive constraints. This is modeled as a probabilistic mixture where an individual uses WADD with probability `w_wadd` and Tallying with probability `1 - w_wadd`.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Expects a (2, n_features) stimulus.\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # WADD score computation\n    sum_val = np.sum(val)\n    if sum_val > 0:\n        score_a_wadd = np.sum(a * val) / sum_val\n        score_b_wadd = np.sum(b * val) / sum_val\n    else:\n        score_a_wadd = np.sum(a)\n        score_b_wadd = np.sum(b)\n    scores_wadd = np.array([score_a_wadd, score_b_wadd])\n    \n    # Tallying score computation\n    score_a_tally = float(np.sum(a > b))\n    score_b_tally = float(np.sum(b > a))\n    scores_tally = np.array([score_a_tally, score_b_tally])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    w_wadd = float(parameters[\"w_wadd\"])\n    \n    # Softmax for WADD\n    z_wadd = beta * (scores_wadd - scores_wadd.max())\n    e_wadd = np.exp(z_wadd)\n    p_wadd = e_wadd / e_wadd.sum()\n    \n    # Softmax for Tallying\n    z_tally = beta * (scores_tally - scores_tally.max())\n    e_tally = np.exp(z_tally)\n    p_tally = e_tally / e_tally.sum()\n    \n    # Mixture of strategies\n    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probs = np.asarray(probabilities, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "w_wadd": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this theory implements a Strategy Selection mechanism. It posits that people use a mixture of heuristics rather than a single fixed one. By blending a compensatory WADD strategy with a non-compensatory Tallying strategy through a mixture parameter `w_wadd`, the model can smoothly interpolate between magnitude-sensitive evaluations and magnitude-insensitive counting, capturing the observed variance in human responses across different experimental designs where one strategy might dominate."
}
```
```

## Usage

```json
{
  "prompt_token_count": 8318,
  "candidates_token_count": 946,
  "total_token_count": 11116
}
```
