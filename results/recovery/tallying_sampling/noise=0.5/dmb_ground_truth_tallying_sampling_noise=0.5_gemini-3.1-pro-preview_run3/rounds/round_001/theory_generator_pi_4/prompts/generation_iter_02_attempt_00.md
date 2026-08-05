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

Propose a 'Mixture of Heuristics' or 'Strategy Selection' theory. This new theory should posit that the population consists of a mixture of decision-makers: a proportion of the trials/subjects rely on a simple Tallying (equal-weight) heuristic, while the rest use a validity-sensitive strategy like WADD or Take-The-Best. Alternatively, propose a 'Rank-Weighted Additive' theory where features are weighted by their ordinal rank rather than their cardinal validities, naturally interpolating between Tallying and WADD.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.797)

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
- Experiment 1: real=0.3172 (var=0.0051) vs this=0.3389 (var=0.0293)
- Experiment 2: real=0.3383 (var=0.0035) vs this=0.3296 (var=0.0225)
- Experiment 3: real=0.6700 (var=0.0082) vs this=0.3683 (var=0.0688)
- Experiment 4: real=0.3233 (var=0.0099) vs this=0.5450 (var=0.0989)


---

### `pi_3` (overall score: 0.774)

**Description**
Tallying (Equal Weight) posits that decision-makers simply count the number of positive features for each option and choose the one with the higher count, completely ignoring the differing validities of the cues. It is a compensatory heuristic that treats all cues as equally important. When tallies are tied, it guesses uniformly. Choice probabilies are generated by applying a softmax function to the tallies, mixed with a uniform lapse rate to account for response errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Calculate the unweighted sum of positive features for each option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3172 (var=0.0051) vs this=0.1508 (var=0.0087)
- Experiment 2: real=0.3383 (var=0.0035) vs this=0.1598 (var=0.0064)
- Experiment 3: real=0.6700 (var=0.0082) vs this=0.8433 (var=0.0144)
- Experiment 4: real=0.3233 (var=0.0099) vs this=0.1500 (var=0.0126)


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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3172 (var=0.0051) vs this=0.8558 (var=0.0110)
- Experiment 2: real=0.3383 (var=0.0035) vs this=0.8242 (var=0.0146)
- Experiment 3: real=0.6700 (var=0.0082) vs this=0.1608 (var=0.0140)
- Experiment 4: real=0.3233 (var=0.0099) vs this=0.8558 (var=0.0136)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2201 -> ACCEPTED
- iter 2: loss=0.5739 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2201 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum != b_sum
    if not mask.any():
        return 0.5
    filtered_data = data[mask]
    a_sum_f = a_sum[mask]
    b_sum_f = b_sum[mask]
    chose_fewer = ((filtered_data['response'] == 0) & (a_sum_f < b_sum_f)) | ((filtered_data['response'] == 1) & (b_sum_f < a_sum_f))
    return float(chose_fewer.mean())
```

**Observed (real) value:** 0.3172 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: 0.4325 (var=0.0302) (Δ vs real +0.1153)
  - iter 2 (most recent): 0.6639 (var=0.0562) (Δ vs real +0.3467)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8558 (var=0.0110)
- pi_2: 0.3389 (var=0.0293)
- pi_3: 0.1508 (var=0.0087)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    diff = a - b
    # Find the index of the first discriminating cue (highest validity first)
    idx = (diff != 0).argmax(axis=1)
    # TTB chooses option A (0) if A > B on this cue, else option B (1)
    ttb_winner = np.where(diff[np.arange(len(diff)), idx] > 0, 0, 1)
    return float((data['response'].values == ttb_winner).mean())
```

**Observed (real) value:** 0.3383 (var=0.0035)
**Previous candidate values (this loop):**
  - iter 1: 0.4910 (var=0.0427) (Δ vs real +0.1527)
  - iter 2 (most recent): 0.6325 (var=0.0257) (Δ vs real +0.2942)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3296 (var=0.0225)
- pi_1: 0.8242 (var=0.0146)
- pi_3: 0.1598 (var=0.0064)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    
    conflict_mask = ((sum_a > sum_b) & (wadd_a < wadd_b)) | ((sum_a < sum_b) & (wadd_a > wadd_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    sum_a_conf = sum_a[conflict_mask]
    sum_b_conf = sum_b[conflict_mask]
    responses = data['response'].values[conflict_mask]
    
    tallying_predictions = (sum_a_conf < sum_b_conf).astype(int)
    
    return float(np.mean(responses == tallying_predictions))
```

**Observed (real) value:** 0.6700 (var=0.0082)
**Previous candidate values (this loop):**
  - iter 1: 0.4875 (var=0.0533) (Δ vs real -0.1825)
  - iter 2 (most recent): 0.1833 (var=0.0310) (Δ vs real -0.4867)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8433 (var=0.0144)
- pi_2: 0.3683 (var=0.0688)
- pi_1: 0.1608 (var=0.0140)

### Experiment 4
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to strings for easy comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify trials where one option has fewer but more valid features (11000)
    # and the other has more but less valid features (00111)
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    total_critical_trials = mask1.sum() + mask2.sum()
    if total_critical_trials == 0:
        return 0.5
        
    # Count how often the subject chose the WADD-favored option (11000)
    wadd_choices = (data.loc[mask1, 'response'] == 0).sum() + (data.loc[mask2, 'response'] == 1).sum()
    
    return float(wadd_choices / total_critical_trials)
```

**Observed (real) value:** 0.3233 (var=0.0099)
**Previous candidate values (this loop):**
  - iter 1: 0.4783 (var=0.0562) (Δ vs real +0.1550)
  - iter 2 (most recent): 0.7808 (var=0.0625) (Δ vs real +0.4575)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5450 (var=0.0989)
- pi_3: 0.1500 (var=0.0126)
- pi_1: 0.8558 (var=0.0136)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Mixture of Heuristics (Tallying and Take-The-Best)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    a, b = stim[0], stim[1]
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
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
- w_ttb: [0.0, 1.0]
- validities: validities

`rationale`: Following the arbiter's suggestion, this theory proposes a Mixture of Heuristics. Looking closely at the empirical data across all four experiments, the real metric values systematically fall at an approximate 3:1 interpolation between pure Tallying and pure Take-The-Best (TTB). For instance, in Experiment 3, Tallying predicts ~0.84 and TTB predicts ~0.16, while the real value is exactly 0.67 (which is 0.75 * 0.84 + 0.25 * 0.16). Similar alignments hold remarkably well for Experiments 1, 2, and 4. Thus, we model choice probabilities as a convex combination of Tallying and TTB governed by a mixing parameter `w_ttb`, representing the proportion of trials or subjects employing a validity-sensitive stopping rule versus an exhaustive equal-weighting rule.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model implements a probability-level mixture of Tallying and Take-The-Best. While this was accepted and improves over the base, the simulated point estimates (0.43, 0.49, 0.48, 0.47) are hovering near 0.5 and fail to hit the observed empirical values (0.31, 0.33, 0.67, 0.32). The model is essentially predicting random choices on average, likely because a probability-level mixture with a broad `w_ttb` range [0.0, 1.0] and independent `beta` scaling for two very different heuristics creates a flat or poorly-behaved loss landscape during subject-level fitting.
Rationale: To achieve tighter fits within this mechanism family, switch from a probability-level mixture to a unified 'Rank-Weighted Additive' rule. Instead of explicitly calculating TTB and Tallying probabilities and mixing them, assign a weight to each cue based on its validity rank. For example, `weight_i = 1 / (rank_i ** gamma)`, where `gamma` is a free parameter (e.g., [0.0, 5.0]). When `gamma=0`, this perfectly matches Tallying (equal weights); as `gamma` increases, it naturally approximates TTB. Compute a single score for each option using these rank-based weights, and pass the difference into a standard softmax. This provides a much smoother interpolation between the two heuristics and eliminates the need to balance two separate probability distributions.

**Outcome of this advice:** iter 1 candidate loss=0.2201 -> iter 2 candidate loss=0.5739 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The Rank-Weighted Additive model (iter 2) was rejected by the gate because it pushed predictions in the wrong direction. The empirical data across all four experiments shows a strong preference (~67-68%) for the Tallying strategy (choosing the option with more positive cues) over validity-based strategies like WADD or TTB (which are chosen ~32-33% of the time). The iter 2 model overshot in favor of the validity-based options. The accepted base from iter 1 (a probability mixture) hovered near 0.5, likely due to a high random-guessing parameter washing out the signal.
Rationale: Since the Rank-Weighted Additive approach was rejected, we will build upon the accepted iter 1 base (the probability-level mixture). To fix the issue of predictions hovering near 0.5 in that base, the proposer should implement a probability-level mixture of Tallying and WADD (or TTB), but strictly limit the random-guessing parameter `epsilon` to [0.0, 0.1]. Use a single shared `beta` parameter [0.1, 10.0] to compute the softmax probabilities for both the Tallying score and the WADD score, and then mix these two probability distributions using a `w_tally` parameter [0.0, 1.0]. This will allow the model to confidently capture the ~67% Tallying behavior observed in the data without decaying into random noise.

**Outcome of this advice:** iter 2 candidate loss=0.5739 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Mixture of Heuristics (Tallying and WADD)",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    w_tally = float(parameters[\"w_tally\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # --- WADD ---\n    scores_wadd = stim @ val\n    z_wadd = beta * (scores_wadd - scores_wadd.max())\n    e_wadd = np.exp(z_wadd)\n    p_wadd = e_wadd / e_wadd.sum()\n        \n    # --- Tallying (Equal Weight) ---\n    scores_tally = stim.sum(axis=1)\n    z_tally = beta * (scores_tally - scores_tally.max())\n    e_tally = np.exp(z_tally)\n    p_tally = e_tally / e_tally.sum()\n    \n    # --- Mixture ---\n    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.1]",
    "w_tally": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we revert to the successful probability-mixture architecture from Iteration 1 but replace TTB with WADD (using objective validities as weights) to capture validity-sensitive compensatory behavior. To address the issue of predictions hovering near 0.5, we restrict the uniform lapse parameter `epsilon` to [0.0, 0.1] and `beta` to [0.1, 10.0]. We also explicitly mix Tallying and WADD probabilities via a `w_tally` parameter. This allows the model to capture the strong (~67%) preference for Tallying over validity-based strategies observed empirically without decaying into random noise."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10585,
  "candidates_token_count": 674,
  "total_token_count": 12473
}
```
