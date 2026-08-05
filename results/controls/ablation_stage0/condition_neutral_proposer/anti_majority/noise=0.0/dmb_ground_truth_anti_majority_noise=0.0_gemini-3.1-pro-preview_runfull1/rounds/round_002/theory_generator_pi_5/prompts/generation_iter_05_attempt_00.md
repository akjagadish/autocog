# generation_iter_05_attempt_00

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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a fundamentally different theory of multi-attribute choice, such as a Configural Weighting model or a Heuristic-Switching model based on the variance of the cues. The new theory should account for why subjects might systematically prefer options with 'moderate' validities or even systematically violate validity-based dominance (as seen in Experiment 1). Consider mechanisms where the subjective weight of a cue is context-dependent (e.g., depends on the presence of other cues) or where subjects use a two-stage process: first comparing the total number of positive features (tallying), and only using validities to break ties or when the difference in counts is small.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.463)

**Description**
People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.1, 200.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 50.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7858 (var=0.0260) vs this=0.7021 (var=0.0525)
- Experiment 2: real=0.2306 (var=0.0195) vs this=0.3344 (var=0.0398)
- Experiment 3: real=0.6521 (var=0.0118) vs this=0.6856 (var=0.0224)
- Experiment 4: real=0.2233 (var=0.0237) vs this=0.2500 (var=0.0332)
- Experiment 5: real=0.1175 (var=0.0070) vs this=0.8662 (var=0.0111)
- Experiment 6: real=0.6593 (var=0.0108) vs this=0.2067 (var=0.0154)


---

### `pi_1` (overall score: 0.370)

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
- Experiment 1: real=0.7858 (var=0.0260) vs this=0.8546 (var=0.0146)
- Experiment 2: real=0.2306 (var=0.0195) vs this=0.1391 (var=0.0083)
- Experiment 3: real=0.6521 (var=0.0118) vs this=0.8827 (var=0.0073)
- Experiment 4: real=0.2233 (var=0.0237) vs this=0.1267 (var=0.0114)
- Experiment 5: real=0.1175 (var=0.0070) vs this=0.8500 (var=0.0070)
- Experiment 6: real=0.6593 (var=0.0108) vs this=0.1437 (var=0.0080)


---

### `pi_4` (overall score: 0.282)

**Description**
Evidence Accumulation with Bounded Threshold: People sample cues sequentially in order of validity. Each discriminating cue adds its validity to the evidence for the favored option. If the absolute difference in evidence between the two options reaches a threshold `theta`, sampling stops and a decision is made based on the accumulated evidence (mimicking Take The Best). If all cues are exhausted without the threshold being reached, the decision is based on the final integrated evidence (mimicking Weighted Additive). This allows for early stopping when initial evidence is strong, but continued sampling when evidence is weak or tied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        if a[j] > b[j]:
            ev_a += val[j]
        elif b[j] > a[j]:
            ev_b += val[j]
            
        # Early stopping if threshold is reached
        if abs(ev_a - ev_b) >= theta:
            break
            
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- theta: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7858 (var=0.0260) vs this=0.8446 (var=0.0633)
- Experiment 2: real=0.2306 (var=0.0195) vs this=0.1831 (var=0.0521)
- Experiment 3: real=0.6521 (var=0.0118) vs this=0.7771 (var=0.0930)
- Experiment 4: real=0.2233 (var=0.0237) vs this=0.1308 (var=0.0399)
- Experiment 5: real=0.1175 (var=0.0070) vs this=0.9531 (var=0.0057)
- Experiment 6: real=0.6593 (var=0.0108) vs this=0.0852 (var=0.0127)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4671 -> ACCEPTED
- iter 2: loss=0.7455 -> REJECTED
- iter 3: loss=0.7297 -> REJECTED
- iter 4: loss=0.7129 -> REJECTED
- iter 5: loss=0.7496 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4671 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd

    def get_predictions(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    preds = data.apply(get_predictions, axis=1)
    
    # Filter for trials where TTB and Tallying make strictly opposite predictions
    mask = (preds['ttb'] != -1) & (preds['tally'] != -1) & (preds['ttb'] != preds['tally'])
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of choices that align with Take The Best on these critical trials
    ttb_choices = (data.loc[mask, 'response'] == preds.loc[mask, 'ttb']).mean()
    return float(ttb_choices)

```

**Observed (real) value:** 0.7858 (var=0.0260)
**Previous candidate values (this loop):**
  - iter 1: 0.3442 (var=0.1088) (Δ vs real -0.4417)
  - iter 2: 0.0967 (var=0.0059) (Δ vs real -0.6892)
  - iter 3: 0.1450 (var=0.0127) (Δ vs real -0.6408)
  - iter 4: 0.1696 (var=0.0294) (Δ vs real -0.6163)
  - iter 5 (most recent): 0.0500 (var=0.0042) (Δ vs real -0.7358)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8546 (var=0.0146)
- pi_2: 0.1412 (var=0.0101)
- pi_3: 0.7021 (var=0.0525)
- pi_4: 0.8446 (var=0.0633)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie
            
        # TTB prediction (cues are ordered by descending validity based on the design)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is None:
            continue  # TTB predicts a tie
            
        # Only consider trials where the two theories make strictly opposite predictions
        if tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            count += 1
            
    if count == 0:
        return 0.5
        
    return float(tally_match / count)

```

**Observed (real) value:** 0.2306 (var=0.0195)
**Previous candidate values (this loop):**
  - iter 1: 0.6009 (var=0.1129) (Δ vs real +0.3703)
  - iter 2: 0.9269 (var=0.0068) (Δ vs real +0.6963)
  - iter 3: 0.8534 (var=0.0105) (Δ vs real +0.6228)
  - iter 4: 0.8847 (var=0.0089) (Δ vs real +0.6541)
  - iter 5 (most recent): 0.9197 (var=0.0116) (Δ vs real +0.6891)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8691 (var=0.0085)
- pi_1: 0.1391 (var=0.0083)
- pi_3: 0.3344 (var=0.0398)
- pi_4: 0.1831 (var=0.0521)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_mat - b_mat
    
    match = 0
    total = 0
    for i in range(len(diff)):
        row_diff = diff[i]
        nonzero_idx = np.nonzero(row_diff)[0]
        if len(nonzero_idx) > 0:
            first_idx = nonzero_idx[0]
            ttb_choice = 0 if row_diff[first_idx] > 0 else 1
            if resp[i] == ttb_choice:
                match += 1
            total += 1
            
    return float(match / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6521 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1: 0.4656 (var=0.0946) (Δ vs real -0.1865)
  - iter 2: 0.1804 (var=0.0042) (Δ vs real -0.4717)
  - iter 3: 0.2098 (var=0.0074) (Δ vs real -0.4423)
  - iter 4: 0.2048 (var=0.0066) (Δ vs real -0.4473)
  - iter 5 (most recent): 0.1654 (var=0.0040) (Δ vs real -0.4867)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8827 (var=0.0073)
- pi_3: 0.6856 (var=0.0224)
- pi_2: 0.2102 (var=0.0075)
- pi_4: 0.7771 (var=0.0930)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    mask = (sum_a == 1) & (sum_b >= 3)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2233 (var=0.0237)
**Previous candidate values (this loop):**
  - iter 1: 0.6383 (var=0.1093) (Δ vs real +0.4150)
  - iter 2: 0.9292 (var=0.0085) (Δ vs real +0.7058)
  - iter 3: 0.8783 (var=0.0102) (Δ vs real +0.6550)
  - iter 4: 0.9292 (var=0.0151) (Δ vs real +0.7058)
  - iter 5 (most recent): 0.9342 (var=0.0101) (Δ vs real +0.7108)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2500 (var=0.0332)
- pi_1: 0.1267 (var=0.0114)
- pi_2: 0.8417 (var=0.0133)
- pi_4: 0.1308 (var=0.0399)

### Experiment 5
**Design**
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Focus exclusively on consensus trials where both models 
        # theoretically predict the TTB option wins for almost all parameters.
        if a == (1, 1, 0, 0) and b == (0, 0, 1, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
        elif a == (1, 0, 1, 0) and b == (0, 1, 0, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
            
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.1175 (var=0.0070)
**Previous candidate values (this loop):**
  - iter 1: 0.5506 (var=0.0723) (Δ vs real +0.4331)
  - iter 2: 0.8000 (var=0.0209) (Δ vs real +0.6825)
  - iter 3: 0.8119 (var=0.0161) (Δ vs real +0.6944)
  - iter 4: 0.7113 (var=0.0630) (Δ vs real +0.5938)
  - iter 5 (most recent): 0.8419 (var=0.0195) (Δ vs real +0.7244)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9531 (var=0.0057)
- pi_3: 0.8662 (var=0.0111)
- pi_1: 0.8500 (var=0.0070)
- pi_2: 0.4831 (var=0.0108)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    moderate_chosen = 0
    tied_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(val * a)
        sum_b = np.sum(val * b)
        
        # Check if it's a tied pair (linear sum of validities is equal)
        if abs(sum_a - sum_b) < 1e-4:
            tied_trials += 1
            
            # Find the best discriminating cue
            for i in range(5):
                if a[i] != b[i]:
                    best_cue_favors_a = (a[i] > b[i])
                    break
            
            # The 'moderate' option is the one NOT favored by the best cue
            chose_a = (row['response'] == 0)
            
            if best_cue_favors_a and not chose_a:
                moderate_chosen += 1
            elif not best_cue_favors_a and chose_a:
                moderate_chosen += 1
                
    if tied_trials == 0:
        return 0.0
    return float(moderate_chosen / tied_trials)
```

**Observed (real) value:** 0.6593 (var=0.0108)
**Previous candidate values (this loop):**
  - iter 1: 0.4870 (var=0.0161) (Δ vs real -0.1722)
  - iter 2: 0.4800 (var=0.0067) (Δ vs real -0.1793)
  - iter 3: 0.5059 (var=0.0044) (Δ vs real -0.1533)
  - iter 4: 0.4948 (var=0.0092) (Δ vs real -0.1644)
  - iter 5 (most recent): 0.5015 (var=0.0053) (Δ vs real -0.1578)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2067 (var=0.0154)
- pi_4: 0.0852 (var=0.0127)
- pi_1: 0.1437 (var=0.0080)
- pi_2: 0.4807 (var=0.0032)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Configural Weighting with Validity-Proportional Discounting: Subjects evaluate options by linearly summing the validities of present features, but apply a configural penalty (or synergy) for cue co-occurrence. Crucially, this interaction term is proportional to the product of the validities of the co-occurring cues. When the configural weight is negative (discounting), multiple highly valid cues strongly penalize each other. This elegantly explains two counter-intuitive phenomena: (1) violating tallying dominance (as in Exp 1), because an option with many cues suffers a massive combinatorial penalty compared to an option with a single strong cue; and (2) preferring 'moderate' validities (as in Exp 5 and 6), because the penalty for combining moderate cues is significantly smaller than the penalty for combining the highest-validity cues, causing the moderate option to emerge victorious.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    omega = float(parameters["omega"])
    
    def get_score(x):
        val_x = val * x
        linear = np.sum(val_x)
        # Efficiently compute sum of pairwise products: 0.5 * ((sum)^2 - sum(squares))
        config = 0.5 * (np.sum(val_x)**2 - np.sum(val_x**2))
        return linear + omega * config

    score_a = get_score(a)
    score_b = get_score(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- omega: [-5.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's insight to explore Configural Weighting to explain preferences for 'moderate' options and violations of validity-based dominance, this theory introduces an interaction term `omega` for cue co-occurrence. The penalty/synergy for having multiple cues is proportional to the product of their validities. With a negative `omega` (discounting), an option with two very high validities (e.g., 0.9 and 0.8) receives a heavy penalty (proportional to 0.72), whereas an option with two moderate validities (e.g., 0.7 and 0.6) receives a much smaller penalty (0.42). This allows the moderate option to win out when linear sums are otherwise close (explaining Exp 5 and 6). Furthermore, an option with many cues (e.g., 3 moderate cues) will accumulate many pairwise penalties, allowing a single strong cue to dominate it, perfectly capturing the violation of tallying dominance in Exp 1.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The proposed Configural Weighting model with Validity-Proportional Discounting was accepted by the loop, establishing a new baseline. However, its predictions are clustering around 0.5 across most experiments (e.g., Exp 1: 0.34 vs 0.78, Exp 5: 0.55 vs 0.11, Exp 6: 0.48 vs 0.65). This suggests that the model is either being washed out by a high upper bound on the lapse rate (epsilon), or that the configural term (omega) is scaling too aggressively and dominating the linear term in unpredictable ways, causing the choice probabilities to become highly variable and effectively random when averaged across the parameter space.
Rationale: Keep the Configural Weighting family prescribed by the arbiter, but make minor adjustments to improve fit. First, constrain the `epsilon` parameter range to [0.0, 0.2] to prevent the model from washing out strong predictions. Second, reconsider the scaling of the configural penalty. Currently, `0.5 * (sum^2 - sum_sq)` scales quadratically with the number of cues, which might overwhelm the linear term for options with many cues. Try normalizing the configural term by the number of active cues, or using the variance of the active validities (as hinted by the arbiter) to control the penalty/synergy. Finally, ensure `omega` has a slightly tighter range (e.g., [-2.0, 2.0]) to prevent it from completely overriding the linear validity sum.

**Outcome of this advice:** iter 1 candidate loss=0.4671 -> iter 2 candidate loss=0.7455 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate was rejected by the accept gate, meaning the attempt to normalize the configural penalty and tighten parameters worsened the fit (loss increased from 0.4671 to 0.7455). Looking at the accepted base (iter 1), the predictions across all experiments hover around 0.5, failing to capture the strong empirical preferences in Exp 1 (real 0.78), Exp 2 (real 0.23), and Exp 4 (real 0.22). The configural weighting approach is struggling to differentiate the choices robustly without washing out.
Rationale: Since tweaking the configural weighting terms worsened the loss, let's pivot to the other mechanism explicitly suggested in the arbiter's recommendation: the two-stage Heuristic-Switching model. Implement a model where subjects first compare the total number of positive features (tallying). If the absolute difference in tally counts is greater than a threshold (e.g., a free parameter `theta` in [0, 2]), the subject chooses based on tallying. If the difference is less than or equal to `theta` (i.e., a tie or near-tie in counts), the subject switches to using the validities (either a linear sum of validities, or Take-The-Best) to break the tie. Use a softmax over the resulting evidence or a mixture model to generate probabilities. This directly addresses the arbiter's prompt and may naturally capture the systematic violations of validity-based dominance seen in the data.

**Outcome of this advice:** iter 2 candidate loss=0.7455 -> iter 3 candidate loss=0.7297 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The two-stage tallying-first heuristic model was rejected by the gate, as it severely missed the empirical patterns across all experiments (e.g., predicting 0.14 vs real 0.78 in Exp 1, and 0.85 vs real 0.23 in Exp 2). The loop will now revert to the accepted base from Iteration 1 (the configural weighting model). The configural model from Iter 1 achieved a better loss (0.4671) but its predictions were still clustering around 0.5. Since our previous attempt to fix it by normalizing the penalty failed, we need a different approach within the context-dependent/configural family.
Rationale: Since the Tallying-first switching model was rejected, the loop resets to the Iter 1 configural base. To improve the configural model without overcomplicating the penalty term (which caused Iter 2 to fail), implement a simpler context-dependent weighting mechanism: Diminishing Returns or Attention-Sharing. For instance, let the subjective value of an option be the sum of its validities divided by the number of positive features raised to a free parameter `alpha` (e.g., `score = sum(validities) / (count ** alpha)` with alpha in [0.0, 2.0]). Alternatively, use a simple fractional power `score = sum(validities ** gamma)`. This directly fulfills the arbiter's suggestion of context-dependent subjective weights while keeping the math stable and avoiding the quadratic blowups that likely ruined the previous configural attempts. Keep epsilon constrained to [0.0, 0.2] to prevent washing out.

**Outcome of this advice:** iter 3 candidate loss=0.7297 -> iter 4 candidate loss=0.7129 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate was REJECTED by the accept gate, meaning the attempt to normalize the configural penalty by the number of active pairs worsened the fit (loss 0.7129 vs running-best 0.4671 from Iter 1). The proposer correctly noted that a simple count-based discounting would fail Experiment 5, but their chosen normalization still failed to capture the empirical phenomena, keeping predictions far from the real data (e.g., Exp 1 is 0.16 vs real 0.78; Exp 2 is 0.88 vs real 0.23). The loop will now revert to the accepted base from Iter 1.
Rationale: Since the loop resets to Iteration 1's unnormalized configural weighting model, we should build directly on that mathematical form but fix the parameter scaling. In Iter 1, the configural term `0.5 * (linear**2 - sum(val_x**2))` scaled quadratically, which likely overpowered the linear term and caused erratic choice probabilities when `omega` was large. Instead of changing the equation or adding complex normalizations, simply restrict the prior range of `omega` to be much smaller (e.g., [-0.2, 0.2] or [-0.5, 0.5]). This ensures the configural penalty acts as a context-dependent modifier rather than dominating the primary validity sum. Keep `epsilon` constrained to [0.0, 0.1] to prevent washing out, and maintain the standard softmax with `beta`.

**Outcome of this advice:** iter 4 candidate loss=0.7129 -> iter 5 candidate loss=0.7496 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate was rejected by the accept gate, meaning that simply tightening the parameter ranges on the quadratic configural penalty worsened the fit (loss 0.7496 vs running-best 0.4671). The predictions for this tightened model are pushed to extremes that completely contradict the empirical data (e.g., predicting 0.05 vs real 0.78 in Exp 1, and 0.91 vs real 0.23 in Exp 2). The quadratic interaction term `sum^2 - sum_sq` is fundamentally failing to capture the choice patterns, even with small parameter bounds.
Rationale: Since tweaking the quadratic configural penalty has repeatedly failed and been rejected by the gate, we must pivot to the other specific mechanism mentioned in the arbiter's recommendation: a model based on the 'variance of the cues'. Implement a Variance-Penalized Linear Model. The subjective score of an option is the linear sum of its present validities minus a penalty proportional to the variance of those present validities. Specifically: `score = np.sum(present_vals) - omega * np.var(present_vals)`. If an option has 0 or 1 positive features, its variance penalty is 0. This elegantly handles Exp 5 (moderate validities have lower variance than a mix of extreme validities, so the moderate option suffers less penalty and wins) and can modulate the dominance of tallying vs TTB based on the dispersion of cue validities. Set `omega` to a free parameter in [0.0, 10.0], maintain the `beta` softmax, and keep `epsilon` in [0.0, 0.1].

**Outcome of this advice:** iter 5 candidate loss=0.7496 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Variance-Penalized Linear Model: Subjects evaluate options by computing the linear sum of the validities of present features, but subtract a penalty proportional to the variance of those present validities. This context-dependent mechanism penalizes options with a highly dispersed set of cues, favoring options with consistent, moderate validities or a single strong cue (which has zero variance).",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    omega = float(parameters[\"omega\"])\n    \n    def get_score(x):\n        present_vals = val[x > 0.5]\n        if len(present_vals) > 1:\n            variance = np.var(present_vals)\n        else:\n            variance = 0.0\n        return np.sum(present_vals) - omega * variance\n\n    score_a = get_score(a)\n    score_b = get_score(b)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "omega": "[0.0, 50.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.1]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice to pivot to a Variance-Penalized Linear Model. Options are evaluated by the linear sum of their present validities minus a penalty proportional to the variance of those present validities. This context-dependent penalty naturally favors options with moderate, consistent validities (as in Exps 5 and 6) and can penalize options with many disparate cues to allow a single strong cue to dominate (Exp 1). The parameter `omega` is expanded up to 50.0 to ensure the penalty can be strong enough to overcome the linear sum when needed, and `epsilon` is constrained to [0.0, 0.1] to prevent washing out."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14570,
  "candidates_token_count": 736,
  "total_token_count": 16682
}
```
