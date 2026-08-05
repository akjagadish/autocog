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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a brand-new theory based on 'Variance Penalization' or 'Risk Aversion' in cue integration. The theory should posit that subjects do not just sum validities (linearly or non-linearly), but actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. This would naturally explain the strong preference for moderate validities in tied-sum scenarios (Exp 1 and 8) and could help explain the anomalous behavior in Exp 7.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.453)

**Description**
Attention-Sharing (Fractional Discounting): Subjects evaluate options by linearly summing the validities of present features, but the overall value is discounted by the number of active cues. This reflects a cognitive bottleneck where attention must be divided among multiple features. The subjective value of an option is its sum of validities divided by `1.0 + kappa * (n_cues - 1)`. This mechanism allows a single highly valid cue to maintain its full weight and potentially dominate an option with multiple weaker cues that dilute each other's impact, explaining violations of tallying dominance while remaining mathematically stable.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    def get_score(x):
        val_x = val * x
        n_cues = np.sum(x)
        if n_cues == 0:
            return 0.0
        linear = np.sum(val_x)
        return linear / (1.0 + kappa * (n_cues - 1.0))

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
- kappa: [0.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.1]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7858 (var=0.0260) vs this=0.6379 (var=0.1062)
- Experiment 2: real=0.2306 (var=0.0195) vs this=0.3372 (var=0.0868)
- Experiment 3: real=0.6521 (var=0.0118) vs this=0.6238 (var=0.0633)
- Experiment 4: real=0.2233 (var=0.0237) vs this=0.4100 (var=0.1130)
- Experiment 5: real=0.1175 (var=0.0070) vs this=0.7750 (var=0.0206)
- Experiment 6: real=0.6593 (var=0.0108) vs this=0.5204 (var=0.0043)
- Experiment 7: real=0.1450 (var=0.0164) vs this=0.5092 (var=0.0103)
- Experiment 8: real=0.3517 (var=0.0118) vs this=0.0158 (var=0.0043)


---

### `pi_3` (overall score: 0.383)

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
- Experiment 7: real=0.1450 (var=0.0164) vs this=0.8650 (var=0.0095)
- Experiment 8: real=0.3517 (var=0.0118) vs this=0.3317 (var=0.0063)


---

### `pi_1` (overall score: 0.326)

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
- Experiment 7: real=0.1450 (var=0.0164) vs this=0.8517 (var=0.0121)
- Experiment 8: real=0.3517 (var=0.0118) vs this=0.3458 (var=0.0129)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4369 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4369 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1 (most recent): 0.5229 (var=0.0590) (Δ vs real -0.2629)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8546 (var=0.0146)
- pi_2: 0.1412 (var=0.0101)
- pi_3: 0.7021 (var=0.0525)
- pi_4: 0.8446 (var=0.0633)
- pi_5: 0.6379 (var=0.1062)

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
  - iter 1 (most recent): 0.4997 (var=0.0685) (Δ vs real +0.2691)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8691 (var=0.0085)
- pi_1: 0.1391 (var=0.0083)
- pi_3: 0.3344 (var=0.0398)
- pi_4: 0.1831 (var=0.0521)
- pi_5: 0.3372 (var=0.0868)

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
  - iter 1 (most recent): 0.5725 (var=0.0507) (Δ vs real -0.0796)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8827 (var=0.0073)
- pi_3: 0.6856 (var=0.0224)
- pi_2: 0.2102 (var=0.0075)
- pi_4: 0.7771 (var=0.0930)
- pi_5: 0.6238 (var=0.0633)

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
  - iter 1 (most recent): 0.5683 (var=0.1014) (Δ vs real +0.3450)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2500 (var=0.0332)
- pi_1: 0.1267 (var=0.0114)
- pi_2: 0.8417 (var=0.0133)
- pi_4: 0.1308 (var=0.0399)
- pi_5: 0.4100 (var=0.1130)

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
  - iter 1 (most recent): 0.8025 (var=0.0184) (Δ vs real +0.6850)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9531 (var=0.0057)
- pi_3: 0.8662 (var=0.0111)
- pi_1: 0.8500 (var=0.0070)
- pi_2: 0.4831 (var=0.0108)
- pi_5: 0.7750 (var=0.0206)

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
  - iter 1 (most recent): 0.6030 (var=0.0331) (Δ vs real -0.0563)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2067 (var=0.0154)
- pi_4: 0.0852 (var=0.0127)
- pi_1: 0.1437 (var=0.0080)
- pi_2: 0.4807 (var=0.0032)
- pi_5: 0.5204 (var=0.0043)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_t1 = (1, 0, 0, 1, 0)
    b_t1 = (0, 1, 1, 0, 0)
    
    a_t2 = (1, 0, 0, 0, 1)
    b_t2 = (0, 1, 0, 1, 0)
    
    def get_val(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == a_t1 and b == b_t1:
            return 1 - row['response']
        elif a == a_t2 and b == b_t2:
            return 1 - row['response']
        elif a == b_t1 and b == a_t1:
            return row['response']
        elif a == b_t2 and b == a_t2:
            return row['response']
        return np.nan
        
    vals = data.apply(get_val, axis=1).dropna()
    if len(vals) == 0:
        return 0.5
    return float(vals.mean())
```

**Observed (real) value:** 0.1450 (var=0.0164)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5292 (var=0.0733) (Δ vs real +0.3842)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5092 (var=0.0103)
- pi_3: 0.8650 (var=0.0095)
- pi_1: 0.8517 (var=0.0121)
- pi_2: 0.4992 (var=0.0106)
- pi_4: 0.9358 (var=0.0158)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify trials where both options have exactly 2 active cues.
    # In the design, these are trials 1, 2, and 3.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask = (a_sums == 2) & (b_sums == 2)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.0
        
    # Group by trial type using string representations of option A
    subset_key = subset['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    devs = []
    for key, group in subset.groupby(subset_key):
        # proportion of times A was chosen (response == 0)
        p_a = np.mean(group['response'] == 0)
        devs.append(abs(p_a - 0.5))
        
    return float(np.mean(devs))
```

**Observed (real) value:** 0.3517 (var=0.0118)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0692 (var=0.0119) (Δ vs real -0.2825)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3317 (var=0.0063)
- pi_5: 0.0158 (var=0.0043)
- pi_1: 0.3458 (var=0.0129)
- pi_2: 0.0192 (var=0.0040)
- pi_4: 0.3908 (var=0.0159)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Variance Penalization (Risk Aversion in Cue Integration): Subjects evaluate options by integrating the validities of present features, but they actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. The subjective value of an option is its non-linearly weighted sum of validities minus a penalty proportional to the variance of its active validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    lambda_pen = float(parameters["lambda_pen"])
    
    def get_score(x):
        active_mask = (x > 0.5)
        if not np.any(active_mask):
            return 0.0
        
        # Weighted sum using exponentiated validities for compensatoriness flexibility
        sum_v = np.sum(val[active_mask] ** gamma)
        
        # Variance penalty using original validities to capture dispersion
        var_v = np.var(val[active_mask]) if np.sum(active_mask) > 1 else 0.0
        
        return sum_v - lambda_pen * var_v

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
- gamma: [0.1, 10.0]
- lambda_pen: [0.0, 20.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: The theory of 'Variance Penalization' posits that decision-makers do not merely compute a weighted sum of evidence, but actively penalize options whose supporting evidence is highly dispersed. An option with moderate, consistent expert ratings is perceived as more reliable than one with a mix of extreme (very high and very low) ratings, reflecting a form of risk aversion in cue integration. By subtracting a penalty proportional to the variance of the active cues' validities, this model naturally explains the strong preference for moderate validities in tied-sum scenarios (Experiments 6, 7, and 8). The inclusion of a gamma exponent allows the model to also capture varying degrees of compensatoriness (e.g., TTB-like behavior in Experiment 1) while remaining sensitive to cue dispersion.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate theory 'Variance Penalization' introduces a penalty for the variance of active cues, which helps it capture the preference for moderate options in tied-sum scenarios (Exp 6). It achieved a new running-best loss, indicating the mechanism is promising. However, it still falls short on key experiments: in Exp 5 and 7, it fails to match the strong empirical deviations from compensatory summation, and in Exp 8 it severely underestimates the deviations from 0.5. A major reason for this is likely the scale of the penalty: because validities are typically bounded between 0.5 and 1.0, their variance is on the order of 0.01. A lambda_pen max of 20.0 only yields a maximum penalty of ~0.2, which is often overpowered by the main sum, especially when gamma > 1.
Rationale: To strengthen the prescribed mechanism family, make two adjustments: 1) Replace the variance penalty with a standard deviation penalty (`np.std(val[active_mask])`), which scales linearly with the validities and produces larger penalty magnitudes. 2) Expand the upper bound of `lambda_pen` significantly (e.g., up to 50.0 or 100.0) to allow the risk-aversion component to dominate when necessary. Keep the rest of the mechanism (including the gamma exponent) intact.

**Outcome of this advice:** iter 1 candidate loss=0.4369 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Variance Penalization (Risk Aversion in Cue Integration): Subjects evaluate options by integrating the validities of present features, but they actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. The subjective value of an option is its non-linearly weighted sum of validities minus a penalty proportional to the standard deviation of its active validities, which scales appropriately to strongly influence decisions.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    lambda_pen = float(parameters[\"lambda_pen\"])\n    \n    def get_score(x):\n        active_mask = (x > 0.5)\n        if not np.any(active_mask):\n            return 0.0\n        \n        # Weighted sum using exponentiated validities for compensatoriness flexibility\n        sum_v = np.sum(val[active_mask] ** gamma)\n        \n        # Standard deviation penalty using original validities to capture dispersion\n        std_v = np.std(val[active_mask]) if np.sum(active_mask) > 1 else 0.0\n        \n        return sum_v - lambda_pen * std_v\n\n    score_a = get_score(a)\n    score_b = get_score(b)\n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "gamma": "[0.1, 10.0]",
    "lambda_pen": "[0.0, 100.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, the penalty term has been changed from variance to standard deviation (`np.std`). Because validities range from 0.5 to 1.0, variance is very small (~0.01), limiting the impact of the penalty. Standard deviation provides a linearly-scaled dispersion measure that can more effectively compete with the weighted sum. Additionally, the upper bound of `lambda_pen` has been increased to 100.0 to allow the risk-aversion component to strongly dominate the choice when necessary, helping to explain the strong deviations observed in Experiments 5, 7, and 8."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14057,
  "candidates_token_count": 863,
  "total_token_count": 15179
}
```
