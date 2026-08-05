# generation_iter_03_attempt_00

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
- The recommendation below acts on THEORY 2 (= `pi_3`).

Propose a new theory based on a 'Lexicographic Semiorder' or 'Cue Difference Threshold' model. In this theory, decision-makers compare options cue-by-cue in order of validity. However, instead of a strict TTB rule, an option is only chosen if the validity of its cue exceeds the alternative by a certain threshold, or if the sheer number of opposing cues doesn't cross a 'tallying deficit' limit. If the top cue's advantage is not seen as decisive enough (e.g., when facing a concentrated block of moderately high cues), they fall back to a compensatory or tallying process. This can explain the non-monotonic preference reversals seen in Experiments 1 and 2 where extreme cue configurations lead to counterintuitive choice rates.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.861)

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
- Experiment 1: real=0.6583 (var=0.0166) vs this=0.8496 (var=0.0117)
- Experiment 2: real=0.2750 (var=0.0193) vs this=0.3260 (var=0.0039)
- Experiment 3: real=0.2278 (var=0.0190) vs this=0.1361 (var=0.0088)
- Experiment 4: real=0.2333 (var=0.0272) vs this=0.1408 (var=0.0158)
- Experiment 5: real=-0.3900 (var=0.1891) vs this=-0.0125 (var=0.0302)
- Experiment 6: real=-0.1983 (var=0.1423) vs this=0.0175 (var=0.0198)


---

### `pi_4` (overall score: 0.855)

**Description**
Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
    cue_order = np.argsort(-val, kind="stable").tolist()
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
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6583 (var=0.0166) vs this=0.7150 (var=0.0215)
- Experiment 2: real=0.2750 (var=0.0193) vs this=0.4446 (var=0.0098)
- Experiment 3: real=0.2278 (var=0.0190) vs this=0.2911 (var=0.0192)
- Experiment 4: real=0.2333 (var=0.0272) vs this=0.2808 (var=0.0246)
- Experiment 5: real=-0.3900 (var=0.1891) vs this=0.1625 (var=0.0470)
- Experiment 6: real=-0.1983 (var=0.1423) vs this=0.0000 (var=0.0209)


---

### `pi_3` (overall score: 0.541)

**Description**
The Weighted Additive (WADD) model with non-linear weight scaling posits that decision makers integrate all available information by computing a weighted sum of features. However, the weighting of cues is not strictly proportional to their log-odds validity. Instead, decision makers apply a non-linear transformation (parameterized by gamma) to the log-odds, allowing them to stretch the weight differential. This permits WADD to approximate lexicographic (TTB-like) choice when gamma > 1, or more uniform (Tallying-like) weighting when gamma < 1, while remaining a fully compensatory integration process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid infinity
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # Compute weighted sum for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the weighted scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [0.5, 2.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.6583 (var=0.0166) vs this=0.7360 (var=0.0309)
- Experiment 2: real=0.2750 (var=0.0193) vs this=0.3937 (var=0.0151)
- Experiment 3: real=0.2278 (var=0.0190) vs this=0.3717 (var=0.0906)
- Experiment 4: real=0.2333 (var=0.0272) vs this=0.3633 (var=0.0726)
- Experiment 5: real=-0.3900 (var=0.1891) vs this=-0.0175 (var=0.0350)
- Experiment 6: real=-0.1983 (var=0.1423) vs this=0.2242 (var=0.1096)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3234 -> ACCEPTED
- iter 2: loss=0.3351 -> REJECTED
- iter 3: loss=0.3340 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.3234 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        
        if winner is not None:
            if row['response'] == winner:
                ttb_consistent += 1
            total += 1
            
    return float(ttb_consistent / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.6583 (var=0.0166)
**Previous candidate values (this loop):**
  - iter 1: 0.6927 (var=0.0202) (Δ vs real +0.0344)
  - iter 2: 0.7271 (var=0.0200) (Δ vs real +0.0687)
  - iter 3 (most recent): 0.6496 (var=0.0745) (Δ vs real -0.0088)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8496 (var=0.0117)
- pi_2: 0.1315 (var=0.0081)
- pi_3: 0.7360 (var=0.0309)
- pi_4: 0.7150 (var=0.0215)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, 1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    matches = (data['response'].values[mask] == tally_pred[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2750 (var=0.0193)
**Previous candidate values (this loop):**
  - iter 1: 0.4058 (var=0.0100) (Δ vs real +0.1308)
  - iter 2: 0.4383 (var=0.0122) (Δ vs real +0.1633)
  - iter 3 (most recent): 0.4446 (var=0.0284) (Δ vs real +0.1696)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8548 (var=0.0066)
- pi_1: 0.3260 (var=0.0039)
- pi_3: 0.3937 (var=0.0151)
- pi_4: 0.4446 (var=0.0098)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    comp_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: A has best cue, B has all others (compensatory is B)
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            comp_choices += (resp == 1)
            total += 1
        # Trial 2: B has best cue, A has next two (compensatory is A)
        elif a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            comp_choices += (resp == 0)
            total += 1
        # Trial 4: A has best available cue, B has next two (compensatory is B)
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            comp_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.0
    return float(comp_choices / total)
```

**Observed (real) value:** 0.2278 (var=0.0190)
**Previous candidate values (this loop):**
  - iter 1: 0.2989 (var=0.0361) (Δ vs real +0.0711)
  - iter 2: 0.3056 (var=0.0389) (Δ vs real +0.0778)
  - iter 3 (most recent): 0.3483 (var=0.0640) (Δ vs real +0.1206)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1361 (var=0.0088)
- pi_3: 0.3717 (var=0.0906)
- pi_2: 0.8578 (var=0.0097)
- pi_4: 0.2911 (var=0.0192)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has only 1 positive feature (the most valid one) 
    # and Option B has 3 or 4 positive features (all lower validity).
    # This corresponds exactly to Trial 1 and Trial 3 in the design.
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = (a_sum == 1) & (b_sum >= 3)
    if not mask.any():
        return 0.0
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2333 (var=0.0272)
**Previous candidate values (this loop):**
  - iter 1: 0.4117 (var=0.0502) (Δ vs real +0.1783)
  - iter 2: 0.3725 (var=0.0696) (Δ vs real +0.1392)
  - iter 3 (most recent): 0.4650 (var=0.1322) (Δ vs real +0.2317)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3633 (var=0.0726)
- pi_1: 0.1408 (var=0.0158)
- pi_2: 0.8700 (var=0.0121)
- pi_4: 0.2808 (var=0.0246)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    def get_p_a(a_str, b_str):
        mask1 = (data['A_str'] == a_str) & (data['B_str'] == b_str)
        mask2 = (data['A_str'] == b_str) & (data['B_str'] == a_str)
        
        chose_a = 0
        total = 0
        
        if mask1.any():
            chose_a += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.any():
            chose_a += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        if total == 0:
            return 0.5
        return chose_a / total

    # Trial 8: Concordant for Strategy Mixture, weak advantage for NL-WADD
    p_a_t8 = get_p_a('10011', '01100')
    
    # Trial 7: Discordant for Strategy Mixture, massive advantage for NL-WADD
    p_a_t7 = get_p_a('11000', '00111')
    
    return float(p_a_t8 - p_a_t7)
```

**Observed (real) value:** -0.3900 (var=0.1891)
**Previous candidate values (this loop):**
  - iter 1: -0.0225 (var=0.0392) (Δ vs real +0.3675)
  - iter 2: -0.0100 (var=0.0362) (Δ vs real +0.3800)
  - iter 3 (most recent): 0.1625 (var=0.0864) (Δ vs real +0.5525)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1625 (var=0.0470)
- pi_3: -0.0175 (var=0.0350)
- pi_1: -0.0125 (var=0.0302)
- pi_2: 0.7450 (var=0.0681)

### Experiment 6
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t7_a = (a_str == '10100') & (b_str == '01011')
    t7_b = (a_str == '01011') & (b_str == '10100')
    
    t36_a = (a_str == '10000') & (b_str == '01111')
    t36_b = (a_str == '01111') & (b_str == '10000')
    
    t7_ttb = pd.concat([
        data.loc[t7_a, 'response'] == 0,
        data.loc[t7_b, 'response'] == 1
    ])
    
    t36_ttb = pd.concat([
        data.loc[t36_a, 'response'] == 0,
        data.loc[t36_b, 'response'] == 1
    ])
    
    t7_rate = t7_ttb.mean() if len(t7_ttb) > 0 else 0.5
    t36_rate = t36_ttb.mean() if len(t36_ttb) > 0 else 0.5
    
    return float(t7_rate - t36_rate)
```

**Observed (real) value:** -0.1983 (var=0.1423)
**Previous candidate values (this loop):**
  - iter 1: 0.2408 (var=0.0895) (Δ vs real +0.4392)
  - iter 2: 0.4067 (var=0.1407) (Δ vs real +0.6050)
  - iter 3 (most recent): 0.1092 (var=0.0590) (Δ vs real +0.3075)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2242 (var=0.1096)
- pi_4: 0.0000 (var=0.0209)
- pi_1: 0.0175 (var=0.0198)
- pi_2: 0.0250 (var=0.0241)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Cue Difference Threshold Theory: Decision-makers evaluate options lexicographically but demand that the best discriminating cue provides a decisive advantage. A cue is deemed decisive if its validity exceeds the best opposing cue by a sufficient threshold, or if the sheer number of opposing cues is small enough (below a tallying deficit limit). If the top cue's advantage is challenged by a concentrated block of moderately high opposing cues (failing both conditions), the decision-maker abandons the non-compensatory heuristic and falls back to a compensatory Weighted Additive (WADD) process to resolve the choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    deficit_limit = int(parameters["deficit_limit"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues favor which option
    favor_a = (a > b)
    favor_b = (b > a)
    
    max_val_a = np.max(val[favor_a]) if np.any(favor_a) else 0.0
    max_val_b = np.max(val[favor_b]) if np.any(favor_b) else 0.0
    
    winner_ttb = 0 if max_val_a > max_val_b else (1 if max_val_b > max_val_a else None)
    
    if winner_ttb is not None:
        # Number of cues favoring the TTB loser
        num_opposing = np.sum(favor_b) if winner_ttb == 0 else np.sum(favor_a)
        
        # Two conditions for TTB to be considered decisive:
        # 1. The validity difference between the best cues of each option is large enough.
        # 2. The number of opposing cues is within the acceptable deficit limit.
        cue_diff_decisive = abs(max_val_a - max_val_b) >= threshold
        deficit_ok = num_opposing <= deficit_limit
        
        if cue_diff_decisive or deficit_ok:
            # Decisive advantage: stick to TTB
            scores = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        else:
            # Not decisive: fall back to compensatory WADD
            val_clipped = np.clip(val, 0.5001, 0.9999)
            log_odds = np.log(val_clipped / (1.0 - val_clipped))
            score_a = np.sum(log_odds * a)
            score_b = np.sum(log_odds * b)
            
            # Normalize compensatory scores by total weight to keep the scale 
            # comparable to the [0, 1] TTB scores for the softmax temperature.
            total_w = np.sum(log_odds)
            scores = np.array([score_a, score_b]) / total_w if total_w > 0 else np.array([0.5, 0.5])
    else:
        scores = np.array([0.5, 0.5])
        
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- threshold: [0.0, 0.5]
- deficit_limit: {0, 1, 2, 3, 4, 5}
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's recommendation, this model implements a Lexicographic Semiorder with a Cue Difference Threshold and a tallying deficit limit. Instead of a monolithic mixture where strategies are chosen probabilistically, this theory posits a deterministic fallback mechanism per trial: decision-makers begin with a lexicographic evaluation (TTB), but only commit to the choice if the top cue's advantage is decisive. Decisiveness is met if the top cue's validity exceeds the opponent's best cue by a learned threshold, OR if the sheer number of opposing cues is small enough. If a concentrated block of moderately high cues challenges the top cue, the decision-maker abandons TTB and resolves the choice using a compensatory Weighted Additive (WADD) process. This condition-based fallback explains the stark non-monotonic preference reversals in Experiments 1, 2, and 5, where specific configurations of opposing cues trigger compensatory evaluation while others do not.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory was accepted by the loop, indicating an improvement in the aggregate loss (now 0.3234). It fits Experiments 1 and 3 well, and shows sensible improvements on Experiments 2 and 4. However, it fails to capture the correct direction for Experiment 6 (real is -0.1983, candidate is +0.2408) and underestimates the magnitude of Experiment 5 (real is -0.39, candidate is -0.02). In Experiment 6, the model predicts higher TTB-consistency on Trial 7 than Trial 36, whereas human data shows the opposite. This suggests the current 'deficit_limit' logic (simply counting opposing cues) makes Trial 7 look 'safer' for TTB than Trial 36, failing to capture why human subjects abandon TTB more readily or choose the compensatory option more frequently in Trial 7.
Rationale: While the current candidate is the running best, the fallback trigger logic needs a minor adjustment within the prescribed family to fix the inverted effect in Experiment 6. Instead of a strict count of opposing cues (`num_opposing`), redefine the 'deficit' to consider the continuous weight of the opposing evidence (e.g., the sum of validities of opposing cues, or the compensatory score difference). Additionally, ensure the compensatory WADD scores are scaled properly (e.g., using raw log-odds differences rather than normalizing by `total_w`, which flattens the confidence of the fallback strategy). Adjusting the deficit definition to be validity-weighted and fixing the WADD scaling should allow the model to capture the correct direction in Experiments 5 and 6 without abandoning the Lexicographic Semiorder family.

**Outcome of this advice:** iter 1 candidate loss=0.3234 -> iter 2 candidate loss=0.3351 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate was REJECTED by the gate because its aggregate loss worsened (0.3351 vs 0.3234). The attempt to redefine the 'deficit' using the sum of log-odds of opposing cues exacerbated the mismatch in Experiment 6, pushing the metric to +0.4067 (real is -0.1983). In Experiment 6, humans are less TTB-consistent on Trial 7 than Trial 36, which is counter-intuitive if we assume a strict WADD or log-odds based fallback, since Trial 36 has more opposing evidence against the TTB winner than Trial 7. The previous advice to use validity-weighted evidence for the deficit limit failed to capture this and was rightfully rejected.
Rationale: Since the log-odds deficit approach was rejected, we should return to the simpler 'sheer number of opposing cues' (tallying deficit) logic from the accepted Iteration 1 base. To improve the fit on Experiments 5 and 6 without breaking the family, change the fallback strategy itself: instead of falling back to a fully compensatory WADD process, fall back to a simple Tallying process (unweighted sum of cues). The arbiter explicitly suggested 'fall back to a compensatory or tallying process'. Using Tallying for the fallback scores (i.e., simply counting the number of positive features for each option) might better capture the non-monotonic preference reversals and the specific trial differences in Experiments 5 and 6. Ensure the Tallying scores are properly scaled by `beta` in the softmax.

**Outcome of this advice:** iter 2 candidate loss=0.3351 -> iter 3 candidate loss=0.3340 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate theory was REJECTED by the gate because the aggregate loss worsened (0.3340 vs the running-best 0.3234 from Iteration 1). The attempt to switch the fallback strategy from WADD to simple Tallying failed to improve the fit and was rejected. The model still struggles with the non-monotonic preference reversals in Experiments 5 and 6. In Experiment 6, humans are less TTB-consistent on Trial 7 than Trial 36, which the hard boolean gating logic based on a strict count of opposing cues struggles to capture because Trial 7 has fewer opposing cues than Trial 36.
Rationale: Since both the log-odds deficit (Iter 2) and the Tallying fallback (Iter 3) were rejected, you must revert to the exact mechanism from the accepted Iteration 1 base: a WADD (compensatory) fallback with a deficit limit based on the sheer number of opposing cues. To improve upon Iteration 1 and capture the nuances of Experiments 5 and 6 without breaking the family, replace the hard boolean switch (`if cue_diff_decisive or deficit_ok`) with a soft-gating mechanism. Compute a continuous probability of using TTB vs WADD based on a logistic function of the cue difference and the opposing cue count. Alternatively, keep the hard switch but ensure the WADD scores are properly calculated as the dot product of features and validities, and apply a dedicated scaling/temperature parameter to the WADD scores so the fallback doesn't wash out the softmax.

**Outcome of this advice:** iter 3 candidate loss=0.3340 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Cue Difference Threshold Theory with Scaled WADD Fallback: Decision-makers evaluate options lexicographically but demand that the best discriminating cue provides a decisive advantage. A cue is deemed decisive if its validity exceeds the best opposing cue by a sufficient threshold, or if the sheer number of opposing cues is small enough (below a tallying deficit limit). If the top cue's advantage is challenged by a concentrated block of moderately high opposing cues, the decision-maker abandons the non-compensatory heuristic and falls back to a compensatory Weighted Additive (WADD) process. By applying a dedicated softmax temperature to the WADD fallback and computing WADD as a direct dot product of validities and features, the fallback strategy can confidently resolve the choice without washing out the probabilities.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    threshold = float(parameters[\"threshold\"])\n    deficit_limit = int(parameters[\"deficit_limit\"])\n    beta_ttb = float(parameters[\"beta_ttb\"])\n    beta_wadd = float(parameters[\"beta_wadd\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Identify which cues favor which option\n    favor_a = (a > b)\n    favor_b = (b > a)\n    \n    max_val_a = np.max(val[favor_a]) if np.any(favor_a) else 0.0\n    max_val_b = np.max(val[favor_b]) if np.any(favor_b) else 0.0\n    \n    winner_ttb = 0 if max_val_a > max_val_b else (1 if max_val_b > max_val_a else None)\n    \n    if winner_ttb is not None:\n        # Number of cues favoring the TTB loser\n        num_opposing = np.sum(favor_b) if winner_ttb == 0 else np.sum(favor_a)\n        \n        # Two conditions for TTB to be considered decisive:\n        # 1. The validity difference between the best cues of each option is large enough.\n        # 2. The number of opposing cues is within the acceptable deficit limit.\n        cue_diff_decisive = abs(max_val_a - max_val_b) >= threshold\n        deficit_ok = num_opposing <= deficit_limit\n        \n        if cue_diff_decisive or deficit_ok:\n            # Decisive advantage: stick to TTB\n            scores = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])\n            z = beta_ttb * (scores - np.max(scores))\n        else:\n            # Not decisive: fall back to compensatory WADD\n            # Calculate WADD directly as the dot product of validities and features\n            score_a = np.sum(val * a)\n            score_b = np.sum(val * b)\n            scores = np.array([score_a, score_b])\n            # Apply dedicated WADD temperature\n            z = beta_wadd * (scores - np.max(scores))\n    else:\n        z = np.array([0.0, 0.0])\n        \n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "threshold": "[0.0, 0.5]",
    "deficit_limit": "{0, 1, 2, 3, 4, 5}",
    "beta_ttb": "[0.1, 15.0]",
    "beta_wadd": "[0.1, 15.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, this minimal edit retains the successful hard-gating logic from the running-best base (Iteration 1) but improves the compensatory WADD fallback mechanism. Instead of transforming validities into log-odds and normalizing by the total weight (which flattened the confidence of the fallback), the model now computes WADD scores simply as the dot product of features and validities. Furthermore, it introduces a dedicated scaling parameter (`beta_wadd`) for the fallback process, separating it from the `beta_ttb` used for lexicographic choices. This allows the model to appropriately scale the compensatory difference so that the fallback strategy doesn't wash out the softmax, better capturing the non-monotonic preference reversals in Experiments 5 and 6."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14175,
  "candidates_token_count": 1312,
  "total_token_count": 17730
}
```
