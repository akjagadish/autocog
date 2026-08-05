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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a 'Constant-Mixture Strategy Selection' theory. Unlike Theory 1, which posits that the probability of using Take-The-Best (TTB) decays as the depth of the first discriminating cue increases, this new theory should propose that subjects have a fixed, individual-specific probability of using TTB versus Tallying on any given trial, completely independent of cue depth. This explains why the depth manipulation in Experiment 1 yielded a 0.0 difference, while preserving the structural symmetry in Experiment 2 (yielding a 0.0 difference) that WADD failed to capture.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.382)

**Description**
Context-Dependent Probabilistic Strategy Selection (Take-the-First, else Tally) with parameterized depth decay. Decision-makers probabilistically switch between a non-compensatory heuristic (TTB) and a compensatory strategy (Tallying) based on choice difficulty (depth of the first discriminating cue). The probability of using TTB starts at a baseline 'alpha' for depth=0 and decays by 'gamma' for deeper cues. Tallying predictions are softened by a temperature parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine TTB prediction and the depth of the first discriminating cue
    cue_order = np.argsort(-val, kind="stable").tolist()
    first_disc_idx = -1
    winner_ttb = -1
    for i, j in enumerate(cue_order):
        if a[j] > b[j]:
            winner_ttb = 0
            first_disc_idx = i
            break
        if b[j] > a[j]:
            winner_ttb = 1
            first_disc_idx = i
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine Tallying prediction using softmax
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    tau = float(parameters["tau"])
    
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)
        
    # Probabilistic strategy switch based on choice difficulty (depth of first discriminating cue)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    if first_disc_idx == -1:
        prob_ttb = 0.0
    else:
        prob_ttb = alpha * (gamma ** first_disc_idx)
        
    p_core = prob_ttb * p_ttb + (1.0 - prob_ttb) * p_tally
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- alpha: [0.5, 1.0]
- gamma: [0.0, 1.0]
- tau: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5360 (var=0.0087) vs this=0.6250 (var=0.0133)
- Experiment 2: real=0.5200 (var=0.0096) vs this=0.4938 (var=0.0156)
- Experiment 3: real=0.4222 (var=0.1296) vs this=0.6064 (var=0.0212)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.1818 (var=0.0465)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.4142 (var=0.0591)
- Experiment 6: real=0.0000 (var=0.0000) vs this=-0.0067 (var=0.0275)


---

### `pi_3` (overall score: 0.316)

**Description**
People evaluate options using a non-linearly Weighted Additive (WADD) strategy. Each option's value is the sum of its features weighted by their perceived importance, which is a non-linear power function of the objective cue validities. This allows a single high-validity cue to balance out multiple lower-validity cues, resulting in compensatory trade-offs and choice probabilities near 0.5 on conflict trials.

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
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
- gamma: [1.0, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5360 (var=0.0087) vs this=0.5487 (var=0.0539)
- Experiment 2: real=0.5200 (var=0.0096) vs this=0.4533 (var=0.0339)
- Experiment 3: real=0.4222 (var=0.1296) vs this=0.5472 (var=0.0395)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.5299 (var=0.1410)
- Experiment 5: real=0.0000 (var=0.0000) vs this=-0.0567 (var=0.0265)
- Experiment 6: real=0.0000 (var=0.0000) vs this=0.1950 (var=0.0591)


---

### `pi_1` (overall score: 0.061)

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
- Experiment 1: real=0.5360 (var=0.0087) vs this=0.8393 (var=0.0102)
- Experiment 2: real=0.5200 (var=0.0096) vs this=0.1409 (var=0.0112)
- Experiment 3: real=0.4222 (var=0.1296) vs this=0.8228 (var=0.0157)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0107 (var=0.0072)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0200 (var=0.0077)
- Experiment 6: real=0.0000 (var=0.0000) vs this=0.0117 (var=0.0236)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # TTB prediction: first discriminating feature in descending validity order
    ttb_preds = np.full(len(data), -1)
    for i in range(a_ratings.shape[1]):
        mask_a = (a_ratings[:, i] > b_ratings[:, i]) & (ttb_preds == -1)
        mask_b = (b_ratings[:, i] > a_ratings[:, i]) & (ttb_preds == -1)
        ttb_preds[mask_a] = 0
        ttb_preds[mask_b] = 1
        
    # Tally prediction: sum of feature-wise wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_preds = np.full(len(data), -1)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Identify trials where TTB and Tallying make opposite predictions
    disagree_mask = (ttb_preds != -1) & (tally_preds != -1) & (ttb_preds != tally_preds)
    
    if not np.any(disagree_mask):
        return 0.5
        
    # Calculate proportion of choices aligning with TTB on these crucial trials
    responses = data['response'].values
    matches = (responses[disagree_mask] == ttb_preds[disagree_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5360 (var=0.0087)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8393 (var=0.0102)
- pi_2: 0.1500 (var=0.0114)
- pi_3: 0.5487 (var=0.0539)
- pi_4: 0.6250 (var=0.0133)

### Experiment 2
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    diff = a_mat - b_mat
    nz = diff != 0
    has_nz = np.any(nz, axis=1)
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    
    ttb_pref = np.where(has_nz, np.where(first_diff > 0, 0, 1), -1)
    
    mask = (tally_pref != -1) & (ttb_pref != -1) & (tally_pref != ttb_pref)
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    tally_prefs = tally_pref[mask]
    
    return float(np.mean(responses == tally_prefs))
```

**Observed (real) value:** 0.5200 (var=0.0096)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8751 (var=0.0092)
- pi_1: 0.1409 (var=0.0112)
- pi_3: 0.4533 (var=0.0339)
- pi_4: 0.4938 (var=0.0156)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.vstack(data['option_a_ratings'].values)
    b_mat = np.vstack(data['option_b_ratings'].values)
    
    # Difference in features
    diff = a_mat - b_mat
    
    # Find the first discriminating cue (TTB logic)
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # WADD logic (with gamma=1)
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    wadd_diff = diff @ val
    wadd_pred = np.where(wadd_diff > 0, 0, 1)
    
    # Identify conflict trials where TTB and WADD (gamma=1) disagree
    conflict = (ttb_pred != wadd_pred) & has_diff
    
    if not conflict.any():
        return 0.5
        
    # Calculate proportion of choices on conflict trials that align with TTB
    responses = data['response'].values
    matches = (responses[conflict] == ttb_pred[conflict]).sum()
    
    return float(matches / conflict.sum())
```

**Observed (real) value:** 0.4222 (var=0.1296)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8228 (var=0.0157)
- pi_3: 0.5472 (var=0.0395)
- pi_2: 0.2175 (var=0.0070)
- pi_4: 0.6064 (var=0.0212)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    choose_a = 1.0 - data['response']
    cov = np.cov(diff, choose_a)[0, 1]
    return float(cov)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5299 (var=0.1410)
- pi_1: 0.0107 (var=0.0072)
- pi_2: 0.8751 (var=0.0369)
- pi_4: 0.1818 (var=0.0465)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    d0_ttb = []
    d3_ttb = []
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_choice = 0 if a[i] > b[i] else 1
                is_ttb = 1 if resp == ttb_choice else 0
                if i == 0:
                    d0_ttb.append(is_ttb)
                elif i == 3:
                    d3_ttb.append(is_ttb)
                break
                
    m0 = np.mean(d0_ttb) if len(d0_ttb) > 0 else 0.0
    m3 = np.mean(d3_ttb) if len(d3_ttb) > 0 else 0.0
    
    return float(m0 - m3)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4142 (var=0.0591)
- pi_3: -0.0567 (var=0.0265)
- pi_1: 0.0200 (var=0.0077)
- pi_2: -0.0208 (var=0.0085)

### Experiment 6
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 4 by their unique Option A profiles
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t1 - p_a_t4)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1950 (var=0.0591)
- pi_4: -0.0067 (var=0.0275)
- pi_1: 0.0117 (var=0.0236)
- pi_2: -0.0133 (var=0.0118)


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
  "theory": "Constant-Mixture Strategy Selection: Decision-makers evaluate options using a probabilistic mixture of a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying). Unlike previous theories that posited strategy selection depends on task difficulty or cue depth, this theory proposes that individuals have a fixed, trait-like probability of using TTB versus Tallying on any given trial, completely independent of how many cues must be searched. This constant mixture explains why empirical data shows zero difference in TTB usage across different cue depths.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Determine Take-The-Best (TTB) prediction\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner_ttb = -1\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner_ttb = 0\n            break\n        if b[j] > a[j]:\n            winner_ttb = 1\n            break\n            \n    if winner_ttb == 0:\n        p_ttb = np.array([1.0, 0.0])\n    elif winner_ttb == 1:\n        p_ttb = np.array([0.0, 1.0])\n    else:\n        p_ttb = np.array([0.5, 0.5])\n        \n    # Determine Tallying prediction using softmax\n    a_wins = float(np.sum(a > b))\n    b_wins = float(np.sum(b > a))\n    scores = np.array([a_wins, b_wins])\n    tau = float(parameters[\"tau\"])\n    \n    z = tau * (scores - np.max(scores))\n    e = np.exp(z)\n    p_tally = e / np.sum(e)\n        \n    # Probabilistic strategy switch (constant mixture, independent of depth)\n    alpha = float(parameters[\"alpha\"])\n    p_core = alpha * p_ttb + (1.0 - alpha) * p_tally\n    \n    # Independent lapse rate\n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "alpha": "[0.0, 1.0]",
    "tau": "[0.1, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this theory implements a Constant-Mixture Strategy Selection model. The previous theory (pi_4) assumed that the probability of using Take-The-Best decayed as the depth of the first discriminating cue increased. However, Experiment 5 demonstrated a 0.0 difference in TTB usage between depth 0 and depth 3, contradicting the depth-decay hypothesis. By replacing the depth-dependent decay with a fixed, individual-specific probability parameter ('alpha'), the new theory perfectly captures the 0.0 depth effect in Experiment 5 while preserving the structural symmetry needed to explain the conflict trial behavior in Experiments 1, 2, and 3."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10472,
  "candidates_token_count": 899,
  "total_token_count": 12745
}
```
