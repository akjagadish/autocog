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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a 'Strategy Mixture' or 'Probabilistic Strategy Selection' theory. Instead of assuming all subjects use a single strategy (either pure TTB or pure WADD), this theory should posit that the population consists of a mixture of decision-makers: some rely on a non-compensatory heuristic (TTB) while others use a compensatory strategy (like Tallying or WADD), or individuals probabilistically switch strategies trial-by-trial based on choice difficulty. A mixture model can capture the ~50% alignment rates in conflict trials (by averaging across the population) while potentially accommodating the flat choice probability curves in specific boundary designs like Experiment 2 if the compensatory subset's behavior cancels out or if strategy selection is context-dependent.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.649)

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


---

### `pi_1` (overall score: 0.058)

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


---

### `pi_2` (overall score: 0.000)

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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5360 (var=0.0087) vs this=0.1500 (var=0.0114)
- Experiment 2: real=0.5200 (var=0.0096) vs this=0.8751 (var=0.0092)
- Experiment 3: real=0.4222 (var=0.1296) vs this=0.2175 (var=0.0070)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.8751 (var=0.0369)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2048 -> ACCEPTED
- iter 2: loss=0.2492 -> REJECTED
- iter 3: loss=0.1620 -> ACCEPTED
- iter 4: loss=0.1699 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.1620 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
**Previous candidate values (this loop):**
  - iter 1: 0.7430 (var=0.0140) (Δ vs real +0.2070)
  - iter 2: 0.6637 (var=0.0213) (Δ vs real +0.1277)
  - iter 3: 0.6223 (var=0.0172) (Δ vs real +0.0863)
  - iter 4 (most recent): 0.6670 (var=0.0173) (Δ vs real +0.1310)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8393 (var=0.0102)
- pi_2: 0.1500 (var=0.0114)
- pi_3: 0.5487 (var=0.0539)

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
**Previous candidate values (this loop):**
  - iter 1: 0.4200 (var=0.0202) (Δ vs real -0.1000)
  - iter 2: 0.3956 (var=0.0269) (Δ vs real -0.1244)
  - iter 3: 0.5084 (var=0.0178) (Δ vs real -0.0116)
  - iter 4 (most recent): 0.4511 (var=0.0210) (Δ vs real -0.0689)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8751 (var=0.0092)
- pi_1: 0.1409 (var=0.0112)
- pi_3: 0.4533 (var=0.0339)

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
**Previous candidate values (this loop):**
  - iter 1: 0.6447 (var=0.0153) (Δ vs real +0.2225)
  - iter 2: 0.6272 (var=0.0251) (Δ vs real +0.2050)
  - iter 3: 0.5864 (var=0.0093) (Δ vs real +0.1642)
  - iter 4 (most recent): 0.5878 (var=0.0156) (Δ vs real +0.1656)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8228 (var=0.0157)
- pi_3: 0.5472 (var=0.0395)
- pi_2: 0.2175 (var=0.0070)

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
**Previous candidate values (this loop):**
  - iter 1: 0.0133 (var=0.0055) (Δ vs real +0.0133)
  - iter 2: 0.2654 (var=0.0580) (Δ vs real +0.2654)
  - iter 3: 0.2063 (var=0.0346) (Δ vs real +0.2063)
  - iter 4 (most recent): 0.1069 (var=0.0252) (Δ vs real +0.1069)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5299 (var=0.1410)
- pi_1: 0.0107 (var=0.0072)
- pi_2: 0.8751 (var=0.0369)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

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

`rationale`: Building on the successful Iter 1 base, we parameterize the depth-dependent mixture rule. Instead of forcing 100% TTB at depth 0, we introduce an 'alpha' parameter to set the baseline P(TTB) at depth 0, and decay it for deeper cues using 'gamma'. We also replace the deterministic Tallying with a softmax version controlled by inverse temperature 'tau'. This allows the model to reduce its overprediction of TTB alignment in Experiments 1 and 3 by mixing in some Tallying even on easy trials, while minimizing the covariance penalty in Experiment 4 by retaining the depth-based structure.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model implements a context-dependent probabilistic strategy selection between TTB and Tallying, using the depth of the first discriminating cue to determine the mixture. This was ACCEPTED by the loop because it successfully captures the 0 covariance in Experiment 4 (by forcing 100% TTB when depth=0). However, it overpredicts TTB alignment in Experiments 1 and 3 (0.74 and 0.64 vs. observed ~0.53 and 0.42). The assumption that depth=0 implies 100% TTB use is too strong, as many conflict trials in Exp 1 and 3 occur at depth=0, yet humans still show significant compensatory behavior.
Rationale: To improve the fit on Experiments 1, 2, and 3 without losing Experiment 4, revise the strategy mixture. Instead of Tallying, use WADD (weighted additive using the provided validities as weights) as the compensatory strategy, and make the mixture proportion a simple fixed population parameter `alpha` (range [0.0, 1.0]) rather than depending on cue depth. In Exp 4, if WADD weights the first cue heavily enough, WADD and TTB will often agree, naturally producing the flat choice curve (0 covariance) without needing to force 100% TTB. Add a temperature parameter `tau` (range [0.1, 5.0]) to the WADD predictions (using softmax) to allow for smooth probabilistic choices, while keeping TTB deterministic. The final prediction should be `(1-epsilon) * (alpha * P_ttb + (1-alpha) * P_wadd) + epsilon * [0.5, 0.5]`.

**Outcome of this advice:** iter 1 candidate loss=0.2048 -> iter 2 candidate loss=0.2492 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The previous attempt (Iter 2) to replace the depth-dependent TTB/Tallying mixture with a fixed TTB/WADD mixture was REJECTED by the gate. The loss increased from 0.2048 to 0.2492. The primary reason for the failure is Experiment 4: the fixed mixture with WADD caused the covariance metric to spike to 0.2654 (real data is 0.0000). In Exp 4, the compensatory strategy's predictions vary across trials as the sum of features changes, which correlates with the difference in sums, breaking the zero covariance. Iter 1 succeeded on Exp 4 by forcing 100% TTB at depth=0, but overpredicted TTB alignment in Exps 1 and 3.
Rationale: Since the fixed mixture failed, we must build on the ACCEPTED Iter 1 base (depth-dependent mixture of TTB and Tallying). To reduce the overpredicted TTB alignment in Exps 1 and 3 without completely ruining Exp 4, parameterize the depth-dependent rule. Instead of forcing P(TTB) = 1.0 at depth=0, make P(TTB) at depth=0 a free parameter `alpha` (range [0.5, 1.0]), and let P(TTB) decay for deeper cues: e.g., P(TTB) = alpha * (gamma ^ depth), where `gamma` is in [0.0, 1.0]. Also, introduce a softmax temperature `tau` for the Tallying strategy to soften its predictions, rather than making it deterministic. This provides enough flexibility to lower TTB alignment in Exps 1/3 while minimizing the covariance penalty in Exp 4.

**Outcome of this advice:** iter 2 candidate loss=0.2492 -> iter 3 candidate loss=0.1620 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate was ACCEPTED by the gate, achieving a new lowest loss of 0.1620. The introduction of the `alpha` parameter successfully reduced the overprediction of TTB alignment in Experiments 1 and 3, and perfectly matched Experiment 2. However, this came at the cost of Experiment 4, where the covariance spiked from ~0.01 to 0.2063 (real data is 0.0000). This occurs because `alpha < 1.0` allows the Tallying strategy to be used even at depth=0, and Tallying's predictions correlate strongly with the linearly changing sum-difference in Experiment 4's design.
Rationale: To fix the covariance spike in Experiment 4 without losing the improvements in Experiments 1-3, we need to make the mixture probability sensitive to the cue's validity, not just its depth. Modify the probabilistic switch so that `prob_ttb` depends on the validity of the first discriminating cue. For example, use `prob_ttb = alpha * val[first_disc_idx] * (gamma ** first_disc_idx)`. In Experiment 4, the first discriminating cue is always the most valid cue, so this change will increase P(TTB) specifically for those trials (flattening the curve and reducing covariance), while still allowing Tallying to mix in during Experiments 1 and 3 when less valid cues are the first to discriminate. Keep the rest of the mechanism intact.

**Outcome of this advice:** iter 3 candidate loss=0.1620 -> iter 4 candidate loss=0.1699 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate (Iteration 4) was REJECTED by the gate because the attempt to scale the TTB mixture probability by the first cue's validity (`disc_val`) worsened the fit on Experiments 1 and 2, while still failing to fully eliminate the covariance in Experiment 4. We must build on the last ACCEPTED base (Iteration 3), which used a simple depth-dependent mixture (`prob_ttb = alpha * (gamma ** first_disc_idx)`) between TTB and an unweighted Tallying strategy.
Rationale: Return to the ACCEPTED Iteration 3 mechanism: a probabilistic mixture between TTB and Tallying where `prob_ttb = alpha * (gamma ** first_disc_idx)`. To fix the covariance spike in Experiment 4 without ruining Experiments 1-3, we should modify the Tallying strategy rather than the mixture rule. In Experiment 4, unweighted Tallying strongly prefers Option B as B accumulates more features, creating a correlation with the sum-difference. Modify Tallying to be a 'Rank-Discounted Tally': instead of a simple sum of wins, weight each cue's win by `delta ** i`, where `i` is the cue's rank (0-indexed) in the validity order, and `delta` is a new parameter in [0.0, 1.0]. When `delta` is small, Tallying heavily discounts deeper cues, preventing the accumulation of low-validity features in Exp 4 from overriding the first cue, thus flattening the choice curve. When `delta` is closer to 1, it behaves like standard Tallying, preserving the compensatory fits for Exps 1 and 3. Keep `alpha`, `gamma`, `tau`, and `epsilon` from Iteration 3.

**Outcome of this advice:** iter 4 candidate loss=0.1699 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Context-Dependent Probabilistic Strategy Selection with Rank-Discounted Tally. Decision-makers probabilistically switch between a non-compensatory heuristic (TTB) and a compensatory strategy (Rank-Discounted Tallying) based on choice difficulty (depth of the first discriminating cue). The probability of using TTB decays for deeper cues. When using the compensatory strategy, individuals tally features but discount the contribution of lower-ranked cues exponentially, preventing an accumulation of weak evidence from overpowering higher-validity cues in extreme cases.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Determine TTB prediction and the depth of the first discriminating cue\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    first_disc_idx = -1\n    winner_ttb = -1\n    for i, j in enumerate(cue_order):\n        if a[j] > b[j]:\n            winner_ttb = 0\n            first_disc_idx = i\n            break\n        if b[j] > a[j]:\n            winner_ttb = 1\n            first_disc_idx = i\n            break\n            \n    if winner_ttb == 0:\n        p_ttb = np.array([1.0, 0.0])\n    elif winner_ttb == 1:\n        p_ttb = np.array([0.0, 1.0])\n    else:\n        p_ttb = np.array([0.5, 0.5])\n        \n    # Determine Rank-Discounted Tallying prediction using softmax\n    delta = float(parameters[\"delta\"])\n    a_wins = 0.0\n    b_wins = 0.0\n    for i, j in enumerate(cue_order):\n        weight = delta ** i\n        if a[j] > b[j]:\n            a_wins += weight\n        elif b[j] > a[j]:\n            b_wins += weight\n            \n    scores = np.array([a_wins, b_wins])\n    tau = float(parameters[\"tau\"])\n    \n    z = tau * (scores - np.max(scores))\n    e = np.exp(z)\n    p_tally = e / np.sum(e)\n        \n    # Probabilistic strategy switch based on choice difficulty (depth of first discriminating cue)\n    gamma = float(parameters[\"gamma\"])\n    alpha = float(parameters[\"alpha\"])\n    if first_disc_idx == -1:\n        prob_ttb = 0.0\n    else:\n        prob_ttb = alpha * (gamma ** first_disc_idx)\n        \n    p_core = prob_ttb * p_ttb + (1.0 - prob_ttb) * p_tally\n    \n    # Independent lapse rate\n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "alpha": "[0.5, 1.0]",
    "gamma": "[0.0, 1.0]",
    "tau": "[0.1, 5.0]",
    "delta": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Building on the successful Iteration 3 base, we modify the Tallying strategy to be a 'Rank-Discounted Tally' where the weight of a cue's win is exponentially discounted by its rank using a new parameter `delta`. This addresses the covariance spike in Experiment 4 by preventing the accumulation of many low-validity features from overriding the first cue, while preserving the compensatory behavior needed to fit Experiments 1 and 3 when `delta` takes higher values. The depth-dependent strategy mixture remains unchanged."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12539,
  "candidates_token_count": 1068,
  "total_token_count": 14867
}
```
