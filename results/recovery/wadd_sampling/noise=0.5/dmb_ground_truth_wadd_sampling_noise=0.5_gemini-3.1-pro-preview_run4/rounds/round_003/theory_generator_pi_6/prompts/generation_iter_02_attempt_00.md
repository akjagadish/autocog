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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a 'Leaky Competing Accumulator' or 'Parallel Cue Integration with Discounting' theory. Instead of a strict mixture of two heuristics (TTB and Tallying) or a strict sequential search, this theory should posit that all cues are processed in parallel, but the evidence provided by each cue is discounted by its rank or by the presence of conflicting higher-validity cues. This allows for a soft blending of compensatory and non-compensatory decision-making without relying on a rigid stochastic mixture, potentially capturing the moderate choice probabilities in Experiments 1 and 2 while maintaining the robust performance of integration models in Experiments 3, 4, 7, and 8.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.829)

**Description**
Strategy Mixture (Take-The-Best and Tallying with Tallying Bias): Decision-makers are heterogeneous in their strategy use, probabilistically switching between strategies. On any given choice, a subject has a probability 'p_ttb' of applying a one-reason lexicographic heuristic (Take-The-Best) and a probability '1 - p_ttb' of applying an equal-weight compensatory heuristic (Tallying). Empirical data suggests that people generally favor Tallying over Take-The-Best in these environments, so the probability of using Take-The-Best is bounded between 10% and 50%, ensuring the mixture slightly favors Tallying to better match observed aggregate choice probabilities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
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
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Tallying Prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_ttb: [0.1, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4850 (var=0.0051) vs this=0.4435 (var=0.0063)
- Experiment 2: real=0.3844 (var=0.0082) vs this=0.3558 (var=0.0123)
- Experiment 3: real=0.5667 (var=0.0123) vs this=0.6022 (var=0.0106)
- Experiment 4: real=0.5000 (var=0.0450) vs this=0.3567 (var=0.0383)
- Experiment 5: real=0.4600 (var=0.0252) vs this=0.3725 (var=0.0117)
- Experiment 6: real=0.4475 (var=0.0246) vs this=0.6181 (var=0.0121)
- Experiment 7: real=0.1619 (var=0.0032) vs this=0.1528 (var=0.0021)
- Experiment 8: real=0.0683 (var=0.0514) vs this=0.0142 (var=0.0162)


---

### `pi_5` (overall score: 0.716)

**Description**
Sequential Evidence Accumulation: Decision-makers inspect cues sequentially in order of validity, accumulating evidence for the favored option. The evidence contributed by each cue is its validity scaled by a non-linear parameter gamma. Search terminates when the absolute evidence difference reaches a threshold theta, or when all cues are exhausted. A choice is then made based on the accumulated evidence with softmax noise. This unified mechanism smoothly interpolates between Take-The-Best (low threshold), Tallying (high threshold, gamma=0), and Weighted Additive (high threshold, gamma>0).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += v[j] * diff
            if abs(E) >= theta:
                break
                
    # E > 0 means A is favored; E < 0 means B is favored
    scores = np.array([E, 0.0])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Add lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- theta: [0.0, 3.0]
- gamma: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4850 (var=0.0051) vs this=0.4838 (var=0.0461)
- Experiment 2: real=0.3844 (var=0.0082) vs this=0.3833 (var=0.0897)
- Experiment 3: real=0.5667 (var=0.0123) vs this=0.6578 (var=0.0216)
- Experiment 4: real=0.5000 (var=0.0450) vs this=0.6617 (var=0.0740)
- Experiment 5: real=0.4600 (var=0.0252) vs this=0.4733 (var=0.0382)
- Experiment 6: real=0.4475 (var=0.0246) vs this=0.5081 (var=0.0288)
- Experiment 7: real=0.1619 (var=0.0032) vs this=0.2899 (var=0.0082)
- Experiment 8: real=0.0683 (var=0.0514) vs this=0.2400 (var=0.0939)


---

### `pi_2` (overall score: 0.458)

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
- Experiment 1: real=0.4850 (var=0.0051) vs this=0.2606 (var=0.0039)
- Experiment 2: real=0.3844 (var=0.0082) vs this=0.1103 (var=0.0071)
- Experiment 3: real=0.5667 (var=0.0123) vs this=0.4739 (var=0.0063)
- Experiment 4: real=0.5000 (var=0.0450) vs this=0.1400 (var=0.0182)
- Experiment 5: real=0.4600 (var=0.0252) vs this=0.4708 (var=0.0078)
- Experiment 6: real=0.4475 (var=0.0246) vs this=0.4919 (var=0.0066)
- Experiment 7: real=0.1619 (var=0.0032) vs this=0.2686 (var=0.0030)
- Experiment 8: real=0.0683 (var=0.0514) vs this=0.0000 (var=0.0104)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4379 -> ACCEPTED
- iter 2: loss=0.1876 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.1876 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    preds = []
    for i in range(len(data)):
        p = -1
        for j in range(4):
            if a[i, j] > b[i, j]:
                p = 0
                break
            elif b[i, j] > a[i, j]:
                p = 1
                break
        preds.append(p)
        
    preds = np.array(preds)
    return float(np.mean(data['response'] == preds))
```

**Observed (real) value:** 0.4850 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: 0.7262 (var=0.0218) (Δ vs real +0.2412)
  - iter 2 (most recent): 0.5450 (var=0.0153) (Δ vs real +0.0600)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8704 (var=0.0083)
- pi_2: 0.2606 (var=0.0039)
- pi_3: 0.4531 (var=0.0128)
- pi_4: 0.4435 (var=0.0063)
- pi_5: 0.4838 (var=0.0461)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    diff = a - b
    # The first index where features differ (since validities are strictly descending)
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_winner = np.where(ttb_val == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    
    valid_mask = (a_wins != b_wins)
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    
    # Focus only on trials where Tallying and Take The Best make strictly opposite predictions
    conflict_mask = valid_mask & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_matches = np.sum(responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(ttb_matches / np.sum(conflict_mask))

```

**Observed (real) value:** 0.3844 (var=0.0082)
**Previous candidate values (this loop):**
  - iter 1: 0.6028 (var=0.0469) (Δ vs real +0.2183)
  - iter 2 (most recent): 0.5197 (var=0.0179) (Δ vs real +0.1353)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1103 (var=0.0071)
- pi_1: 0.8622 (var=0.0073)
- pi_3: 0.3056 (var=0.0217)
- pi_4: 0.3558 (var=0.0123)
- pi_5: 0.3833 (var=0.0897)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ties = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        a_w = sum(1 for x, y in zip(a, b) if x > y)
        b_w = sum(1 for x, y in zip(a, b) if y > x)
        ties.append(a_w == b_w)
        
    tie_data = data[ties]
    if len(tie_data) == 0:
        return 0.5
        
    # In tie trials, A is designed to have higher-validity features than B.
    # Tallying predicts 50% A (response == 0) because the tallies are tied.
    # WADD predicts > 50% A because A's WADD score is higher.
    return float(np.mean(tie_data['response'] == 0))
```

**Observed (real) value:** 0.5667 (var=0.0123)
**Previous candidate values (this loop):**
  - iter 1: 0.8056 (var=0.0140) (Δ vs real +0.2389)
  - iter 2 (most recent): 0.6678 (var=0.0168) (Δ vs real +0.1011)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8056 (var=0.0141)
- pi_2: 0.4739 (var=0.0063)
- pi_1: 0.8617 (var=0.0094)
- pi_4: 0.6022 (var=0.0106)
- pi_5: 0.6578 (var=0.0216)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Project option_a_ratings to string for hashability and comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1 is uniquely identified by Option A having exactly these ratings
    t1_mask = a_str == '00111'
    
    if t1_mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option B was chosen on this trial
    return float(data.loc[t1_mask, 'response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0450)
**Previous candidate values (this loop):**
  - iter 1: 0.8750 (var=0.0173) (Δ vs real +0.3750)
  - iter 2 (most recent): 0.7000 (var=0.0331) (Δ vs real +0.2000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1400 (var=0.0182)
- pi_3: 0.7300 (var=0.0663)
- pi_1: 0.8367 (var=0.0158)
- pi_4: 0.3567 (var=0.0383)
- pi_5: 0.6617 (var=0.0740)

### Experiment 5
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_trial_1 = (a_tuples == (0, 1, 1, 0)) & (b_tuples == (1, 0, 0, 1))
    is_trial_7 = (a_tuples == (1, 0, 0, 1)) & (b_tuples == (0, 1, 1, 0))
    
    wadd_choices = 0
    total_trials = 0
    
    if is_trial_1.sum() > 0:
        wadd_choices += (data.loc[is_trial_1, 'response'] == 0).sum()
        total_trials += is_trial_1.sum()
        
    if is_trial_7.sum() > 0:
        wadd_choices += (data.loc[is_trial_7, 'response'] == 1).sum()
        total_trials += is_trial_7.sum()
        
    if total_trials == 0:
        return 0.5
        
    return float(wadd_choices / total_trials)
```

**Observed (real) value:** 0.4600 (var=0.0252)
**Previous candidate values (this loop):**
  - iter 1: 0.2450 (var=0.0284) (Δ vs real -0.2150)
  - iter 2 (most recent): 0.3767 (var=0.0165) (Δ vs real -0.0833)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5608 (var=0.0263)
- pi_4: 0.3725 (var=0.0117)
- pi_1: 0.1492 (var=0.0115)
- pi_2: 0.4708 (var=0.0078)
- pi_5: 0.4733 (var=0.0382)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 0]
    # Mixture predicts A (0), WADD predicts B (1)
    t1_match = (a_str == '10010') & (data['response'] == 0)
    
    # Trial 2: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 0, 1]
    # Mixture predicts B (1), WADD predicts A (0)
    t2_match = (a_str == '01100') & (data['response'] == 1)
    
    valid_trials = (a_str == '10010') | (a_str == '01100')
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float((t1_match.sum() + t2_match.sum()) / valid_trials.sum())
```

**Observed (real) value:** 0.4475 (var=0.0246)
**Previous candidate values (this loop):**
  - iter 1: 0.6075 (var=0.0750) (Δ vs real +0.1600)
  - iter 2 (most recent): 0.6119 (var=0.0304) (Δ vs real +0.1644)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6181 (var=0.0121)
- pi_3: 0.2075 (var=0.0236)
- pi_1: 0.8588 (var=0.0118)
- pi_2: 0.4919 (var=0.0066)
- pi_5: 0.5081 (var=0.0288)

### Experiment 7
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # The most valid cue (cue 0) always discriminates in this design.
    # Determine the Take-The-Best (TTB) winner for each trial (0 for A, 1 for B).
    a_v0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_v0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_winner = (b_v0 > a_v0).astype(int)
    
    # Record whether the subject chose the TTB winner
    chose_ttb = (data['response'] == ttb_winner).astype(float)
    
    # Create a safe, hashable string identifier for each unique trial type
    trial_id = data['option_a_ratings'].apply(lambda x: "".join([str(v) for v in x])) + "_" + \
               data['option_b_ratings'].apply(lambda x: "".join([str(v) for v in x]))
               
    df_temp = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_id': trial_id,
        'chose_ttb': chose_ttb
    })
    
    # Calculate the proportion of times the TTB winner was chosen per subject, per trial type
    means = df_temp.groupby(['subject_id', 'trial_id'])['chose_ttb'].mean()
    
    # Metric: Mean absolute deviation of these choice probabilities from 0.5
    return float((means - 0.5).abs().mean())

```

**Observed (real) value:** 0.1619 (var=0.0032)
**Previous candidate values (this loop):**
  - iter 1: 0.2792 (var=0.0140) (Δ vs real +0.1173)
  - iter 2 (most recent): 0.1771 (var=0.0064) (Δ vs real +0.0152)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2899 (var=0.0082)
- pi_4: 0.1528 (var=0.0021)
- pi_1: 0.3562 (var=0.0087)
- pi_2: 0.2686 (var=0.0030)
- pi_3: 0.3166 (var=0.0126)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Project list representations to strings for safe masking
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
    t2_mask = a_str == '01110'
    # Trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
    t4_mask = a_str == '00111'
    
    # Calculate probability of choosing Option A (response == 0)
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean() if t2_mask.any() else 0.5
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean() if t4_mask.any() else 0.5
    
    # Return the difference in probability of choosing A
    return float(p_a_t2 - p_a_t4)
```

**Observed (real) value:** 0.0683 (var=0.0514)
**Previous candidate values (this loop):**
  - iter 1: 0.2550 (var=0.1015) (Δ vs real +0.1867)
  - iter 2 (most recent): 0.1467 (var=0.0429) (Δ vs real +0.0783)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0142 (var=0.0162)
- pi_5: 0.2400 (var=0.0939)
- pi_1: -0.0017 (var=0.0104)
- pi_2: 0.0000 (var=0.0104)
- pi_3: 0.5367 (var=0.0965)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Parallel Cue Integration with Rank Discounting: Decision-makers process all available cues in parallel rather than strictly sequentially, but they discount the evidence provided by each cue based on its validity rank. The weight of a cue is a function of its validity (scaled non-linearly) and an exponential decay based on its rank order. This mechanism allows for a soft blending of compensatory and non-compensatory decision-making: strong rank discounting mimics Take-The-Best, while weak discounting with varying validity sensitivity smoothly interpolates between Tallying and Weighted Additive strategies, avoiding the need for a rigid probabilistic mixture.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    discount_rate = float(parameters["discount_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for rank, j in enumerate(cue_order):
        diff = a[j] - b[j]
        weight = (val[j] ** gamma) * (discount_rate ** rank)
        E += weight * diff
        
    scores = np.array([E, 0.0])
    
    # Softmax for choice probability
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Apply lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- discount_rate: [0.0, 1.0]
- gamma: [0.0, 2.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.6]
- validities: validities

`rationale`: Following the critic's advice, the upper bounds for the scaling parameters `beta` and `gamma` have been reduced (to 5.0 and 2.0, respectively), and the upper bound for `epsilon` was slightly increased to 0.6. This minimal edit softens the predictions, preventing the decision rule from behaving too deterministically and over-weighting accumulated evidence differences, thereby better matching the moderate choice probabilities observed in the human data while keeping the core rank-discounting logic intact.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory implements the prescribed Parallel Cue Integration with Rank Discounting mechanism. However, across almost all experiments, the simulated metrics are significantly more extreme than the observed human data. For instance, in Experiment 4, the simulated value is 0.875 compared to the real value of 0.500; in Experiment 3, it is 0.805 compared to 0.566. This suggests that the decision rule is behaving too deterministically and over-weighting the accumulated evidence differences.
Rationale: While the mechanism successfully captures the qualitative blending of strategies, the parameter ranges allow for excessively deterministic behavior. To soften the predictions and better match the moderate choice probabilities observed in the data, restrict the upper bounds on the scaling parameters. Specifically, reduce the maximum for `beta` (e.g., from 20.0 to 5.0) and `gamma` (e.g., from 5.0 to 2.0). You may also want to allow `epsilon` to play a stronger role. Keep the core rank-discounting logic exactly as is, but tune the parameter ranges to prevent the model from overshooting.

**Outcome of this advice:** iter 1 candidate loss=0.4379 -> iter 2 candidate loss=0.1876 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The previous suggestion to reduce the upper bounds of the scaling parameters `beta` and `gamma` was highly successful, cutting the aggregate loss by more than half (0.4379 down to 0.1876). However, the model still consistently overshoots the moderate choice probabilities observed in the human data. For example, in Experiment 4 the simulated metric is 0.700 compared to the real 0.500, and in Experiment 6 it is 0.612 compared to the real 0.448. The model is still slightly too deterministic and overly sensitive to cue validity differences.
Rationale: Since the previous reduction in `beta` and `gamma` significantly improved the fit, continue in this exact direction. Reduce the upper bound of `beta` further (e.g., from 5.0 down to 2.5 or 3.0) and reduce the upper bound of `gamma` (e.g., from 2.0 down to 1.0 or 1.5). This will further soften the evidence accumulation and bring the simulated metrics closer to the human ground truth. Keep the core Parallel Cue Integration with Rank Discounting mechanism exactly as it is.

**Outcome of this advice:** iter 2 candidate loss=0.1876 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Parallel Cue Integration with Rank Discounting: Decision-makers process all available cues in parallel rather than strictly sequentially, but they discount the evidence provided by each cue based on its validity rank. The weight of a cue is a function of its validity (scaled non-linearly) and an exponential decay based on its rank order. This mechanism allows for a soft blending of compensatory and non-compensatory decision-making: strong rank discounting mimics Take-The-Best, while weak discounting with varying validity sensitivity smoothly interpolates between Tallying and Weighted Additive strategies, avoiding the need for a rigid probabilistic mixture.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    discount_rate = float(parameters[\"discount_rate\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Order cues by descending validity\n    cue_order = np.argsort(-val, kind=\"stable\")\n    \n    E = 0.0\n    for rank, j in enumerate(cue_order):\n        diff = a[j] - b[j]\n        weight = (val[j] ** gamma) * (discount_rate ** rank)\n        E += weight * diff\n        \n    scores = np.array([E, 0.0])\n    \n    # Softmax for choice probability\n    z = beta * (scores - np.max(scores))\n    e_vals = np.exp(z)\n    p = e_vals / np.sum(e_vals)\n    \n    # Apply lapse rate\n    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "discount_rate": "[0.0, 1.0]",
    "gamma": "[0.0, 1.5]",
    "beta": "[0.1, 3.0]",
    "epsilon": "[0.0, 0.6]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, the upper bounds for the scaling parameters `beta` and `gamma` have been further reduced (from 5.0 to 3.0, and from 2.0 to 1.5, respectively). This edit continues to soften the evidence accumulation and choice determinism, aiming to better match the moderate choice probabilities observed in the human data while keeping the core rank-discounting logic exactly intact."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14009,
  "candidates_token_count": 715,
  "total_token_count": 15057
}
```
