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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace the Logistic Choice Model with a 'Strategy Selection / Adaptive Toolbox' theory. Instead of assuming a single compensatory mechanism with non-linear weights, this new theory should posit that decision-makers probabilistically select between different simple heuristics on a trial-by-trial or subject-by-subject basis. For example, a mixture model where agents predominantly use Take-The-Best but have a certain probability of falling back to 'Tallying' (simply counting the number of positive features without weighting them by validity) when confidence is low. This preserves the non-compensatory nature of the data while allowing for slight deviations from strict TTB.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.983)

**Description**
Strict Take-The-Best: Decision-makers process information lexicographically, ranking features by their validity and choosing deterministically based on the first discriminating feature, with a constant probability of random guessing (lapse).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(-validities, kind='mergesort')
    
    epsilon = float(parameters["epsilon"])
    
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8667 (var=0.0063) vs this=0.8733 (var=0.0099)
- Experiment 2: real=0.8767 (var=0.0106) vs this=0.8900 (var=0.0137)
- Experiment 3: real=0.8644 (var=0.0213) vs this=0.8811 (var=0.0109)
- Experiment 4: real=0.1800 (var=0.0309) vs this=0.1600 (var=0.0200)
- Experiment 5: real=0.8700 (var=0.0128) vs this=0.8706 (var=0.0072)
- Experiment 6: real=0.8250 (var=0.0157) vs this=0.8650 (var=0.0065)
- Experiment 7: real=-0.0073 (var=0.0102) vs this=0.0040 (var=0.0114)
- Experiment 8: real=-0.0300 (var=0.0180) vs this=-0.0008 (var=0.0144)


---

### `pi_3` (overall score: 0.960)

**Description**
Take-The-Best (TTB) assumes that decision-makers do not integrate all information or simply count features. Instead, they rank features by their subjective or instructed validity and compare options lexicographically. They stop at the first feature that discriminates between the two options and choose the one with the higher value on that feature. If all features tie, they guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-validities, kind='mergesort')
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    if score_a > score_b:
        p_core = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8667 (var=0.0063) vs this=0.8450 (var=0.0131)
- Experiment 2: real=0.8767 (var=0.0106) vs this=0.8683 (var=0.0128)
- Experiment 3: real=0.8644 (var=0.0213) vs this=0.8889 (var=0.0114)
- Experiment 4: real=0.1800 (var=0.0309) vs this=0.1633 (var=0.0283)
- Experiment 5: real=0.8700 (var=0.0128) vs this=0.8781 (var=0.0057)
- Experiment 6: real=0.8250 (var=0.0157) vs this=0.8694 (var=0.0069)
- Experiment 7: real=-0.0073 (var=0.0102) vs this=0.0100 (var=0.0104)
- Experiment 8: real=-0.0300 (var=0.0180) vs this=-0.0175 (var=0.0144)


---

### `pi_5` (overall score: 0.702)

**Description**
Logistic Choice Model (Weighted Additive with Non-linear Validity Scaling): Decision-makers integrate all available information by computing a weighted sum of the features for each option. However, the subjective weights placed on these features are a non-linear scaling of their objective validities, governed by an exponent parameter (gamma). This allows the compensatory process to approximate non-compensatory heuristics when gamma is large, as the highest-validity features will dominate the weighted sum. The choice is made probabilistically by passing these weighted sums through a softmax function, modulated by an inverse temperature parameter (beta) to capture choice determinism, along with a baseline lapse rate (epsilon) for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities to compute subjective weights
    weights = validities ** gamma
    
    # Compute weighted sum of features using subjective weights
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    p_core = np.exp(z)
    p_core /= p_core.sum()
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.2]
- gamma: [0.1, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8667 (var=0.0063) vs this=0.8667 (var=0.0102)
- Experiment 2: real=0.8767 (var=0.0106) vs this=0.9017 (var=0.0119)
- Experiment 3: real=0.8644 (var=0.0213) vs this=0.9111 (var=0.0094)
- Experiment 4: real=0.1800 (var=0.0309) vs this=0.1167 (var=0.0381)
- Experiment 5: real=0.8700 (var=0.0128) vs this=0.8700 (var=0.0059)
- Experiment 6: real=0.8250 (var=0.0157) vs this=0.9017 (var=0.0039)
- Experiment 7: real=-0.0073 (var=0.0102) vs this=0.0570 (var=0.0153)
- Experiment 8: real=-0.0300 (var=0.0180) vs this=0.1925 (var=0.0170)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2044 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2044 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_favored_choices = 0
    tied_trials = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == (0, 1, 0, 1) and b == (1, 1, 0, 0):
            wadd_favored = 1
        elif a == (1, 0, 1, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        elif a == (1, 1, 1, 0) and b == (0, 1, 1, 1):
            wadd_favored = 0
        elif a == (1, 1, 0, 1) and b == (1, 1, 1, 0):
            wadd_favored = 1
        else:
            continue
            
        tied_trials += 1
        if row['response'] == wadd_favored:
            wadd_favored_choices += 1
            
    if tied_trials == 0:
        return 0.5
    return float(wadd_favored_choices / tied_trials)
```

**Observed (real) value:** 0.8667 (var=0.0063)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7467 (var=0.0137) (Δ vs real -0.1200)
**Other theories' values on this metric (for reference):**
- pi_1: 0.5083 (var=0.0129)
- pi_2: 0.6717 (var=0.0328)
- pi_3: 0.8450 (var=0.0131)
- pi_4: 0.8733 (var=0.0099)
- pi_5: 0.8667 (var=0.0102)

### Experiment 2
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.8, 0.85, 0.55])
    wadd_consistent = 0
    tally_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying predicts a tie
        if a_wins == b_wins:
            score_a = np.sum(a * validities)
            score_b = np.sum(b * validities)
            
            if score_a > score_b:
                wadd_choice = 0
            elif score_b > score_a:
                wadd_choice = 1
            else:
                continue
                
            if row['response'] == wadd_choice:
                wadd_consistent += 1
            tally_ties += 1
            
    if tally_ties == 0:
        return 0.5
    return float(wadd_consistent / tally_ties)
```

**Observed (real) value:** 0.8767 (var=0.0106)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7617 (var=0.0175) (Δ vs real -0.1150)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5183 (var=0.0651)
- pi_1: 0.4733 (var=0.0257)
- pi_3: 0.8683 (var=0.0128)
- pi_4: 0.8900 (var=0.0137)
- pi_5: 0.9017 (var=0.0119)

### Experiment 3
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
    # TTB chooses A (f0 discriminates). WADD might choose B if w2 is high and w0/w3 are low.
    mask_t12 = (a_tuples == (1, 0, 0, 1)) & (b_tuples == (0, 0, 1, 0))
    
    # Trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses A (f2 discriminates). WADD might choose B if w1 > w2.
    mask_t15 = (a_tuples == (1, 0, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    # Trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
    # TTB chooses B (f0 discriminates). WADD might choose A if w2 is high and w0 is low.
    mask_t4 = (a_tuples == (0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0))
    
    ttb_aligned = 0
    total = 0
    
    if mask_t12.sum() > 0:
        ttb_aligned += (data.loc[mask_t12, 'response'] == 0).sum()
        total += mask_t12.sum()
        
    if mask_t15.sum() > 0:
        ttb_aligned += (data.loc[mask_t15, 'response'] == 0).sum()
        total += mask_t15.sum()
        
    if mask_t4.sum() > 0:
        ttb_aligned += (data.loc[mask_t4, 'response'] == 1).sum()
        total += mask_t4.sum()
        
    if total == 0:
        return 0.5
        
    return float(ttb_aligned) / total
```

**Observed (real) value:** 0.8644 (var=0.0213)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7978 (var=0.0122) (Δ vs real -0.0667)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8889 (var=0.0114)
- pi_2: 0.5856 (var=0.0243)
- pi_1: 0.6067 (var=0.0148)
- pi_4: 0.8811 (var=0.0109)
- pi_5: 0.9111 (var=0.0094)

### Experiment 4
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 11 is the most discriminative trial:
    # Option A: [1, 1, 0, 0] vs Option B: [0, 1, 1, 1]
    # TTB looks at the highest validity feature (F0: 0.95), where A=1 and B=0, so TTB strictly prefers A (response 0).
    # WADD computes a weighted sum. Since F1 cancels out, WADD compares w0*0.95 (for A) against w2*0.58 + w3*0.55 (for B).
    # Because the weights w0, w2, w3 are independently drawn from U(0,1), the sum w2*0.58 + w3*0.55 is frequently greater than w0*0.95.
    # Thus, WADD will choose B (response 1) more than half the time, whereas TTB will choose B almost never (only on lapses).
    
    a_target = [1, 1, 0, 0]
    b_target = [0, 1, 1, 1]
    
    mask = (data['option_a_ratings'].apply(lambda x: list(x) == a_target) & 
            data['option_b_ratings'].apply(lambda x: list(x) == b_target))
            
    if not mask.any():
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1800 (var=0.0309)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2767 (var=0.0429) (Δ vs real +0.0967)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6033 (var=0.1021)
- pi_3: 0.1633 (var=0.0283)
- pi_1: 0.8633 (var=0.0219)
- pi_4: 0.1600 (var=0.0200)
- pi_5: 0.1167 (var=0.0381)

### Experiment 5
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.55, 0.65, 0.55])
    order = np.argsort(-validities, kind='mergesort')
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = -1
        for idx in order:
            if a[idx] > b[idx]:
                ttb_choice = 0
                break
            elif b[idx] > a[idx]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            if row['response'] == ttb_choice:
                match_count += 1
            total += 1
            
    if total == 0:
        return 0.5
    return match_count / total
```

**Observed (real) value:** 0.8700 (var=0.0128)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8544 (var=0.0053) (Δ vs real -0.0156)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8781 (var=0.0057)
- pi_4: 0.8706 (var=0.0072)
- pi_1: 0.6744 (var=0.0036)
- pi_2: 0.6769 (var=0.0163)
- pi_5: 0.8700 (var=0.0059)

### Experiment 6
**Design**
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def choice_matches_ttb(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        for idx in [0, 2, 1, 3]:
            if a[idx] > b[idx]:
                return 1 if response == 0 else 0
            elif b[idx] > a[idx]:
                return 1 if response == 1 else 0
        return 0.5
    
    matches = data.apply(choice_matches_ttb, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.8250 (var=0.0157)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8300 (var=0.0057) (Δ vs real +0.0050)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8650 (var=0.0065)
- pi_3: 0.8694 (var=0.0069)
- pi_1: 0.7454 (var=0.0054)
- pi_2: 0.7217 (var=0.0153)
- pi_5: 0.9017 (var=0.0039)

### Experiment 7
**Design**
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    # Trials where WADD has a very small difference in weighted sums
    low_diff_trials = {
        ((0, 1, 0, 0), (0, 0, 1, 0)): 1,  # TTB chooses B (response 1)
        ((0, 1, 1, 0), (0, 0, 1, 1)): 0   # TTB chooses A (response 0)
    }
    
    # Trials where WADD has a very large difference in weighted sums
    high_diff_trials = {
        ((1, 0, 0, 0), (1, 1, 1, 0)): 1,  # TTB chooses B
        ((1, 1, 1, 1), (0, 1, 1, 0)): 0,  # TTB chooses A
        ((1, 0, 1, 0), (1, 1, 1, 1)): 1,  # TTB chooses B
        ((1, 1, 1, 0), (0, 0, 1, 0)): 0   # TTB chooses A
    }
    
    def get_group(row):
        pair = (row['a_tup'], row['b_tup'])
        if pair in low_diff_trials:
            return 'low', low_diff_trials[pair]
        if pair in high_diff_trials:
            return 'high', high_diff_trials[pair]
        return 'none', -1
        
    groups = data.apply(get_group, axis=1)
    data['group'] = [g[0] for g in groups]
    data['ttb'] = [g[1] for g in groups]
    
    low_data = data[data['group'] == 'low']
    high_data = data[data['group'] == 'high']
    
    if len(low_data) == 0 or len(high_data) == 0:
        return 0.0
        
    low_acc = (low_data['response'] == low_data['ttb']).mean()
    high_acc = (high_data['response'] == high_data['ttb']).mean()
    
    return float(high_acc - low_acc)
```

**Observed (real) value:** -0.0073 (var=0.0102)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0923 (var=0.0159) (Δ vs real +0.0997)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0100 (var=0.0104)
- pi_5: 0.0570 (var=0.0153)
- pi_1: 0.3607 (var=0.0291)
- pi_2: 0.2397 (var=0.0292)
- pi_4: 0.0040 (var=0.0114)

### Experiment 8
**Design**
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert lists to strings for fast comparison
    data_a = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    data_b = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Strong trials: large difference in weighted sums under WADD
    strong_0 = (data_a == "1011") & (data_b == "0000")
    strong_1 = ((data_a == "0100") & (data_b == "1111")) | ((data_a == "0010") & (data_b == "1111"))
    strong_2 = (data_a == "1111") & (data_b == "0001")
    
    # Weak trials: very small difference in weighted sums under WADD
    weak_0 = ((data_a == "0110") & (data_b == "0101")) | ((data_a == "1110") & (data_b == "1011"))
    
    strong_correct = 0
    strong_total = 0
    
    if strong_0.any():
        strong_correct += (data.loc[strong_0, 'response'] == 0).sum()
        strong_total += strong_0.sum()
    if strong_1.any():
        strong_correct += (data.loc[strong_1, 'response'] == 1).sum()
        strong_total += strong_1.sum()
    if strong_2.any():
        strong_correct += (data.loc[strong_2, 'response'] == 0).sum()
        strong_total += strong_2.sum()
        
    weak_correct = 0
    weak_total = 0
    if weak_0.any():
        weak_correct += (data.loc[weak_0, 'response'] == 0).sum()
        weak_total += weak_0.sum()
        
    strong_acc = strong_correct / strong_total if strong_total > 0 else 0.0
    weak_acc = weak_correct / weak_total if weak_total > 0 else 0.0
    
    return float(strong_acc - weak_acc)
```

**Observed (real) value:** -0.0300 (var=0.0180)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1083 (var=0.0167) (Δ vs real +0.1383)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1925 (var=0.0170)
- pi_3: -0.0175 (var=0.0144)
- pi_1: 0.3917 (var=0.0440)
- pi_2: 0.3425 (var=0.0904)
- pi_4: -0.0008 (var=0.0144)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Adaptive Toolbox (Strategy Selection): Decision-makers do not rely on a single, fixed compensatory or non-compensatory mechanism. Instead, they probabilistically select from a 'toolbox' of simple heuristics on a trial-by-trial basis. In this domain, agents predominantly use the lexicographic Take-The-Best (TTB) heuristic, but with some probability they fall back to a simple 'Tallying' heuristic (counting the number of positive features without weighting them by validity). This mixture preserves the largely non-compensatory choice patterns observed in the data while naturally accounting for specific systematic deviations from strict TTB that a uniform lapse rate cannot explain.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    epsilon = float(parameters["epsilon"])
    p_tally = float(parameters["p_tally"])
    
    # Take-The-Best (TTB) Mechanism
    order = np.argsort(-validities, kind='mergesort')
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Mechanism (Equal Weights)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Probabilistic Strategy Selection
    p_mix = (1.0 - p_tally) * p_ttb + p_tally * p_tally_arr
    
    # Incorporate baseline lapse rate (random guessing)
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- epsilon: [0.0, 0.5]
- p_tally: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's feedback, this theory abandons the single-mechanism Logistic Choice Model (Weighted Additive) in favor of the Adaptive Toolbox framework. It implements a probabilistic mixture model where decision-makers predominantly rely on Take-The-Best (TTB) but occasionally switch to Tallying (unweighted feature summation). This explains the slight deviations from strict TTB observed in experiments (e.g., Experiment 4, where the real value is 0.18, higher than what pure TTB with lapses predicts) by introducing structured, heuristic-driven noise rather than relying purely on random lapses or fully compensatory integration.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate theory successfully implemented the arbiter's recommended Adaptive Toolbox family, mixing Take-The-Best (TTB) and Tallying. This was accepted by the gate and improved the aggregate loss. However, the unconditional mixing of Tallying introduces a systematic artifact in Experiments 7 and 8. In these experiments, the metric measures the difference in accuracy between 'high difference' and 'low difference' trials. Because Tallying tends to agree with TTB on high-difference trials but ties on low-difference trials, an unconditional `p_tally` artificially inflates accuracy on high-diff trials relative to low-diff trials, yielding a difference of ~0.10 (whereas human data shows ~0.0). Additionally, the unconditional mix dilutes overall accuracy in Experiments 1-3.
Rationale: To fix the artifacts in Experiments 7 and 8 while maintaining the prescribed Adaptive Toolbox family, operationalize the arbiter's suggestion to fall back to Tallying 'when confidence is low'. Instead of a fixed `p_tally` across all trials, make the probability of using Tallying inversely proportional to the validity of the feature that TTB uses to discriminate. For instance, if TTB terminates on a highly valid feature (e.g., validity > 0.8), the probability of falling back to Tallying should be near zero; if it terminates on a weak feature, the fallback probability should be higher. Introduce a parameter (like a validity threshold or a scaling factor) to control this confidence-dependent fallback. This minor adjustment within the family will preserve TTB's strictness on high-validity trials (fixing Exps 1-3, 7, 8) while allowing Tallying to explain variance on lower-confidence trials.

**Outcome of this advice:** iter 1 candidate loss=0.2044 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Adaptive Toolbox with Confidence-Dependent Strategy Selection: Decision-makers predominantly use the Take-The-Best (TTB) heuristic, but probabilistically fall back to Tallying only when their confidence is low—specifically, when the validity of the first discriminating feature falls below a subjective threshold.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n\n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    p_tally = float(parameters[\"p_tally\"])\n    threshold = float(parameters[\"threshold\"])\n    \n    # Take-The-Best (TTB) Mechanism\n    order = np.argsort(-validities, kind='mergesort')\n    p_ttb = np.array([0.5, 0.5])\n    discrim_validity = 0.5\n    for idx in order:\n        if a[idx] > b[idx]:\n            p_ttb = np.array([1.0, 0.0])\n            discrim_validity = validities[idx]\n            break\n        elif b[idx] > a[idx]:\n            p_ttb = np.array([0.0, 1.0])\n            discrim_validity = validities[idx]\n            break\n            \n    # Tallying Mechanism (Equal Weights)\n    sum_a = np.sum(a)\n    sum_b = np.sum(b)\n    if sum_a > sum_b:\n        p_tally_arr = np.array([1.0, 0.0])\n    elif sum_b > sum_a:\n        p_tally_arr = np.array([0.0, 1.0])\n    else:\n        p_tally_arr = np.array([0.5, 0.5])\n        \n    # Confidence-dependent fallback to Tallying\n    if discrim_validity < threshold:\n        actual_p_tally = p_tally\n    else:\n        actual_p_tally = 0.0\n        \n    # Probabilistic Strategy Selection\n    p_mix = (1.0 - actual_p_tally) * p_ttb + actual_p_tally * p_tally_arr\n    \n    # Incorporate baseline lapse rate (random guessing)\n    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "epsilon": "[0.0, 0.5]",
    "p_tally": "[0.0, 0.5]",
    "threshold": "[0.5, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, we introduce a confidence-dependent fallback mechanism. Instead of unconditionally mixing Tallying on every trial, decision-makers only fall back to Tallying if the validity of the discriminating feature found by Take-The-Best (TTB) is below a certain subjective `threshold`. If TTB terminates on a highly valid feature, the decision is made with high confidence using pure TTB, effectively reducing the Tallying probability to zero. This preserves TTB's strictness on high-validity trials (resolving the artifacts in Experiments 7 and 8 and maintaining accuracy in Experiments 1-3) while still allowing Tallying to explain systematic deviations on lower-confidence trials."
}
```
```

## Usage

```json
{
  "prompt_token_count": 15762,
  "candidates_token_count": 979,
  "total_token_count": 19344
}
```
