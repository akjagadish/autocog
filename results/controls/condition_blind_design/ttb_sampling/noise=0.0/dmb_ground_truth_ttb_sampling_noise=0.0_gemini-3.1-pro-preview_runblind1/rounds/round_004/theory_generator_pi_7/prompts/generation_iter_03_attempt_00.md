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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Probabilistic Lexicographic Model' (or Probabilistic Feature Selection). Instead of a deterministic fallback to a completely different strategy like Tallying, assume that subjects always use a lexicographic stopping rule, but the *order* in which they search through features is probabilistic and weighted by the features' validities. This preserves the core TTB mechanism that fits the data well, while allowing for the slight behavioral deviations from pure deterministic TTB by occasionally checking a lower-validity feature first.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.977)

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
- Experiment 9: real=0.0200 (var=0.0496) vs this=-0.0267 (var=0.0315)
- Experiment 10: real=0.1200 (var=0.0100) vs this=0.1033 (var=0.0232)


---

### `pi_3` (overall score: 0.962)

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
- Experiment 9: real=0.0200 (var=0.0496) vs this=0.0533 (var=0.0316)
- Experiment 10: real=0.1200 (var=0.0100) vs this=0.1100 (var=0.0207)


---

### `pi_6` (overall score: 0.895)

**Description**
Adaptive Toolbox with Confidence-Dependent Strategy Selection: Decision-makers predominantly use the Take-The-Best (TTB) heuristic, but probabilistically fall back to Tallying only when their confidence is low—specifically, when the validity of the first discriminating feature falls below a subjective threshold.

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
    threshold = float(parameters["threshold"])
    
    # Take-The-Best (TTB) Mechanism
    order = np.argsort(-validities, kind='mergesort')
    p_ttb = np.array([0.5, 0.5])
    discrim_validity = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_validity = validities[idx]
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_validity = validities[idx]
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
        
    # Confidence-dependent fallback to Tallying
    if discrim_validity < threshold:
        actual_p_tally = p_tally
    else:
        actual_p_tally = 0.0
        
    # Probabilistic Strategy Selection
    p_mix = (1.0 - actual_p_tally) * p_ttb + actual_p_tally * p_tally_arr
    
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
- threshold: [0.5, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8667 (var=0.0063) vs this=0.8375 (var=0.0109)
- Experiment 2: real=0.8767 (var=0.0106) vs this=0.8800 (var=0.0164)
- Experiment 3: real=0.8644 (var=0.0213) vs this=0.8544 (var=0.0091)
- Experiment 4: real=0.1800 (var=0.0309) vs this=0.1300 (var=0.0292)
- Experiment 5: real=0.8700 (var=0.0128) vs this=0.8415 (var=0.0052)
- Experiment 6: real=0.8250 (var=0.0157) vs this=0.8694 (var=0.0075)
- Experiment 7: real=-0.0073 (var=0.0102) vs this=0.0337 (var=0.0151)
- Experiment 8: real=-0.0300 (var=0.0180) vs this=0.0750 (var=0.0184)
- Experiment 9: real=0.0200 (var=0.0496) vs this=0.0700 (var=0.0390)
- Experiment 10: real=0.1200 (var=0.0100) vs this=0.1833 (var=0.0403)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3929 -> ACCEPTED
- iter 2: loss=0.2076 -> ACCEPTED
- iter 3: loss=0.2219 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.2076 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.7650 (var=0.0149) (Δ vs real -0.1017)
  - iter 2: 0.9450 (var=0.0031) (Δ vs real +0.0783)
  - iter 3 (most recent): 0.9350 (var=0.0031) (Δ vs real +0.0683)
**Other theories' values on this metric (for reference):**
- pi_1: 0.5083 (var=0.0129)
- pi_2: 0.6717 (var=0.0328)
- pi_3: 0.8450 (var=0.0131)
- pi_4: 0.8733 (var=0.0099)
- pi_5: 0.8667 (var=0.0102)
- pi_6: 0.8375 (var=0.0109)

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
  - iter 1: 0.6933 (var=0.0254) (Δ vs real -0.1833)
  - iter 2: 0.9233 (var=0.0061) (Δ vs real +0.0467)
  - iter 3 (most recent): 0.9467 (var=0.0052) (Δ vs real +0.0700)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5183 (var=0.0651)
- pi_1: 0.4733 (var=0.0257)
- pi_3: 0.8683 (var=0.0128)
- pi_4: 0.8900 (var=0.0137)
- pi_5: 0.9017 (var=0.0119)
- pi_6: 0.8800 (var=0.0164)

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
  - iter 1: 0.7056 (var=0.0172) (Δ vs real -0.1589)
  - iter 2: 0.9300 (var=0.0057) (Δ vs real +0.0656)
  - iter 3 (most recent): 0.9456 (var=0.0033) (Δ vs real +0.0811)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8889 (var=0.0114)
- pi_2: 0.5856 (var=0.0243)
- pi_1: 0.6067 (var=0.0148)
- pi_4: 0.8811 (var=0.0109)
- pi_5: 0.9111 (var=0.0094)
- pi_6: 0.8544 (var=0.0091)

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
  - iter 1: 0.2633 (var=0.0445) (Δ vs real +0.0833)
  - iter 2: 0.0433 (var=0.0098) (Δ vs real -0.1367)
  - iter 3 (most recent): 0.0833 (var=0.0114) (Δ vs real -0.0967)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6033 (var=0.1021)
- pi_3: 0.1633 (var=0.0283)
- pi_1: 0.8633 (var=0.0219)
- pi_4: 0.1600 (var=0.0200)
- pi_5: 0.1167 (var=0.0381)
- pi_6: 0.1300 (var=0.0292)

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
  - iter 1: 0.7735 (var=0.0091) (Δ vs real -0.0965)
  - iter 2: 0.9348 (var=0.0010) (Δ vs real +0.0648)
  - iter 3 (most recent): 0.9513 (var=0.0014) (Δ vs real +0.0813)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8781 (var=0.0057)
- pi_4: 0.8706 (var=0.0072)
- pi_1: 0.6744 (var=0.0036)
- pi_2: 0.6769 (var=0.0163)
- pi_5: 0.8700 (var=0.0059)
- pi_6: 0.8415 (var=0.0052)

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
  - iter 1: 0.7579 (var=0.0040) (Δ vs real -0.0671)
  - iter 2: 0.8962 (var=0.0026) (Δ vs real +0.0713)
  - iter 3 (most recent): 0.9417 (var=0.0012) (Δ vs real +0.1167)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8650 (var=0.0065)
- pi_3: 0.8694 (var=0.0069)
- pi_1: 0.7454 (var=0.0054)
- pi_2: 0.7217 (var=0.0153)
- pi_5: 0.9017 (var=0.0039)
- pi_6: 0.8694 (var=0.0075)

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
  - iter 1: 0.2113 (var=0.0336) (Δ vs real +0.2187)
  - iter 2: 0.0077 (var=0.0068) (Δ vs real +0.0150)
  - iter 3 (most recent): -0.0023 (var=0.0057) (Δ vs real +0.0050)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0100 (var=0.0104)
- pi_5: 0.0570 (var=0.0153)
- pi_1: 0.3607 (var=0.0291)
- pi_2: 0.2397 (var=0.0292)
- pi_4: 0.0040 (var=0.0114)
- pi_6: 0.0337 (var=0.0151)

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
  - iter 1: 0.2042 (var=0.0229) (Δ vs real +0.2342)
  - iter 2: 0.0650 (var=0.0105) (Δ vs real +0.0950)
  - iter 3 (most recent): 0.0108 (var=0.0044) (Δ vs real +0.0408)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1925 (var=0.0170)
- pi_3: -0.0175 (var=0.0144)
- pi_1: 0.3917 (var=0.0440)
- pi_2: 0.3425 (var=0.0904)
- pi_4: -0.0008 (var=0.0144)
- pi_6: 0.0750 (var=0.0184)

### Experiment 9
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert option ratings to string for easy matching
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 13: A=[0, 1, 0, 1], B=[0, 0, 1, 0]
    # Both TTB and Tallying predict A.
    t13_mask = (data['A_str'] == '0101') & (data['B_str'] == '0010')
    
    # Trial 12: A=[0, 1, 0, 1], B=[0, 0, 1, 1]
    # TTB predicts A, but Tallying predicts a Tie.
    t12_mask = (data['A_str'] == '0101') & (data['B_str'] == '0011')
    
    # response == 0 means choice A
    p_a_t13 = (data[t13_mask]['response'] == 0).mean()
    p_a_t12 = (data[t12_mask]['response'] == 0).mean()
    
    if pd.isna(p_a_t13) or pd.isna(p_a_t12):
        return 0.0
        
    return float(p_a_t13 - p_a_t12)
```

**Observed (real) value:** 0.0200 (var=0.0496)
**Previous candidate values (this loop):**
  - iter 1: 0.0533 (var=0.0683) (Δ vs real +0.0333)
  - iter 2: 0.0100 (var=0.0271) (Δ vs real -0.0100)
  - iter 3 (most recent): -0.0300 (var=0.0174) (Δ vs real -0.0500)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0533 (var=0.0316)
- pi_6: 0.0700 (var=0.0390)
- pi_1: 0.3767 (var=0.0509)
- pi_2: 0.1400 (var=0.0871)
- pi_4: -0.0267 (var=0.0315)
- pi_5: 0.0633 (var=0.0743)

### Experiment 10
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_trial_3 = (
        (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0)) &
        (data['option_b_ratings'].apply(tuple) == (1, 0, 1, 1))
    )
    if is_trial_3.sum() == 0:
        return 0.0
    return float(data[is_trial_3]['response'].mean())

```

**Observed (real) value:** 0.1200 (var=0.0100)
**Previous candidate values (this loop):**
  - iter 1: 0.5300 (var=0.0374) (Δ vs real +0.4100)
  - iter 2: 0.4967 (var=0.0472) (Δ vs real +0.3767)
  - iter 3 (most recent): 0.5000 (var=0.0500) (Δ vs real +0.3800)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1833 (var=0.0403)
- pi_3: 0.1100 (var=0.0207)
- pi_1: 0.8567 (var=0.0322)
- pi_2: 0.6200 (var=0.0789)
- pi_4: 0.1033 (var=0.0232)
- pi_5: 0.7133 (var=0.0723)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Probabilistic Lexicographic Search: Decision-makers consistently employ a lexicographic stopping rule (stopping at the first discriminating feature), but the order in which they search through features is probabilistic rather than strictly deterministic. The probability of checking a feature is weighted by its validity, meaning higher-validity features are overwhelmingly more likely to be searched first, but lower-validity features are occasionally encountered first.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between the two options
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, guess randomly
        p_core = np.array([0.5, 0.5])
    else:
        # Calculate the probability of each discriminating feature being the FIRST one encountered
        # using a softmax over the validities of the discriminating features.
        discrim_validities = validities[discrim]
        logits = gamma * discrim_validities
        
        # Numerically stable softmax
        logits -= np.max(logits)
        w_discrim = np.exp(logits)
        probs_discrim = w_discrim / np.sum(w_discrim)
        
        p_a = 0.0
        p_b = 0.0
        
        discrim_indices = np.where(discrim)[0]
        for idx, p_feat in zip(discrim_indices, probs_discrim):
            if a[idx] > b[idx]:
                p_a += p_feat
            elif b[idx] > a[idx]:
                p_b += p_feat
                
        p_core = np.array([p_a, p_b])
        
    # Incorporate baseline lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [10.0, 100.0]
- epsilon: [0.0, 0.2]
- validities: validities

`rationale`: Following the critic's feedback, the softmax temperature parameter 'gamma' range has been increased to [10.0, 100.0], and the lapse rate 'epsilon' range has been restricted to [0.0, 0.2]. This minimal edit ensures that the model more closely approximates deterministic Take-The-Best, as high-validity features will be heavily favored in the probabilistic search order, significantly reducing the excessively frequent deviations from TTB that caused the previous iteration to underfit the empirical data.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model successfully implements the Probabilistic Feature Selection mechanism prescribed by the arbiter. However, it significantly underfits the empirical data compared to prior deterministic TTB models (e.g., on Experiment 10, the real value is 0.12, but the candidate predicts 0.53; on Experiment 2, real is 0.88, candidate is 0.69). The deviations from deterministic TTB are too frequent, suggesting that the probability of selecting a lower-validity feature first is too high.
Rationale: The candidate was accepted as the first in the loop, but its fit is too noisy because the softmax temperature `gamma` is too low, making the lexicographic search order too random. Adjust the parameter ranges to make the model approximate deterministic TTB more closely: increase the range of `gamma` (e.g., [10.0, 50.0] or [10.0, 100.0]) so that the highest-validity discriminating feature is heavily favored, and keep `epsilon` small (e.g., [0.0, 0.2]). This will preserve the probabilistic framework while restoring the strong predictive accuracy of the core TTB mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.3929 -> iter 2 candidate loss=0.2076 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate successfully implements the Probabilistic Feature Selection mechanism and improved significantly over the previous iteration, as reflected by the accepted loss of 0.2076. By increasing `gamma`, the model approximates deterministic TTB much better in most experiments (e.g., Exp 1, Exp 2, Exp 3). However, it still fails significantly on Experiment 10, predicting roughly random choice (~0.50) whereas the empirical data shows a strong preference for Option A (0.12). This likely occurs because the raw validities of the discriminating features in this experiment are very close to each other. When raw validities are close, a softmax over them (even with high gamma) can distribute probabilities too evenly, causing the model to lose the strict lexicographic preference that humans exhibit here.
Rationale: To fix the underfitting on Experiment 10 while remaining strictly within the prescribed Probabilistic Lexicographic family, modify the `predict` function to compute the softmax logits using the log-odds of the validities rather than the raw validities: `logits = gamma * np.log(discrim_validities / (1.0 - discrim_validities + 1e-6))`. Alternatively, you could base the softmax on the ordinal ranks of the validities. Transforming the validities into log-odds stretches the differences between competing features (especially at the high end), ensuring that the highest-validity feature strongly dominates the probabilistic search order even when numerical validities are close. Keep the current parameter ranges for `gamma` and `epsilon`.

**Outcome of this advice:** iter 2 candidate loss=0.2076 -> iter 3 candidate loss=0.2219 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate model attempted to solve the underfitting on Experiment 10 by applying a log-odds transformation to the validities before the softmax. However, this candidate was REJECTED by the accept-gate, as its overall loss (0.2219) was worse than the running-best base from Iteration 2 (0.2076). The log-odds scaling failed to properly separate the probabilities in Experiment 10 and slightly degraded performance on other experiments. We must therefore abandon the log-odds approach and try a different in-family adjustment on top of the Iteration 2 base to fix Experiment 10.
Rationale: Since the log-odds transformation was rejected, we will build on the Iteration 2 base (which uses raw validities in the softmax). To fix the underfitting on Experiment 10 (where the model predicts ~0.50 but the data is 0.12), the model needs a stronger winner-take-all dynamic when validities are close. Instead of transforming the validities, significantly increase the range of the temperature parameter `gamma`. Change the `gamma` parameter range to `[50.0, 500.0]`. This will allow the model to learn a much sharper softmax distribution over the raw validities, ensuring that the highest-validity discriminating feature strictly dominates the search order even when numerical differences are small, without introducing new nonlinear transformations that might distort other experiments.

**Outcome of this advice:** iter 3 candidate loss=0.2219 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Probabilistic Lexicographic Search: Decision-makers consistently employ a lexicographic stopping rule (stopping at the first discriminating feature), but the order in which they search through features is probabilistic rather than strictly deterministic. The probability of checking a feature is weighted by its validity, meaning higher-validity features are overwhelmingly more likely to be searched first, but lower-validity features are occasionally encountered first.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n\n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Identify which features discriminate between the two options\n    discrim = (a != b)\n    \n    if not np.any(discrim):\n        # If no features discriminate, guess randomly\n        p_core = np.array([0.5, 0.5])\n    else:\n        # Calculate the probability of each discriminating feature being the FIRST one encountered\n        # using a softmax over the validities of the discriminating features.\n        discrim_validities = validities[discrim]\n        logits = gamma * discrim_validities\n        \n        # Numerically stable softmax\n        logits -= np.max(logits)\n        w_discrim = np.exp(logits)\n        probs_discrim = w_discrim / np.sum(w_discrim)\n        \n        p_a = 0.0\n        p_b = 0.0\n        \n        discrim_indices = np.where(discrim)[0]\n        for idx, p_feat in zip(discrim_indices, probs_discrim):\n            if a[idx] > b[idx]:\n                p_a += p_feat\n            elif b[idx] > a[idx]:\n                p_b += p_feat\n                \n        p_core = np.array([p_a, p_b])\n        \n    # Incorporate baseline lapse rate (random guessing)\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "gamma": "[50.0, 500.0]",
    "epsilon": "[0.0, 0.2]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, we abandon the rejected log-odds transformation and return to using the raw validities in the softmax. To address the underfitting on Experiment 10 where validities are very close, we significantly increase the maximum value of the temperature parameter 'gamma' to [50.0, 500.0]. This allows the model to learn a much sharper distribution over the raw validities, ensuring the highest-validity discriminating feature strongly dominates the probabilistic search order even when numerical differences are small."
}
```
```

## Usage

```json
{
  "prompt_token_count": 19653,
  "candidates_token_count": 829,
  "total_token_count": 21265
}
```
