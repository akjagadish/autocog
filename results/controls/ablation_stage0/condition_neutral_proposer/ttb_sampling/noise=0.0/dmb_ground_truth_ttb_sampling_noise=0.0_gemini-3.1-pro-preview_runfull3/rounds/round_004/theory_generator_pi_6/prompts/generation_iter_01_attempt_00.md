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

Propose a 'Probabilistic Cue Search' (or Probabilistic TTB) theory. Instead of a deterministic search order based strictly on validities, assume that the order in which cues are evaluated is probabilistic, weighted by their validities (e.g., a softmax over validities to pick the first cue to evaluate). Once a cue is selected, if it discriminates, it determines the choice. This maintains the fast-and-frugal, non-compensatory nature of TTB that fits the data so well, but allows for systematic deviations that depend on how close the cue validities are to one another, providing a more principled source of variability than a flat lapse rate.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.979)

**Description**
People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1625 (var=0.0238) vs this=0.1525 (var=0.0155)
- Experiment 2: real=0.8217 (var=0.0115) vs this=0.8517 (var=0.0129)
- Experiment 3: real=0.1200 (var=0.0109) vs this=0.1575 (var=0.0131)
- Experiment 4: real=0.8567 (var=0.0164) vs this=0.8517 (var=0.0167)
- Experiment 5: real=0.8833 (var=0.0156) vs this=0.8257 (var=0.0098)
- Experiment 6: real=0.3138 (var=0.0086) vs this=0.2778 (var=0.0048)
- Experiment 7: real=-0.0033 (var=0.0067) vs this=-0.0256 (var=0.0049)
- Experiment 8: real=0.1543 (var=0.0099) vs this=0.1414 (var=0.0113)
- Experiment 9: real=0.8250 (var=0.0092) vs this=0.8503 (var=0.0087)
- Experiment 10: real=0.1573 (var=0.0116) vs this=0.1631 (var=0.0121)


---

### `pi_4` (overall score: 0.899)

**Description**
Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially. However, unlike the strict deterministic order of Take-The-Best, the search order is probabilistic. The probability of retrieving and evaluating a cue is proportional to an exponential function of its validity (a Plackett-Luce choice process). Because non-discriminating cues simply prompt the decision maker to continue searching, the probability that the decision is ultimately determined by a specific discriminating cue is exactly proportional to its weight relative to all other discriminating cues. This model elegantly bridges random search (Tallying) and deterministic search (TTB) through a single search-temperature parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating cues
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # Due to the Plackett-Luce property of sampling without replacement,
        # the probability that a particular discriminating cue is encountered 
        # before any other discriminating cue is simply its proportional weight
        # among the set of discriminating cues. Non-discriminating cues drop out.
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        # Calculate weights for discriminating cues with numerical stability
        weights = np.zeros_like(validities)
        weights[discrim_mask] = np.exp(gamma * (validities[discrim_mask] - max_v))
        
        w_A = np.sum(weights[diff > 0])
        w_B = np.sum(weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- gamma: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1625 (var=0.0238) vs this=0.1512 (var=0.0213)
- Experiment 2: real=0.8217 (var=0.0115) vs this=0.8258 (var=0.0142)
- Experiment 3: real=0.1200 (var=0.0109) vs this=0.1750 (var=0.0157)
- Experiment 4: real=0.8567 (var=0.0164) vs this=0.8383 (var=0.0134)
- Experiment 5: real=0.8833 (var=0.0156) vs this=0.7540 (var=0.0179)
- Experiment 6: real=0.3138 (var=0.0086) vs this=0.3367 (var=0.0090)
- Experiment 7: real=-0.0033 (var=0.0067) vs this=0.0683 (var=0.0122)
- Experiment 8: real=0.1543 (var=0.0099) vs this=0.2632 (var=0.0287)
- Experiment 9: real=0.8250 (var=0.0092) vs this=0.6878 (var=0.0232)
- Experiment 10: real=0.1573 (var=0.0116) vs this=0.3009 (var=0.0271)


---

### `pi_5` (overall score: 0.843)

**Description**
Two-Stage Confidence-Threshold Strategy Selection: Decision-makers default to the fast and frugal Take-The-Best (TTB) heuristic, evaluating options based solely on the most valid discriminating cue. However, if the validity of this top discriminating cue falls below a subjective confidence threshold, the decision-maker deems the single-cue evidence insufficient and falls back to a compensatory Weighted Additive (WADD) strategy that integrates all available cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    confidence_threshold = float(parameters["confidence_threshold"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    scores = np.zeros(2)
    if np.any(discrim_mask):
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        if max_v >= confidence_threshold:
            # Strategy 1: Take-The-Best (TTB)
            top_idx = np.where((discrim_mask) & (validities == max_v))[0][0]
            if stim[0, top_idx] > stim[1, top_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
        else:
            # Strategy 2: Weighted Additive (WADD) fallback
            wadd_scores = stim @ validities
            if wadd_scores[0] > wadd_scores[1]:
                scores[0] = 1.0
            elif wadd_scores[1] > wadd_scores[0]:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
- confidence_threshold: [0.5, 0.8]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1625 (var=0.0238) vs this=0.1812 (var=0.0183)
- Experiment 2: real=0.8217 (var=0.0115) vs this=0.8517 (var=0.0092)
- Experiment 3: real=0.1200 (var=0.0109) vs this=0.1775 (var=0.0165)
- Experiment 4: real=0.8567 (var=0.0164) vs this=0.8250 (var=0.0226)
- Experiment 5: real=0.8833 (var=0.0156) vs this=0.8263 (var=0.0127)
- Experiment 6: real=0.3138 (var=0.0086) vs this=0.3253 (var=0.0072)
- Experiment 7: real=-0.0033 (var=0.0067) vs this=-0.0139 (var=0.0059)
- Experiment 8: real=0.1543 (var=0.0099) vs this=0.1714 (var=0.0121)
- Experiment 9: real=0.8250 (var=0.0092) vs this=0.5619 (var=0.0643)
- Experiment 10: real=0.1573 (var=0.0116) vs this=0.4102 (var=0.0669)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1697 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1697 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    score = 0
    count = 0
    for _, row in data.iterrows():
        A = tuple(row['option_a_ratings'])
        B = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers B (3 to 2 wins), WADD prefers A (1.85 to 1.65)
        if A == (1, 1, 0, 0, 0) and B == (0, 0, 1, 1, 1):
            score += (resp == 1)
            count += 1
        # Trial 8: Tallying prefers A (2 to 1 wins), WADD prefers B (0.95 to 1.15)
        elif A == (0, 0, 1, 1, 0) and B == (1, 0, 0, 0, 0):
            score += (resp == 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.1625 (var=0.0238)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1925 (var=0.0331) (Δ vs real +0.0300)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8525 (var=0.0142)
- pi_2: 0.4650 (var=0.0585)
- pi_2_1: 0.4863 (var=0.0035)
- pi_3: 0.1525 (var=0.0155)
- pi_4: 0.1512 (var=0.0213)
- pi_5: 0.1812 (var=0.0183)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    trial_1 = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    trial_2 = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 1, 0, 0, 0))
    
    t1_wadd_aligned = (data['response'] == 0) & trial_1
    t2_wadd_aligned = (data['response'] == 1) & trial_2
    
    wadd_choices = t1_wadd_aligned.sum() + t2_wadd_aligned.sum()
    total_dissociation = trial_1.sum() + trial_2.sum()
    
    return float(wadd_choices / total_dissociation) if total_dissociation > 0 else 0.5
```

**Observed (real) value:** 0.8217 (var=0.0115)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7942 (var=0.0264) (Δ vs real -0.0275)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5617 (var=0.0991)
- pi_1: 0.1533 (var=0.0142)
- pi_2_1: 0.9658 (var=0.0033)
- pi_3: 0.8517 (var=0.0129)
- pi_4: 0.8258 (var=0.0142)
- pi_5: 0.8517 (var=0.0092)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_choices = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_str = ''.join(map(str, a))
        b_str = ''.join(map(str, b))
        
        is_t1 = (a_str == '11000' and b_str == '00111')
        is_t2 = (a_str == '00111' and b_str == '11000')
        
        if is_t1:
            tally_choices.append(1 if row['response'] == 1 else 0)
        elif is_t2:
            tally_choices.append(1 if row['response'] == 0 else 0)
            
    if not tally_choices:
        return 0.5
        
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.1200 (var=0.0109)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1975 (var=0.0198) (Δ vs real +0.0775)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8308 (var=0.0159)
- pi_2_1: 0.0933 (var=0.0101)
- pi_2: 0.4717 (var=0.0916)
- pi_3: 0.1575 (var=0.0131)
- pi_4: 0.1750 (var=0.0157)
- pi_5: 0.1775 (var=0.0165)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target = [1, 1, 0, 0, 0, 0]
    competitor = [0, 0, 1, 1, 1, 0]
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return (a == target and b == competitor) or (a == competitor and b == target)
        
    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    def chose_target(row):
        a = list(row['option_a_ratings'])
        chose_a = (row['response'] == 0)
        return 1.0 if (a == target) == chose_a else 0.0
        
    return float(subset.apply(chose_target, axis=1).mean())
```

**Observed (real) value:** 0.8567 (var=0.0164)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.8150 (var=0.0309) (Δ vs real -0.0417)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.9167 (var=0.0125)
- pi_1: 0.1300 (var=0.0112)
- pi_2: 0.5783 (var=0.0840)
- pi_3: 0.8517 (var=0.0167)
- pi_4: 0.8383 (var=0.0134)
- pi_5: 0.8250 (var=0.0226)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    ttb_match = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'], dtype=float)
        b = np.array(row['option_b_ratings'], dtype=float)
        
        # TTB choice
        ttb_choice = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice == -1:
            continue
            
        # WADD choice
        wadd_a = np.dot(a, validities)
        wadd_b = np.dot(b, validities)
        if wadd_a == wadd_b:
            continue
        wadd_choice = 0 if wadd_a > wadd_b else 1
        
        # Only consider trials where the two models fundamentally disagree
        if ttb_choice != wadd_choice:
            disagree_count += 1
            if row['response'] == ttb_choice:
                ttb_match += 1
                
    if disagree_count == 0:
        return 0.5
    return float(ttb_match / disagree_count)
```

**Observed (real) value:** 0.8833 (var=0.0156)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6260 (var=0.0296) (Δ vs real -0.2573)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8257 (var=0.0098)
- pi_2_1: 0.0233 (var=0.0006)
- pi_1: 0.1103 (var=0.0068)
- pi_2: 0.2860 (var=0.0176)
- pi_4: 0.7540 (var=0.0179)
- pi_5: 0.8263 (var=0.0127)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.65, 0.95, 0.55, 0.85, 0.75])
    
    def is_wadd_choice(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        pred = 0 if score_a > score_b else 1
        return row['response'] == pred

    return float(data.apply(is_wadd_choice, axis=1).mean())
```

**Observed (real) value:** 0.3138 (var=0.0086)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4316 (var=0.0124) (Δ vs real +0.1178)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.9707 (var=0.0005)
- pi_3: 0.2778 (var=0.0048)
- pi_1: 0.8429 (var=0.0124)
- pi_2: 0.7264 (var=0.0102)
- pi_4: 0.3367 (var=0.0090)
- pi_5: 0.3253 (var=0.0072)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    pair_str = a_str + "_" + b_str
    
    ttb_choices = {
        '10000_01000': 0,
        '10000_01100': 0,
        '10000_01110': 0,
        '10000_01111': 0,
        '11000_10100': 0,
        '11000_10111': 0,
        '11100_11010': 0,
        '11100_11011': 0,
        '11111_00000': 0,
        '01111_10000': 1
    }
    
    low_opposing = {
        '10000_01000',
        '11000_10100',
        '11100_11010',
        '11111_00000'
    }
    
    high_opposing = {
        '10000_01110',
        '10000_01111',
        '11000_10111',
        '01111_10000'
    }
    
    data['ttb_choice'] = pair_str.map(ttb_choices)
    data['is_ttb_match'] = (data['response'] == data['ttb_choice']).astype(float)
    
    is_low = pair_str.isin(low_opposing)
    is_high = pair_str.isin(high_opposing)
    
    low_match = data.loc[is_low, 'is_ttb_match'].mean()
    high_match = data.loc[is_high, 'is_ttb_match'].mean()
    
    if pd.isna(low_match) or pd.isna(high_match):
        return 0.0
        
    return float(low_match - high_match)
```

**Observed (real) value:** -0.0033 (var=0.0067)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0761 (var=0.0153) (Δ vs real +0.0794)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0256 (var=0.0049)
- pi_4: 0.0683 (var=0.0122)
- pi_1: 0.4650 (var=0.0182)
- pi_2: 0.4117 (var=0.0282)
- pi_2_1: 0.8700 (var=0.0089)
- pi_5: -0.0139 (var=0.0059)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    compensatory_choices = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        
        diff = np.array(A) - np.array(B)
        discrim_mask = diff != 0
        if not np.any(discrim_mask):
            continue
            
        # Top cue is the first non-zero in diff (since validities are monotonically decreasing)
        top_idx = np.where(discrim_mask)[0][0]
        top_favors_A = (diff[top_idx] > 0)
        
        count_A = np.sum(diff > 0)
        count_B = np.sum(diff < 0)
        
        # Identify trials where the top cue opposes the simple majority of discriminating cues
        if top_favors_A and count_B > count_A:
            chosen_majority = (row['response'] == 1)
            compensatory_choices.append(chosen_majority)
        elif (not top_favors_A) and count_A > count_B:
            chosen_majority = (row['response'] == 0)
            compensatory_choices.append(chosen_majority)
            
    if not compensatory_choices:
        return 0.0
        
    return float(np.mean(compensatory_choices))
```

**Observed (real) value:** 0.1543 (var=0.0099)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3589 (var=0.0339) (Δ vs real +0.2046)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2632 (var=0.0287)
- pi_3: 0.1414 (var=0.0113)
- pi_1: 0.8496 (var=0.0134)
- pi_2: 0.7243 (var=0.0172)
- pi_2_1: 0.9718 (var=0.0011)
- pi_5: 0.1714 (var=0.0121)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.75, 0.65, 0.6, 0.55, 0.5])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    ttb_match = 0
    valid_count = 0
    
    for i in range(len(responses)):
        d = a_ratings[i] - b_ratings[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) == 0:
            continue
            
        top_idx = nonzero[0]
        ttb_pred = 0 if d[top_idx] > 0 else 1
        max_v = validities[top_idx]
        
        wadd_a = np.sum(a_ratings[i] * validities)
        wadd_b = np.sum(b_ratings[i] * validities)
        if wadd_a == wadd_b:
            continue
        wadd_pred = 0 if wadd_a > wadd_b else 1
        
        # Focus on trials where TTB and WADD disagree, and top cue validity < 0.9
        if ttb_pred != wadd_pred and max_v < 0.9:
            valid_count += 1
            if responses[i] == ttb_pred:
                ttb_match += 1
                
    if valid_count == 0:
        return 0.5
        
    return float(ttb_match / valid_count)
```

**Observed (real) value:** 0.8250 (var=0.0092)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5537 (var=0.0132) (Δ vs real -0.2712)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8503 (var=0.0087)
- pi_5: 0.5619 (var=0.0643)
- pi_1: 0.1484 (var=0.0145)
- pi_2: 0.2506 (var=0.0142)
- pi_2_1: 0.0294 (var=0.0007)
- pi_4: 0.6878 (var=0.0232)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where option A is favored only by a single cue (which is not the most valid cue overall)
    # These are trials where TTB picks A, but WADD picks B, and the max validity is <= 0.75.
    is_target = data['option_a_ratings'].apply(lambda x: sum(x) == 1 and x[0] == 0)
    if not is_target.any():
        return 0.0
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.1573 (var=0.0116)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4249 (var=0.0228) (Δ vs real +0.2676)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4102 (var=0.0669)
- pi_3: 0.1631 (var=0.0121)
- pi_1: 0.8231 (var=0.0147)
- pi_2: 0.7004 (var=0.0260)
- pi_2_1: 0.9747 (var=0.0010)
- pi_4: 0.3009 (var=0.0271)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially, but the search order is probabilistic rather than deterministic. The probability of evaluating a cue is weighted by a softmax over its validity. The first cue encountered that discriminates between the options completely determines the choice. This maintains the fast-and-frugal, non-compensatory nature of Take-The-Best, but allows for systematic deviations that depend on how close the cue validities are to one another.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # The probability of a discriminating cue being the *first* discriminating cue encountered
        # in a sampling-without-replacement process is proportional to its weight relative to all
        # other discriminating cues.
        discrim_validities = validities[discrim_mask]
        z = gamma * discrim_validities
        z -= np.max(z) # for numerical stability
        weights = np.exp(z)
        
        full_weights = np.zeros_like(validities)
        full_weights[discrim_mask] = weights
        
        w_A = np.sum(full_weights[diff > 0])
        w_B = np.sum(full_weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- gamma: [0.0, 25.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Implements the Probabilistic Cue Search (PCS) mechanism as requested by the arbiter. It replaces the confidence-threshold fallback strategy with a probabilistic search order based on a softmax over cue validities. The first discriminating cue determines the choice. This provides a principled source of variability sensitive to the relative closeness of cue validities, maintaining the fast-and-frugal nature of TTB while elegantly capturing systematic deviations from deterministic search.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Probabilistic Cue Search (PCS) model was accepted and successfully introduces a principled source of variability. However, it systematically underfits the degree of determinism seen in human choices on several experiments. In Experiments 5, 8, 9, and 10, the human data strongly aligns with the deterministic Take-The-Best (TTB) predictions (e.g., Exp 9 real=0.825 vs cand=0.553; Exp 10 real=0.157 vs cand=0.424). The current softmax formulation over cue validities with gamma capped at 25.0 does not allow the model to become deterministic enough when cue validities are moderately separated.
Rationale: The candidate is conceptually on the right track within the prescribed PCS family, but the temperature/scaling parameter `gamma` is too restricted. Expand the parameter range for `gamma` significantly (e.g., [0.0, 100.0] or [0.0, 200.0]) so the softmax can approximate deterministic TTB when required to fit the high TTB-match rates in Experiments 5 and 9. Retain the same mechanism otherwise.

**Outcome of this advice:** iter 1 candidate loss=0.1697 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially, but the search order is probabilistic rather than deterministic. The probability of evaluating a cue is weighted by a softmax over its validity. The first cue encountered that discriminates between the options completely determines the choice. This maintains the fast-and-frugal, non-compensatory nature of Take-The-Best, but allows for systematic deviations that depend on how close the cue validities are to one another.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    diff = stim[0] - stim[1]\n    discrim_mask = diff != 0\n    \n    if not np.any(discrim_mask):\n        p_core = np.array([0.5, 0.5])\n    else:\n        # The probability of a discriminating cue being the *first* discriminating cue encountered\n        # in a sampling-without-replacement process is proportional to its weight relative to all\n        # other discriminating cues.\n        discrim_validities = validities[discrim_mask]\n        z = gamma * discrim_validities\n        z -= np.max(z) # for numerical stability\n        weights = np.exp(z)\n        \n        full_weights = np.zeros_like(validities)\n        full_weights[discrim_mask] = weights\n        \n        w_A = np.sum(full_weights[diff > 0])\n        w_B = np.sum(full_weights[diff < 0])\n        \n        total_w = w_A + w_B\n        if total_w == 0:\n            p_core = np.array([0.5, 0.5])\n        else:\n            p_core = np.array([w_A / total_w, w_B / total_w])\n            \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "gamma": "[0.0, 200.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Expanded the parameter range for `gamma` to `[0.0, 200.0]` as requested by the critic. This allows the probabilistic cue search to approximate deterministic Take-The-Best (TTB) more closely when required by the data, fitting the high determinism observed in several experiments while maintaining the principled variability of the PCS mechanism."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16938,
  "candidates_token_count": 788,
  "total_token_count": 18143
}
```
