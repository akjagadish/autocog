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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a 'Probabilistic Heuristic Mixture' theory. Instead of assuming any form of Compensatory/WADD integration, propose that on each trial, decision-makers probabilistically sample a single simple heuristic (e.g., Take The Best, Tallying, or random guessing) from a fixed distribution. When heuristics agree, the choice is deterministic; when they conflict, the mixture naturally averages out to ~0.5 choice probabilities without requiring an explicit 'conflict detection' mechanism or computing WADD as a backup (which Theory 2 assumes). This provides a computationally simpler, purely non-compensatory alternative to Theory 2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.951)

**Description**
Conflict-Induced Guessing with Evidence Threshold: Decision-makers evaluate options using simple heuristics (Take The Best and Tallying) but also monitor the overall Weighted Additive (WADD) evidence. When the heuristics make strict opposing predictions, OR when the overall WADD evidence difference between the options is too small to confidently discriminate, the decision-maker experiences uncertainty and resorts to random guessing. This captures the pervasive ~0.5 choice probabilities and low variance across both strict heuristic conflict trials and trials with nominally agreeing heuristics but weak overall evidence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Determine Take The Best (TTB) winner
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Determine Tallying winner
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Compute WADD difference
    wadd_a = np.sum(a * val)
    wadd_b = np.sum(b * val)
    wadd_diff = abs(wadd_a - wadd_b)
    threshold = float(parameters["threshold"])
        
    # Check for strict conflict or insufficient WADD evidence
    conflict = (ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner)
    
    if conflict or (wadd_diff <= threshold):
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: rely on the agreed winner (or the one that isn't tied)
        winner = ttb_winner if ttb_winner is not None else tally_winner
        if winner == 0:
            p_core = np.array([1.0, 0.0])
        elif winner == 1:
            p_core = np.array([0.0, 1.0])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.0, 0.2]
- threshold: [0.0, 15.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5075 (var=0.0051) vs this=0.5012 (var=0.0057)
- Experiment 2: real=0.5079 (var=0.0012) vs this=0.5029 (var=0.0028)
- Experiment 3: real=0.5012 (var=0.0025) vs this=0.4947 (var=0.0040)
- Experiment 4: real=0.5175 (var=0.0039) vs this=0.4984 (var=0.0040)
- Experiment 5: real=0.5212 (var=0.0057) vs this=0.4950 (var=0.0080)
- Experiment 6: real=0.0400 (var=0.0147) vs this=0.0474 (var=0.0345)
- Experiment 7: real=0.4758 (var=0.0084) vs this=0.5037 (var=0.0055)
- Experiment 8: real=0.5108 (var=0.0021) vs this=0.5702 (var=0.0183)
- Experiment 9: real=0.4808 (var=0.0053) vs this=0.5012 (var=0.0061)
- Experiment 10: real=0.4884 (var=0.0172) vs this=0.4947 (var=0.0142)


---

### `pi_4` (overall score: 0.595)

**Description**
Decision-makers use a Compensatory / Weighted Additive (WADD) strategy, where an overall value for each option is computed by weighting each feature by a subjective transformation of its validity (validity raised to a power) and summing these weighted features. On strict conflict trials, the high weight of the single discriminating cue roughly balances the combined weights of multiple opposing lower-validity cues. This produces a near-zero difference in overall value, naturally resulting in a choice probability near 0.5 without requiring a mixture of distinct heuristics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities
    w = val ** gamma
    
    val_a = np.sum(a * w)
    val_b = np.sum(b * w)
    
    beta = float(parameters["beta"])
    scores = np.array([val_a, val_b])
    
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    p = (1.0 - epsilon) * p + epsilon * 0.5
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- gamma: [3.0, 8.0]
- beta: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5075 (var=0.0051) vs this=0.4896 (var=0.0059)
- Experiment 2: real=0.5079 (var=0.0012) vs this=0.4508 (var=0.0034)
- Experiment 3: real=0.5012 (var=0.0025) vs this=0.5044 (var=0.0053)
- Experiment 4: real=0.5175 (var=0.0039) vs this=0.4888 (var=0.0054)
- Experiment 5: real=0.5212 (var=0.0057) vs this=0.5500 (var=0.0066)
- Experiment 6: real=0.0400 (var=0.0147) vs this=-0.1453 (var=0.0248)
- Experiment 7: real=0.4758 (var=0.0084) vs this=0.6053 (var=0.0084)
- Experiment 8: real=0.5108 (var=0.0021) vs this=0.6498 (var=0.0064)
- Experiment 9: real=0.4808 (var=0.0053) vs this=0.6046 (var=0.0074)
- Experiment 10: real=0.4884 (var=0.0172) vs this=0.5842 (var=0.0111)


---

### `pi_5` (overall score: 0.590)

**Description**
Top-K Tallying with Restricted Scope: Decision-makers simplify complex multi-attribute choices by focusing only on the top K most valid cues, ignoring all lower-validity information. By restricting K to a narrow band (1, 2, or 3) and allowing for a high degree of random guessing (lapse rate), the model produces consistent behavior across individuals while naturally capturing the pervasive ~0.5 choice probabilities on strict conflict trials where the top cues often tie or balance out.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(parameters["k"])
    
    # Ensure k doesn't exceed the number of available features
    k = min(k, len(val))
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally positive features among top K
    a_score = np.sum(a[top_k_cues])
    b_score = np.sum(b[top_k_cues])
    
    if a_score > b_score:
        p_core = np.array([1.0, 0.0])
    elif b_score > a_score:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- k: {1, 2, 3}
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5075 (var=0.0051) vs this=0.5554 (var=0.0210)
- Experiment 2: real=0.5079 (var=0.0012) vs this=0.4131 (var=0.0099)
- Experiment 3: real=0.5012 (var=0.0025) vs this=0.5609 (var=0.0198)
- Experiment 4: real=0.5175 (var=0.0039) vs this=0.5409 (var=0.0195)
- Experiment 5: real=0.5212 (var=0.0057) vs this=0.4744 (var=0.0821)
- Experiment 6: real=0.0400 (var=0.0147) vs this=-0.0242 (var=0.2223)
- Experiment 7: real=0.4758 (var=0.0084) vs this=0.3032 (var=0.0245)
- Experiment 8: real=0.5108 (var=0.0021) vs this=0.4331 (var=0.0048)
- Experiment 9: real=0.4808 (var=0.0053) vs this=0.6092 (var=0.0283)
- Experiment 10: real=0.4884 (var=0.0172) vs this=0.5126 (var=0.0560)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for row in data.itertuples():
        a = row.option_a_ratings
        b = row.option_b_ratings
        
        # Determine Take The Best (TTB) prediction
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # Determine Tallying prediction
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        # Only consider trials where TTB and Tallying make strict and opposing predictions
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if row.response == ttb_winner:
                ttb_matches += 1
            total += 1
            
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5075 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8567 (var=0.0074)
- pi_2: 0.1517 (var=0.0113)
- pi_3: 0.5162 (var=0.0080)
- pi_4: 0.4896 (var=0.0059)
- pi_5: 0.5554 (var=0.0210)
- pi_6: 0.5012 (var=0.0057)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    tally_pred = (b_sums > a_sums).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.5079 (var=0.0012)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8596 (var=0.0101)
- pi_1: 0.1427 (var=0.0087)
- pi_3: 0.4919 (var=0.0060)
- pi_4: 0.4508 (var=0.0034)
- pi_5: 0.4131 (var=0.0099)
- pi_6: 0.5029 (var=0.0028)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        a_sum = sum(a)
        b_sum = sum(b)
        tally_winner = 0 if a_sum > b_sum else (1 if b_sum > a_sum else None)
        
        if tally_winner is not None and ttb_winner != tally_winner:
            ttb_match.append(1 if resp == ttb_winner else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.5012 (var=0.0025)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8475 (var=0.0088)
- pi_3: 0.5178 (var=0.0082)
- pi_2: 0.1009 (var=0.0067)
- pi_4: 0.5044 (var=0.0053)
- pi_5: 0.5609 (var=0.0198)
- pi_6: 0.4947 (var=0.0040)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    # TTB winner: first cue where options differ (since validities are strictly descending)
    diff = a_mat - b_mat
    nz = diff != 0
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    ttb_winner = np.where(first_diff > 0, 0, 1)
    
    # Tallying winner: option with more total feature wins
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    tally_winner = np.full(len(diff), -1)
    tally_winner[a_wins > b_wins] = 0
    tally_winner[b_wins > a_wins] = 1
    
    # Identify strict conflict trials where TTB and Tallying favor different options
    conflict = (ttb_winner != tally_winner) & (tally_winner != -1)
    
    if not np.any(conflict):
        return 0.5
        
    # Return the proportion of times the subject chose the TTB-favored option on conflict trials
    match = (resp[conflict] == ttb_winner[conflict])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5175 (var=0.0039)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5044 (var=0.0045)
- pi_1: 0.8375 (var=0.0119)
- pi_2: 0.1388 (var=0.0100)
- pi_4: 0.4888 (var=0.0054)
- pi_5: 0.5409 (var=0.0195)
- pi_6: 0.4984 (var=0.0040)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    choices_01100 = (data.loc[t1_mask, 'response'] == 1).sum() + (data.loc[t2_mask, 'response'] == 0).sum()
    total = t1_mask.sum() + t2_mask.sum()
    
    return float(choices_01100 / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5212 (var=0.0057)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5500 (var=0.0066)
- pi_3: 0.0488 (var=0.0021)
- pi_1: 0.1462 (var=0.0140)
- pi_2: 0.1412 (var=0.0103)
- pi_5: 0.4744 (var=0.0821)
- pi_6: 0.4950 (var=0.0080)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: target is [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]
    t1_mask1 = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t1_mask2 = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    t1_chose_target = (t1_mask1 & (data['response'] == 0)).sum() + (t1_mask2 & (data['response'] == 1)).sum()
    t1_total = t1_mask1.sum() + t1_mask2.sum()
    p1 = t1_chose_target / t1_total if t1_total > 0 else 0.5
    
    # Trial 2: target is [0, 1, 1, 1, 1] vs [1, 0, 0, 0, 0]
    t2_mask1 = (a_tuples == (0, 1, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    t2_mask2 = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 1, 1, 1))
    
    t2_chose_target = (t2_mask1 & (data['response'] == 0)).sum() + (t2_mask2 & (data['response'] == 1)).sum()
    t2_total = t2_mask1.sum() + t2_mask2.sum()
    p2 = t2_chose_target / t2_total if t2_total > 0 else 0.5
    
    return float(p1 - p2)
```

**Observed (real) value:** 0.0400 (var=0.0147)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4495 (var=0.0153)
- pi_4: -0.1453 (var=0.0248)
- pi_1: 0.6958 (var=0.0398)
- pi_2: -0.0316 (var=0.0130)
- pi_5: -0.0242 (var=0.2223)
- pi_6: 0.0474 (var=0.0345)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sums = data['option_b_ratings'].apply(sum)
    mask = b_sums == 4
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4758 (var=0.0084)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6053 (var=0.0084)
- pi_5: 0.3032 (var=0.0245)
- pi_1: 0.1358 (var=0.0111)
- pi_2: 0.8584 (var=0.0110)
- pi_3: 0.5026 (var=0.0070)
- pi_6: 0.5037 (var=0.0055)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    wadd_target = {
        (1, 1, 0, 0, 0, 0): 1,
        (0, 0, 1, 1, 1, 1): 0,
        (1, 0, 1, 0, 0, 0): 1,
        (1, 0, 1, 1, 1, 1): 0,
        (0, 1, 0, 1, 1, 1): 0,
        (0, 1, 0, 0, 0, 0): 1
    }
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    targets = a_tuples.map(wadd_target)
    matches = (data['response'] == targets)
    
    return float(matches.mean())
```

**Observed (real) value:** 0.5108 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4331 (var=0.0048)
- pi_4: 0.6498 (var=0.0064)
- pi_1: 0.6160 (var=0.0014)
- pi_2: 0.8600 (var=0.0085)
- pi_3: 0.7969 (var=0.0020)
- pi_6: 0.5702 (var=0.0183)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_chosen = 0
    total = 0
    
    t3_target = (1, 1, 0, 0, 0)
    t3_other  = (0, 0, 1, 1, 1)
    
    t5_target = (0, 1, 1, 1, 1)
    t5_other  = (1, 0, 0, 0, 0)
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Check if trial is T3/4
        if (a == t3_target and b == t3_other) or (a == t3_other and b == t3_target):
            total += 1
            if (row['response'] == 0 and a == t3_target) or (row['response'] == 1 and b == t3_target):
                target_chosen += 1
                
        # Check if trial is T5/6
        elif (a == t5_target and b == t5_other) or (a == t5_other and b == t5_target):
            total += 1
            if (row['response'] == 0 and a == t5_target) or (row['response'] == 1 and b == t5_target):
                target_chosen += 1
                
    if total == 0:
        return 0.5
    return float(target_chosen) / total
```

**Observed (real) value:** 0.4808 (var=0.0053)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6046 (var=0.0074)
- pi_6: 0.5012 (var=0.0061)
- pi_1: 0.5038 (var=0.0029)
- pi_2: 0.5121 (var=0.0040)
- pi_3: 0.4771 (var=0.0057)
- pi_5: 0.6092 (var=0.0283)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 2 by option_a_ratings
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    if mask.sum() == 0:
        return 0.0
    
    # Return the proportion of times A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4884 (var=0.0172)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4947 (var=0.0142)
- pi_4: 0.5842 (var=0.0111)
- pi_1: 0.1621 (var=0.0128)
- pi_2: 0.8558 (var=0.0124)
- pi_3: 0.4726 (var=0.0124)
- pi_5: 0.5126 (var=0.0560)


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
  "theory": "Probabilistic Heuristic Mixture: Decision-makers do not rely on a single deterministic strategy or a complex compensatory integration (like WADD) with explicit conflict detection. Instead, on each trial, they probabilistically sample a single simple heuristic from a subjective repertoire (e.g., Take The Best, Tallying, or Random Guessing). When these heuristics agree on an option, the mixture produces a high probability for that option. When they conflict, the mixture naturally averages out to near 0.5 choice probabilities, seamlessly explaining the high uncertainty on conflict trials without requiring a secondary fallback mechanism.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    w_ttb = float(parameters[\"w_ttb\"])\n    w_tally = float(parameters[\"w_tally\"])\n    w_guess = float(parameters[\"w_guess\"])\n    \n    total_w = w_ttb + w_tally + w_guess\n    if total_w == 0:\n        p_ttb, p_tally, p_guess = 0.0, 0.0, 1.0\n    else:\n        p_ttb = w_ttb / total_w\n        p_tally = w_tally / total_w\n        p_guess = w_guess / total_w\n        \n    cue_order = np.argsort(-val, kind=\"stable\")\n    ttb_prob_a = 0.5\n    for j in cue_order:\n        if a[j] > b[j]:\n            ttb_prob_a = 1.0\n            break\n        elif b[j] > a[j]:\n            ttb_prob_a = 0.0\n            break\n            \n    a_wins = np.sum(a > b)\n    b_wins = np.sum(b > a)\n    if a_wins > b_wins:\n        tally_prob_a = 1.0\n    elif b_wins > a_wins:\n        tally_prob_a = 0.0\n    else:\n        tally_prob_a = 0.5\n        \n    prob_a = p_ttb * ttb_prob_a + p_tally * tally_prob_a + p_guess * 0.5\n    prob_a = np.clip(prob_a, 1e-5, 1.0 - 1e-5)\n    \n    return np.array([prob_a, 1.0 - prob_a])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "w_ttb": "[0.0, 10.0]",
    "w_tally": "[0.0, 10.0]",
    "w_guess": "[0.0, 10.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's guidance, this theory completely abandons the compensatory (WADD) integration and explicit conflict detection mechanisms. Instead, it posits that decision-makers probabilistically sample from a set of simple, non-compensatory heuristics (Take The Best, Tallying, and Random Guessing) on each trial. By mixing these heuristic predictions based on individual-level weights, the model naturally produces deterministic choices when heuristics agree, and ~0.5 choice probabilities when they conflict, capturing the behavioral data through a computationally simpler and purely heuristic-driven process."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12675,
  "candidates_token_count": 937,
  "total_token_count": 15714
}
```
