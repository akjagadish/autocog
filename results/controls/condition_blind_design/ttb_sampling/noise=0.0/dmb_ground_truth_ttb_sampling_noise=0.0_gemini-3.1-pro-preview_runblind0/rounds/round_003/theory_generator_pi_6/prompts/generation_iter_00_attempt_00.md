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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a 'Hybrid TTB-WADD' theory. Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model should compute the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blend them using a subject-level mixing parameter. This allows the model to capture the dominant TTB-like behavior observed in Experiments 1, 2, and 5, while simultaneously accounting for the subtle compensatory effects of lower-validity cues seen in Experiments 3 and 8.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.921)

**Description**
Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1467 (var=0.0250) vs this=0.1900 (var=0.0422)
- Experiment 2: real=0.8200 (var=0.0532) vs this=0.8733 (var=0.0284)
- Experiment 3: real=0.8433 (var=0.0296) vs this=0.8117 (var=0.0236)
- Experiment 4: real=0.1333 (var=0.0156) vs this=0.1550 (var=0.0122)
- Experiment 5: real=-0.0067 (var=0.0433) vs this=0.0233 (var=0.0245)
- Experiment 6: real=-0.0733 (var=0.0624) vs this=0.0600 (var=0.0786)
- Experiment 7: real=0.8733 (var=0.0317) vs this=0.8667 (var=0.0267)
- Experiment 8: real=0.8300 (var=0.0186) vs this=0.8250 (var=0.0184)


---

### `pi_5` (overall score: 0.691)

**Description**
Take-Two with Conditional Fallback: Decision-makers evaluate the top two most valid features. If these two features agree or one favors an option while the other ties, that option is chosen. If they conflict (each option wins one), the decision-maker probabilistically mixes between reverting to the 1st feature and the 3rd feature. If the top two features tie (neither option wins on either), the decision-maker falls back to a simple tally of all features to break the tie.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-Two expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Evaluate the top two features
    f1, f2 = order[0], order[1]
    
    wins_a = (a[f1] > b[f1]) + (a[f2] > b[f2])
    wins_b = (b[f1] > a[f1]) + (b[f2] > a[f2])
    
    if wins_a > wins_b:
        score_a, score_b = 1.0, 0.0
    elif wins_b > wins_a:
        score_a, score_b = 0.0, 1.0
    else:
        if wins_a == 1 and wins_b == 1:
            # Conflict in top 2 features
            gamma = float(parameters["gamma"])
            
            # F1 preference (revert to most valid feature)
            score_a_f1, score_b_f1 = 0.5, 0.5
            if a[f1] > b[f1]:
                score_a_f1, score_b_f1 = 1.0, 0.0
            elif b[f1] > a[f1]:
                score_a_f1, score_b_f1 = 0.0, 1.0
                
            # F3 preference
            score_a_f3, score_b_f3 = 0.5, 0.5
            if len(order) > 2:
                f3 = order[2]
                if a[f3] > b[f3]:
                    score_a_f3, score_b_f3 = 1.0, 0.0
                elif b[f3] > a[f3]:
                    score_a_f3, score_b_f3 = 0.0, 1.0
                    
            score_a = gamma * score_a_f1 + (1.0 - gamma) * score_a_f3
            score_b = gamma * score_b_f1 + (1.0 - gamma) * score_b_f3
        else:
            # Tie in top 2 features (0 wins each)
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            score_a, score_b = 0.5, 0.5
            if tally_a > tally_b:
                score_a, score_b = 1.0, 0.0
            elif tally_b > tally_a:
                score_a, score_b = 0.0, 1.0
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1467 (var=0.0250) vs this=0.0492 (var=0.0481)
- Experiment 2: real=0.8200 (var=0.0532) vs this=0.8267 (var=0.0333)
- Experiment 3: real=0.8433 (var=0.0296) vs this=0.6233 (var=0.0584)
- Experiment 4: real=0.1333 (var=0.0156) vs this=0.1383 (var=0.0146)
- Experiment 5: real=-0.0067 (var=0.0433) vs this=-0.0067 (var=0.0377)
- Experiment 6: real=-0.0733 (var=0.0624) vs this=0.2867 (var=0.2223)
- Experiment 7: real=0.8733 (var=0.0317) vs this=0.8233 (var=0.0249)
- Experiment 8: real=0.8300 (var=0.0186) vs this=0.5683 (var=0.1160)


---

### `pi_4` (overall score: 0.540)

**Description**
Rank-Weighted Additive Theory: Decision-makers evaluate options using a compensatory but steeply decaying weighting scheme. Instead of using raw validities as weights, they rank features by their validity and assign exponentially decaying weights based on their rank (e.g., w_k = decay_rate^{-k}). This creates a 'soft' lexicographic strategy that largely mimics Take The Best by making the most valid cue dominant, but allows for compensation if multiple lower-ranked cues unanimously oppose the top cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Weighted Additive expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    decay_rate = float(parameters["decay_rate"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Assign exponentially decaying weights based on rank
    weights = np.zeros_like(validities)
    for k, idx in enumerate(order):
        weights[idx] = decay_rate ** (-k)
        
    # Compute weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- decay_rate: [1.5, 4.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1467 (var=0.0250) vs this=0.2533 (var=0.0507)
- Experiment 2: real=0.8200 (var=0.0532) vs this=0.8433 (var=0.0360)
- Experiment 3: real=0.8433 (var=0.0296) vs this=0.7917 (var=0.0253)
- Experiment 4: real=0.1333 (var=0.0156) vs this=0.3250 (var=0.0406)
- Experiment 5: real=-0.0067 (var=0.0433) vs this=0.2433 (var=0.0769)
- Experiment 6: real=-0.0733 (var=0.0624) vs this=0.0000 (var=0.1344)
- Experiment 7: real=0.8733 (var=0.0317) vs this=0.8167 (var=0.0325)
- Experiment 8: real=0.8300 (var=0.0186) vs this=0.8183 (var=0.0280)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.76, 0.92, 0.55])
    
    # Extract matrices of A and B features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying differences
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = a_wins - b_wins
    
    # WADD differences (expected without subject-specific weights)
    wadd_diff = np.sum((a_mat - b_mat) * validities, axis=1)
    
    # 1 if chose A, 0 if chose B
    chose_a = 1.0 - data['response'].values
    
    # We compare choice probabilities within trials where Tallying predicts the exact same difference.
    # For tally_diff == 1, WADD predicts some trials favor A more strongly than others.
    mask1_high = (tally_diff == 1) & (wadd_diff > 0.85)
    mask1_low = (tally_diff == 1) & (wadd_diff < 0.85)
    
    diff1 = 0.0
    if np.any(mask1_high) and np.any(mask1_low):
        diff1 = np.mean(chose_a[mask1_high]) - np.mean(chose_a[mask1_low])
        
    # For tally_diff == -1, WADD predicts some trials favor B more strongly than others.
    mask_m1_high = (tally_diff == -1) & (wadd_diff > -0.85)
    mask_m1_low = (tally_diff == -1) & (wadd_diff < -0.85)
    
    diff_m1 = 0.0
    if np.any(mask_m1_high) and np.any(mask_m1_low):
        diff_m1 = np.mean(chose_a[mask_m1_high]) - np.mean(chose_a[mask_m1_low])
        
    # Under Tallying, both diff1 and diff_m1 should be 0.
    # Under WADD, both diff1 and diff_m1 should be positive.
    return float(diff1 + diff_m1)
```

**Observed (real) value:** 0.1467 (var=0.0250)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0017 (var=0.0438)
- pi_2: 0.0792 (var=0.0977)
- pi_3: 0.1900 (var=0.0422)
- pi_4: 0.2533 (var=0.0507)
- pi_5: 0.0492 (var=0.0481)

### Experiment 2
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Tallying predicts a tie but WADD predicts a strong preference.
    # Trial 8: A=[0, 1, 0, 1], B=[1, 1, 0, 0]
    # A wins on feature 4 (validity 0.55). B wins on feature 1 (validity 0.95).
    # Tallying sees 1 win for A and 1 win for B, predicting exactly 50% choice for B.
    # WADD sees B's win on the most important feature as outweighing A's win on the least important, predicting >50% choice for B.
    is_target = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1]) & \
                data['option_b_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0])
    
    if is_target.sum() == 0:
        return 0.5
        
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.8200 (var=0.0532)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6533 (var=0.0887)
- pi_1: 0.4967 (var=0.0417)
- pi_3: 0.8733 (var=0.0284)
- pi_4: 0.8433 (var=0.0360)
- pi_5: 0.8267 (var=0.0333)

### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 3: A=[0, 1, 1, 0], B=[1, 0, 0, 0]
        # TTB chooses B (due to F1), WADD might choose A (due to F2+F3)
        if a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
        elif a == (1, 0, 0, 0) and b == (0, 1, 1, 0):
            if resp == 0:
                ttb_match += 1
            total += 1
            
        # Trial 11: A=[1, 0, 1, 0], B=[0, 1, 1, 1]
        # TTB chooses A (due to F1), WADD might choose B (due to F2+F4)
        elif a == (1, 0, 1, 0) and b == (0, 1, 1, 1):
            if resp == 0:
                ttb_match += 1
            total += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 1, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)

```

**Observed (real) value:** 0.8433 (var=0.0296)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8117 (var=0.0236)
- pi_2: 0.3750 (var=0.0726)
- pi_1: 0.1367 (var=0.0147)
- pi_4: 0.7917 (var=0.0253)
- pi_5: 0.6233 (var=0.0584)

### Experiment 4
**Design**
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert feature lists to tuples to allow element-wise comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the two trials where WADD's compensatory nature opposes TTB's lexicographic rule
    # Trial 10: TTB chooses B (due to feature 2), WADD leans A (features 3 + 4 compensate for feature 2)
    is_trial_10 = (a_tuples == (0, 0, 1, 1)) & (b_tuples == (0, 1, 0, 0))
    # Trial 14: TTB chooses A (due to feature 2), WADD leans B (features 3 + 4 compensate for feature 2)
    is_trial_14 = (a_tuples == (1, 1, 0, 0)) & (b_tuples == (1, 0, 1, 1))
    
    # Calculate the proportion of choices that align with the WADD compensatory prediction
    wadd_choice_10 = (data.loc[is_trial_10, 'response'] == 0).mean()
    wadd_choice_14 = (data.loc[is_trial_14, 'response'] == 1).mean()
    
    # Handle edge cases where a subject might have missing data for these specific trials
    if pd.isna(wadd_choice_10): wadd_choice_10 = 0.5
    if pd.isna(wadd_choice_14): wadd_choice_14 = 0.5
    
    return float((wadd_choice_10 + wadd_choice_14) / 2.0)
```

**Observed (real) value:** 0.1333 (var=0.0156)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5783 (var=0.0843)
- pi_3: 0.1550 (var=0.0122)
- pi_1: 0.8317 (var=0.0199)
- pi_4: 0.3250 (var=0.0406)
- pi_5: 0.1383 (var=0.0146)

### Experiment 5
**Design**
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Trial 14: A=[1, 1, 1, 1], B=[0, 0, 0, 0] -> TTB predicts A
    t14 = data[(data['A_tuple'] == (1, 1, 1, 1)) & (data['B_tuple'] == (0, 0, 0, 0))]
    # Trial 7: A=[0, 0, 1, 0], B=[0, 0, 1, 1] -> TTB predicts B
    t7 = data[(data['A_tuple'] == (0, 0, 1, 0)) & (data['B_tuple'] == (0, 0, 1, 1))]
    
    if len(t14) == 0 or len(t7) == 0:
        return 0.0
        
    p_A_14 = (t14['response'] == 0).mean()
    p_B_7 = (t7['response'] == 1).mean()
    
    return float(p_A_14 - p_B_7)
```

**Observed (real) value:** -0.0067 (var=0.0433)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0233 (var=0.0245)
- pi_4: 0.2433 (var=0.0769)
- pi_1: 0.0067 (var=0.0377)
- pi_2: 0.1200 (var=0.0623)
- pi_5: -0.0067 (var=0.0377)

### Experiment 6
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    
    def prob_choose_1(op1, op2):
        mask1 = (data['A'] == op1) & (data['B'] == op2)
        mask2 = (data['A'] == op2) & (data['B'] == op1)
        
        choices_op1 = 0
        total = 0
        
        if mask1.sum() > 0:
            choices_op1 += (data.loc[mask1, 'response'] == 0).sum()
            total += mask1.sum()
        if mask2.sum() > 0:
            choices_op1 += (data.loc[mask2, 'response'] == 1).sum()
            total += mask2.sum()
            
        return choices_op1 / total if total > 0 else np.nan

    # Trial 1: op1=(1,0,0,0), op2=(0,1,1,0)
    # Highest validity cue favors op1, but two lower cues favor op2.
    p_t1 = prob_choose_1((1,0,0,0), (0,1,1,0))
    
    # Trial 3: op1=(1,0,0,0), op2=(0,0,1,0)
    # Highest validity cue favors op1, only one lower cue favors op2.
    p_t3 = prob_choose_1((1,0,0,0), (0,0,1,0))
    
    # Trial 2: op1=(1,1,0,0), op2=(0,1,0,1)
    # Highest validity cue favors op1, lowest cue favors op2.
    p_t2 = prob_choose_1((1,1,0,0), (0,1,0,1))
    
    # Trial 13: op1=(1,1,0,0), op2=(0,1,0,0)
    # Highest validity cue favors op1, no cues favor op2.
    p_t13 = prob_choose_1((1,1,0,0), (0,1,0,0))
    
    val1 = (p_t3 - p_t1) if not np.isnan(p_t3 - p_t1) else 0.0
    val2 = (p_t13 - p_t2) if not np.isnan(p_t13 - p_t2) else 0.0
    
    return float(val1 + val2)
```

**Observed (real) value:** -0.0733 (var=0.0624)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0000 (var=0.1344)
- pi_3: 0.0600 (var=0.0786)
- pi_1: 0.7300 (var=0.1788)
- pi_2: 0.4233 (var=0.1591)
- pi_5: 0.2867 (var=0.2223)

### Experiment 7
**Design**
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target(a, b):
        return list(a) == [1, 0, 1, 0] and list(b) == [0, 1, 1, 0]
        
    def is_target_rev(a, b):
        return list(a) == [0, 1, 1, 0] and list(b) == [1, 0, 1, 0]

    fwd = data.apply(lambda row: is_target(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    rev = data.apply(lambda row: is_target_rev(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    
    n_fwd = fwd.sum()
    n_rev = rev.sum()
    
    if n_fwd + n_rev == 0:
        return 0.0
        
    chose_target_fwd = (data.loc[fwd, 'response'] == 0).sum()
    chose_target_rev = (data.loc[rev, 'response'] == 1).sum()
    
    return float(chose_target_fwd + chose_target_rev) / (n_fwd + n_rev)
```

**Observed (real) value:** 0.8733 (var=0.0317)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8667 (var=0.0267)
- pi_5: 0.8233 (var=0.0249)
- pi_1: 0.5133 (var=0.0509)
- pi_2: 0.4900 (var=0.1327)
- pi_4: 0.8167 (var=0.0325)

### Experiment 8
**Design**
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    b0 = data['option_b_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    b1 = data['option_b_ratings'].apply(lambda x: x[1])
    
    mask_A = (a0 > b0) & (a1 < b1)
    mask_B = (a0 < b0) & (a1 > b1)
    
    subset_A = data[mask_A]
    subset_B = data[mask_B]
    
    total_conflict = len(subset_A) + len(subset_B)
    if total_conflict == 0:
        return 0.5
        
    chose_f0_A = (subset_A['response'] == 0).sum()
    chose_f0_B = (subset_B['response'] == 1).sum()
    
    return float((chose_f0_A + chose_f0_B) / total_conflict)
```

**Observed (real) value:** 0.8300 (var=0.0186)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5683 (var=0.1160)
- pi_3: 0.8250 (var=0.0184)
- pi_1: 0.1483 (var=0.0248)
- pi_2: 0.3533 (var=0.0507)
- pi_4: 0.8183 (var=0.0280)


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
  "theory": "Hybrid TTB-WADD Theory: Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model computes the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blends them using a subject-level mixing parameter.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Hybrid model expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # --- TTB Probability ---\n    # Rank features by validity in descending order\n    order = np.argsort(validities)[::-1]\n    ttb_p = np.array([0.5, 0.5])\n    for idx in order:\n        if a[idx] > b[idx]:\n            ttb_p = np.array([1.0, 0.0])\n            break\n        elif b[idx] > a[idx]:\n            ttb_p = np.array([0.0, 1.0])\n            break\n            \n    # --- WADD Probability ---\n    score_a = np.sum(a * validities)\n    score_b = np.sum(b * validities)\n    wadd_scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * (wadd_scores - np.max(wadd_scores))\n    e = np.exp(z)\n    wadd_p = e / np.sum(e)\n    \n    # --- Blend ---\n    mix_rate = float(parameters[\"mix_rate\"])\n    p_core = mix_rate * ttb_p + (1.0 - mix_rate) * wadd_p\n    \n    # --- Lapse Rate ---\n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * 0.5\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "mix_rate": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's recommendation, this theory implements a Hybrid TTB-WADD model. It explicitly computes choice probabilities under both a pure Take-The-Best heuristic and a compensatory Weighted Additive rule, and combines them using a subject-specific mixing parameter (`mix_rate`). This allows the model to flexibly capture both dominant lexicographic behavior and subtle compensatory effects depending on the subject's blend of strategies."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14450,
  "candidates_token_count": 813,
  "total_token_count": 16934
}
```
