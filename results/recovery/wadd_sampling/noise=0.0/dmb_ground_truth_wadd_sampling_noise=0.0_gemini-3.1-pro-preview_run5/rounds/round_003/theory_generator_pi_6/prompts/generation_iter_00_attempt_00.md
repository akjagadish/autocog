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

Propose a new theory based on Sequential Feature Sampling or Evidence-Dependent Noise. Instead of probabilistically mixing distinct strategies (which violates log-odds linearity), the new theory should assume a single mechanism where features are sequentially sampled proportional to their validities until a threshold is reached, or where the decision noise (temperature in the softmax) scales with the variance or conflict among the features. This would maintain the log-odds linearity observed in Experiment 2 (since the core process remains additive) while naturally flattening the extremeness across trials with varying evidence magnitudes (as seen in Experiment 7), since higher evidence trials would also generate higher decision noise.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.669)

**Description**
Validity-Thresholded Tallying: Decision-makers simplify choices by ignoring cues with low validities and applying a simple tallying heuristic to the remaining high-validity cues. A subjective threshold determines which cues are considered reliable enough to use. For the included cues, the decision-maker counts how many times each option has a higher feature value than the other. The option with more wins among the thresholded cues is chosen. This boundedly rational strategy combines the frugality and robustness of tallying with the validity-sensitivity of weighted additive models, effectively breaking ties in favor of options that excel on more valid cues without requiring complex mental arithmetic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Thresholded Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues meet the subjective validity threshold
    valid_mask = val >= tau
    
    if not np.any(valid_mask):
        # If no cues are reliable enough, the decision-maker has no preference
        scores = np.zeros(2)
    else:
        # Tally wins only on the included cues
        a_filtered = stim[0, valid_mask]
        b_filtered = stim[1, valid_mask]
        
        a_wins = float(np.sum(a_filtered > b_filtered))
        b_wins = float(np.sum(b_filtered > a_filtered))
        
        scores = np.array([a_wins, b_wins])
        
    # Softmax over the tally scores
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
- tau: [0.0, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3292 (var=0.0132) vs this=0.3800 (var=0.0403)
- Experiment 2: real=0.7356 (var=0.0121) vs this=0.6919 (var=0.0412)
- Experiment 3: real=0.6989 (var=0.0397) vs this=0.6472 (var=0.0365)
- Experiment 4: real=0.1850 (var=0.2183) vs this=0.1887 (var=0.0688)
- Experiment 5: real=0.0167 (var=0.1200) vs this=0.2633 (var=0.1040)
- Experiment 6: real=0.0750 (var=0.0053) vs this=0.1146 (var=0.0067)
- Experiment 7: real=0.1109 (var=0.0106) vs this=0.1157 (var=0.0163)
- Experiment 8: real=0.1182 (var=4.9895) vs this=1.6297 (var=3.1031)


---

### `pi_3` (overall score: 0.623)

**Description**
Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3292 (var=0.0132) vs this=0.3406 (var=0.0096)
- Experiment 2: real=0.7356 (var=0.0121) vs this=0.7931 (var=0.0137)
- Experiment 3: real=0.6989 (var=0.0397) vs this=0.7222 (var=0.0223)
- Experiment 4: real=0.1850 (var=0.2183) vs this=0.2562 (var=0.0685)
- Experiment 5: real=0.0167 (var=0.1200) vs this=0.4017 (var=0.0430)
- Experiment 6: real=0.0750 (var=0.0053) vs this=0.0397 (var=0.0026)
- Experiment 7: real=0.1109 (var=0.0106) vs this=0.0867 (var=0.0027)
- Experiment 8: real=0.1182 (var=4.9895) vs this=0.9495 (var=4.2008)


---

### `pi_5` (overall score: 0.520)

**Description**
Strategy Selection (WADD and Probabilistic Tallying): Decision-makers probabilistically alternate between a purely compensatory strategy (Weighted Additive) and a simpler Tallying heuristic on a trial-by-trial basis. The Tallying heuristic is probabilistic, using a softmax over win counts to generate choice probabilities rather than deterministic choices. This mixture allows individuals to exhibit graded sensitivity to cue evidence on some trials while defaulting to unweighted, softer cue-counting on others, effectively explaining both the high tallying agreement in certain environments and the near-zero extremeness differences in others.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD Strategy
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    wadd_weights = val ** gamma
    wadd_scores = np.dot(stim, wadd_weights)
    
    z = beta * (wadd_scores - np.max(wadd_scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Tallying Strategy (Probabilistic)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    tally_scores = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
        
    # Mixture
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- w_wadd: [0.0, 1.0]
- gamma: [0.1, 5.0]
- beta: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3292 (var=0.0132) vs this=0.3192 (var=0.0153)
- Experiment 2: real=0.7356 (var=0.0121) vs this=0.7722 (var=0.0140)
- Experiment 3: real=0.6989 (var=0.0397) vs this=0.6344 (var=0.0170)
- Experiment 4: real=0.1850 (var=0.2183) vs this=0.1863 (var=0.0460)
- Experiment 5: real=0.0167 (var=0.1200) vs this=0.3783 (var=0.0717)
- Experiment 6: real=0.0750 (var=0.0053) vs this=0.0429 (var=0.0023)
- Experiment 7: real=0.1109 (var=0.0106) vs this=0.0983 (var=0.0078)
- Experiment 8: real=0.1182 (var=4.9895) vs this=2.2855 (var=3.1568)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    cue_order = np.argsort(-validities)
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            if row['response'] == ttb_winner:
                ttb_matches += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_matches / total)

```

**Observed (real) value:** 0.3292 (var=0.0132)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8598 (var=0.0061)
- pi_2: 0.1840 (var=0.0046)
- pi_3: 0.3406 (var=0.0096)
- pi_4: 0.3800 (var=0.0403)
- pi_5: 0.3192 (var=0.0153)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        tally_a = sum(1 for x, y in zip(a, b) if x > y)
        tally_b = sum(1 for x, y in zip(a, b) if y > x)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for x, y in zip(a, b):
            if x > y:
                ttb_pred = 0
                break
            elif y > x:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            agreements.append(1 if resp == tally_pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.7356 (var=0.0121)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8589 (var=0.0069)
- pi_1: 0.1317 (var=0.0105)
- pi_3: 0.7931 (var=0.0137)
- pi_4: 0.6919 (var=0.0412)
- pi_5: 0.7722 (var=0.0140)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify trials where Tallying sees an exact tie (2 wins vs 2 wins)
    t4_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 0))
    t5_mask = (a_tuples == (0, 0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0, 0))
    t6_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    
    # Under WADD, these trials are NOT ties, because the validities are strictly ordered:
    # w0 > w2 and w1 > w3, etc. 
    # WADD strictly prefers A in T4, B in T5, and A in T6 regardless of the gamma parameter.
    
    c4 = np.sum((data['response'] == 0) & t4_mask)
    c5 = np.sum((data['response'] == 1) & t5_mask)
    c6 = np.sum((data['response'] == 0) & t6_mask)
    
    total = np.sum(t4_mask) + np.sum(t5_mask) + np.sum(t6_mask)
    
    if total == 0:
        return 0.5
        
    return float((c4 + c5 + c6) / total)
```

**Observed (real) value:** 0.6989 (var=0.0397)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7222 (var=0.0223)
- pi_2: 0.4917 (var=0.0061)
- pi_1: 0.8544 (var=0.0096)
- pi_4: 0.6472 (var=0.0365)
- pi_5: 0.6344 (var=0.0170)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    t1_optA = (0, 0, 1, 1, 1)
    t1_optB = (1, 1, 0, 0, 0)
    
    t5_optA = (0, 1, 1, 1, 0)
    t5_optB = (1, 0, 0, 0, 1)
    
    t1_choices = []
    t5_choices = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_optA and b == t1_optB:
            t1_choices.append(1 if resp == 0 else 0)
        elif a == t1_optB and b == t1_optA:
            t1_choices.append(1 if resp == 1 else 0)
            
        elif a == t5_optA and b == t5_optB:
            t5_choices.append(1 if resp == 0 else 0)
        elif a == t5_optB and b == t5_optA:
            t5_choices.append(1 if resp == 1 else 0)
            
    if not t1_choices or not t5_choices:
        return 0.0
        
    return float(np.mean(t5_choices) - np.mean(t1_choices))
```

**Observed (real) value:** 0.1850 (var=0.2183)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0238 (var=0.0101)
- pi_3: 0.2562 (var=0.0685)
- pi_1: -0.0175 (var=0.0158)
- pi_4: 0.1887 (var=0.0688)
- pi_5: 0.1863 (var=0.0460)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project sequence column to string for hashable equality checks
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def subj_score(df):
        # Identify the extreme trials (T7 and T8)
        m7 = df['a_str'] == '10110'
        m8 = df['a_str'] == '01001'
        
        # Identify the weak/tie trials (T5 and T6)
        m5 = df['a_str'] == '10001'
        m6 = df['a_str'] == '01100'
        
        # Calculate probability of choosing Option A on each trial
        p7 = float((df.loc[m7, 'response'] == 0).mean()) if m7.any() else 0.5
        p8 = float((df.loc[m8, 'response'] == 0).mean()) if m8.any() else 0.5
        p5 = float((df.loc[m5, 'response'] == 0).mean()) if m5.any() else 0.5
        p6 = float((df.loc[m6, 'response'] == 0).mean()) if m6.any() else 0.5
        
        # Calculate 'extremeness' (absolute deviation from indifference)
        ext7 = abs(p7 - 0.5)
        ext8 = abs(p8 - 0.5)
        ext5 = abs(p5 - 0.5)
        ext6 = abs(p6 - 0.5)
        
        # Contrast extreme trials vs weak trials
        return (ext7 + ext8) - (ext5 + ext6)
        
    data_copy = data.copy()
    data_copy['a_str'] = a_str
    
    if 'subject_id' in data_copy.columns:
        return float(data_copy.groupby('subject_id').apply(subj_score).mean())
    return subj_score(data_copy)
```

**Observed (real) value:** 0.0167 (var=0.1200)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4017 (var=0.0430)
- pi_4: 0.2633 (var=0.1040)
- pi_1: 0.0233 (var=0.0331)
- pi_2: 0.5200 (var=0.0504)
- pi_5: 0.3783 (var=0.0717)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_t56_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trials 5 and 6 pit {1, 2} against {3, 4, 5}
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return row['response'] == 0
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return row['response'] == 1
            
        return np.nan
        
    choices = data.apply(get_t56_choice, axis=1)
    
    df_temp = pd.DataFrame({
        'subject_id': data['subject_id'],
        'choice': choices
    }).dropna()
    
    if len(df_temp) == 0:
        return 0.0
        
    # Calculate each subject's proportion of choosing the {1, 2} option
    subj_means = df_temp.groupby('subject_id')['choice'].mean()
    
    # Calculate extremeness as the squared deviation from 0.5
    extremeness = (subj_means - 0.5) ** 2
    
    return float(extremeness.mean())
```

**Observed (real) value:** 0.0750 (var=0.0053)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1146 (var=0.0067)
- pi_3: 0.0397 (var=0.0026)
- pi_1: 0.1486 (var=0.0054)
- pi_2: 0.1426 (var=0.0053)
- pi_5: 0.0429 (var=0.0023)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # These 6 trials all share the exact same Tallying difference (+1 for Option A).
    # However, their WADD evidence differences vary widely (from +0.1 to +1.3 under linear weights).
    tally_plus_A = ['10011', '01110', '11100', '10110', '00111', '11010']
    
    df_filtered = data[data['A_str'].isin(tally_plus_A)]
    if df_filtered.empty:
        return 0.0
        
    p_A = df_filtered.groupby('A_str')['response'].apply(lambda x: (x == 0).mean())
    
    if len(p_A) < 2:
        return 0.0
        
    # We measure the standard deviation of the choice probabilities across these 6 trials.
    return float(p_A.std())
```

**Observed (real) value:** 0.1109 (var=0.0106)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0867 (var=0.0027)
- pi_5: 0.0983 (var=0.0078)
- pi_1: 0.3449 (var=0.0104)
- pi_2: 0.0074 (var=0.0020)
- pi_4: 0.1157 (var=0.0163)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the three critical trials
    def is_t15(a, b): return tuple(a) == (1, 0, 0, 1, 1) and tuple(b) == (0, 1, 1, 0, 0)
    def is_t4(a, b):  return tuple(a) == (1, 0, 0, 0, 0) and tuple(b) == (0, 1, 1, 1, 1)
    def is_t2(a, b):  return tuple(a) == (1, 0, 0, 0, 0) and tuple(b) == (0, 1, 1, 0, 0)
    
    data['is_t15'] = data.apply(lambda row: is_t15(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_t4'] = data.apply(lambda row: is_t4(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    data['is_t2'] = data.apply(lambda row: is_t2(row['option_a_ratings'], row['option_b_ratings']), axis=1)
    
    def subject_metric(subj_df):
        # Count choices for A (response == 0)
        y_15 = (1.0 - subj_df.loc[subj_df['is_t15'], 'response']).sum()
        y_4 = (1.0 - subj_df.loc[subj_df['is_t4'], 'response']).sum()
        y_2 = (1.0 - subj_df.loc[subj_df['is_t2'], 'response']).sum()
        
        n_15 = subj_df['is_t15'].sum()
        n_4 = subj_df['is_t4'].sum()
        n_2 = subj_df['is_t2'].sum()
        
        if n_15 == 0 or n_4 == 0 or n_2 == 0:
            return 0.0
            
        # Laplace smoothing to aggressively reduce variance from extreme proportions
        p_15 = (y_15 + 1.0) / (n_15 + 2.0)
        p_4 = (y_4 + 1.0) / (n_4 + 2.0)
        p_2 = (y_2 + 1.0) / (n_2 + 2.0)
        
        L_15 = np.log(p_15 / (1.0 - p_15))
        L_4 = np.log(p_4 / (1.0 - p_4))
        L_2 = np.log(p_2 / (1.0 - p_2))
        
        return L_15 + L_4 - 2 * L_2
        
    subject_scores = data.groupby('subject_id').apply(subject_metric)
    return float(subject_scores.mean())
```

**Observed (real) value:** 0.1182 (var=4.9895)
**Other theories' values on this metric (for reference):**
- pi_5: 2.2855 (var=3.1568)
- pi_3: 0.9495 (var=4.2008)
- pi_1: -0.1233 (var=1.2183)
- pi_2: 2.6479 (var=2.8010)
- pi_4: 1.6297 (var=3.1031)


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
  "theory": "Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Stimulus expects shape (2, n_features); got {stim.shape}.\")\n        \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    theta = float(parameters[\"theta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Subjective feature weighting\n    weights = val ** gamma\n    \n    # Calculate weighted sums of evidence for each option\n    scores = np.dot(stim, weights)\n    \n    # Calculate conflict: total weight of features where the options differ\n    diff = np.abs(stim[0] - stim[1])\n    conflict = np.dot(diff, weights)\n    \n    # Effective beta scales inversely with conflict (higher conflict = more noise)\n    beta_eff = beta / (1.0 + theta * conflict)\n    \n    # Softmax over scores with max-subtraction for numerical stability\n    z = beta_eff * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "gamma": "[0.0, 5.0]",
    "beta": "[0.1, 20.0]",
    "theta": "[0.0, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "The previous theory (pi_5) attempted to capture graded sensitivity and flattening of extremeness by probabilistically mixing a Weighted Additive (WADD) strategy and a Tallying heuristic. However, as the arbiter noted, probabilistically mixing distinct strategies violates log-odds linearity because the log-odds of a mixture of exponentials is not linear with respect to the evidence. To address this, the new theory proposes an Evidence-Dependent Noise model. It assumes a single, unified WADD mechanism where decision-makers integrate all features according to subjectively transformed validities. Crucially, the decision noise (temperature in the softmax) is not constant; it scales proportionally with the total conflict or variance among the features (calculated as the total weight of features on which the options differ). This maintains the core additive process, preserving the log-odds linearity observed in Experiment 2. At the same time, it naturally flattens the extremeness across trials with varying evidence magnitudes (as seen in Experiment 7), because trials with higher evidence differences also tend to have higher conflict, which increases decision noise and tempers the choice probabilities."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12963,
  "candidates_token_count": 911,
  "total_token_count": 16342
}
```
