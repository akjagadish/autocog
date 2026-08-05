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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace the WADD theory with a 'Take-The-Best (TTB) with high noise' theory, or a 'Mixture of TTB and Tallying' theory. WADD completely failed its own adversarial test in Experiment 2, showing no compensatory validity weighting orthogonal to Tallying. A non-compensatory strategy like TTB, which evaluates features sequentially by validity and stops at the first discriminating cue, might better capture the subtle deviations from chance across the other experiments without predicting a WADD-like compensatory signal.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.858)

**Description**
Noisy Equal Weights (Tallying): Decision-makers find integrating specific cue validities too cognitively demanding. Instead, they evaluate options by simply counting the number of positive features (1s) for each option, treating all cues equally, and preferring the option with the higher total. However, their choices are dominated by an extremely high lapse rate (epsilon ~0.90-1.0), meaning they almost always guess randomly. This accounts for the observed aggregate behavior being essentially at chance across multiple experimental metrics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count the number of positive features (1s) for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4000 (var=0.2400) vs this=0.5267 (var=0.0404)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5150 (var=0.0252)
- Experiment 3: real=0.5046 (var=0.0133) vs this=0.5295 (var=0.0038)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.5500 (var=0.0228)
- Experiment 5: real=-0.0231 (var=0.3337) vs this=0.0706 (var=0.0235)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.5600 (var=0.0353)
- Experiment 7: real=0.5200 (var=0.0274) vs this=0.5333 (var=0.0138)
- Experiment 8: real=0.5133 (var=0.1109) vs this=0.4900 (var=0.0055)
- Experiment 9: real=0.5333 (var=0.0267) vs this=0.5044 (var=0.0192)
- Experiment 10: real=0.0000 (var=0.0000) vs this=-0.0004 (var=0.0000)


---

### `pi_4` (overall score: 0.851)

**Description**
Noisy Take-The-Best: Decision-makers evaluate options lexicographically, starting with the most valid cue and stopping at the first cue that discriminates between the options. However, the execution of this heuristic is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic TTB predictions with heavy noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity
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
        p_core = np.ones(2) / 2.0
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.7, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4000 (var=0.2400) vs this=0.5433 (var=0.0453)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5333 (var=0.0169)
- Experiment 3: real=0.5046 (var=0.0133) vs this=0.5444 (var=0.0027)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4033 (var=0.0218)
- Experiment 5: real=-0.0231 (var=0.3337) vs this=0.0442 (var=0.0232)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.4533 (var=0.0445)
- Experiment 7: real=0.5200 (var=0.0274) vs this=0.4289 (var=0.0140)
- Experiment 8: real=0.5133 (var=0.1109) vs this=0.5628 (var=0.0087)
- Experiment 9: real=0.5333 (var=0.0267) vs this=0.5567 (var=0.0236)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.0002 (var=0.0000)


---

### `pi_6` (overall score: 0.728)

**Description**
Noisy Weighted Additive (WADD): Decision-makers compute a fully compensatory, weighted sum of the features for each option, using the provided cue validities as weights. They then choose based on the difference in these weighted sums. However, the execution of this strategy is highly stochastic, characterized by a massive lapse rate. Because the observed aggregate behavior is very close to random guessing (~0.50), the theory posits that subjects frequently suffer from attentional lapses or guess uniformly, effectively washing out the deterministic WADD predictions with heavy noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    scores = np.sum(stim * val, axis=1)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return np.random.choice(len(p), p=p)


`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.8, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4000 (var=0.2400) vs this=0.4567 (var=0.0298)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5483 (var=0.0236)
- Experiment 3: real=0.5046 (var=0.0133) vs this=0.5351 (var=0.0033)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4883 (var=0.0164)
- Experiment 5: real=-0.0231 (var=0.3337) vs this=0.0698 (var=0.0171)
- Experiment 6: real=0.5200 (var=0.2496) vs this=0.5267 (var=0.0393)
- Experiment 7: real=0.5200 (var=0.0274) vs this=0.5111 (var=0.0158)
- Experiment 8: real=0.5133 (var=0.1109) vs this=0.5244 (var=0.0051)
- Experiment 9: real=0.5333 (var=0.0267) vs this=0.5211 (var=0.0144)
- Experiment 10: real=0.0000 (var=0.0000) vs this=-0.0006 (var=0.0000)


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
    # Identify trials where TTB and Tallying make opposite deterministic predictions.
    # Specifically, Trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
    # TTB: Feature 0 (highest validity) favors A, so TTB chooses A.
    # Tallying: A wins on Feature 0, but B wins on Features 1 and 2. Tallying chooses B.
    is_reversal_trial = data.apply(
        lambda row: tuple(row['option_a_ratings']) == (1, 0, 0, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 1),
        axis=1
    )
    if is_reversal_trial.sum() == 0:
        return 0.5
    
    # Return the proportion of times option A was chosen on this trial.
    # TTB will have a value close to 1, while Tallying will have a value close to 0.
    return float((data.loc[is_reversal_trial, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4000 (var=0.2400)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9000 (var=0.0167)
- pi_2: 0.1567 (var=0.0260)
- pi_3: 0.3467 (var=0.0398)
- pi_4: 0.5433 (var=0.0453)
- pi_5: 0.5267 (var=0.0404)
- pi_6: 0.4567 (var=0.0298)

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
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_trial_8 = (a_str == '0101') & (b_str == '1100')
    is_trial_11 = (a_str == '1101') & (b_str == '1011')
    
    ttb_consistent_8 = data[is_trial_8]['response'] == 1
    ttb_consistent_11 = data[is_trial_11]['response'] == 0
    
    total_trials = len(ttb_consistent_8) + len(ttb_consistent_11)
    if total_trials == 0:
        return 0.5
        
    ttb_consistent_count = ttb_consistent_8.sum() + ttb_consistent_11.sum()
    return float(ttb_consistent_count / total_trials)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5150 (var=0.0221)
- pi_1: 0.8617 (var=0.0152)
- pi_3: 0.5383 (var=0.0262)
- pi_4: 0.5333 (var=0.0169)
- pi_5: 0.5150 (var=0.0252)
- pi_6: 0.5483 (var=0.0236)

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
    
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['pair'] = data['A_str'] + data['B_str']
    
    tally_0 = {'11010001', '01101000', '11110000', '11111101', '00110001', '11100101', '10111001', '01100100'}
    tally_1 = {'10101011', '00010101', '10100111', '11001101', '11001110'}
    
    is_0 = data['pair'].isin(tally_0)
    is_1 = data['pair'].isin(tally_1)
    
    matches = (is_0 & (data['response'] == 0)) | (is_1 & (data['response'] == 1))
    valid = is_0 | is_1
    
    return float(matches.sum() / valid.sum()) if valid.sum() > 0 else 0.5
```

**Observed (real) value:** 0.5046 (var=0.0133)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6451 (var=0.0098)
- pi_2: 0.8569 (var=0.0102)
- pi_1: 0.7438 (var=0.0075)
- pi_4: 0.5444 (var=0.0027)
- pi_5: 0.5295 (var=0.0038)
- pi_6: 0.5351 (var=0.0033)

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
    import numpy as np
    
    favored_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 10: A=[0, 0, 1, 1], B=[0, 1, 0, 0]
        # Tallying difference = 1 (A wins 2 features, B wins 1)
        # WADD difference = 0.27 (A=1.21, B=0.94)
        if a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            favored_chosen.append(1 if resp == 0 else 0)
            
        # Trial 14: A=[1, 1, 0, 0], B=[1, 0, 1, 1]
        # Tallying difference = 1 (B wins 2 features, A wins 1)
        # WADD difference = 0.27 (B=2.16, A=1.89)
        elif a == (1, 1, 0, 0) and b == (1, 0, 1, 1):
            favored_chosen.append(1 if resp == 1 else 0)
            
    if not favored_chosen:
        return 0.5
    return float(np.mean(favored_chosen))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8583 (var=0.0192)
- pi_3: 0.5483 (var=0.0225)
- pi_1: 0.1667 (var=0.0258)
- pi_4: 0.4033 (var=0.0218)
- pi_5: 0.5500 (var=0.0228)
- pi_6: 0.4883 (var=0.0164)

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
    import numpy as np
    val = np.array([0.95, 0.94, 0.91, 0.55])
    
    score_a = data['option_a_ratings'].apply(lambda x: np.sum(val * np.array(x)))
    score_b = data['option_b_ratings'].apply(lambda x: np.sum(val * np.array(x)))
    
    chosen_score = np.where(data['response'] == 0, score_a, score_b)
    unchosen_score = np.where(data['response'] == 0, score_b, score_a)
    
    return float(np.mean(chosen_score - unchosen_score))
```

**Observed (real) value:** -0.0231 (var=0.3337)
**Other theories' values on this metric (for reference):**
- pi_3: 0.3962 (var=0.0554)
- pi_4: 0.0442 (var=0.0232)
- pi_1: 0.6652 (var=0.0412)
- pi_2: 0.6891 (var=0.0214)
- pi_5: 0.0706 (var=0.0235)
- pi_6: 0.0698 (var=0.0171)

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
    import numpy as np
    
    # Identify Trial 1, which is the only trial where WADD and TTB have opposite deterministic predictions.
    # Trial 1: A = [0, 1, 1, 0], B = [1, 0, 0, 0]
    # WADD favors A (Score A = 0.92 + 0.87 = 1.79 > Score B = 0.95)
    # TTB favors B (Cue 0 discriminates for B)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 1, 0]) & \
                 data['option_b_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0])
                 
    if not is_trial_1.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_trial_1, 'response'] == 0))
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4533 (var=0.0445)
- pi_3: 0.6500 (var=0.0447)
- pi_1: 0.1400 (var=0.0226)
- pi_2: 0.8400 (var=0.0344)
- pi_5: 0.5600 (var=0.0353)
- pi_6: 0.5267 (var=0.0393)

### Experiment 7
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
    import numpy as np
    
    tally_choices = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        sum_a = sum(a)
        sum_b = sum(b)
        
        tally_favors_a = sum_a > sum_b
        tally_favors_b = sum_b > sum_a
        
        # TTB evaluates cues in order 0, 1, 2, 3 based on validities
        ttb_favors_a = False
        ttb_favors_b = False
        for i in range(4):
            if a[i] > b[i]:
                ttb_favors_a = True
                break
            elif b[i] > a[i]:
                ttb_favors_b = True
                break
                
        # Check if the two models disagree
        if (tally_favors_a and ttb_favors_b) or (tally_favors_b and ttb_favors_a):
            resp = row['response']
            # Record 1 if the subject chose the Tallying prediction, 0 if TTB
            if tally_favors_a and resp == 0:
                tally_choices.append(1)
            elif tally_favors_b and resp == 1:
                tally_choices.append(1)
            else:
                tally_choices.append(0)
                
    if not tally_choices:
        return 0.5
        
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.5200 (var=0.0274)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5333 (var=0.0138)
- pi_4: 0.4289 (var=0.0140)
- pi_1: 0.1633 (var=0.0184)
- pi_2: 0.8200 (var=0.0162)
- pi_3: 0.6033 (var=0.0149)
- pi_6: 0.5111 (var=0.0158)

### Experiment 8
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_pred(a, b):
        for i in range(4):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5
        
    def get_ew_pred(a, b):
        sa, sb = sum(a), sum(b)
        if sa > sb: return 0
        if sb > sa: return 1
        return 0.5
        
    ttb_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb = get_ttb_pred(a, b)
        ew = get_ew_pred(a, b)
        
        if ttb != ew and ttb != 0.5:
            ttb_match.append(1 if resp == ttb else 0)
            
    if not ttb_match:
        return 0.5
    return np.mean(ttb_match)
```

**Observed (real) value:** 0.5133 (var=0.1109)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5628 (var=0.0087)
- pi_5: 0.4900 (var=0.0055)
- pi_1: 0.8550 (var=0.0085)
- pi_2: 0.4378 (var=0.0056)
- pi_3: 0.5356 (var=0.0070)
- pi_6: 0.5244 (var=0.0051)

### Experiment 9
**Design**
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 1, 0) and b == (1, 0, 1, 1):
            score += (1 if resp == 0 else 0)
            count += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 1, 1):
            score += (1 if resp == 1 else 0)
            count += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 0):
            score += (1 if resp == 1 else 0)
            count += 1
            
    return score / count if count > 0 else 0.5
```

**Observed (real) value:** 0.5333 (var=0.0267)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5044 (var=0.0192)
- pi_6: 0.5211 (var=0.0144)
- pi_1: 0.8378 (var=0.0211)
- pi_2: 0.5067 (var=0.0104)
- pi_3: 0.5467 (var=0.0160)
- pi_4: 0.5567 (var=0.0236)

### Experiment 10
**Design**
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    val = np.array([0.95, 0.74, 0.64, 0.55])
    
    # Extract options as 2D arrays
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # Calculate the difference in scores between B and A for both theories
    D_T = np.sum(B, axis=1) - np.sum(A, axis=1)
    D_W = np.sum(B * val, axis=1) - np.sum(A * val, axis=1)
    
    df = pd.DataFrame({
        'D_T': D_T,
        'D_W': D_W,
        'y': data['response'].values
    })
    
    # Isolate the WADD signal that is strictly orthogonal to ANY Tallying effect
    # by subtracting the mean WADD difference within each exact Tallying difference group.
    df['D_W_mean'] = df.groupby('D_T')['D_W'].transform('mean')
    df['w'] = df['D_W'] - df['D_W_mean']
    
    # The metric is the projection of the choices onto this orthogonalized WADD component.
    return float((df['w'] * df['y']).mean())

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0006 (var=0.0000)
- pi_5: -0.0004 (var=0.0000)
- pi_1: -0.0007 (var=0.0000)
- pi_2: 0.0002 (var=0.0000)
- pi_3: 0.0006 (var=0.0000)
- pi_4: 0.0002 (var=0.0000)


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
  "theory": "Mixture of Take-The-Best and Tallying: Decision-makers rely on a mixture of two simple heuristics rather than a fully compensatory weighted additive strategy. Depending on the trial or individual, they either use Take-The-Best (evaluating features sequentially by validity and stopping at the first discriminating cue) or Tallying (simply counting the number of positive features). The overall behavior is highly stochastic due to a large lapse rate (frequent random guessing), which effectively washes out the deterministic heuristic predictions to match the near-chance aggregate observed across experiments.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    # Sort cues by descending validity\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    \n    # TTB prediction\n    ttb_winner = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            ttb_winner = 0\n            break\n        elif b[j] > a[j]:\n            ttb_winner = 1\n            break\n            \n    if ttb_winner == 0:\n        p_ttb = np.array([1.0, 0.0])\n    elif ttb_winner == 1:\n        p_ttb = np.array([0.0, 1.0])\n    else:\n        p_ttb = np.array([0.5, 0.5])\n        \n    # Tallying prediction\n    score_a = np.sum(a)\n    score_b = np.sum(b)\n    if score_a > score_b:\n        p_tally = np.array([1.0, 0.0])\n    elif score_b > score_a:\n        p_tally = np.array([0.0, 1.0])\n    else:\n        p_tally = np.array([0.5, 0.5])\n        \n    w_ttb = float(parameters[\"w_ttb\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally\n    \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p /= p.sum()\n    return np.random.choice(len(p), p=p)\n",
  "parameters": {
    "w_ttb": "[0.0, 1.0]",
    "epsilon": "[0.8, 1.0]",
    "validities": "validities"
  },
  "rationale": "The previous WADD theory completely failed the adversarial test in Experiment 10, predicting a compensatory signal orthogonal to Tallying that was not present in the empirical data. To resolve this, this new theory replaces WADD with a mixture of two non-compensatory/equal-weight heuristics: Take-The-Best (TTB) and Tallying. By blending these two strategies and maintaining the high lapse rate (epsilon) needed to capture the nearly random empirical choices, the model can flexibly capture the subtle deviations from chance across all experiments without predicting the flawed compensatory WADD signal."
}
```
```

## Usage

```json
{
  "prompt_token_count": 15137,
  "candidates_token_count": 915,
  "total_token_count": 18187
}
```
