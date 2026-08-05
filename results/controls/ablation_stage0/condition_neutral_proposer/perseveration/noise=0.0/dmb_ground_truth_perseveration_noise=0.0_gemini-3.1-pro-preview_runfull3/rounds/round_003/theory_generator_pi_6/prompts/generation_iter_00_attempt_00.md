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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a 'Strict Random Guessing' theory. While Theory 1 allows for varying spatial position biases across subjects, the zero variance in many observed metrics suggests subjects might be uniformly guessing with exactly p=0.5. The new theory should posit that in the absence of feedback, subjects completely disengage and simply flip a coin on every trial (predicting exactly [0.5, 0.5] with no free parameters), ignoring stimulus, history, and spatial position alike.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.979)

**Description**
Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


`parameters`:
- bias_a: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4969 (var=0.0020)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5031 (var=0.0019)
- Experiment 3: real=0.4250 (var=0.1350) vs this=0.5221 (var=0.0492)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4985 (var=0.0017)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0001)
- Experiment 6: real=0.0000 (var=0.0000) vs this=0.0088 (var=0.0068)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0067 (var=0.0001)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0890 (var=0.0033)


---

### `pi_3` (overall score: 0.567)

**Description**
People make decisions by computing the expected value of each option, integrating both the magnitude of the features and their subjectively weighted validities. In a Weighted Additive (WADD) strategy, every feature contributes to an option's total score proportionally to a non-linear transformation of its cue validity, capturing subjective distortion of probabilities or weights. Choice probabilities are generated via a softmax function over these weighted sums, with an additional lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = np.power(val, gamma)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- gamma: [0.1, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.3513 (var=0.0200)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.6421 (var=0.0265)
- Experiment 3: real=0.4250 (var=0.1350) vs this=0.2840 (var=0.0195)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.2888 (var=0.0249)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0389 (var=0.0042)
- Experiment 6: real=0.0000 (var=0.0000) vs this=0.4004 (var=0.0734)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0095 (var=0.0002)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0764 (var=0.0030)


---

### `pi_5` (overall score: 0.415)

**Description**
Sequential Dependency / Alternation: In the absence of correctness feedback and when facing complex multi-cue choices, subjects often disengage from evaluating the actual features of the options. Instead of relying on a static spatial position bias, subjects exhibit sequential dependencies in their choices, such as a tendency to repeat their previous choice (inertia) or to systematically alternate between Option A and Option B. This history-dependent strategy completely ignores the stimulus validities and values, making decisions solely based on the temporal sequence of past actions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    
    if not history["response"]:
        # First trial: no history, predict uniformly
        return np.array([0.5, 0.5])
    
    last_resp = history["response"][-1]
    
    if last_resp == 0:
        # Last response was Option A
        prob_a = p_repeat
        prob_b = 1.0 - p_repeat
    else:
        # Last response was Option B
        prob_a = 1.0 - p_repeat
        prob_b = p_repeat
        
    return np.array([prob_a, prob_b])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- p_repeat: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4921 (var=0.0019)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5050 (var=0.0019)
- Experiment 3: real=0.4250 (var=0.1350) vs this=0.5083 (var=0.0148)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4885 (var=0.0022)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0004 (var=0.0001)
- Experiment 6: real=0.0000 (var=0.0000) vs this=-0.0104 (var=0.0072)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.3092 (var=0.0846)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.5241 (var=0.0691)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    preds = []
    for i in range(len(data)):
        a_vec = data['option_a_ratings'].iloc[i]
        b_vec = data['option_b_ratings'].iloc[i]
        pred = -1
        for j in range(len(a_vec)):
            if a_vec[j] > b_vec[j]:
                pred = 0
                break
            elif b_vec[j] > a_vec[j]:
                pred = 1
                break
        preds.append(pred)
        
    return float(np.mean(data['response'].values == np.array(preds)))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8444 (var=0.0101)
- pi_2: 0.1360 (var=0.0099)
- pi_3: 0.3513 (var=0.0200)
- pi_4: 0.4969 (var=0.0020)
- pi_5: 0.4921 (var=0.0019)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        if row['response'] == pred:
            match_count += 1
        total += 1
        
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8760 (var=0.0081)
- pi_1: 0.1310 (var=0.0075)
- pi_3: 0.6421 (var=0.0265)
- pi_4: 0.5031 (var=0.0019)
- pi_5: 0.5050 (var=0.0019)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_aligned(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                return row['response'] == ttb_pred
        return False
        
    return float(data.apply(is_ttb_aligned, axis=1).mean())
```

**Observed (real) value:** 0.4250 (var=0.1350)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8413 (var=0.0121)
- pi_3: 0.2840 (var=0.0195)
- pi_2: 0.1325 (var=0.0080)
- pi_4: 0.5221 (var=0.0492)
- pi_5: 0.5083 (var=0.0148)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                break
    return float(matches / len(data))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2888 (var=0.0249)
- pi_1: 0.8498 (var=0.0105)
- pi_2: 0.1650 (var=0.0102)
- pi_4: 0.4985 (var=0.0017)
- pi_5: 0.4885 (var=0.0022)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + df['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    trial_means = df.groupby('trial_type')['response'].mean()
    return float(np.var(trial_means))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0000 (var=0.0001)
- pi_3: 0.0389 (var=0.0042)
- pi_1: 0.1203 (var=0.0046)
- pi_2: 0.0711 (var=0.0008)
- pi_5: 0.0004 (var=0.0001)

### Experiment 6
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    val = np.array([0.9, 0.8, 0.7, 0.6])
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    diff = score_a - score_b
    
    choose_a = 1.0 - data['response'].values
    
    mask_a = diff > 0.01
    mask_b = diff < -0.01
    
    p_a = np.mean(choose_a[mask_a]) if np.sum(mask_a) > 0 else 0.0
    p_b = np.mean(choose_a[mask_b]) if np.sum(mask_b) > 0 else 0.0
    
    return float(p_a - p_b)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4004 (var=0.0734)
- pi_4: 0.0088 (var=0.0068)
- pi_1: 0.7275 (var=0.0367)
- pi_2: 0.3721 (var=0.0116)
- pi_5: -0.0104 (var=0.0072)

### Experiment 7
**Design**
  A=[1, 0, 0]  B=[0, 1, 1]
  A=[0, 1, 1]  B=[1, 0, 0]
  A=[1, 1, 0]  B=[0, 0, 1]
  A=[0, 0, 1]  B=[1, 1, 0]
  A=[1, 0, 1]  B=[0, 1, 0]
  A=[0, 1, 0]  B=[1, 0, 1]
  A=[1, 1, 1]  B=[0, 0, 0]
  A=[0, 0, 0]  B=[1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def subject_metric(df):
        resp = df['response'].values
        if len(resp) < 2:
            return 0.0
        
        idx_0 = np.where(resp[:-1] == 0)[0]
        idx_1 = np.where(resp[:-1] == 1)[0]
        
        if len(idx_0) == 0 or len(idx_1) == 0:
            # If a subject only ever gives one response, there is no variance
            # in transitions. For Theory 1 (Position Bias), this corresponds to 
            # extreme bias and a true difference of 0. 
            return 0.0
            
        p0_given_0 = np.mean(resp[idx_0 + 1] == 0)
        p0_given_1 = np.mean(resp[idx_1 + 1] == 0)
        
        return float((p0_given_0 - p0_given_1)**2)

    return float(data.groupby('subject_id').apply(subject_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0067 (var=0.0001)
- pi_5: 0.3092 (var=0.0846)
- pi_1: 0.0057 (var=0.0000)
- pi_2: 0.0097 (var=0.0001)
- pi_3: 0.0095 (var=0.0002)

### Experiment 8
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def subj_metric(df):
        responses = df['response'].values
        if len(responses) < 2:
            return 0.0
        r_t = responses[1:]
        r_t_minus_1 = responses[:-1]
        
        n00 = np.sum((r_t_minus_1 == 0) & (r_t == 0))
        n01 = np.sum((r_t_minus_1 == 0) & (r_t == 1))
        n10 = np.sum((r_t_minus_1 == 1) & (r_t == 0))
        n11 = np.sum((r_t_minus_1 == 1) & (r_t == 1))
        
        p_0_given_0 = n00 / (n00 + n01) if (n00 + n01) > 0 else np.nan
        p_0_given_1 = n10 / (n10 + n11) if (n10 + n11) > 0 else np.nan
        
        if np.isnan(p_0_given_0) or np.isnan(p_0_given_1):
            return 0.0
        
        return abs(p_0_given_0 - p_0_given_1)

    return float(data.groupby('subject_id').apply(subj_metric).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5241 (var=0.0691)
- pi_4: 0.0890 (var=0.0033)
- pi_1: 0.0793 (var=0.0028)
- pi_2: 0.0906 (var=0.0044)
- pi_3: 0.0764 (var=0.0030)


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
  "theory": "Strict Random Guessing: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects completely disengage from the task. They ignore the stimulus features, spatial positions, and their own choice history, instead falling back on a pure uniform random guessing strategy (flipping a coin) on every single trial.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    # Predict exactly 0.5 for both options regardless of stimulus or history\n    return np.array([0.5, 0.5])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {},
  "rationale": "Following the arbiter's feedback, the prior theories (such as spatial bias or sequential alternation) introduced inter-subject variances that were not present in the observed data, which had exactly 0.0000 variance on many metrics. A Strict Random Guessing theory with exactly p=0.5 and zero free parameters directly models complete task disengagement, aiming to eliminate the spurious variance introduced by subject-specific bias or history dependence parameters."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9981,
  "candidates_token_count": 306,
  "total_token_count": 11290
}
```
