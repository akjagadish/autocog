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
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace the Position Bias theory with a 'Sequential Pattern' theory, such as 'Response Alternation' or 'Response Persistence'. Since the overall choice proportion is exactly 50%, participants might not be guessing independently on each trial but instead following a sequential heuristic (e.g., alternating between A and B, or repeating the previous choice). This would maintain the 50% overall choice probability for Option A, but introduce a distinct trial-by-trial autocorrelation that can be tested against independent random guessing.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.885)

**Description**
Single-Cue Heuristic with Very High Lapse: Participants avoid the cognitive overload of integrating multiple cues by relying exclusively on the single most valid cue. They compare the two options on this single best feature, choosing the one that is superior. If the options tie on this cue, they guess randomly. However, due to task disengagement, lack of feedback, or cognitive fatigue, participants exhibit a near-total lapse rate, meaning they almost entirely guess randomly, with only a tiny residual trace of the single-cue strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Single-Cue expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single most valid cue
    best_cue_idx = np.argmax(val)
    
    a_val = stim[0, best_cue_idx]
    b_val = stim[1, best_cue_idx]
    
    epsilon = float(parameters["epsilon"])
    
    # Deterministic choice based on the single best cue
    if a_val > b_val:
        p_core = np.array([1.0, 0.0])
    elif b_val > a_val:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend with random guessing based on the lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.9, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5179 (var=0.0027)
- Experiment 2: real=0.5042 (var=0.0108) vs this=0.4779 (var=0.0021)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.4756 (var=0.0022)
- Experiment 4: real=0.4985 (var=0.0000) vs this=0.4878 (var=0.0029)
- Experiment 5: real=0.5008 (var=0.0004) vs this=0.5185 (var=0.0020)
- Experiment 6: real=0.4990 (var=0.0006) vs this=0.5161 (var=0.0059)
- Experiment 7: real=-3.6043 (var=0.0395) vs this=8.0042 (var=0.4383)
- Experiment 8: real=891.0000 (var=80.8704) vs this=1860.0000 (var=14.1200)
- Experiment 9: real=0.5000 (var=0.0000) vs this=0.5048 (var=0.0034)
- Experiment 10: real=-2.0012 (var=0.0000) vs this=-1.9426 (var=0.5036)


---

### `pi_4` (overall score: 0.864)

**Description**
Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
(none)

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5046 (var=0.0028)
- Experiment 2: real=0.5042 (var=0.0108) vs this=0.4954 (var=0.0020)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5029 (var=0.0019)
- Experiment 4: real=0.4985 (var=0.0000) vs this=0.5069 (var=0.0030)
- Experiment 5: real=0.5008 (var=0.0004) vs this=0.4856 (var=0.0028)
- Experiment 6: real=0.4990 (var=0.0006) vs this=0.4894 (var=0.0064)
- Experiment 7: real=-3.6043 (var=0.0395) vs this=0.6981 (var=0.2895)
- Experiment 8: real=891.0000 (var=80.8704) vs this=1743.0000 (var=12.9604)
- Experiment 9: real=0.5000 (var=0.0000) vs this=0.4873 (var=0.0023)
- Experiment 10: real=-2.0012 (var=0.0000) vs this=0.1791 (var=0.5921)


---

### `pi_3` (overall score: 0.847)

**Description**
Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a weighted sum of their features, where the weights directly correspond to the cue validities. The probability of choosing an option is then determined by the difference in these weighted sums via a softmax function, subject to decision noise. To capture empirical data where choices appear highly noisy or evenly balanced across experimental manipulations, the decision noise parameters (softmax inverse temperature and random lapse rate) can span the full range from deterministic choice to complete uniform guessing. In this iteration, the inverse temperature is constrained to very low values, enforcing highly stochastic behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute the weighted sum of features for each option
    wadd_a = np.sum(stim[0] * val)
    wadd_b = np.sum(stim[1] * val)
    
    scores = np.array([wadd_a, wadd_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 0.5]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4742 (var=0.0035)
- Experiment 2: real=0.5042 (var=0.0108) vs this=0.5233 (var=0.0024)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5192 (var=0.0039)
- Experiment 4: real=0.4985 (var=0.0000) vs this=0.5147 (var=0.0036)
- Experiment 5: real=0.5008 (var=0.0004) vs this=0.5494 (var=0.0039)
- Experiment 6: real=0.4990 (var=0.0006) vs this=0.5956 (var=0.0133)
- Experiment 7: real=-3.6043 (var=0.0395) vs this=-17.0167 (var=0.2624)
- Experiment 8: real=891.0000 (var=80.8704) vs this=1775.0000 (var=10.5700)
- Experiment 9: real=0.5000 (var=0.0000) vs this=0.4931 (var=0.0019)
- Experiment 10: real=-2.0012 (var=0.0000) vs this=-1.9949 (var=0.6275)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice: first cue that discriminates determines choice
        ttb_choice = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice is None:
            ttb_choice = 0.5
            
        if ttb_choice == row['response']:
            matches.append(1.0)
        elif ttb_choice == 0.5:
            matches.append(0.5)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8527 (var=0.0108)
- pi_2: 0.1663 (var=0.0108)
- pi_3: 0.4742 (var=0.0035)
- pi_4: 0.5046 (var=0.0028)
- pi_5: 0.5179 (var=0.0027)
- pi_6: 0.5115 (var=0.0030)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    valid_trials = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        if a_wins > b_wins:
            if r == 0:
                matches += 1
            valid_trials += 1
        elif b_wins > a_wins:
            if r == 1:
                matches += 1
            valid_trials += 1
    return float(matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.5042 (var=0.0108)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8631 (var=0.0101)
- pi_1: 0.1667 (var=0.0111)
- pi_3: 0.5233 (var=0.0024)
- pi_4: 0.4954 (var=0.0020)
- pi_5: 0.4779 (var=0.0021)
- pi_6: 0.5031 (var=0.0017)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1421 (var=0.0096)
- pi_3: 0.5192 (var=0.0039)
- pi_2: 0.8592 (var=0.0081)
- pi_4: 0.5029 (var=0.0019)
- pi_5: 0.4756 (var=0.0022)
- pi_6: 0.4567 (var=0.0045)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4985 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5147 (var=0.0036)
- pi_1: 0.1318 (var=0.0101)
- pi_2: 0.6425 (var=0.0025)
- pi_4: 0.5069 (var=0.0030)
- pi_5: 0.4878 (var=0.0029)
- pi_6: 0.4543 (var=0.0032)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    diffs = data.apply(get_wadd_diff, axis=1)
    
    # WADD predicts choice A (0) when diffs > 0, and choice B (1) when diffs < 0
    is_correct = ((diffs > 0) & (data['response'] == 0)) | ((diffs < 0) & (data['response'] == 1))
    
    return float(is_correct.mean())
```

**Observed (real) value:** 0.5008 (var=0.0004)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4856 (var=0.0028)
- pi_3: 0.5494 (var=0.0039)
- pi_1: 0.6802 (var=0.0030)
- pi_2: 0.7510 (var=0.0058)
- pi_5: 0.5185 (var=0.0020)
- pi_6: 0.5100 (var=0.0025)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    val_a = A.dot(val)
    val_b = B.dot(val)
    
    # The expected choice under deterministic WADD
    expected = (val_a < val_b).astype(int)
    correct = (data['response'].values == expected).astype(float)
    
    # Weight the accuracy by the absolute difference in weighted sums
    diff = np.abs(val_a - val_b)
    
    # Return the weighted accuracy
    return float(np.sum(correct * diff) / np.sum(diff))
```

**Observed (real) value:** 0.4990 (var=0.0006)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5956 (var=0.0133)
- pi_4: 0.4894 (var=0.0064)
- pi_1: 0.8439 (var=0.0127)
- pi_2: 0.7734 (var=0.0079)
- pi_5: 0.5161 (var=0.0059)
- pi_6: 0.4887 (var=0.0054)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Feature 0 is the single most valid cue (validity 0.95)
    a_cue_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue_0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Identify trials where the subject chose the option endorsed by the best cue
    aligned = (chose_a & (a_cue_0 > b_cue_0)) | (chose_b & (b_cue_0 > a_cue_0))
    
    # The Single-Cue model with very high lapse predicts a slight bias (mean p~0.525)
    # toward the best cue, whereas Zero-Intelligence predicts exactly p=0.5.
    # We compute the total log-likelihood ratio (LLR) of the choices under the 
    # Single-Cue hypothesis (p=0.525) versus the Zero-Intelligence hypothesis (p=0.5).
    # This is the optimal test statistic (Neyman-Pearson) for discriminating the two.
    llr = aligned * np.log(0.525 / 0.5) + (~aligned) * np.log(0.475 / 0.5)
    
    return float(llr.sum())
```

**Observed (real) value:** -3.6043 (var=0.0395)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6981 (var=0.2895)
- pi_5: 8.0042 (var=0.4383)
- pi_1: 151.8241 (var=1.1059)
- pi_2: -173.7474 (var=0.9048)
- pi_3: -17.0167 (var=0.2624)
- pi_6: -13.1134 (var=0.2220)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract the first cue (the single most valid cue) for both options
    cue0_a = data['option_a_ratings'].apply(lambda x: x[0])
    cue0_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Filter to trials where cue 0 discriminates between the two options
    valid_trials = cue0_a != cue0_b
    if not valid_trials.any():
        return 0.0
        
    subset = data[valid_trials]
    cue0_a_sub = cue0_a[valid_trials]
    cue0_b_sub = cue0_b[valid_trials]
    
    # Response == 0 means A, Response == 1 means B
    # Check if the subject's choice aligns with the option favored by cue 0
    aligned = (subset['response'] == (cue0_b_sub > cue0_a_sub).astype(int))
    
    # Return the TOTAL sum of aligned choices instead of the mean.
    # This aggregates the small residual signal into a large cumulative effect.
    return float(aligned.sum())
```

**Observed (real) value:** 891.0000 (var=80.8704)
**Other theories' values on this metric (for reference):**
- pi_5: 1860.0000 (var=14.1200)
- pi_4: 1743.0000 (var=12.9604)
- pi_1: 3084.0000 (var=57.2576)
- pi_2: 1376.0000 (var=11.2496)
- pi_3: 1775.0000 (var=10.5700)
- pi_6: 1817.0000 (var=14.6644)

### Experiment 9
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4873 (var=0.0023)
- pi_6: 0.5379 (var=0.0027)
- pi_1: 0.4963 (var=0.0012)
- pi_2: 0.5048 (var=0.0019)
- pi_3: 0.4931 (var=0.0019)
- pi_5: 0.5048 (var=0.0034)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Count choices for Option A
    k = np.sum(data['response'] == 0)
    n = len(data)
    
    # The Advocated Theory (Position Bias) posits p_A is uniformly distributed between 0.5 and 0.6.
    # The Competing Theory (Zero-Intelligence) posits p_A is exactly 0.5.
    # We construct the log-likelihood ratio (LLR) of the observed choices under PB vs ZI.
    # This is the theoretically optimal test statistic (Neyman-Pearson lemma) for discriminating them.
    
    ps = np.linspace(0.5, 0.6, 51)
    
    # For each candidate p in the PB theory, calculate the log likelihood ratio compared to p=0.5
    # LLR_p = k * log(p / 0.5) + (n - k) * log((1 - p) / 0.5)
    log_terms = k * np.log(2 * ps) + (n - k) * np.log(2 * (1 - ps))
    
    # Integrate out the uniform prior over p using the log-sum-exp trick for numerical stability
    max_log = np.max(log_terms)
    llr = max_log + np.log(np.mean(np.exp(log_terms - max_log)))
    
    return float(llr)
```

**Observed (real) value:** -2.0012 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 14.6204 (var=1.1961)
- pi_4: 0.1791 (var=0.5921)
- pi_1: -0.6077 (var=0.5334)
- pi_2: -1.6852 (var=0.3263)
- pi_3: -1.9949 (var=0.6275)
- pi_5: -1.9426 (var=0.5036)


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
  "theory": "Sequential Pattern Theory: Participants facing difficult or unrewarded binary choices often fall back on sequential heuristics rather than evaluating the options independently on each trial. They may exhibit response persistence (repeating the same choice) or response alternation (switching back and forth between Option A and Option B). This strategy maintains an overall 50% choice proportion for each option, but introduces significant trial-by-trial autocorrelation, explaining sequential dependencies in the choice data that independent random guessing (Zero-Intelligence) fails to capture.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    alpha = float(parameters[\"alpha\"])\n    \n    if len(history[\"response\"]) == 0:\n        return np.array([0.5, 0.5])\n        \n    prev_response = history[\"response\"][-1]\n    \n    p = np.zeros(2)\n    p[prev_response] = alpha\n    p[1 - prev_response] = 1.0 - alpha\n    \n    return p",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "alpha": "[0.0, 1.0]"
  },
  "rationale": "Following the arbiter's suggestion, this theory replaces the Position Bias model with a Sequential Pattern model. Instead of an independent bias for Option A or B, participants are modeled as having a sequential dependency (autocorrelation) in their choices, represented by the parameter 'alpha'. An alpha > 0.5 indicates response persistence, while alpha < 0.5 indicates response alternation. This captures the exact 50% overall choice proportion while providing a mechanistic explanation for trial-by-trial choice sequences that deviate from pure independent random guessing."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11917,
  "candidates_token_count": 459,
  "total_token_count": 14754
}
```
