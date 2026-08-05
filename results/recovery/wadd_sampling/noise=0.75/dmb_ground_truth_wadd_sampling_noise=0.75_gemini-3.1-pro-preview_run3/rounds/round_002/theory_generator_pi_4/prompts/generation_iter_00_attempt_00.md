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
- THEORY 2 = `pi_2_1`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace the WADD theory with the 'Take-The-Best' (TTB) heuristic. TTB offers a classic, non-compensatory alternative to Tallying. Under TTB, the decision-maker considers cues one at a time in descending order of validity, stopping at the first cue that discriminates between the two options and choosing the option favored by that cue. This will provide a strong non-compensatory baseline that respects cue ordering, contrasting perfectly with Tallying (which is compensatory but ignores cue ordering).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2_1` (overall score: 0.971)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.0, 1.0]
- epsilon: [0.4, 0.9]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4425 (var=0.0035) vs this=0.4529 (var=0.0047)
- Experiment 2: real=0.4578 (var=0.0041) vs this=0.4439 (var=0.0058)
- Experiment 3: real=0.5100 (var=0.0385) vs this=0.4667 (var=0.0156)
- Experiment 4: real=0.0025 (var=0.0611) vs this=0.0075 (var=0.0281)
- Experiment 5: real=-0.0700 (var=0.1048) vs this=-0.0417 (var=0.0484)
- Experiment 6: real=0.0800 (var=0.0525) vs this=-0.0100 (var=0.0405)


---

### `pi_2` (overall score: 0.522)

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
- Experiment 1: real=0.4425 (var=0.0035) vs this=0.1537 (var=0.0079)
- Experiment 2: real=0.4578 (var=0.0041) vs this=0.1403 (var=0.0096)
- Experiment 3: real=0.5100 (var=0.0385) vs this=0.5333 (var=0.0175)
- Experiment 4: real=0.0025 (var=0.0611) vs this=0.0050 (var=0.0284)
- Experiment 5: real=-0.0700 (var=0.1048) vs this=-0.0350 (var=0.0370)
- Experiment 6: real=0.0800 (var=0.0525) vs this=-0.0167 (var=0.0397)


---

### `pi_3` (overall score: 0.134)

**Description**
People make decisions by computing a weighted sum of the features for each option, where the weights correspond to a non-linear scaling of the subjective validities of the cues. This Weighted Additive (WADD) strategy allows for compensatory decision-making, where multiple weak cues can override a single strong cue, but the non-linear scaling (gamma parameter) flexibly tunes how strongly higher-validity cues dominate over weaker ones. A softmax choice rule on these weighted sums, along with a lapse rate, introduces response noise and accounts for the intermediate choice patterns observed in human data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    # Calculate weighted sum for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4425 (var=0.0035) vs this=0.5642 (var=0.0484)
- Experiment 2: real=0.4578 (var=0.0041) vs this=0.4869 (var=0.0441)
- Experiment 3: real=0.5100 (var=0.0385) vs this=0.8183 (var=0.0255)
- Experiment 4: real=0.0025 (var=0.0611) vs this=0.6450 (var=0.0596)
- Experiment 5: real=-0.0700 (var=0.1048) vs this=0.2900 (var=0.0917)
- Experiment 6: real=0.0800 (var=0.0525) vs this=0.6567 (var=0.0541)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5
    
    ttb_preds = data.apply(ttb_choice, axis=1)
    return float(np.mean(data['response'] == ttb_preds))
```

**Observed (real) value:** 0.4425 (var=0.0035)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8519 (var=0.0124)
- pi_2: 0.1537 (var=0.0079)
- pi_3: 0.5642 (var=0.0484)
- pi_2_1: 0.4529 (var=0.0047)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned_choices = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            disagreement_trials += 1
            if row['response'] == ttb_pred:
                ttb_aligned_choices += 1
                
    if disagreement_trials == 0:
        return 0.5
        
    return float(ttb_aligned_choices / disagreement_trials)
```

**Observed (real) value:** 0.4578 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1403 (var=0.0096)
- pi_1: 0.8478 (var=0.0126)
- pi_3: 0.4869 (var=0.0441)
- pi_2_1: 0.4439 (var=0.0058)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    opt1 = (1, 1, 0, 0, 0)
    opt2 = (0, 0, 1, 1, 0)
    
    chose_opt1 = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == opt1 and b == opt2:
            chose_opt1.append(1 if row['response'] == 0 else 0)
        elif a == opt2 and b == opt1:
            chose_opt1.append(1 if row['response'] == 1 else 0)
            
    if not chose_opt1:
        return 0.5
    return sum(chose_opt1) / len(chose_opt1)
```

**Observed (real) value:** 0.5100 (var=0.0385)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8183 (var=0.0255)
- pi_2: 0.5333 (var=0.0175)
- pi_1: 0.8433 (var=0.0157)
- pi_2_1: 0.4667 (var=0.0156)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = data['A_str'] == '11000'
    t4_mask = data['A_str'] == '00011'
    
    # response == 0 means subject chose A
    p_A_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_A_t4 = 1.0 - data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(p_A_t1): p_A_t1 = 0.5
    if pd.isna(p_A_t4): p_A_t4 = 0.5
    
    return float(p_A_t1 - p_A_t4)
```

**Observed (real) value:** 0.0025 (var=0.0611)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0050 (var=0.0284)
- pi_3: 0.6450 (var=0.0596)
- pi_1: 0.7150 (var=0.0733)
- pi_2_1: 0.0075 (var=0.0281)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project lists to strings for hashable matching
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 1: A=[1,0,0,0,0], B=[0,1,1,1,0]. Tallying tally diff = 3 - 1 = 2.
    t1_mask = (a_str == "10000") & (b_str == "01110")
    # Trial 2: A=[1,0,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 1 = 2.
    t2_mask = (a_str == "10000") & (b_str == "00111")
    
    # Trial 7: A=[1,0,1,0,0], B=[0,1,0,1,1]. Tallying tally diff = 3 - 2 = 1.
    t7_mask = (a_str == "10100") & (b_str == "01011")
    # Trial 8: A=[1,1,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 2 = 1.
    t8_mask = (a_str == "11000") & (b_str == "00111")
    
    p_b_t1 = data.loc[t1_mask, 'response'].mean()
    p_b_t2 = data.loc[t2_mask, 'response'].mean()
    p_b_t7 = data.loc[t7_mask, 'response'].mean()
    p_b_t8 = data.loc[t8_mask, 'response'].mean()
    
    # Handle missing trial types gracefully
    p_b_t1 = 0.5 if pd.isna(p_b_t1) else p_b_t1
    p_b_t2 = 0.5 if pd.isna(p_b_t2) else p_b_t2
    p_b_t7 = 0.5 if pd.isna(p_b_t7) else p_b_t7
    p_b_t8 = 0.5 if pd.isna(p_b_t8) else p_b_t8
    
    # Tallying predicts exactly 0 difference for both pairs. 
    # WADD predicts positive differences because B has higher validity features in T1 vs T2, and T7 vs T8.
    return float((p_b_t1 - p_b_t2) + (p_b_t7 - p_b_t8))
```

**Observed (real) value:** -0.0700 (var=0.1048)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2900 (var=0.0917)
- pi_2_1: -0.0417 (var=0.0484)
- pi_1: -0.0150 (var=0.0613)
- pi_2: -0.0350 (var=0.0370)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify Trial 1: A=[1, 1, 0, 0, 0, 0], B=[0, 0, 0, 1, 1, 0]
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 0))
    
    # Identify Trial 4: A=[0, 0, 0, 1, 1, 1], B=[1, 1, 1, 0, 0, 0]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    # Handle edge cases where a subject might have missed these trials (should not happen in full design)
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t4): p_a_t4 = 0.5
    
    return float(p_a_t1 - p_a_t4)

```

**Observed (real) value:** 0.0800 (var=0.0525)
**Other theories' values on this metric (for reference):**
- pi_2_1: -0.0100 (var=0.0405)
- pi_3: 0.6567 (var=0.0541)
- pi_1: 0.6517 (var=0.0744)
- pi_2: -0.0167 (var=0.0397)


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
  "theory": "Take-The-Best (TTB) heuristic: People make decisions by considering cues sequentially in descending order of their validity. They stop at the first cue that discriminates between the two options and choose the option favored by that cue, completely ignoring all remaining (less valid) cues. If no cues discriminate, they guess. Response noise is captured through a constant lapse rate.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"TTB expects a (2, n_features) stimulus.\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Sort indices by validity descending\n    order = np.argsort(val)[::-1]\n    \n    chosen = None\n    for idx in order:\n        if a[idx] > b[idx]:\n            chosen = 0\n            break\n        elif b[idx] > a[idx]:\n            chosen = 1\n            break\n            \n    if chosen == 0:\n        p_core = np.array([1.0, 0.0])\n    elif chosen == 1:\n        p_core = np.array([0.0, 1.0])\n    else:\n        p_core = np.array([0.5, 0.5])\n        \n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = len(p_core)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's feedback, this implements the Take-The-Best (TTB) heuristic, which offers a strict non-compensatory baseline. Unlike Tallying (which gives equal weight to all discriminating cues) and WADD (which is fully compensatory), TTB respects cue ordering but completely ignores all cues after the first discriminating one. This captures decision-making strategies where the single most reliable piece of evidence dictates the choice."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10090,
  "candidates_token_count": 633,
  "total_token_count": 12279
}
```
