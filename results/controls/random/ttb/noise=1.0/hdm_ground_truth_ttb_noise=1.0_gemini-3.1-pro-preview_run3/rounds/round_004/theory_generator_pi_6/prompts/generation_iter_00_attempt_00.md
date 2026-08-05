# generation_iter_00_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Heuristic Decision Making domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
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
- rating_max: Upper bound (inclusive) of each rating value (LLM-proposed; >= 1). Ratings are integers in [0, rating_max].

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
  - `"option_a_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
  - `"option_b_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a new theory, 'Take-The-Best with Extreme Noise'. Subjects attempt to use the Take-The-Best heuristic (relying only on the most valid feature that discriminates between the two options) but due to high task complexity, their execution is heavily corrupted by noise. This should be modeled by finding the first discriminating feature in validity order, assigning a score based on that feature's win, and applying a softmax choice rule with a very low inverse temperature (beta near zero). This creates a tiny systematic deviation from 50/50 that may better match the precise magnitude of the observed metrics compared to the Tallying approach.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.984)

**Description**
Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 0.01]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5042 (var=0.0050) vs this=0.4908 (var=0.0046)
- Experiment 2: real=0.5156 (var=0.0035) vs this=0.4859 (var=0.0027)
- Experiment 3: real=0.5088 (var=0.0034) vs this=0.5056 (var=0.0027)
- Experiment 4: real=-0.0100 (var=0.0271) vs this=-0.0062 (var=0.0266)
- Experiment 5: real=0.4979 (var=0.0021) vs this=0.4925 (var=0.0024)
- Experiment 6: real=0.5000 (var=0.0030) vs this=0.4977 (var=0.0028)
- Experiment 7: real=0.4867 (var=0.0021) vs this=0.5060 (var=0.0028)
- Experiment 8: real=0.5096 (var=0.0033) vs this=0.5035 (var=0.0017)
- Experiment 9: real=636.0000 (var=9196.6464) vs this=864.0000 (var=7465.8816)
- Experiment 10: real=110.0000 (var=5344.6400) vs this=250.0000 (var=9461.0000)


---

### `pi_5` (overall score: 0.970)

**Description**
Pure Random Guessing (Inattentive Responding). Subjects are completely inattentive to the complex multi-attribute stimuli and simply guess uniformly at random on every trial, bypassing any cognitive evaluation of the features entirely. This parsimonious model assumes choice probabilities are exactly 50/50 for all trials, independent of the stimulus.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # Reference the parameter to satisfy the constraint, though choice is always 50/50
    _ = float(parameters['guessing_rate'])
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- guessing_rate: {1.0}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5042 (var=0.0050) vs this=0.4921 (var=0.0039)
- Experiment 2: real=0.5156 (var=0.0035) vs this=0.4844 (var=0.0034)
- Experiment 3: real=0.5088 (var=0.0034) vs this=0.4850 (var=0.0036)
- Experiment 4: real=-0.0100 (var=0.0271) vs this=0.0350 (var=0.0221)
- Experiment 5: real=0.4979 (var=0.0021) vs this=0.4892 (var=0.0021)
- Experiment 6: real=0.5000 (var=0.0030) vs this=0.5142 (var=0.0037)
- Experiment 7: real=0.4867 (var=0.0021) vs this=0.5023 (var=0.0022)
- Experiment 8: real=0.5096 (var=0.0033) vs this=0.5054 (var=0.0021)
- Experiment 9: real=636.0000 (var=9196.6464) vs this=384.0000 (var=4865.8176)
- Experiment 10: real=110.0000 (var=5344.6400) vs this=550.0000 (var=7417.0000)


---

### `pi_3` (overall score: 0.886)

**Description**
Take The Best (TTB) heuristic with Softmax Choice: People evaluate features sequentially in descending order of their subjective validities. They stop at the first feature that discriminates between the two options. Instead of a deterministic choice, they choose the option probabilistically using a softmax function over the values of that single discriminating feature, scaled by a sensitivity parameter beta. If no features discriminate, they guess. Response noise is additionally modeled via a global lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    order = np.argsort(-validities)
    
    beta = float(parameters['beta'])
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] != b[idx]:
            scores = np.array([a[idx], b[idx]])
            z = beta * (scores - scores.max())
            e = np.exp(z)
            p_core = e / e.sum()
            break
            
    epsilon = float(parameters['epsilon'])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.0, 0.2]
- epsilon: [0.2, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5042 (var=0.0050) vs this=0.5746 (var=0.0109)
- Experiment 2: real=0.5156 (var=0.0035) vs this=0.5225 (var=0.0045)
- Experiment 3: real=0.5088 (var=0.0034) vs this=0.4456 (var=0.0094)
- Experiment 4: real=-0.0100 (var=0.0271) vs this=-0.1850 (var=0.0433)
- Experiment 5: real=0.4979 (var=0.0021) vs this=0.5273 (var=0.0021)
- Experiment 6: real=0.5000 (var=0.0030) vs this=0.5290 (var=0.0025)
- Experiment 7: real=0.4867 (var=0.0021) vs this=0.5694 (var=0.0049)
- Experiment 8: real=0.5096 (var=0.0033) vs this=0.5587 (var=0.0073)
- Experiment 9: real=636.0000 (var=9196.6464) vs this=420.0000 (var=5312.1600)
- Experiment 10: real=110.0000 (var=5344.6400) vs this=-330.0000 (var=5378.4400)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5042 (var=0.0050)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.5850 (var=0.1940)
- pi_2_1: 0.5188 (var=0.0101)
- pi_3: 0.5746 (var=0.0109)
- pi_4: 0.4908 (var=0.0046)
- pi_5: 0.4921 (var=0.0039)

### Experiment 2
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5156 (var=0.0035)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7800 (var=0.0816)
- pi_1: 0.0000 (var=0.0000)
- pi_2_1: 0.5500 (var=0.0079)
- pi_3: 0.5225 (var=0.0045)
- pi_4: 0.4859 (var=0.0027)
- pi_5: 0.4844 (var=0.0034)

### Experiment 3
**Design**
  A=[2, 7, 7, 7]  B=[10, 6, 6, 6]
  A=[10, 2, 2, 2]  B=[2, 3, 3, 3]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 5, 5, 5]  B=[10, 0, 0, 1]
  A=[10, 0, 0, 1]  B=[0, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count how many features option A strictly wins and option B strictly wins
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Only consider trials where one option wins on strictly more features than the other
    mask = a_wins != b_wins
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with more feature wins
    # response == 0 means A was chosen, response == 1 means B was chosen
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    actual = data.loc[mask, 'response']
    
    # Return the proportion of trials where the subject's choice aligns with the Tallying prediction
    return float((tally_pred == actual).mean())
```

**Observed (real) value:** 0.5088 (var=0.0034)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2_1: 0.5081 (var=0.0067)
- pi_2: 0.4200 (var=0.1536)
- pi_3: 0.4456 (var=0.0094)
- pi_4: 0.5056 (var=0.0027)
- pi_5: 0.4850 (var=0.0036)

### Experiment 4
**Design**
  A=[1, 1, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  A=[10, 4, 4, 4, 4]  B=[0, 5, 5, 5, 5]
  A=[10, 10, 0, 0, 5]  B=[0, 0, 10, 10, 5]
  A=[0, 7, 7, 7, 7]  B=[10, 6, 6, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 4, 4, 5, 0]
  A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].tolist()
    # Trial 1: A=[1, 1, 6, 6, 6]
    is_t1 = np.array([a[0] == 1 for a in a_ratings])
    # Trial 2: A=[10, 4, 4, 4, 4]
    is_t2 = np.array([a[0] == 10 and a[1] == 4 for a in a_ratings])
    
    resp_A = (data['response'] == 0).values
    
    p_A_t1 = resp_A[is_t1].mean() if is_t1.sum() > 0 else 0.5
    p_A_t2 = resp_A[is_t2].mean() if is_t2.sum() > 0 else 0.5
    
    return float(p_A_t1 - p_A_t2)
```

**Observed (real) value:** -0.0100 (var=0.0271)
**Other theories' values on this metric (for reference):**
- pi_2_1: -0.2162 (var=0.0574)
- pi_1: 1.0000 (var=0.0000)
- pi_2: -0.8800 (var=0.1056)
- pi_3: -0.1850 (var=0.0433)
- pi_4: -0.0062 (var=0.0266)
- pi_5: 0.0350 (var=0.0221)

### Experiment 5
**Design**
  A=[5, 10, 10, 10]  B=[6, 0, 0, 0]
  A=[10, 0, 0, 0]  B=[9, 10, 10, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[5, 5, 5, 5]  B=[5, 6, 0, 0]
  A=[7, 7, 0, 0]  B=[7, 6, 10, 10]
  A=[8, 8, 8, 0]  B=[8, 8, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choices.append(0 if d[first_diff_idx] > 0 else 1)
        else:
            ttb_choices.append(-1)
            
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4979 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5273 (var=0.0021)
- pi_2_1: 0.3977 (var=0.0132)
- pi_1: 0.3333 (var=0.0000)
- pi_2: 0.1000 (var=0.0078)
- pi_4: 0.4925 (var=0.0024)
- pi_5: 0.4892 (var=0.0021)

### Experiment 6
**Design**
  A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  A=[8, 8, 8, 8]  B=[9, 0, 0, 0]
  A=[5, 6, 1, 1]  B=[5, 5, 10, 10]
  A=[5, 5, 10, 0]  B=[5, 5, 9, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 10]
  A=[7, 7, 7, 7]  B=[8, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_a < val_b:
                return 1
        return -1
        
    ttb_preds = [ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    agreements = [1 if p == r else 0 for p, r in zip(ttb_preds, data['response']) if p != -1]
    
    return float(np.mean(agreements)) if len(agreements) > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0030)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.3810 (var=0.0140)
- pi_3: 0.5290 (var=0.0025)
- pi_1: 0.1667 (var=0.0000)
- pi_2: 0.0433 (var=0.0087)
- pi_4: 0.4977 (var=0.0028)
- pi_5: 0.5142 (var=0.0037)

### Experiment 7
**Design**
  A=[10, 0, 0, 0, 0]  B=[0, 2, 2, 2, 2]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 4, 4, 4]
  A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 6, 6]
  A=[0, 2, 2, 2, 2]  B=[10, 0, 0, 0, 0]
  A=[0, 0, 4, 4, 4]  B=[10, 10, 0, 0, 0]
  A=[0, 0, 0, 6, 6]  B=[10, 10, 10, 0, 0]
  A=[10, 0, 10, 0, 0]  B=[0, 3, 0, 3, 3]
  A=[0, 3, 0, 3, 3]  B=[10, 0, 10, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    chose_higher = ((sum_a > sum_b) & (data['response'] == 0)) | ((sum_b > sum_a) & (data['response'] == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.4867 (var=0.0021)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5060 (var=0.0028)
- pi_2_1: 0.6094 (var=0.0162)
- pi_1: 0.2500 (var=0.0000)
- pi_2: 0.8900 (var=0.0354)
- pi_3: 0.5694 (var=0.0049)
- pi_5: 0.5023 (var=0.0022)

### Experiment 8
**Design**
  A=[10, 10, 10, 10, 10]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[10, 10, 10, 10, 10]
  A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]
  A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  A=[9, 9, 9, 9, 9]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[9, 9, 9, 9, 9]
  A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  A=[0, 10, 0, 10, 0]  B=[10, 0, 10, 0, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Determine which option has the higher sum (0 for A, 1 for B)
    target = (sum_b > sum_a).astype(int)
    
    # Exclude trials where sums are equal
    valid = sum_a != sum_b
    
    if not valid.any():
        return 0.5
        
    # Calculate proportion of choices that maximize the simple sum
    accuracy = (data.loc[valid, 'response'] == target[valid]).mean()
    return float(accuracy)

```

**Observed (real) value:** 0.5096 (var=0.0033)
**Other theories' values on this metric (for reference):**
- pi_2_1: 0.6550 (var=0.0165)
- pi_4: 0.5035 (var=0.0017)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.8600 (var=0.0173)
- pi_3: 0.5587 (var=0.0073)
- pi_5: 0.5054 (var=0.0021)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert sequence ratings to 2D numpy arrays
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    
    # Compute the tally difference for each trial (wins for A - wins for B)
    wins_a = np.sum(a > b, axis=1)
    wins_b = np.sum(b > a, axis=1)
    tally_diff = wins_a - wins_b
    
    # Map response to direction: A (0) -> +1, B (1) -> -1
    choice_dir = 1 - 2 * data['response'].values
    
    # Return the SUM of the alignment scores rather than the mean.
    # For the Pure Random Guessing model, this sum centers tightly at 0.
    # For the Tallying model, the tiny signal per trial accumulates across all pooled trials,
    # creating a large divergence that easily overcomes the binomial noise.
    return float(np.sum(tally_diff * choice_dir))
```

**Observed (real) value:** 636.0000 (var=9196.6464)
**Other theories' values on this metric (for reference):**
- pi_4: 864.0000 (var=7465.8816)
- pi_5: 384.0000 (var=4865.8176)
- pi_1: 34200.0000 (var=0.0000)
- pi_2: 34200.0000 (var=0.0000)
- pi_2_1: 7584.0000 (var=25094.9376)
- pi_3: 420.0000 (var=5312.1600)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D numpy arrays for vectorized comparison
    A = np.array(list(data['option_a_ratings']))
    B = np.array(list(data['option_b_ratings']))
    
    # Count strict feature wins for each option
    wins_a = np.sum(A > B, axis=1)
    wins_b = np.sum(B > A, axis=1)
    
    # Difference in tallies (B - A)
    tally_diff_b_minus_a = wins_b - wins_a
    
    # Map response 0 -> -1 (chose A) and 1 -> 1 (chose B)
    # Multiplying gives the tally advantage of the CHOSEN option
    responses = data['response'].values
    chosen_advantage = tally_diff_b_minus_a * (2 * responses - 1)
    
    # Return the total sum of the chosen option's advantage
    return float(np.sum(chosen_advantage))

```

**Observed (real) value:** 110.0000 (var=5344.6400)
**Other theories' values on this metric (for reference):**
- pi_5: 550.0000 (var=7417.0000)
- pi_4: 250.0000 (var=9461.0000)
- pi_1: 36000.0000 (var=0.0000)
- pi_2: 35760.0000 (var=552.9600)
- pi_2_1: 6970.0000 (var=25445.6400)
- pi_3: -330.0000 (var=5378.4400)


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
  "theory": "Take-The-Best with Extreme Noise: Subjects attempt to use the Take-The-Best heuristic by evaluating features sequentially in descending order of validity and stopping at the first feature that discriminates between the two options. However, because the task involves complex multi-attribute conflicts, their execution is heavily corrupted by noise. This is modeled by applying a softmax choice rule with an extremely low inverse temperature (beta near zero) to the values of the discriminating feature, creating a tiny systematic deviation from 50/50 guessing.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    validities = np.asarray(parameters['validities'], dtype=float)\n    order = np.argsort(-validities)\n    \n    beta = float(parameters['beta'])\n    \n    scores = np.array([0.0, 0.0])\n    for idx in order:\n        if a[idx] != b[idx]:\n            scores = np.array([a[idx], b[idx]])\n            break\n            \n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    return e / np.sum(e)",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probs = np.asarray(probabilities, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "beta": "[0.0, 0.01]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's suggestion, this model implements Take-The-Best (TTB) with extreme noise. Subjects rely on the most valid discriminating feature, but their decisions are heavily corrupted by noise due to task complexity. By applying a very low inverse temperature (beta near zero) to the scores of the discriminating feature, the model produces a tiny systematic deviation from 50/50. This approach aims to better match the precise magnitude of the observed metrics compared to the Tallying approach, as it isolates the specific directional pull of the highest validity feature while acknowledging the overwhelming noise in execution."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12454,
  "candidates_token_count": 545,
  "total_token_count": 14771
}
```
