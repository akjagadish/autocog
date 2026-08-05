# generation_iter_01_attempt_00

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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Theory 2 (Response Inertia) is degenerate because subjects do not show sequential dependencies. Since pure random guessing (Theory 1) fits the data almost perfectly, a good competitor must also predict near-chance performance but through a different mechanism. Propose a 'High-Noise Salience' or 'Occasional Max-Value' theory: subjects generally guess randomly due to cognitive overload, but on a small fraction of trials, their attention is captured by the single highest rating present in the stimulus array, leading them to choose the option possessing that maximum value. This provides a feature-dependent alternative that still largely mimics random guessing.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.978)

**Description**
In complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. Thus, behavior resembles zero-intelligence random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
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
- Experiment 1: real=0.4917 (var=0.0021) vs this=0.5052 (var=0.0023)
- Experiment 2: real=0.5000 (var=0.0073) vs this=0.4900 (var=0.0087)
- Experiment 3: real=0.4829 (var=0.0017) vs this=0.4860 (var=0.0027)
- Experiment 4: real=0.5017 (var=0.0022) vs this=0.5185 (var=0.0025)
- Experiment 5: real=0.4768 (var=0.0049) vs this=0.5019 (var=0.0036)
- Experiment 6: real=0.0005 (var=0.0001) vs this=0.0004 (var=0.0001)
- Experiment 7: real=0.4917 (var=0.0039) vs this=0.5071 (var=0.0052)
- Experiment 8: real=-0.0008 (var=0.0015) vs this=-0.0018 (var=0.0012)
- Experiment 9: real=0.0436 (var=0.0008) vs this=0.0347 (var=0.0007)
- Experiment 10: real=0.0457 (var=0.0018) vs this=0.0425 (var=0.0011)


---

### `pi_5` (overall score: 0.744)

**Description**
Single Random Feature heuristic: On each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all. An epsilon parameter allows blending this strategy with pure random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    epsilon = float(parameters["epsilon"])
    n_features = stim.shape[1]
    
    wins = (stim[0] > stim[1]).sum()
    losses = (stim[0] < stim[1]).sum()
    ties = (stim[0] == stim[1]).sum()
    
    p_A_core = (wins + 0.5 * ties) / n_features
    p_B_core = (losses + 0.5 * ties) / n_features
    
    probs = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * probs + epsilon * 0.5

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4917 (var=0.0021) vs this=0.5938 (var=0.0040)
- Experiment 2: real=0.5000 (var=0.0073) vs this=0.4238 (var=0.0118)
- Experiment 3: real=0.4829 (var=0.0017) vs this=0.4506 (var=0.0046)
- Experiment 4: real=0.5017 (var=0.0022) vs this=0.4773 (var=0.0032)
- Experiment 5: real=0.4768 (var=0.0049) vs this=0.6692 (var=0.0134)
- Experiment 6: real=0.0005 (var=0.0001) vs this=0.0313 (var=0.0014)
- Experiment 7: real=0.4917 (var=0.0039) vs this=0.6933 (var=0.0177)
- Experiment 8: real=-0.0008 (var=0.0015) vs this=0.0975 (var=0.0042)
- Experiment 9: real=0.0436 (var=0.0008) vs this=0.0411 (var=0.0009)
- Experiment 10: real=0.0457 (var=0.0018) vs this=0.0451 (var=0.0008)


---

### `pi_6` (overall score: 0.468)

**Description**
Response Inertia and Alternation: Subjects in complex, feedback-free multi-attribute choice tasks ignore the stimulus features completely. Instead, their choices are driven by sequential dependencies, where they either exhibit perseveration (a tendency to repeat their previous choice) or alternation (a tendency to switch choices). This results in aggregate choice probabilities that appear as 50/50 random guessing on feature-based metrics, but are mechanistically generated by trial-to-trial memory rather than zero-memory noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    p_repeat = float(parameters["p_repeat"])
    
    if not history or len(history.get("response", [])) == 0:
        return np.array([0.5, 0.5])
        
    last_resp = history["response"][-1]
    
    if last_resp == 0:
        p_A = p_repeat
    else:
        p_A = 1.0 - p_repeat
        
    return np.array([p_A, 1.0 - p_A])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- p_repeat: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4917 (var=0.0021) vs this=0.4931 (var=0.0029)
- Experiment 2: real=0.5000 (var=0.0073) vs this=0.5156 (var=0.0188)
- Experiment 3: real=0.4829 (var=0.0017) vs this=0.5085 (var=0.0031)
- Experiment 4: real=0.5017 (var=0.0022) vs this=0.4988 (var=0.0034)
- Experiment 5: real=0.4768 (var=0.0049) vs this=0.4978 (var=0.0046)
- Experiment 6: real=0.0005 (var=0.0001) vs this=0.0003 (var=0.0001)
- Experiment 7: real=0.4917 (var=0.0039) vs this=0.5242 (var=0.0050)
- Experiment 8: real=-0.0008 (var=0.0015) vs this=-0.0070 (var=0.0011)
- Experiment 9: real=0.0436 (var=0.0008) vs this=0.2408 (var=0.0192)
- Experiment 10: real=0.0457 (var=0.0018) vs this=0.2303 (var=0.0232)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1432 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1432 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 4, 4]  B=[5, 3, 3]
  A=[5, 0, 0]  B=[2, 1, 1]
  A=[2, 5, 0]  B=[3, 1, 1]
  A=[0, 4, 4]  B=[2, 3, 3]
  A=[0, 5, 5]  B=[3, 4, 4]
  A=[4, 1, 0]  B=[1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_pred = (b_wins > a_wins).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.4917 (var=0.0021)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4090 (var=0.0043) (Δ vs real -0.0827)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.3067 (var=0.1115)
- pi_3: 0.2800 (var=0.0057)
- pi_4: 0.5052 (var=0.0023)
- pi_5: 0.5938 (var=0.0040)
- pi_6: 0.4931 (var=0.0029)

### Experiment 2
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[10, 2, 2]  B=[8, 3, 3]
  A=[9, 5, 4]  B=[1, 5, 8]
  A=[5, 5, 5]  B=[0, 8, 8]
  A=[1, 9, 9]  B=[8, 8, 8]
  A=[9, 10, 1]  B=[10, 1, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        resp = row['response']
        # Trial 1: A=[2, 6, 6], B=[10, 5, 5]
        if a[0] == 2:
            wadd_match.append(1 if resp == 1 else 0)
        # Trial 5: A=[1, 9, 9], B=[8, 8, 8]
        elif a[0] == 1:
            wadd_match.append(1 if resp == 1 else 0)
    if not wadd_match:
        return 0.5
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.5000 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5094 (var=0.0052) (Δ vs real +0.0094)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8900 (var=0.0829)
- pi_1: 0.0000 (var=0.0000)
- pi_3: 0.8675 (var=0.0085)
- pi_4: 0.4900 (var=0.0087)
- pi_5: 0.4238 (var=0.0118)
- pi_6: 0.5156 (var=0.0188)

### Experiment 3
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[5, 10, 10]  B=[6, 0, 0]
  A=[8, 2, 10]  B=[8, 3, 0]
  A=[0, 8, 8]  B=[1, 0, 0]
  A=[10, 0, 0]  B=[9, 9, 9]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[2, 10, 10]  B=[3, 2, 2]
  A=[7, 8, 0]  B=[7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = np.zeros(len(data), dtype=int) - 1
    for i in range(len(data)):
        for j in range(a_ratings.shape[1]):
            if diff[i, j] > 0:
                ttb_choices[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_choices[i] = 1
                break
                
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    match = (responses[valid] == ttb_choices[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4829 (var=0.0017)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3615 (var=0.0062) (Δ vs real -0.1215)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8508 (var=0.0143)
- pi_2: 0.0200 (var=0.0046)
- pi_1: 0.1250 (var=0.0000)
- pi_4: 0.4860 (var=0.0027)
- pi_5: 0.4506 (var=0.0046)
- pi_6: 0.5085 (var=0.0031)

### Experiment 4
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[10, 2, 0]  B=[10, 1, 10]
  A=[1, 10, 10]  B=[2, 0, 0]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[0, 8, 8]  B=[1, 1, 1]
  A=[7, 7, 7]  B=[7, 8, 0]
  A=[4, 9, 0]  B=[4, 10, 0]
  A=[0, 0, 10]  B=[0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (6, 0, 0) and b == (5, 10, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (10, 2, 0) and b == (10, 1, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (1, 10, 10) and b == (2, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (5, 5, 5) and b == (6, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 8, 8) and b == (1, 1, 1):
            ttb_match += (resp == 1)
            count += 1
        elif a == (7, 7, 7) and b == (7, 8, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (4, 9, 0) and b == (4, 10, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 0, 10) and b == (0, 1, 0):
            ttb_match += (resp == 1)
            count += 1
    if count == 0:
        return 0.5
    return ttb_match / count

```

**Observed (real) value:** 0.5017 (var=0.0022)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4771 (var=0.0029) (Δ vs real -0.0246)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1450 (var=0.0059)
- pi_3: 0.8462 (var=0.0150)
- pi_1: 0.2500 (var=0.0000)
- pi_4: 0.5185 (var=0.0025)
- pi_5: 0.4773 (var=0.0032)
- pi_6: 0.4988 (var=0.0034)

### Experiment 5
**Design**
  A=[10, 10, 10]  B=[0, 0, 0]
  A=[0, 0, 0]  B=[10, 10, 10]
  A=[9, 8, 7]  B=[1, 2, 3]
  A=[1, 2, 3]  B=[9, 8, 7]
  A=[10, 0, 10]  B=[0, 10, 0]
  A=[0, 10, 0]  B=[10, 0, 10]
  A=[8, 2, 5]  B=[2, 8, 5]
  A=[2, 8, 5]  B=[8, 2, 5]
  A=[5, 5, 5]  B=[5, 5, 5]
  A=[6, 4, 8]  B=[4, 6, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    correct = np.where(sum_a > sum_b, 0, 1)
    
    return float(np.mean(data.loc[mask, 'response'] == correct[mask]))
```

**Observed (real) value:** 0.4768 (var=0.0049)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6743 (var=0.0129) (Δ vs real +0.1975)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5019 (var=0.0036)
- pi_2: 0.9657 (var=0.0111)
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.8448 (var=0.0111)
- pi_5: 0.6692 (var=0.0134)
- pi_6: 0.4978 (var=0.0046)

### Experiment 6
**Design**
  A=[10, 10, 10, 10]  B=[0, 0, 0, 0]
  A=[8, 8, 8, 8]  B=[2, 2, 2, 2]
  A=[6, 6, 6, 6]  B=[4, 4, 4, 4]
  A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  A=[4, 4, 4, 4]  B=[6, 6, 6, 6]
  A=[2, 2, 2, 2]  B=[8, 8, 8, 8]
  A=[0, 0, 0, 0]  B=[10, 10, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 0]
  A=[0, 10, 10, 0]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[9, 2, 2, 2]  B=[2, 9, 9, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for each unique trial pair
    trial_keys = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    
    # Calculate the probability of choosing option A (response == 0) for each unique trial
    p_choose_A = (data['response'] == 0).groupby(trial_keys).mean()
    
    # Return the variance of these probabilities across different trial types
    return float(np.var(p_choose_A))
```

**Observed (real) value:** 0.0005 (var=0.0001)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0311 (var=0.0013) (Δ vs real +0.0307)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1872 (var=0.0000)
- pi_4: 0.0004 (var=0.0001)
- pi_1: 0.2431 (var=0.0000)
- pi_3: 0.1076 (var=0.0035)
- pi_5: 0.0313 (var=0.0014)
- pi_6: 0.0003 (var=0.0001)

### Experiment 7
**Design**
  A=[5, 5, 5, 5]  B=[0, 0, 0, 0]
  A=[5, 5, 5, 0]  B=[0, 0, 0, 5]
  A=[5, 5, 0, 0]  B=[0, 0, 5, 5]
  A=[5, 0, 0, 0]  B=[0, 5, 5, 5]
  A=[0, 0, 0, 0]  B=[5, 5, 5, 5]
  A=[3, 3, 3, 3]  B=[3, 3, 3, 3]
  A=[4, 2, 4, 2]  B=[2, 4, 2, 4]
  A=[1, 4, 1, 4]  B=[4, 1, 4, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wins_a = (a_ratings > b_ratings).sum(axis=1)
    wins_b = (b_ratings > a_ratings).sum(axis=1)
    
    mask = wins_a != wins_b
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    wa = wins_a[mask]
    wb = wins_b[mask]
    
    correct = ((wa > wb) & (responses == 0)) | ((wb > wa) & (responses == 1))
    return float(np.mean(correct))
```

**Observed (real) value:** 0.4917 (var=0.0039)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6388 (var=0.0061) (Δ vs real +0.1471)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5071 (var=0.0052)
- pi_5: 0.6933 (var=0.0177)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.9750 (var=0.0056)
- pi_3: 0.6683 (var=0.0054)
- pi_6: 0.5242 (var=0.0050)

### Experiment 8
**Design**
  A=[8, 8, 8, 8, 8]  B=[2, 2, 2, 2, 2]
  A=[8, 8, 8, 8, 2]  B=[2, 2, 2, 2, 8]
  A=[8, 8, 8, 2, 2]  B=[2, 2, 2, 8, 8]
  A=[8, 8, 5, 2, 2]  B=[2, 2, 5, 8, 8]
  A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]
  A=[8, 2, 2, 2, 2]  B=[2, 8, 8, 8, 8]
  A=[2, 2, 2, 2, 2]  B=[8, 8, 8, 8, 8]
  A=[8, 8, 8, 8, 5]  B=[2, 2, 2, 2, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    # Calculate the SRF predicted probability of choosing A
    wins = np.sum(a > b, axis=1)
    ties = np.sum(a == b, axis=1)
    p_A = (wins + 0.5 * ties) / a.shape[1]
    
    # Map response to +1 for choosing A (response=0) and -1 for choosing B (response=1)
    choice_sign = 1 - 2 * data['response']
    
    # Calculate the covariance-like product between choice and SRF advantage
    # Random Guessing expected value = 0
    # SRF expected value > 0
    return float(np.mean(choice_sign * (p_A - 0.5)))
```

**Observed (real) value:** -0.0008 (var=0.0015)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0852 (var=0.0039) (Δ vs real +0.0860)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0975 (var=0.0042)
- pi_4: -0.0018 (var=0.0012)
- pi_1: 0.2750 (var=0.0000)
- pi_2: 0.2595 (var=0.0003)
- pi_3: 0.1249 (var=0.0010)
- pi_6: -0.0070 (var=0.0011)

### Experiment 9
**Design**
  A=[6, 2, 2, 2]  B=[5, 10, 10, 10]
  A=[3, 3, 3, 0]  B=[2, 2, 2, 10]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 10, 10, 10]  B=[9, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[10, 0, 10, 10]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  A=[10, 0, 10, 0]  B=[0, 10, 0, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    deviations = []
    for _, df in data.groupby('subject_id'):
        resp = df['response'].values
        if len(resp) > 1:
            repeats = np.mean(resp[1:] == resp[:-1])
            deviations.append(abs(repeats - 0.5))
    return float(np.mean(deviations)) if deviations else 0.0
```

**Observed (real) value:** 0.0436 (var=0.0008)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0408 (var=0.0011) (Δ vs real -0.0027)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0347 (var=0.0007)
- pi_6: 0.2408 (var=0.0192)
- pi_1: 0.0368 (var=0.0000)
- pi_2: 0.0672 (var=0.0051)
- pi_3: 0.1116 (var=0.0041)
- pi_5: 0.0411 (var=0.0009)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def subj_score(df):
        resp = df['response'].values
        if len(resp) < 2:
            return 0.0
        repeats = np.mean(resp[1:] == resp[:-1])
        return float(abs(repeats - 0.5))
        
    return float(data.groupby('subject_id').apply(subj_score).mean())
```

**Observed (real) value:** 0.0457 (var=0.0018)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0429 (var=0.0009) (Δ vs real -0.0027)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2303 (var=0.0232)
- pi_4: 0.0425 (var=0.0011)
- pi_1: 0.0368 (var=0.0000)
- pi_2: 0.0478 (var=0.0011)
- pi_3: 0.1097 (var=0.0045)
- pi_5: 0.0451 (var=0.0008)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Occasional Max-Value Heuristic: In complex, feedback-free choice environments, cognitive overload causes subjects to default to random guessing on the vast majority of trials. However, on a small fraction of trials, their attention is captured by the single highest rating present anywhere in the stimulus array. When this happens, they choose the option possessing that maximum value. If the maximum values of both options are tied, they revert to random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    epsilon = float(parameters["epsilon"])
    
    A_ratings = np.array(state[0])
    B_ratings = np.array(state[1])
    
    max_A = np.max(A_ratings)
    max_B = np.max(B_ratings)
    
    if max_A > max_B:
        p_A_max = 1.0
    elif max_B > max_A:
        p_A_max = 0.0
    else:
        p_A_max = 0.5
        
    p_A = (1.0 - epsilon) * 0.5 + epsilon * p_A_max
    
    return np.array([p_A, 1.0 - p_A])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.0, 1.0]

`rationale`: Following the arbiter's diagnosis, this 'Occasional Max-Value' theory bridges the gap between pure random guessing and feature-based processing. Because subjects do not show the sequential dependencies predicted by the Response Inertia theory (pi_6), we instead introduce a feature-dependent mechanism that still primarily behaves like noise. Most of the time, choices are 50/50. But with probability epsilon, the subject is drawn to the single most salient (highest) rating in the entire stimulus array. This captures any slight deviations from chance performance that align with extreme feature values, without over-predicting compensatory integration.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The Occasional Max-Value Heuristic successfully captures the lack of sequential dependencies (Experiments 9 and 10) and provides a feature-dependent mechanism. However, it introduces too much systematic bias towards the option with the maximum rating. For instance, in Experiments 5, 7, and 8, the candidate's simulated values (0.6743, 0.6388, 0.0852) deviate significantly from the near-chance empirical ground truth (0.4768, 0.4917, -0.0008). The model is over-predicting the max-value effect because the parameter range for epsilon allows it to be too large.
Rationale: The current mechanism is conceptually sound and follows the arbiter's recommendation, but its parameterization allows for too much deviation from random guessing. To fix the over-prediction of max-value choices in Experiments 1, 3, 5, 7, and 8, tighten the parameter range for `epsilon` to be much smaller (e.g., `[0.0, 0.1]` or `[0.0, 0.15]`). This will ensure the model remains primarily a random-guessing baseline while still allowing for the occasional feature-dependent capture, bringing the metrics back in line with the near-chance empirical data.

**Outcome of this advice:** iter 1 candidate loss=0.1432 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Occasional Max-Value Heuristic: In complex, feedback-free choice environments, cognitive overload causes subjects to default to random guessing on the vast majority of trials. However, on a small fraction of trials, their attention is captured by the single highest rating present anywhere in the stimulus array. When this happens, they choose the option possessing that maximum value. If the maximum values of both options are tied, they revert to random guessing.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    epsilon = float(parameters[\"epsilon\"])\n    \n    A_ratings = np.array(state[0])\n    B_ratings = np.array(state[1])\n    \n    max_A = np.max(A_ratings)\n    max_B = np.max(B_ratings)\n    \n    if max_A > max_B:\n        p_A_max = 1.0\n    elif max_B > max_A:\n        p_A_max = 0.0\n    else:\n        p_A_max = 0.5\n        \n    p_A = (1.0 - epsilon) * 0.5 + epsilon * p_A_max\n    \n    return np.array([p_A, 1.0 - p_A])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "epsilon": "[0.0, 0.15]"
  },
  "rationale": "Following the critic's diagnosis, the previous candidate's parameterization of epsilon ([0.0, 1.0]) allowed for too much systematic bias towards the option with the maximum rating, causing deviations from the near-chance empirical ground truth. I have applied a minimal edit to tighten the parameter range for epsilon to [0.0, 0.15]. This ensures the model remains primarily a random-guessing baseline while still allowing for the occasional feature-dependent capture, bringing the metrics back in line with the empirical data."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13228,
  "candidates_token_count": 529,
  "total_token_count": 14189
}
```
