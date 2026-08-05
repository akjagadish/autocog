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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Theory 1 is falsified because the first trial choice does not align with the Tallying heuristic. Propose a brand-new theory that serves as a stronger, more plausible competitor to the Position Bias theory (Theory 2). For instance, consider a 'Take-The-Best (TTB) Anchoring' theory where subjects evaluate the very first trial using the single most valid cue (rather than tallying all cues) and then anchor on that choice for the remainder of the experiment. Alternatively, propose a theory where subjects use a 'Satisficing' rule on the first trial (e.g., choosing the first option that meets a certain threshold on the top cues) before disengaging.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.955)

**Description**
Task Disengagement / Position Bias Theory: Subjects find the fictitious product choice task unengaging or too cognitively demanding to evaluate systematically. Instead of using the expert ratings, they adopt a completely deterministic, zero-effort heuristic of exclusively selecting the same option position (e.g., always choosing Option A or always Option B) on every single trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    pref = int(parameters["preferred_option"])
    n_opts = np.asarray(state).shape[0]
    
    probs = np.zeros(n_opts)
    if pref < n_opts:
        probs[pref] = 1.0
    else:
        probs[0] = 1.0
        
    return probs

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- preferred_option: {0, 1}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 9: real=0.4400 (var=0.2464) vs this=0.5800 (var=0.2436)
- Experiment 10: real=0.4800 (var=0.2496) vs this=0.5200 (var=0.2496)


---

### `pi_6` (overall score: 0.497)

**Description**
First Impression Anchoring: Subjects find the task repetitive and cognitively demanding. To minimize effort, they genuinely evaluate the options on the very first trial using a simple tallying heuristic (choosing the option with the most positive features, or guessing if tied). They then use this initial choice as an anchor, deterministically repeating the same position choice (e.g., always choosing A or always B) for the remainder of the experiment. This results in a completely zero within-subject variance while maintaining an even 50/50 split across the population.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # Dummy parameter to satisfy the contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # On the first trial, evaluate based on simple feature tallying
    if len(history.get("response", [])) == 0:
        sums = np.sum(stim, axis=1)
        max_sum = np.max(sums)
        is_max = (sums == max_sum).astype(float)
        return is_max / np.sum(is_max)
    else:
        # On all subsequent trials, repeat the first trial's choice
        first_resp = int(history["response"][0])
        probs = np.zeros(n_opts)
        if first_resp < n_opts:
            probs[first_resp] = 1.0
        else:
            probs[0] = 1.0
        return probs

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- dummy: {1.0}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 9: real=0.4400 (var=0.2464) vs this=1.0000 (var=0.0000)
- Experiment 10: real=0.4800 (var=0.2496) vs this=0.0000 (var=0.0000)


---

### `pi_4` (overall score: 0.386)

**Description**
Overload/Indifference Theory: When faced with adversarial choices where one highly valid cue conflicts with multiple weakly valid cues, subjects experience cognitive overload or perceive the options as perfectly tied. This leads to a breakdown of compensatory or non-compensatory decision strategies, resulting in pure random guessing (a 50/50 choice probability on every trial).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # Read dummy parameter to satisfy the parameter reference contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # Pure random guessing due to cognitive overload / indifference
    return np.ones(n_opts) / n_opts


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- dummy: {1.0}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4833 (var=0.0028)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5021 (var=0.0022)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.4900 (var=0.0024)
- Experiment 4: real=0.5000 (var=0.0000) vs this=0.4919 (var=0.0026)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.1110 (var=0.0010)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.5106 (var=0.0098)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.2468 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.2478 (var=0.0000)
- Experiment 9: real=0.4400 (var=0.2464) vs this=0.3600 (var=0.2304)
- Experiment 10: real=0.4800 (var=0.2496) vs this=0.5092 (var=0.0027)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues ordered 0 to 4 based on validities)
        ttb_winner = None
        for j in range(5):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for i in range(5) if a[i] > b[i])
        b_wins = sum(1 for i in range(5) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only evaluate on trials where the two heuristics strictly oppose
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_match += 1
            total += 1
            
    return float(ttb_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8489 (var=0.0092)
- pi_2: 0.1450 (var=0.0073)
- pi_3: 0.4317 (var=0.0054)
- pi_4: 0.4833 (var=0.0028)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    matches = (data['response'] == tally_pred)
    return float(matches.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8506 (var=0.0083)
- pi_1: 0.1396 (var=0.0108)
- pi_3: 0.5667 (var=0.0047)
- pi_4: 0.5021 (var=0.0022)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_has_f1 = data['option_b_ratings'].apply(lambda x: x[0] == 1).astype(int)
    chose_f1 = (data['response'] == b_has_f1)
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6669 (var=0.0149)
- pi_2: 0.1710 (var=0.0074)
- pi_1: 0.8294 (var=0.0095)
- pi_4: 0.4900 (var=0.0024)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    chose_f1 = ((a_f1 == 1) & (data['response'] == 0)) | ((b_f1 == 1) & (data['response'] == 1))
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1502 (var=0.0066)
- pi_3: 0.6677 (var=0.0158)
- pi_1: 0.8237 (var=0.0102)
- pi_4: 0.4919 (var=0.0026)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(tuple)
    trial_means = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    subj_devs = (trial_means - 0.5).abs().groupby('subject_id').mean()
    return float(subj_devs.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1671 (var=0.0056)
- pi_4: 0.1110 (var=0.0010)
- pi_1: 0.3329 (var=0.0124)
- pi_2: 0.3294 (var=0.0075)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    dom_a = (a_sum > 0) & (b_sum == 0)
    dom_b = (a_sum == 0) & (b_sum > 0)
    subset = data[dom_a | dom_b]
    if len(subset) == 0:
        return 0.5
    correct = ((dom_a) & (data['response'] == 0)) | ((dom_b) & (data['response'] == 1))
    return float(correct[dom_a | dom_b].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5106 (var=0.0098)
- pi_3: 0.6669 (var=0.0240)
- pi_1: 0.8431 (var=0.0143)
- pi_2: 0.8662 (var=0.0079)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5000 (var=0.0000)

### Experiment 7
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
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2471 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_1: 0.2479 (var=0.0000)
- pi_2: 0.2475 (var=0.0000)
- pi_4: 0.2468 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0000 (var=0.0000)
- pi_3: 0.2478 (var=0.0000)
- pi_1: 0.2489 (var=0.0000)
- pi_2: 0.2490 (var=0.0000)
- pi_4: 0.2478 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    first_trials = data.groupby('subject_id').first()
    
    matches = []
    for _, row in first_trials.iterrows():
        a_sum = sum(row['option_a_ratings'])
        b_sum = sum(row['option_b_ratings'])
        
        pred = 0 if a_sum > b_sum else 1
        matches.append(1.0 if row['response'] == pred else 0.0)
        
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4400 (var=0.2464)
**Other theories' values on this metric (for reference):**
- pi_6: 1.0000 (var=0.0000)
- pi_5: 0.5800 (var=0.2436)
- pi_1: 0.8600 (var=0.1204)
- pi_2: 0.8200 (var=0.1476)
- pi_3: 0.5400 (var=0.2484)
- pi_4: 0.3600 (var=0.2304)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.4800 (var=0.2496)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5200 (var=0.2496)
- pi_6: 0.0000 (var=0.0000)
- pi_1: 0.3683 (var=0.0021)
- pi_2: 0.1675 (var=0.0084)
- pi_3: 0.4406 (var=0.0058)
- pi_4: 0.5092 (var=0.0027)


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
  "theory": "Take-The-Best (TTB) Anchoring: Subjects find the task repetitive and cognitively demanding. To minimize effort, they evaluate the options on the very first trial using a simple non-compensatory heuristic (Take-The-Best), relying only on the single most valid cue to make their choice. They then use this initial choice as an anchor, deterministically repeating the same position choice (e.g., always choosing A or always B) for the remainder of the experiment, effectively disengaging from further cue evaluation.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    import json\n    \n    stim = np.asarray(state, dtype=float)\n    n_opts = stim.shape[0]\n    \n    # On the first trial, evaluate based on the single most valid cue (TTB)\n    if len(history.get(\"response\", [])) == 0:\n        val_str = parameters.get(\"validities\", \"[]\")\n        try:\n            validities = json.loads(val_str)\n        except:\n            validities = []\n            \n        if len(validities) > 0:\n            best_cue = np.argmax(validities)\n            if stim[0, best_cue] > stim[1, best_cue]:\n                probs = np.zeros(n_opts)\n                probs[0] = 1.0\n                return probs\n            elif stim[1, best_cue] > stim[0, best_cue]:\n                probs = np.zeros(n_opts)\n                probs[1] = 1.0\n                return probs\n            else:\n                return np.ones(n_opts) / n_opts\n        else:\n            return np.ones(n_opts) / n_opts\n    else:\n        # On all subsequent trials, repeat the first trial's choice\n        first_resp = int(history[\"response\"][0])\n        probs = np.zeros(n_opts)\n        if first_resp < n_opts:\n            probs[first_resp] = 1.0\n        else:\n            probs[0] = 1.0\n        return probs",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities"
  },
  "rationale": "The arbiter noted that the previous 'First Impression Anchoring' theory failed because its first trial choice (based on Tallying) did not align well with the empirical data (e.g., yielding 1.0 on Experiment 9 instead of the observed 0.44). Following the arbiter's suggestion, this new theory instantiates 'Take-The-Best (TTB) Anchoring'. Subjects evaluate the very first trial using only the single most valid cue (guessing if tied), and then anchor on that position choice for the rest of the experiment. This preserves the zero within-subject variance seen in Experiments 7 and 8 while providing a more plausible, non-compensatory mechanism for the initial choice."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10833,
  "candidates_token_count": 767,
  "total_token_count": 15379
}
```
