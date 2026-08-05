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

Propose a 'Biased Constant Choice' theory. Like Theory 1, it should posit that subjects completely ignore the stimuli and choose a single option (A or B) constantly across all trials, resulting in zero within-subject variance. However, unlike Theory 1 which assumes a symmetric 50/50 split in the population's preferred option, this new theory should allow for a population-level bias (e.g., a higher probability of preferring Option B over Option A). This will preserve the perfect 0.0000 predictions for within-subject variance and deviation metrics, while better capturing the slight shift away from 0.5 in unbalanced designs like Experiment 4.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.922)

**Description**
Strong Position Bias / Constant Choice: Due to the lack of trial-by-trial feedback and low engagement, subjects adopt a degenerate strategy of always choosing the same option (e.g., always Option A or always Option B) regardless of the cues. This leads to choice probabilities of 1.0 or 0.0 for a given subject across all trials, perfectly explaining the near-zero within-subject variance across trial types and the extreme choice probabilities observed.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # The subject has a strict preference for either Option A (0) or Option B (1)
    pref = int(parameters["preferred_option"])
    
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- preferred_option: {0, 1}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 2: real=0.4400 (var=0.0242) vs this=0.5333 (var=0.0267)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.5000 (var=0.0000)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 9: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)


---

### `pi_6` (overall score: 0.815)

**Description**
Mixture of Constant Choice and Take-The-Best: Subjects primarily exhibit a degenerate strategy of relying on a fixed position preference (always choosing Option A or always Option B) due to low engagement or lack of trial-by-trial feedback. However, on a small fraction of trials, they lapse into using a single-cue heuristic (Take-The-Best), where they compare the options on the most valid cue. This mixture maintains the near-zero variance in choice proportions across most experiments while capturing the slight preference for TTB over Tallying in disagreement trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    pref = int(parameters["preferred_option"])
    epsilon = float(parameters["epsilon"])
    
    # Constant choice probabilities
    p_const = np.array([1.0, 0.0]) if pref == 0 else np.array([0.0, 1.0])
    
    # Take-The-Best (TTB) prediction
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    ttb_pred = -1
    for idx in order:
        if a[idx] > b[idx]:
            ttb_pred = 0
            break
        elif b[idx] > a[idx]:
            ttb_pred = 1
            break
            
    if ttb_pred == 0:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_pred == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_const + epsilon * p_ttb


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- preferred_option: {0, 1}
- epsilon: [0.0, 0.25]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5577 (var=0.0016)
- Experiment 2: real=0.4400 (var=0.0242) vs this=0.3797 (var=0.0204)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0087 (var=0.0021)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0039 (var=0.0000)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.4437 (var=0.0026)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.4302 (var=0.0047)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.0525 (var=0.0014)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.1350 (var=0.0166)
- Experiment 9: real=0.0000 (var=0.0000) vs this=0.1339 (var=0.0061)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.0360 (var=0.0006)


---

### `pi_4` (overall score: 0.312)

**Description**
Random Guessing / High-Noise Tallying: Due to the lack of trial-by-trial feedback and potentially low engagement, subjects largely ignore cue validities and feature values. Instead, they make choices that are nearly indistinguishable from uniform random guessing, which can be modeled as a simple unweighted tally of features heavily diluted by extreme decision noise and high lapse rates. This explains the ~0.5 agreement with Take The Best, ~0.44 agreement with Tallying, and the near-zero variance in choice proportions across different trial types.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of winning features
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    # Blend with uniform guessing based on epsilon
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- beta: [0.0, 0.05]
- epsilon: [0.8, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4904 (var=0.0021)
- Experiment 2: real=0.4400 (var=0.0242) vs this=0.4936 (var=0.0033)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0180 (var=0.0007)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0178 (var=0.0001)
- Experiment 5: real=0.5000 (var=0.0000) vs this=0.1154 (var=0.0011)
- Experiment 6: real=0.5000 (var=0.0000) vs this=0.1131 (var=0.0010)
- Experiment 7: real=0.0000 (var=0.0000) vs this=0.2477 (var=0.0000)
- Experiment 8: real=0.0000 (var=0.0000) vs this=0.0233 (var=0.0411)
- Experiment 9: real=0.0000 (var=0.0000) vs this=0.0011 (var=0.0131)
- Experiment 10: real=0.0000 (var=0.0000) vs this=0.1150 (var=0.0005)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = -1
        # Features are in descending order of validity: [0.9, 0.8, 0.7, 0.6]
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred != -1:
            agreements.append(1.0 if resp == ttb_pred else 0.0)
            
    return float(np.mean(agreements)) if agreements else 0.0

```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8390 (var=0.0119)
- pi_2: 0.2194 (var=0.0070)
- pi_3: 0.6073 (var=0.0263)
- pi_4: 0.4904 (var=0.0021)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.5577 (var=0.0016)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    tallying_agreements = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        # Take The Best (TTB) prediction
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Check for strict opposition between the two heuristics
        if tally_pred != -1 and ttb_pred != -1 and tally_pred != ttb_pred:
            disagreement_trials += 1
            if row['response'] == tally_pred:
                tallying_agreements += 1
                
    if disagreement_trials == 0:
        return 0.5
    return tallying_agreements / disagreement_trials

```

**Observed (real) value:** 0.4400 (var=0.0242)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8644 (var=0.0154)
- pi_1: 0.1231 (var=0.0109)
- pi_3: 0.4475 (var=0.0386)
- pi_4: 0.4936 (var=0.0033)
- pi_5: 0.5333 (var=0.0267)
- pi_6: 0.3797 (var=0.0204)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for trial types based on Option A's features
    # (Since each of the 8 unique trial types has a distinct Option A, this uniquely identifies the trial type)
    trial_keys = data['option_a_ratings'].apply(tuple)
    
    # Calculate whether the subject chose Option A (response == 0)
    chose_A = (data['response'] == 0).astype(float)
    
    # Combine into a temporary DataFrame to group
    df_temp = pd.DataFrame({'trial_key': trial_keys, 'chose_A': chose_A})
    
    # Compute the proportion of Option A choices for each trial type
    props = df_temp.groupby('trial_key')['chose_A'].mean()
    
    # Return the standard deviation of these choice proportions across the 8 trial types
    return float(np.std(props))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0115 (var=0.0013)
- pi_3: 0.1069 (var=0.0046)
- pi_2: 0.2512 (var=0.0037)
- pi_4: 0.0180 (var=0.0007)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0087 (var=0.0021)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Create a hashable key for trial types based on option A (which is unique for all 8 pairs)
    keys = data['option_a_ratings'].apply(tuple)
    # Compute the mean response (proportion of B choices) per subject per trial type
    means = data.assign(trial_key=keys).groupby(['subject_id', 'trial_key'])['response'].mean()
    # Compute the variance of these means across the 8 trial types for each subject, then average
    return float(means.groupby('subject_id').var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0357 (var=0.0010)
- pi_1: 0.0089 (var=0.0000)
- pi_2: 0.1012 (var=0.0022)
- pi_4: 0.0178 (var=0.0001)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0039 (var=0.0000)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a hashable string key for each unique trial type
    a_str = data['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    df = data.copy()
    df['trial_key'] = a_str + "_" + b_str
    
    # Calculate the proportion of choosing option B (response = 1) for each subject and trial type
    subject_trial_means = df.groupby(['subject_id', 'trial_key'])['response'].mean()
    
    # Calculate the absolute difference from 0.5 (extremeness of choice probability)
    abs_devs = (subject_trial_means - 0.5).abs()
    
    # Return the average extremeness across all subjects and trial types
    return float(abs_devs.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1154 (var=0.0011)
- pi_3: 0.3287 (var=0.0070)
- pi_1: 0.3577 (var=0.0090)
- pi_2: 0.3631 (var=0.0066)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.4437 (var=0.0026)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a hashable string representation for each unique trial type
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    trial_type = a_str + "_" + b_str
    
    # Response is 0 for A, 1 for B. We want proportion of choosing A.
    chose_A = 1 - data['response']
    
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_type': trial_type,
        'chose_A': chose_A
    })
    
    # Calculate the proportion of choosing A for each subject and trial type
    subj_trial = df.groupby(['subject_id', 'trial_type'])['chose_A'].mean()
    
    # Calculate the absolute deviation from 0.5 (random guessing)
    abs_dev = (subj_trial - 0.5).abs()
    
    # Average across trial types for each subject
    subj_dev = abs_dev.groupby('subject_id').mean()
    
    # Return the grand mean across subjects
    return float(subj_dev.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2754 (var=0.0056)
- pi_4: 0.1131 (var=0.0010)
- pi_1: 0.3533 (var=0.0063)
- pi_2: 0.3208 (var=0.0067)
- pi_5: 0.5000 (var=0.0000)
- pi_6: 0.4302 (var=0.0047)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data.groupby('subject_id')['response'].var(ddof=0).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0000 (var=0.0000)
- pi_3: 0.2491 (var=0.0000)
- pi_1: 0.2488 (var=0.0000)
- pi_2: 0.2488 (var=0.0000)
- pi_4: 0.2477 (var=0.0000)
- pi_6: 0.0525 (var=0.0014)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    t1_mask = (sum_a == 5) & (sum_b == 0)
    t2_mask = (sum_a == 0) & (sum_b == 5)
    
    m1 = data[t1_mask].groupby('subject_id')['response'].mean()
    m2 = data[t2_mask].groupby('subject_id')['response'].mean()
    
    diff = m2 - m1
    return float(diff.mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7250 (var=0.0528)
- pi_5: 0.0000 (var=0.0000)
- pi_1: 0.6900 (var=0.0636)
- pi_2: 0.7317 (var=0.0473)
- pi_4: 0.0233 (var=0.0411)
- pi_6: 0.1350 (var=0.0166)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_ttb_pred(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
        
    data['ttb'] = data.apply(get_ttb_pred, axis=1)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        b_trials = subj_df[subj_df['ttb'] == 1]
        a_trials = subj_df[subj_df['ttb'] == 0]
        
        if len(b_trials) > 0 and len(a_trials) > 0:
            diff = b_trials['response'].mean() - a_trials['response'].mean()
            subj_diffs.append(diff)
            
    if not subj_diffs:
        return 0.0
        
    return float(np.mean(subj_diffs))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.1339 (var=0.0061)
- pi_1: 0.7117 (var=0.0434)
- pi_2: -0.7489 (var=0.0318)
- pi_3: 0.2683 (var=0.0813)
- pi_4: 0.0011 (var=0.0131)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        mode = df['response'].mode().iloc[0]
        score = 0
        for _, row in df.iterrows():
            if row['response'] != mode:
                a = row['option_a_ratings']
                b = row['option_b_ratings']
                ttb_pred = -1
                for i in range(len(a)):
                    if a[i] > b[i]:
                        ttb_pred = 0
                        break
                    elif b[i] > a[i]:
                        ttb_pred = 1
                        break
                if ttb_pred == row['response']:
                    score += 1
        return score / len(df)
    return float(data.groupby('subject_id').apply(subject_score).mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0360 (var=0.0006)
- pi_5: 0.0000 (var=0.0000)
- pi_1: 0.2110 (var=0.0006)
- pi_2: 0.0281 (var=0.0005)
- pi_3: 0.1508 (var=0.0034)
- pi_4: 0.1150 (var=0.0005)


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
  "theory": "Biased Constant Choice: Due to low engagement or a lack of trial-by-trial feedback, subjects adopt a degenerate strategy of picking one option and sticking to it for the entire experiment. However, the initial selection of this preferred option is not necessarily a symmetric 50/50 split; subjects exhibit a population-level bias (e.g., a baseline preference for Option B). They make a biased probabilistic choice on their very first trial and then deterministically repeat that exact choice on all subsequent trials, resulting in zero within-subject variance while capturing asymmetric population preferences.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    # If the subject has already made a choice in a previous trial, they stick to it deterministically\n    if len(history[\"response\"]) > 0:\n        first_choice = history[\"response\"][0]\n        if first_choice == 0:\n            return np.array([1.0, 0.0])\n        else:\n            return np.array([0.0, 1.0])\n            \n    # On the very first trial, the subject chooses an option based on a population-level bias\n    p_a = float(parameters[\"p_a\"])\n    return np.array([p_a, 1.0 - p_a])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "p_a": "[0.0, 1.0]"
  },
  "rationale": "To address the arbiter's feedback, we need a theory that maintains the perfect 0.0000 within-subject variance of Theory 1 (pi_5) while allowing the population-level preference to deviate from a strict 50/50 split. By introducing a continuous parameter 'p_a' that governs the choice probability only on the very first trial, the model can capture population-level biases (e.g., towards Option B in unbalanced designs). By forcing all subsequent trials to deterministically copy the first trial's response via the history object, we guarantee that every simulated subject exhibits exactly zero within-subject variance, perfectly matching the empirical data on variance metrics."
}
```
```

## Usage

```json
{
  "prompt_token_count": 12327,
  "candidates_token_count": 559,
  "total_token_count": 16345
}
```
