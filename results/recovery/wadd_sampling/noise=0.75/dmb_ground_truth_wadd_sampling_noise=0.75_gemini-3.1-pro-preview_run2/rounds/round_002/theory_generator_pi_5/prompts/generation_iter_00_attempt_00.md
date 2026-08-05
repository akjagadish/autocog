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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a new theory: 'Strategy Mixture of TTB and WADD'. Since the data consistently falls halfway between the predictions of a purely non-compensatory approach (TTB) and a purely compensatory approach with log-odds weighting (WADD), the new theory should posit that decision-makers draw from a repertoire containing both TTB and WADD. By using a mixture parameter to blend TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums), this theory will naturally predict the intermediate (~0.50) choice proportions observed on conflict trials where TTB and WADD make opposite predictions.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.840)

**Description**
Strategy Mixture Theory: People do not rely exclusively on a single decision strategy. Instead, individuals or trials draw from a repertoire of strategies, specifically a mixture of a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). The probability of using TTB versus Tallying is governed by a mixture parameter, allowing the model to capture intermediate choice proportions on conflict trials where the two strategies make opposite predictions. Tallying treats all cues equally, ensuring it provides a genuine compensatory contrast to TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Tallying (Compensatory) Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_comp = np.array([a_wins, b_wins])
    
    z_comp = beta * (scores_comp - np.max(scores_comp))
    e_comp = np.exp(z_comp)
    p_comp = e_comp / np.sum(e_comp)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_comp
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.4392 (var=0.0459)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.5032 (var=0.0323)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.5413 (var=0.0378)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.5035 (var=0.0409)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.1897 (var=0.0224)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.2092 (var=0.0138)


---

### `pi_3` (overall score: 0.486)

**Description**
People make decisions by computing a subjective value for each option through a Weighted Additive (WADD) process. They take a weighted sum of the features for each option, where the weight of each feature is the log-odds of its provided cue validity. This transformation appropriately scales probabilities into additive evidence. The probability of choosing an option is then determined by a softmax over the options' subjective values, combined with a lapse rate for random errors. This compensatory mechanism allows multiple lower-validity cues to jointly outweigh a single high-validity cue, naturally predicting the graded, intermediate behavior observed on conflict trials compared to non-compensatory heuristics like Take The Best or Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds to represent additive evidence
    # Clip to avoid division by zero or log(0)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.4117 (var=0.0076)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.4159 (var=0.0049)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.6379 (var=0.0059)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.6508 (var=0.0056)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.7826 (var=0.0190)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.8163 (var=0.0124)


---

### `pi_2` (overall score: 0.150)

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
- Experiment 1: real=0.4775 (var=0.0054) vs this=0.1396 (var=0.0094)
- Experiment 2: real=0.5340 (var=0.0037) vs this=0.8454 (var=0.0089)
- Experiment 3: real=0.5275 (var=0.0041) vs this=0.8562 (var=0.0088)
- Experiment 4: real=0.5571 (var=0.0045) vs this=0.8621 (var=0.0056)
- Experiment 5: real=0.4738 (var=0.0085) vs this=0.1344 (var=0.0102)
- Experiment 6: real=0.4550 (var=0.0058) vs this=0.2746 (var=0.0075)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (assumes features are sorted by descending validity)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pred = 0
        elif sum_b > sum_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        # Focus strictly on conflict trials where the theories make opposite predictions
        if ttb_pred is not None and tally_pred is not None and ttb_pred != tally_pred:
            matches.append(1 if row['response'] == ttb_pred else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.4775 (var=0.0054)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8421 (var=0.0086)
- pi_2: 0.1396 (var=0.0094)
- pi_3: 0.4117 (var=0.0076)
- pi_4: 0.4392 (var=0.0459)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        tally_pred = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        cue_order = np.argsort(-val)
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            total += 1
            
    return tally_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5340 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8454 (var=0.0089)
- pi_1: 0.1371 (var=0.0093)
- pi_3: 0.4159 (var=0.0049)
- pi_4: 0.5032 (var=0.0323)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    w = np.array([2.19722458, 1.38629436, 1.09861229, 0.40546511])
    
    wadd_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        if ttb_winner is not None and ttb_winner != wadd_winner:
            subject_choice = row['response']
            wadd_choices.append(1 if subject_choice == wadd_winner else 0)
            
    if len(wadd_choices) == 0:
        return 0.5
        
    return float(np.mean(wadd_choices))
```

**Observed (real) value:** 0.5275 (var=0.0041)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1442 (var=0.0139)
- pi_3: 0.6379 (var=0.0059)
- pi_2: 0.8562 (var=0.0088)
- pi_4: 0.5413 (var=0.0378)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5571 (var=0.0045)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6508 (var=0.0056)
- pi_1: 0.1527 (var=0.0092)
- pi_2: 0.8621 (var=0.0056)
- pi_4: 0.5035 (var=0.0409)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify critical trials where TTB and Tallying both prefer Option A, but WADD prefers Option B.
    # These are trials where Option A has the highest validity cue (A[0] == 1), Option B has the next two highest (B[1] == 1, B[2] == 1),
    # and Option A has more positive cues overall (sum(A) > sum(B)).
    is_critical = data.apply(
        lambda row: sum(row['option_a_ratings']) > sum(row['option_b_ratings']) 
                    and row['option_a_ratings'][0] == 1 
                    and row['option_b_ratings'][1] == 1,
        axis=1
    )
    if not is_critical.any():
        return 0.5
    
    # Return the proportion of times Option B was chosen on these critical trials.
    # Strategy Mixture will be close to 0 (since both TTB and Tallying prefer A).
    # WADD will be close to 1 (since the log-odds of cues 2 and 3 outweigh cue 1 and the minor cues).
    return float(data.loc[is_critical, 'response'].mean())
```

**Observed (real) value:** 0.4738 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1897 (var=0.0224)
- pi_3: 0.7826 (var=0.0190)
- pi_1: 0.1410 (var=0.0149)
- pi_2: 0.1344 (var=0.0102)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_profile = (0, 1, 1, 0, 0)
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == target_profile:
            total += 1
            if row['response'] == 0:
                matches += 1
        elif b == target_profile:
            total += 1
            if row['response'] == 1:
                matches += 1
    if total == 0:
        return 0.5
    return float(matches / total)
```

**Observed (real) value:** 0.4550 (var=0.0058)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8163 (var=0.0124)
- pi_4: 0.2092 (var=0.0138)
- pi_1: 0.1737 (var=0.0115)
- pi_2: 0.2746 (var=0.0075)


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
  "theory": "Strategy Mixture of TTB and WADD: Decision-makers draw from a repertoire of strategies, specifically mixing a non-compensatory heuristic (Take The Best) and a fully compensatory one (Weighted Additive). By probabilistically blending TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums of all cues), the model naturally predicts the intermediate choice proportions observed on conflict trials where the two strategies make opposite predictions.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be a (2, n_features) array.\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    beta_ttb = float(parameters[\"beta_ttb\"])\n    beta_wadd = float(parameters[\"beta_wadd\"])\n    epsilon = float(parameters[\"epsilon\"])\n    p_ttb_weight = float(parameters[\"p_ttb\"])\n    \n    # --- Take The Best (TTB) Prediction ---\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    winner_ttb = None\n    for j in cue_order:\n        if a[j] > b[j]:\n            winner_ttb = 0\n            break\n        if b[j] > a[j]:\n            winner_ttb = 1\n            break\n            \n    if winner_ttb is None:\n        p_ttb = np.array([0.5, 0.5])\n    else:\n        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])\n        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))\n        e_ttb = np.exp(z_ttb)\n        p_ttb = e_ttb / np.sum(e_ttb)\n        \n    # --- Weighted Additive (WADD) Prediction ---\n    # Transform validities to log-odds to represent additive evidence\n    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)\n    w = np.log(val_clipped / (1.0 - val_clipped))\n    v_a = np.sum(a * w)\n    v_b = np.sum(b * w)\n    scores_wadd = np.array([v_a, v_b])\n    \n    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))\n    e_wadd = np.exp(z_wadd)\n    p_wadd = e_wadd / np.sum(e_wadd)\n    \n    # --- Mixture ---\n    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_wadd\n    \n    n_opts = p_mixed.shape[0]\n    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / np.sum(probabilities)\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "beta_ttb": "[0.01, 10.0]",
    "beta_wadd": "[0.01, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "p_ttb": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's recommendation, we propose a Strategy Mixture of Take The Best (TTB) and Weighted Additive (WADD). Unlike the previous mixture model that used Tallying, mixing TTB with WADD captures the tension between purely non-compensatory decision-making and optimal log-odds compensatory integration. The model naturally predicts intermediate choice proportions on conflict trials where TTB and WADD make opposite predictions, and allowing distinct response noise parameters (beta_ttb and beta_wadd) accommodates the different scaling of their respective evidence values."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9609,
  "candidates_token_count": 1118,
  "total_token_count": 13559
}
```
