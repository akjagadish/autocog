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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Propose a 'Weighted Additive Model (WADD)' to replace Probabilistic Cue Selection. Unlike pure Tallying (which ignores validities) and Random Cue (which samples a single cue), WADD integrates all available features by computing a sum of cue values weighted by their subjective validities. This allows the model to predict indifference (near 0.5) when a higher tally is perfectly offset by lower validities (as in Experiment 1), while still capturing the nuanced sensitivity (or lack thereof, depending on the softmax temperature) across different trial types. Ensure the model includes a softmax temperature and a lapse rate to accommodate the high degree of noise observed in the data (many metrics hover near 0.5).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.786)

**Description**
Probabilistic Cue Selection (Random Cue) posits that decision-makers do not deterministically use the most valid cue (like Take-The-Best) nor do they integrate all cues simultaneously (like WADD). Instead, on each trial, they sample a single cue with a probability proportional to its subjective validity. They then choose the option favored by that sampled cue, guessing uniformly if the sampled cue ties. This single-cue sampling process naturally generates probabilistic choices across trials, producing choice shares near 0.50 for conflict trials where different cues favor different options, without relying on extreme softmax noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Subjective validities used as sampling weights
    subj_weights = val * w
    sum_w = np.sum(subj_weights)
    
    if sum_w <= 1e-9:
        p_core = np.array([0.5, 0.5])
    else:
        p_cue = subj_weights / sum_w
        a, b = stim[0], stim[1]
        
        p_a = 0.0
        for j in range(n_features):
            if a[j] > b[j]:
                p_a += p_cue[j]
            elif a[j] == b[j]:
                p_a += p_cue[j] * 0.5
                
        p_core = np.array([p_a, 1.0 - p_a])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 0.5]
- weights: [(0.0, 1.0)] * n_features
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5071 (var=0.0028) vs this=0.4753 (var=0.0043)
- Experiment 2: real=0.4822 (var=0.0067) vs this=0.4017 (var=0.0165)
- Experiment 3: real=0.6489 (var=0.0031) vs this=0.7252 (var=0.0057)
- Experiment 4: real=0.5233 (var=0.0124) vs this=0.6092 (var=0.0121)
- Experiment 5: real=0.5075 (var=0.0019) vs this=0.4548 (var=0.0037)
- Experiment 6: real=0.0133 (var=0.1054) vs this=0.2583 (var=0.1245)


---

### `pi_4` (overall score: 0.534)

**Description**
Equal Weights (Tallying)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    
    # Count the number of positive features for each option
    tally_a = np.sum(stim[0])
    tally_b = np.sum(stim[1])
    
    if tally_a > tally_b:
        scores = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        scores = np.array([0.0, 1.0])
    else:
        # Guess randomly if there is a tie
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary tally score
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Independent lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5071 (var=0.0028) vs this=0.4778 (var=0.0027)
- Experiment 2: real=0.4822 (var=0.0067) vs this=0.3261 (var=0.0239)
- Experiment 3: real=0.6489 (var=0.0031) vs this=0.6970 (var=0.0079)
- Experiment 4: real=0.5233 (var=0.0124) vs this=0.6708 (var=0.0215)
- Experiment 5: real=0.5075 (var=0.0019) vs this=0.3387 (var=0.0148)
- Experiment 6: real=0.0133 (var=0.1054) vs this=-0.0233 (var=0.0692)


---

### `pi_2` (overall score: 0.164)

**Description**
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5071 (var=0.0028) vs this=0.4631 (var=0.0165)
- Experiment 2: real=0.4822 (var=0.0067) vs this=0.3089 (var=0.0415)
- Experiment 3: real=0.6489 (var=0.0031) vs this=0.8207 (var=0.0112)
- Experiment 4: real=0.5233 (var=0.0124) vs this=0.7192 (var=0.0279)
- Experiment 5: real=0.5075 (var=0.0019) vs this=0.3965 (var=0.0177)
- Experiment 6: real=0.0133 (var=0.1054) vs this=0.5833 (var=0.3689)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_cue1_winner(row):
        if row['option_a_ratings'][0] > row['option_b_ratings'][0]:
            return 0
        elif row['option_b_ratings'][0] > row['option_a_ratings'][0]:
            return 1
        return -1
    
    cue1_winners = data.apply(get_cue1_winner, axis=1)
    valid_trials = cue1_winners != -1
    if not valid_trials.any():
        return 0.5
    
    return np.mean(data.loc[valid_trials, 'response'] == cue1_winners[valid_trials])
```

**Observed (real) value:** 0.5071 (var=0.0028)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8364 (var=0.0110)
- pi_2: 0.4631 (var=0.0165)
- pi_3: 0.4753 (var=0.0043)
- pi_4: 0.4778 (var=0.0027)

### Experiment 2
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
            
    if conflict_trials == 0:
        return 0.5
    return float(ttb_matches / conflict_trials)
```

**Observed (real) value:** 0.4822 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3089 (var=0.0415)
- pi_1: 0.8200 (var=0.0128)
- pi_3: 0.4017 (var=0.0165)
- pi_4: 0.3261 (var=0.0239)

### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['chose_a'] = (data['response'] == 0).astype(float)
    
    # We focus on the '3v1' trials where WADD predicts near-deterministic choices
    target_trials = ['0111', '1110', '1011']
    df_target = data[data['a_str'].isin(target_trials)]
    
    if df_target.empty:
        return 0.5
        
    # Calculate the proportion of A choices per subject and trial type
    grouped = df_target.groupby(['subject_id', 'a_str'])['chose_a'].mean()
    
    # Consistency is how far the choice proportion is from 0.5 (i.e., max(P(A), P(B)))
    consistency = np.maximum(grouped, 1.0 - grouped)
    
    return float(consistency.mean())
```

**Observed (real) value:** 0.6489 (var=0.0031)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7252 (var=0.0057)
- pi_2: 0.8207 (var=0.0112)
- pi_1: 0.8637 (var=0.0103)
- pi_4: 0.6970 (var=0.0079)

### Experiment 4
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Objective validities used in the experiment
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # Reconstruct the feature matrices
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate objective scores and differences
    score_a = a_ratings @ v
    score_b = b_ratings @ v
    diff = score_a - score_b
    
    # Select "easy" trials with a large score difference (abs(diff) >= 0.65)
    # In this design, this cleanly isolates Trials 7, 9, and 10.
    mask = np.abs(diff) >= 0.65
    
    if not np.any(mask):
        return 0.5
        
    # Determine if the subject chose the option with the higher objective score
    is_correct = ((diff > 0) & (data['response'] == 0)) | ((diff < 0) & (data['response'] == 1))
    
    # Return the simple accuracy on these easy trials
    return float(is_correct[mask].mean())
```

**Observed (real) value:** 0.5233 (var=0.0124)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7192 (var=0.0279)
- pi_3: 0.6092 (var=0.0121)
- pi_1: 0.6033 (var=0.0059)
- pi_4: 0.6708 (var=0.0215)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))

```

**Observed (real) value:** 0.5075 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4548 (var=0.0037)
- pi_4: 0.3387 (var=0.0148)
- pi_1: 0.7346 (var=0.0044)
- pi_2: 0.3965 (var=0.0177)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    m1 = a_str == '11000'
    m2 = a_str == '10100'
    m4 = a_str == '01111'
    m7 = a_str == '00111'
    
    def get_p_a(mask):
        subset = data.loc[mask, 'response']
        if len(subset) == 0:
            return 0.5
        return float((subset == 0).mean())
        
    p1 = get_p_a(m1)
    p2 = get_p_a(m2)
    p4 = get_p_a(m4)
    p7 = get_p_a(m7)
    
    return float((p4 - p7) + (p1 - p2))
```

**Observed (real) value:** 0.0133 (var=0.1054)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0233 (var=0.0692)
- pi_3: 0.2583 (var=0.1245)
- pi_1: -0.0333 (var=0.0550)
- pi_2: 0.5833 (var=0.3689)


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
  "theory": "Weighted Additive Model (WADD) with Subjective Validities: Decision-makers integrate all available features by computing a weighted sum of cue values for each option. The weights are subjective validities, modeled as a non-linear transformation of the objective validities. Choices are made probabilistically using a softmax function over the options' weighted sums, along with an independent lapse rate to account for random guessing, capturing the high degree of noise observed in the data.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Transform objective validities into subjective weights\n    subj_weights = validities ** gamma\n    \n    # Compute WADD scores\n    scores = stim @ subj_weights\n    \n    # Softmax over scores\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Incorporate lapse rate\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "gamma": "[0.0, 5.0]",
    "beta": "[0.0, 10.0]",
    "epsilon": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's recommendation, this Weighted Additive Model (WADD) integrates all cues by computing a sum of cue values weighted by their subjective validities. Subjective validities are formalized as a power transformation of objective validities (via the 'gamma' parameter) to allow for over- or under-weighting of cue differences. To capture the fact that many of the observed metrics hover near 0.5, the model incorporates both a softmax temperature ('beta') and an independent lapse rate ('epsilon'). This allows the model to predict indifference when conflicting tallies and validities offset each other or when decision noise is high."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9364,
  "candidates_token_count": 607,
  "total_token_count": 10999
}
```
