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

Replace Theory 2 with a 'Tallying with Noise' or 'Equal Weighting' theory. Since subjects do not exhibit the strong spatial biases posited by Theory 2 (as shown by Exps 1 and 2), but also do not follow the TTB predictions in Exps 9 and 10, a theory that assumes subjects simply count the number of positive features for each option (Tallying) and choose the one with the most (subject to noise) might better capture the specific trial-by-trial deviations while maintaining a 0.5 baseline choice rate across symmetrically balanced designs.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.800)

**Description**
Cognitive Overload / Random Choice: When faced with multiple conflicting binary cues and explicitly stated validities, subjects experience cognitive overload or lack sufficient motivation to integrate the information systematically. Consequently, their decision-making process collapses into pure random guessing, treating both options as equally likely to be chosen regardless of their specific feature values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess randomly.
    # We extract the lapse_rate parameter to satisfy the parameter mapping requirement.
    lapse_rate = float(parameters["lapse_rate"])
    
    # Always return a uniform distribution over the two options.
    return np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- lapse_rate: {1.0}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.4977 (var=0.0023)
- Experiment 2: real=0.5020 (var=0.0043) vs this=0.4993 (var=0.0037)
- Experiment 3: real=0.5125 (var=0.0033) vs this=0.4954 (var=0.0043)
- Experiment 4: real=0.5067 (var=0.0022) vs this=0.5055 (var=0.0021)
- Experiment 5: real=0.5167 (var=0.0071) vs this=0.5294 (var=0.0053)
- Experiment 6: real=0.5183 (var=0.0092) vs this=0.4875 (var=0.0094)
- Experiment 7: real=1198.0000 (var=26.7936) vs this=2441.0000 (var=21.3476)
- Experiment 8: real=2191200.0000 (var=136485.2736) vs this=8548800.0000 (var=129352.0896)
- Experiment 9: real=0.0383 (var=0.0013) vs this=0.0408 (var=0.0010)
- Experiment 10: real=0.2472 (var=0.0000) vs this=0.2479 (var=0.0000)


---

### `pi_5` (overall score: 0.795)

**Description**
Take-The-Best (TTB) heuristic with extreme noise: Decision-makers avoid cognitive overload by not integrating all information. Instead, they search through cues in descending order of their explicitly stated validities. The choice is based entirely on the first cue that discriminates between the two options. However, due to the high cognitive demands of the task or lack of motivation, their behavior is overwhelmingly stochastic, requiring extremely high lapse rates and very low softmax temperatures to capture the near-random empirical choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue found
        p_core = np.array([0.5, 0.5])
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * scores
        z -= np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- beta: [0.0, 1.0]
- epsilon: [0.8, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.5177 (var=0.0018)
- Experiment 2: real=0.5020 (var=0.0043) vs this=0.4823 (var=0.0041)
- Experiment 3: real=0.5125 (var=0.0033) vs this=0.5308 (var=0.0049)
- Experiment 4: real=0.5067 (var=0.0022) vs this=0.4840 (var=0.0031)
- Experiment 5: real=0.5167 (var=0.0071) vs this=0.5128 (var=0.0059)
- Experiment 6: real=0.5183 (var=0.0092) vs this=0.5092 (var=0.0080)
- Experiment 7: real=1198.0000 (var=26.7936) vs this=2507.0000 (var=22.7204)
- Experiment 8: real=2191200.0000 (var=136485.2736) vs this=8558400.0000 (var=203345.5104)
- Experiment 9: real=0.0383 (var=0.0013) vs this=0.0417 (var=0.0013)
- Experiment 10: real=0.2472 (var=0.0000) vs this=0.2478 (var=0.0000)


---

### `pi_6` (overall score: 0.407)

**Description**
Biased Random Guessing: Due to cognitive overload from processing multiple conflicting binary cues with explicit validities, subjects abandon systematic cue integration and resort to random guessing. However, their guessing is not perfectly uniform; instead, they exhibit a parameterized spatial or positional bias (e.g., a baseline preference for Option A over Option B due to reading order). This allows the model to capture near-random choice behavior while gracefully absorbing slight empirical deviations from exactly 50%.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess, but with a spatial/positional bias.
    p_a = float(parameters["p_a"])
    
    # Return the biased probabilities for Option A and Option B.
    return np.array([p_a, 1.0 - p_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_a: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4900 (var=0.0012) vs this=0.4875 (var=0.0209)
- Experiment 2: real=0.5020 (var=0.0043) vs this=0.5023 (var=0.0053)
- Experiment 3: real=0.5125 (var=0.0033) vs this=0.4854 (var=0.0964)
- Experiment 4: real=0.5067 (var=0.0022) vs this=0.4902 (var=0.0142)
- Experiment 5: real=0.5167 (var=0.0071) vs this=0.4761 (var=0.0114)
- Experiment 6: real=0.5183 (var=0.0092) vs this=0.5125 (var=0.0049)
- Experiment 7: real=1198.0000 (var=26.7936) vs this=2394.0000 (var=62.7456)
- Experiment 8: real=2191200.0000 (var=136485.2736) vs this=8524800.0000 (var=124320.1536)
- Experiment 9: real=0.0383 (var=0.0013) vs this=0.2175 (var=0.0191)
- Experiment 10: real=0.2472 (var=0.0000) vs this=0.1603 (var=0.0044)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0012)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8229 (var=0.0112)
- pi_2: 0.2154 (var=0.0039)
- pi_3: 0.5325 (var=0.0022)
- pi_4: 0.4977 (var=0.0023)
- pi_5: 0.5177 (var=0.0018)
- pi_6: 0.4875 (var=0.0209)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_tallying = 0
    total_strict = 0
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
            continue
        total_strict += 1
        if row['response'] == tally_pred:
            match_tallying += 1
    return match_tallying / total_strict if total_strict > 0 else 0.5
```

**Observed (real) value:** 0.5020 (var=0.0043)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8493 (var=0.0099)
- pi_1: 0.1607 (var=0.0132)
- pi_3: 0.3943 (var=0.0036)
- pi_4: 0.4993 (var=0.0037)
- pi_5: 0.4823 (var=0.0041)
- pi_6: 0.5023 (var=0.0053)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        # These are trials 1, 3, 4, and 5 from the design.
        # In these trials, WADD strongly prefers Option A due to log-odds weighting of highly valid cues,
        # whereas Tallying strictly prefers Option B because B has more winning features.
        return a in [
            (1, 0, 0, 0, 0),
            (1, 0, 1, 0, 0),
            (0, 1, 0, 0, 0),
            (1, 0, 0, 1, 0)
        ]
        
    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.5125 (var=0.0033)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7296 (var=0.0132)
- pi_2: 0.1363 (var=0.0068)
- pi_1: 0.8471 (var=0.0177)
- pi_4: 0.4954 (var=0.0043)
- pi_5: 0.5308 (var=0.0049)
- pi_6: 0.4854 (var=0.0964)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the number of winning features for each option
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Filter out trials where Tallying predicts a tie
    mask = a_wins != b_wins
    filtered_data = data[mask]
    if len(filtered_data) == 0:
        return 0.5
        
    a_wins_f = a_wins[mask]
    b_wins_f = b_wins[mask]
    
    # Determine the Tallying prediction (0 for A, 1 for B)
    tally_prediction = (b_wins_f > a_wins_f).astype(int)
    
    # Return the proportion of choices that align with the Tallying prediction
    return (filtered_data['response'] == tally_prediction).mean()
```

**Observed (real) value:** 0.5067 (var=0.0022)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8426 (var=0.0130)
- pi_3: 0.4448 (var=0.0025)
- pi_1: 0.2457 (var=0.0038)
- pi_4: 0.5055 (var=0.0021)
- pi_5: 0.4840 (var=0.0031)
- pi_6: 0.4902 (var=0.0142)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    valid_trials = sum_a != sum_b
    if not valid_trials.any():
        return 0.5
        
    df_valid = data[valid_trials]
    sa = sum_a[valid_trials]
    sb = sum_b[valid_trials]
    
    chose_a = (df_valid['response'] == 0)
    chose_b = (df_valid['response'] == 1)
    
    correct = (chose_a & (sa > sb)) | (chose_b & (sb > sa))
    return float(correct.mean())
```

**Observed (real) value:** 0.5167 (var=0.0071)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8550 (var=0.0107)
- pi_4: 0.5294 (var=0.0053)
- pi_1: 0.6144 (var=0.0052)
- pi_2: 0.8617 (var=0.0104)
- pi_5: 0.5128 (var=0.0059)
- pi_6: 0.4761 (var=0.0114)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    correct = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 1, 1, 1, 1) and b == (0, 0, 0, 0, 0):
            if row['response'] == 0:
                correct += 1
            total += 1
        elif a == (0, 0, 0, 0, 0) and b == (1, 1, 1, 1, 1):
            if row['response'] == 1:
                correct += 1
            total += 1
    return float(correct / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5183 (var=0.0092)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4875 (var=0.0094)
- pi_3: 0.8425 (var=0.0095)
- pi_1: 0.8617 (var=0.0132)
- pi_2: 0.8583 (var=0.0135)
- pi_5: 0.5092 (var=0.0080)
- pi_6: 0.5125 (var=0.0049)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    first_diff_idx = np.argmax(diff != 0, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_winner = (first_diff_val < 0).astype(int)
    matches = (data['response'].values == ttb_winner)
    
    return float(np.sum(matches))
```

**Observed (real) value:** 1198.0000 (var=26.7936)
**Other theories' values on this metric (for reference):**
- pi_5: 2507.0000 (var=22.7204)
- pi_4: 2441.0000 (var=21.3476)
- pi_1: 4040.0000 (var=87.0800)
- pi_2: 2363.0000 (var=6.7524)
- pi_3: 3417.0000 (var=20.8244)
- pi_6: 2394.0000 (var=62.7456)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    is_disc = np.any(diff != 0, axis=1)
    
    if not np.any(is_disc):
        return 0.0
        
    diff_disc = diff[is_disc]
    responses = data['response'].values[is_disc]
    
    first_nonzero_idx = np.argmax(diff_disc != 0, axis=1)
    first_diff = diff_disc[np.arange(len(diff_disc)), first_nonzero_idx]
    
    ttb_winner = np.where(first_diff > 0, 0, 1)
    correct = np.sum(responses == ttb_winner)
    
    return float(correct * len(data))

```

**Observed (real) value:** 2191200.0000 (var=136485.2736)
**Other theories' values on this metric (for reference):**
- pi_4: 8548800.0000 (var=129352.0896)
- pi_5: 8558400.0000 (var=203345.5104)
- pi_1: 14764800.0000 (var=464943.5136)
- pi_2: 2366400.0000 (var=456011.3664)
- pi_3: 6801600.0000 (var=147312.2304)
- pi_6: 8524800.0000 (var=124320.1536)

### Experiment 9
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    subj_means = data.groupby('subject_id')['response'].mean()
    return float(abs(subj_means - 0.5).mean())
```

**Observed (real) value:** 0.0383 (var=0.0013)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0417 (var=0.0013)
- pi_6: 0.2175 (var=0.0191)
- pi_1: 0.0325 (var=0.0008)
- pi_2: 0.0327 (var=0.0012)
- pi_3: 0.0417 (var=0.0011)
- pi_4: 0.0408 (var=0.0010)

### Experiment 10
**Design**
  A=[1, 0, 0]  B=[0, 1, 1]
  A=[0, 1, 1]  B=[1, 0, 0]
  A=[1, 1, 0]  B=[1, 0, 1]
  A=[1, 0, 1]  B=[1, 1, 0]
  A=[1, 1, 1]  B=[1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the proportion of Option B choices for each subject
    subject_means = data.groupby('subject_id')['response'].mean()
    
    # Calculate p * (1 - p) for each subject
    # This represents the variance of the Bernoulli distribution of their choices
    subject_bernoulli_vars = subject_means * (1.0 - subject_means)
    
    # Return the average of these subject-level variances.
    # By averaging inside the metric, the pooled point_estimate will correctly
    # reflect the mean of the subject-level metrics, ensuring a valid t-test.
    return float(subject_bernoulli_vars.mean())
```

**Observed (real) value:** 0.2472 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1603 (var=0.0044)
- pi_5: 0.2478 (var=0.0000)
- pi_1: 0.2485 (var=0.0000)
- pi_2: 0.2477 (var=0.0000)
- pi_3: 0.2487 (var=0.0000)
- pi_4: 0.2479 (var=0.0000)


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
  "theory": "Tallying with Noise: Subjects avoid the cognitive overload of weighting cues by their explicitly stated validities. Instead, they use an equal-weighting (Tallying) heuristic, simply counting the number of positive features for each option and favoring the one with the higher count. The decision is subject to random noise (lapse rate) and choice stochasticity (softmax temperature), which captures trial-by-trial deviations without introducing systematic spatial biases.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    \n    tally_a = np.sum(a)\n    tally_b = np.sum(b)\n    \n    beta = float(parameters[\"beta\"])\n    scores = np.array([tally_a, tally_b]) * beta\n    \n    scores -= np.max(scores)\n    p = np.exp(scores)\n    p_core = p / np.sum(p)\n    \n    epsilon = float(parameters[\"epsilon\"])\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "beta": "[0.0, 10.0]",
    "epsilon": "[0.0, 1.0]"
  },
  "rationale": "By adopting a Tallying (equal weighting) heuristic combined with decision noise, this model perfectly addresses the arbiter's feedback. It eliminates the arbitrary spatial biases that caused poor fits in symmetric experiments (like Exps 1, 2, 9, and 10), while still capturing the trial-by-trial variance through the difference in positive cue counts. The combination of a softmax temperature (beta) and a lapse rate (epsilon) ensures the model can flexibly capture both sensitivity to tally differences and baseline random guessing due to cognitive overload."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11628,
  "candidates_token_count": 529,
  "total_token_count": 12759
}
```
