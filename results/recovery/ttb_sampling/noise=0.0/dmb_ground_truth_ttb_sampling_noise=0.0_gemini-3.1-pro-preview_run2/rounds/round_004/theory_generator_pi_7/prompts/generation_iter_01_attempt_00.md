# generation_iter_01_attempt_00

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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a brand-new theory to replace the Strategy Selection (TTB + Tallying) mixture. Since the data strongly supports a non-compensatory lexicographic process with no significant Tallying influence, the new theory should explore variations of cue-based processing. For example, consider a Probabilistic Cue Search model where decision-makers do not always search in strict descending order of validity, but instead sample cues probabilistically weighted by their validities; or a Sequential Evidence Accumulation model where cues are processed one by one but the threshold for a decision depends on the validity of the cue. This would provide a more nuanced alternative to pure TTB without relying on compensatory tallying.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.965)

**Description**
Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1575 (var=0.0090) vs this=0.1037 (var=0.0066)
- Experiment 2: real=0.1400 (var=0.0213) vs this=0.1263 (var=0.0110)
- Experiment 3: real=0.8538 (var=0.0067) vs this=0.8667 (var=0.0052)
- Experiment 4: real=0.8292 (var=0.0119) vs this=0.8700 (var=0.0069)
- Experiment 5: real=-0.0062 (var=0.0059) vs this=0.0256 (var=0.0083)
- Experiment 6: real=0.0070 (var=0.0102) vs this=-0.0076 (var=0.0133)
- Experiment 7: real=-0.0133 (var=0.0058) vs this=-0.0028 (var=0.0057)
- Experiment 8: real=-0.0079 (var=0.0084) vs this=0.0090 (var=0.0053)
- Experiment 9: real=-0.1642 (var=0.6574) vs this=-0.4917 (var=0.9874)
- Experiment 10: real=9.0000 (var=17.2704) vs this=1.0000 (var=21.4596)


---

### `pi_6` (overall score: 0.955)

**Description**
Strategy Selection: Decision-makers predominantly rely on a non-compensatory Take The Best (TTB) heuristic but probabilistically mix in a simple compensatory Tallying strategy. Unlike validity-dependent lapse models, this mixture maintains flat adherence across different validities of discriminating cues, while allowing small but consistent deviations in choice probabilities when TTB and Tallying conflict. The probability of tallying is kept very small to align with the near-zero deviations observed in experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(-validities, kind='stable')
    p_ttb_choice = np.array([0.5, 0.5])
    for i in order:
        if a[i] > b[i]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    p_tally = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of strategies
    p_core = (1.0 - p_tally) * p_ttb_choice + p_tally * p_tally_choice
    
    # Apply uniform response noise (lapse)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- p_tally: [0.0, 0.05]
- epsilon: [0.0, 0.4]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1575 (var=0.0090) vs this=0.1092 (var=0.0053)
- Experiment 2: real=0.1400 (var=0.0213) vs this=0.1275 (var=0.0087)
- Experiment 3: real=0.8538 (var=0.0067) vs this=0.8827 (var=0.0037)
- Experiment 4: real=0.8292 (var=0.0119) vs this=0.8992 (var=0.0044)
- Experiment 5: real=-0.0062 (var=0.0059) vs this=0.0250 (var=0.0057)
- Experiment 6: real=0.0070 (var=0.0102) vs this=0.0343 (var=0.0102)
- Experiment 7: real=-0.0133 (var=0.0058) vs this=-0.0017 (var=0.0052)
- Experiment 8: real=-0.0079 (var=0.0084) vs this=-0.0117 (var=0.0056)
- Experiment 9: real=-0.1642 (var=0.6574) vs this=4.7706 (var=1.1655)
- Experiment 10: real=9.0000 (var=17.2704) vs this=27.0000 (var=15.5684)


---

### `pi_5` (overall score: 0.887)

**Description**
Take The Best with Validity-Dependent Confidence: Decision makers use a non-compensatory, lexicographic strategy (Take The Best) to choose between options. However, their execution of this strategy is noisy, and the degree of noise (lapse rate) depends on the validity of the discriminating cue. When options are discriminated by a highly valid cue, confidence is high and the lapse rate is low. When the discriminating cue has low validity, confidence is lower, leading to a higher probability of guessing or lapsing. This maintains the non-compensatory nature of TTB while naturally explaining variations in performance across different trial structures.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(-validities, kind='stable')
    
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    p_core = np.array([0.5, 0.5])
    v_discriminating = None
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            v_discriminating = validities[i]
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            v_discriminating = validities[i]
            break
            
    if v_discriminating is not None:
        # Lapse rate increases as validity of the discriminating cue decreases
        lapse = epsilon + gamma * (1.0 - v_discriminating)
        lapse = max(0.0, min(1.0, lapse))
        return (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    else:
        # Guess if no cues discriminate
        return np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1575 (var=0.0090) vs this=0.1529 (var=0.0076)
- Experiment 2: real=0.1400 (var=0.0213) vs this=0.1300 (var=0.0136)
- Experiment 3: real=0.8538 (var=0.0067) vs this=0.8073 (var=0.0066)
- Experiment 4: real=0.8292 (var=0.0119) vs this=0.8081 (var=0.0080)
- Experiment 5: real=-0.0062 (var=0.0059) vs this=-0.0150 (var=0.0080)
- Experiment 6: real=0.0070 (var=0.0102) vs this=0.0327 (var=0.0199)
- Experiment 7: real=-0.0133 (var=0.0058) vs this=0.1250 (var=0.0101)
- Experiment 8: real=-0.0079 (var=0.0084) vs this=0.1471 (var=0.0176)
- Experiment 9: real=-0.1642 (var=0.6574) vs this=15.1459 (var=1.0516)
- Experiment 10: real=9.0000 (var=17.2704) vs this=34.0000 (var=32.3776)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1968 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1968 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying strongly prefers A, WADD generally prefers B
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 2: Tallying strongly prefers B, WADD generally prefers A
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            score += resp
            count += 1
        # Trial 3: Tallying is indifferent (50/50), WADD generally prefers B
        elif a == (0, 1, 0, 0, 1) and b == (1, 0, 1, 0, 0):
            score += (1 - resp)
            count += 1
        # Trial 4: Tallying is indifferent (50/50), WADD generally prefers A
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 0, 1):
            score += resp
            count += 1
            
    return float(score / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.1575 (var=0.0090)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2288 (var=0.0141) (Δ vs real +0.0713)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6808 (var=0.0049)
- pi_2: 0.5171 (var=0.0377)
- pi_3: 0.1037 (var=0.0066)
- pi_4: 0.1358 (var=0.0050)
- pi_5: 0.1529 (var=0.0076)
- pi_6: 0.1092 (var=0.0053)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert option A ratings to a string for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1], B = [1, 1, 0, 0, 0]
    t1_mask = a_str == '00111'
    
    if t1_mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1400 (var=0.0213)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1888 (var=0.0305) (Δ vs real +0.0487)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5000 (var=0.0906)
- pi_1: 0.8550 (var=0.0190)
- pi_3: 0.1263 (var=0.0110)
- pi_4: 0.1800 (var=0.0163)
- pi_5: 0.1300 (var=0.0136)
- pi_6: 0.1275 (var=0.0087)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1
        
    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    matches = (np.array(data['response']) == np.array(ttb_choices))
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8538 (var=0.0067)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6793 (var=0.0292) (Δ vs real -0.1744)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8667 (var=0.0052)
- pi_2: 0.3989 (var=0.0150)
- pi_1: 0.2884 (var=0.0044)
- pi_4: 0.8404 (var=0.0062)
- pi_5: 0.8073 (var=0.0066)
- pi_6: 0.8827 (var=0.0037)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is not None:
            matches.append(1 if resp == ttb_pred else 0)
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.8292 (var=0.0119)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7385 (var=0.0162) (Δ vs real -0.0906)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4113 (var=0.0159)
- pi_3: 0.8700 (var=0.0069)
- pi_1: 0.3394 (var=0.0015)
- pi_4: 0.8740 (var=0.0051)
- pi_5: 0.8081 (var=0.0080)
- pi_6: 0.8992 (var=0.0044)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Agreement trials (where TTB and Tallying agree)
    # Trial 2: A='11100', B='00010' -> Both choose A (response 0)
    # Trial 5: A='00010', B='11100' -> Both choose B (response 1)
    t2_mask = (a_str == '11100') & (b_str == '00010')
    t5_mask = (a_str == '00010') & (b_str == '11100')
    
    errors = 0
    total_agree = 0
    if t2_mask.sum() > 0:
        errors += (data.loc[t2_mask, 'response'] == 1).sum()
        total_agree += t2_mask.sum()
    if t5_mask.sum() > 0:
        errors += (data.loc[t5_mask, 'response'] == 0).sum()
        total_agree += t5_mask.sum()
        
    e_agree = errors / total_agree if total_agree > 0 else 0
    
    # Conflict trials (where TTB and Tallying completely disagree)
    # Trial 1: A='10000', B='01110' -> TTB chooses A (0), Tallying chooses B (1)
    # Trial 4: A='01110', B='10000' -> TTB chooses B (1), Tallying chooses A (0)
    t1_mask = (a_str == '10000') & (b_str == '01110')
    t4_mask = (a_str == '01110') & (b_str == '10000')
    
    tally_choices = 0
    total_conflict = 0
    if t1_mask.sum() > 0:
        tally_choices += (data.loc[t1_mask, 'response'] == 1).sum()
        total_conflict += t1_mask.sum()
    if t4_mask.sum() > 0:
        tally_choices += (data.loc[t4_mask, 'response'] == 0).sum()
        total_conflict += t4_mask.sum()
        
    p_tally = tally_choices / total_conflict if total_conflict > 0 else 0
    
    # Subtracting agreement error rate controls for the lapse rate (epsilon)
    return float(p_tally - e_agree)
```

**Observed (real) value:** -0.0062 (var=0.0059)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2150 (var=0.0270) (Δ vs real +0.2213)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0256 (var=0.0083)
- pi_4: 0.1125 (var=0.0126)
- pi_1: 0.7450 (var=0.0265)
- pi_2: 0.4794 (var=0.1093)
- pi_5: -0.0150 (var=0.0080)
- pi_6: 0.0250 (var=0.0057)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements_match = []
    conflicts_match = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues are already ordered by validity 0 to 4)
        ttb_choice = -1
        for i in range(5):
            if A[i] > B[i]:
                ttb_choice = 0
                break
            elif B[i] > A[i]:
                ttb_choice = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for a, b in zip(A, B) if a > b)
        b_wins = sum(1 for a, b in zip(A, B) if b > a)
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            tally_choice = -1
            
        if ttb_choice == -1:
            continue
            
        is_match = 1 if resp == ttb_choice else 0
        
        if ttb_choice == tally_choice:
            agreements_match.append(is_match)
        elif tally_choice != -1 and ttb_choice != tally_choice:
            conflicts_match.append(is_match)
            
    if not agreements_match or not conflicts_match:
        return 0.0
        
    return float(np.mean(agreements_match) - np.mean(conflicts_match))
```

**Observed (real) value:** 0.0070 (var=0.0102)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1752 (var=0.0274) (Δ vs real +0.1683)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1660 (var=0.0107)
- pi_3: -0.0076 (var=0.0133)
- pi_1: 0.6848 (var=0.0484)
- pi_2: 0.5581 (var=0.0657)
- pi_5: 0.0327 (var=0.0199)
- pi_6: 0.0343 (var=0.0102)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_disc_cue(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return i
        return -1
        
    def get_ttb_favored(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return 0 if a[i] > b[i] else 1
        return -1

    disc_cues = data.apply(get_disc_cue, axis=1)
    ttb_favored = data.apply(get_ttb_favored, axis=1)
    
    adherence = (data['response'] == ttb_favored).astype(float)
    
    high_val_mask = disc_cues.isin([0, 1])
    low_val_mask = disc_cues.isin([3, 4])
    
    if high_val_mask.sum() == 0 or low_val_mask.sum() == 0:
        return 0.0
        
    high_val_adherence = adherence[high_val_mask].mean()
    low_val_adherence = adherence[low_val_mask].mean()
    
    return float(high_val_adherence - low_val_adherence)
```

**Observed (real) value:** -0.0133 (var=0.0058)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.1172 (var=0.0228) (Δ vs real -0.1039)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0028 (var=0.0057)
- pi_5: 0.1250 (var=0.0101)
- pi_1: -0.5483 (var=0.0193)
- pi_2: -0.4478 (var=0.0623)
- pi_4: -0.1072 (var=0.0096)
- pi_6: -0.0017 (var=0.0052)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_disc_cue(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                return i
        return -1

    cues = data.apply(get_disc_cue, axis=1)
    # In all designed trials, option A is favored by the highest discriminating cue.
    # Thus, '1 - response' computes the TTB accuracy (1 if chose A, 0 if chose B).
    acc = 1.0 - data['response']
    
    high_val = acc[cues.isin([0, 1])]
    low_val = acc[cues.isin([3, 4])]
    
    if len(high_val) == 0 or len(low_val) == 0:
        return 0.0
        
    return float(high_val.mean() - low_val.mean())
```

**Observed (real) value:** -0.0079 (var=0.0084)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.1635 (var=0.0287) (Δ vs real -0.1556)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1471 (var=0.0176)
- pi_3: 0.0090 (var=0.0053)
- pi_1: -0.6175 (var=0.0154)
- pi_2: -0.4679 (var=0.0622)
- pi_4: -0.0940 (var=0.0106)
- pi_6: -0.0117 (var=0.0056)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB choice (lexicographic)
    diff = a_ratings - b_ratings
    weights = 2 ** np.arange(diff.shape[1])[::-1]
    ttb_vals = diff.dot(weights)
    ttb_choice = np.where(ttb_vals > 0, 0, 1)
    
    # Tallying choice (compensatory)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    responses = data['response'].values
    errors = (responses != ttb_choice).astype(float)
    
    agree_mask = (ttb_choice == tally_choice) & (ttb_choice != -1)
    conflict_mask = (ttb_choice != tally_choice) & (tally_choice != -1) & (ttb_choice != -1)
    
    if np.sum(agree_mask) == 0 or np.sum(conflict_mask) == 0:
        return 0.0
        
    agree_err = errors[agree_mask]
    conflict_err = errors[conflict_mask]
    
    mean_diff = np.mean(conflict_err) - np.mean(agree_err)
    
    # Compute stable standard error using the pooled overall error rate
    p_pool = np.mean(errors)
    if p_pool <= 0.0 or p_pool >= 1.0:
        return 0.0
        
    var_pool = p_pool * (1.0 - p_pool)
    se = np.sqrt(var_pool * (1.0 / len(agree_err) + 1.0 / len(conflict_err)))
    
    z_stat = mean_diff / se
    
    # Scale by sqrt(N) to project the pooled statistical power
    n_subj = data['subject_id'].nunique()
    
    return float(z_stat * np.sqrt(n_subj))
```

**Observed (real) value:** -0.1642 (var=0.6574)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 53.0451 (var=0.8201) (Δ vs real +53.2094)
**Other theories' values on this metric (for reference):**
- pi_3: -0.4917 (var=0.9874)
- pi_6: 4.7706 (var=1.1655)
- pi_1: 276.0128 (var=2.7805)
- pi_2: 173.1854 (var=4.9730)
- pi_4: 81.5885 (var=0.5030)
- pi_5: 15.1459 (var=1.0516)

### Experiment 10
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agree_ttb_match = 0
    conflict_ttb_match = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Take The Best (TTB) prediction (cues are ordered by validity)
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        if ttb_pred == -1:
            continue
            
        ttb_match = 1 if row['response'] == ttb_pred else 0
        
        if ttb_pred == tally_pred:
            agree_ttb_match += ttb_match
        else:
            conflict_ttb_match += ttb_match
            
    # Weight agree matches by 2 because there are exactly twice as many conflict trials as agree trials.
    # This cumulative score scales with the dataset size, amplifying the signal.
    return float(agree_ttb_match * 2 - conflict_ttb_match)
```

**Observed (real) value:** 9.0000 (var=17.2704)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 581.0000 (var=60.9556) (Δ vs real +572.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 27.0000 (var=15.5684)
- pi_3: 1.0000 (var=21.4596)
- pi_1: 2411.0000 (var=92.7316)
- pi_2: 1570.0000 (var=276.6400)
- pi_4: 563.0000 (var=39.8724)
- pi_5: 34.0000 (var=32.3776)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Probabilistic Cue Selection: Decision-makers evaluate options by identifying all cues that discriminate between them. Instead of strictly applying the highest-validity cue (as in pure Take The Best) or equally weighting all cues (as in Tallying), they probabilistically select one discriminating cue to drive their choice. The probability of selecting a particular cue is proportional to the exponential of its validity, controlled by a temperature parameter. This creates a soft-lexicographic process where multiple lower-validity cues can occasionally outweigh a single higher-validity cue without relying on an explicit compensatory strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating cues
    disc = (a != b)
    
    if not np.any(disc):
        return np.array([0.5, 0.5])
        
    # Compute weights for discriminating cues using stable exponentiation
    v_disc = validities[disc]
    max_v = np.max(v_disc)
    
    w = np.zeros_like(validities)
    w[disc] = np.exp(beta * (validities[disc] - max_v))
    
    # Aggregate weights for cues favoring Option A and Option B
    w_a = np.sum(w[a > b])
    w_b = np.sum(w[b > a])
    
    total_w = w_a + w_b
    if total_w == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([w_a / total_w, w_b / total_w])
        
    # Apply uniform response noise (lapse)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.0, 30.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's guidance to avoid explicit Strategy Selection mixtures while maintaining a non-compensatory flavor, this model instantiates a Probabilistic Cue Search process. By selecting a single discriminating cue probabilistically (weighted by validity via a softmax function), the model naturally predicts high adherence to the most valid cue (mimicking TTB) while allowing occasional choices driven by lower-validity cues. This elegantly captures the soft deviations observed in the data without requiring a compensatory Tallying mechanism.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate theory was accepted by the loop because it offers a better fit than the previous baseline, but a closer look at Experiments 9 and 10 reveals a massive deviation from human data. The model predicts scores of 53.0 and 581.0 respectively, whereas human data is near 0 or slightly negative/positive. By summing the exponentiated validities of all discriminating cues (`w_a = np.sum(...)`), the model mathematically collapses into a compensatory model with exponential weights. This allows multiple lower-validity cues to easily outvote the highest-validity cue on conflict trials, which explicitly contradicts the robust non-compensatory (TTB-like) behavior observed in the empirical data.
Rationale: While the Probabilistic Cue Selection concept fits the arbiter's recommendation, the current implementation aggregates weights across all cues, making it too compensatory and blowing out Exps 9 and 10. To stay within the prescribed family but fix this, revise the mechanism to be strictly sequential rather than aggregative. For example, implement a noisy sequential search: the decision-maker considers cues in descending order of validity. If a cue discriminates, they adopt its recommendation with probability `gamma` (a parameter, possibly scaled by validity), and with `1 - gamma` they ignore it and evaluate the next cue. This maintains the soft-lexicographic, cue-based search without allowing a tally of lower cues to deterministically outvote a higher cue.

**Outcome of this advice:** iter 1 candidate loss=0.1968 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Noisy Sequential Cue Search: Decision-makers evaluate options using a soft-lexicographic process. They consider cues sequentially in descending order of validity. If a cue discriminates between options, they adopt its recommendation with a fixed probability (gamma). With probability 1 - gamma, they ignore the cue and continue their search. If all cues are exhausted without a decision, they guess. This produces probabilistic TTB-like behavior without relying on a compensatory tallying mechanism.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n    \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    gamma = float(parameters[\"gamma\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Sort cues in descending order of validity\n    order = np.argsort(-validities, kind='stable')\n    \n    p_a = 0.0\n    p_b = 0.0\n    p_reach = 1.0\n    \n    # Noisy sequential search\n    for i in order:\n        if a[i] > b[i]:\n            p_a += p_reach * gamma\n            p_reach *= (1.0 - gamma)\n        elif b[i] > a[i]:\n            p_b += p_reach * gamma\n            p_reach *= (1.0 - gamma)\n            \n    # If all cues are evaluated or skipped, guess with remaining probability\n    p_a += p_reach * 0.5\n    p_b += p_reach * 0.5\n    \n    p_core = np.array([p_a, p_b])\n        \n    # Apply uniform response noise (lapse)\n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=float)\n    probabilities = probabilities / probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))",
  "parameters": {
    "gamma": "[0.0, 1.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we replaced the aggregative exponential weighting with a noisy sequential search. The decision-maker considers cues in descending order of validity. When a cue discriminates, they adopt its recommendation with probability `gamma`. With probability `1 - gamma`, they ignore it and continue searching. This maintains a soft-lexicographic process without mathematically collapsing into a compensatory model, preventing lower cues from deterministically outvoting a higher-validity cue and fixing the blowout on Experiments 9 and 10."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16758,
  "candidates_token_count": 747,
  "total_token_count": 18870
}
```
