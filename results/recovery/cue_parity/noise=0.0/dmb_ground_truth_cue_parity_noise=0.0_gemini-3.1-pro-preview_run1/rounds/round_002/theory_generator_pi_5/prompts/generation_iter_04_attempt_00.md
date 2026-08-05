# generation_iter_04_attempt_00

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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a brand-new theory to replace THEORY 1. The new theory must account for the fact that subjects systematically deviate from random guessing when tallies are tied, but do not strictly follow a validity-weighted tie-breaker (showing reversals in preference in Experiments 1 and 2 compared to Experiment 5). Consider a theory based on configural feature processing, an Exemplar-based model (where options are compared to past instances rather than evaluated by cue validities), or a weighted additive model (WADD) where cue weights are dynamically adjusted based on the specific feature distribution or contrast in the current trial (e.g., a salience-weighted model).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.514)

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
- Experiment 1: real=0.5711 (var=0.0014) vs this=0.1771 (var=0.0092)
- Experiment 2: real=0.6890 (var=0.0042) vs this=0.8565 (var=0.0079)
- Experiment 3: real=0.6200 (var=0.0045) vs this=0.5033 (var=0.0060)
- Experiment 4: real=0.8542 (var=0.0086) vs this=0.8442 (var=0.0115)
- Experiment 5: real=0.3850 (var=0.0061) vs this=0.4792 (var=0.0045)
- Experiment 6: real=0.3250 (var=0.0053) vs this=0.5100 (var=0.0036)


---

### `pi_4` (overall score: 0.477)

**Description**
People primarily compare multi-attribute options using a Tallying heuristic, counting the number of features on which each option is strictly better. The option with the higher tally is chosen. However, if the tallies are tied, the decision-maker falls back to a compensatory tie-breaking mechanism, evaluating the options based on the weighted sum of their features. The weights correspond to feature validities centered at chance and non-linearly scaled by a parameter gamma, allowing flexible adjustment of the tie-breaker's sensitivity to validity differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary stage: Tallying feature wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins != b_wins:
        scores = np.array([a_wins, b_wins])
    else:
        # Secondary stage: Validity-weighted tie-breaker with non-linear scaling
        val = np.asarray(parameters["validities"], dtype=float)
        gamma = float(parameters["gamma"])
        centered_val = val - 0.5
        w = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
        tie_scale = float(parameters["tie_scale"])
        scores = tie_scale * np.array([np.sum(a * w), np.sum(b * w)])
        
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.8]
- tie_scale: [1.0, 20.0]
- gamma: [0.1, 5.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5711 (var=0.0014) vs this=0.2876 (var=0.0121)
- Experiment 2: real=0.6890 (var=0.0042) vs this=0.7345 (var=0.0158)
- Experiment 3: real=0.6200 (var=0.0045) vs this=0.6611 (var=0.0274)
- Experiment 4: real=0.8542 (var=0.0086) vs this=0.7354 (var=0.0174)
- Experiment 5: real=0.3850 (var=0.0061) vs this=0.6854 (var=0.0190)
- Experiment 6: real=0.3250 (var=0.0053) vs this=0.6600 (var=0.0167)


---

### `pi_3` (overall score: 0.433)

**Description**
People evaluate options by computing a weighted sum of all available features, where each feature is weighted by a non-linearly scaled version of its validity centered at chance (0.5). This ensures that uninformative cues contribute nothing, and allows a scaling parameter gamma to control the shift between compensatory and non-compensatory decision-making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly, centered at chance (0.5)
    centered_val = val - 0.5
    weights = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
    
    # Compute the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- gamma: [0.1, 5.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5711 (var=0.0014) vs this=0.5142 (var=0.0156)
- Experiment 2: real=0.6890 (var=0.0042) vs this=0.4655 (var=0.0215)
- Experiment 3: real=0.6200 (var=0.0045) vs this=0.6611 (var=0.0165)
- Experiment 4: real=0.8542 (var=0.0086) vs this=0.4475 (var=0.0249)
- Experiment 5: real=0.3850 (var=0.0061) vs this=0.6587 (var=0.0183)
- Experiment 6: real=0.3250 (var=0.0053) vs this=0.6917 (var=0.0211)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2441 -> ACCEPTED
- iter 2: loss=0.2561 -> REJECTED
- iter 3: loss=0.2894 -> REJECTED
- iter 4: loss=0.2936 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2441 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5711 (var=0.0014)
**Previous candidate values (this loop):**
  - iter 1: 0.2158 (var=0.0089) (Δ vs real -0.3553)
  - iter 2: 0.3780 (var=0.0192) (Δ vs real -0.1931)
  - iter 3: 0.2442 (var=0.0131) (Δ vs real -0.3269)
  - iter 4 (most recent): 0.4422 (var=0.0494) (Δ vs real -0.1289)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8500 (var=0.0114)
- pi_2: 0.1771 (var=0.0092)
- pi_3: 0.5142 (var=0.0156)
- pi_4: 0.2876 (var=0.0121)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = np.where(a_wins > b_wins, 0, 1)
    responses = np.array(data['response'].tolist())
    
    matches = (responses[valid_mask] == tally_preds[valid_mask])
    return float(np.mean(matches))

```

**Observed (real) value:** 0.6890 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.8730 (var=0.0084) (Δ vs real +0.1840)
  - iter 2: 0.6385 (var=0.0224) (Δ vs real -0.0505)
  - iter 3: 0.8043 (var=0.0085) (Δ vs real +0.1153)
  - iter 4 (most recent): 0.5158 (var=0.0549) (Δ vs real -0.1732)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8565 (var=0.0079)
- pi_1: 0.1787 (var=0.0103)
- pi_3: 0.4655 (var=0.0215)
- pi_4: 0.7345 (var=0.0158)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.7, 0.65, 0.6, 0.55])
    w = val - 0.5
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Isolate trials where Tallying sees a tie
        if a_wins == b_wins:
            score_a = np.sum(a * w)
            score_b = np.sum(b * w)
            
            if score_a > score_b:
                target = 0
            elif score_b > score_a:
                target = 1
            else:
                continue
                
            if row['response'] == target:
                match_count += 1
            total += 1
            
    if total == 0:
        return 0.5
        
    return match_count / total
```

**Observed (real) value:** 0.6200 (var=0.0045)
**Previous candidate values (this loop):**
  - iter 1: 0.4561 (var=0.1092) (Δ vs real -0.1639)
  - iter 2: 0.3917 (var=0.0747) (Δ vs real -0.2283)
  - iter 3: 0.3583 (var=0.0915) (Δ vs real -0.2617)
  - iter 4 (most recent): 0.5333 (var=0.1286) (Δ vs real -0.0867)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6611 (var=0.0165)
- pi_2: 0.5033 (var=0.0060)
- pi_1: 0.8411 (var=0.0219)
- pi_4: 0.6611 (var=0.0274)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t3 = (a_str == '10001') & (b_str == '01110')
    t5 = (a_str == '01011') & (b_str == '10100')
    t6 = (a_str == '10010') & (b_str == '01101')
    
    score = 0.0
    count = 0
    
    if t3.any():
        score += data.loc[t3, 'response'].mean()
        count += 1
    if t5.any():
        score += (1 - data.loc[t5, 'response']).mean()
        count += 1
    if t6.any():
        score += data.loc[t6, 'response'].mean()
        count += 1
        
    return score / count if count > 0 else 0.5
```

**Observed (real) value:** 0.8542 (var=0.0086)
**Previous candidate values (this loop):**
  - iter 1: 0.8546 (var=0.0145) (Δ vs real +0.0004)
  - iter 2: 0.6375 (var=0.0456) (Δ vs real -0.2167)
  - iter 3: 0.8171 (var=0.0185) (Δ vs real -0.0371)
  - iter 4 (most recent): 0.5388 (var=0.0658) (Δ vs real -0.3154)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8442 (var=0.0115)
- pi_3: 0.4475 (var=0.0249)
- pi_1: 0.1571 (var=0.0118)
- pi_4: 0.7354 (var=0.0174)

### Experiment 5
**Design**
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    correct = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1
        if a == (1, 0, 1, 0) and b == (0, 1, 0, 1):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 1, 0, 1) and b == (1, 0, 1, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
        # Trial 3
        elif a == (1, 1, 0, 0) and b == (0, 0, 1, 1):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 0, 1, 1) and b == (1, 1, 0, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
        # Trial 4
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 0):
            correct.append(1 if row['response'] == 0 else 0)
        elif a == (0, 0, 1, 0) and b == (0, 1, 0, 0):
            correct.append(1 if row['response'] == 1 else 0)
            
    if not correct:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.3850 (var=0.0061)
**Previous candidate values (this loop):**
  - iter 1: 0.3729 (var=0.0793) (Δ vs real -0.0121)
  - iter 2: 0.3438 (var=0.0619) (Δ vs real -0.0413)
  - iter 3: 0.4512 (var=0.0976) (Δ vs real +0.0662)
  - iter 4 (most recent): 0.5150 (var=0.1065) (Δ vs real +0.1300)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6854 (var=0.0190)
- pi_2: 0.4792 (var=0.0045)
- pi_1: 0.8662 (var=0.0120)
- pi_3: 0.6587 (var=0.0183)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tied_mask = a_wins == b_wins
    
    if not np.any(tied_mask):
        return 0.5
        
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_val = np.sum(a_ratings * val, axis=1)
    b_val = np.sum(b_ratings * val, axis=1)
    
    preferred = np.where(a_val > b_val, 0, np.where(b_val > a_val, 1, -1))
    
    valid_mask = tied_mask & (preferred != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    
    alignment = responses[valid_mask] == preferred[valid_mask]
    
    return float(np.mean(alignment))

```

**Observed (real) value:** 0.3250 (var=0.0053)
**Previous candidate values (this loop):**
  - iter 1: 0.4058 (var=0.0817) (Δ vs real +0.0808)
  - iter 2: 0.4267 (var=0.0716) (Δ vs real +0.1017)
  - iter 3: 0.4525 (var=0.0761) (Δ vs real +0.1275)
  - iter 4 (most recent): 0.4479 (var=0.0875) (Δ vs real +0.1229)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5100 (var=0.0036)
- pi_4: 0.6600 (var=0.0167)
- pi_1: 0.6758 (var=0.0048)
- pi_3: 0.6917 (var=0.0211)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Salience-Weighted Tallying: Decision-makers primarily rely on a non-compensatory Tallying heuristic, simply counting the number of features where one option dominates the other. However, when options are tied in their tally, a secondary compensatory evaluation is triggered. In this tie-breaking phase, the weights of the features are dynamically adjusted based on their salience rather than strictly their objective validity. Specifically, validities are non-linearly scaled by a parameter gamma that can be negative, allowing less predictive features to become disproportionately salient and overweighted during conflict resolution. This mechanism accounts for preference reversals where individuals sometimes favor options with multiple lower-validity cues over those with fewer high-validity cues when overall tallies are tied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary stage: Tallying feature wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins != b_wins:
        scores = np.array([a_wins, b_wins])
    else:
        # Secondary stage: Salience-weighted tie-breaker
        val = np.asarray(parameters["validities"], dtype=float)
        gamma = float(parameters["gamma"])
        centered_val = val - 0.5
        
        # Non-linear scaling allows for negative gamma (overweighting less valid cues)
        w = np.sign(centered_val) * ((np.abs(centered_val) + 1e-4) ** gamma)
        
        tie_scale = float(parameters["tie_scale"])
        scores = tie_scale * np.array([np.sum(a * w), np.sum(b * w)])
        
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [-5.0, 5.0]
- tie_scale: [0.1, 10.0]
- validities: validities

`rationale`: The arbiter observed that while subjects deviate from random guessing during tied tallies, they do not strictly follow a validity-weighted tie-breaker, as evidenced by preference reversals in Experiments 5 and 6 (where accuracy on validity-based tie-breakers drops well below 0.5). To account for this, the model extends the Tallying-with-tie-breaker framework by allowing the non-linear scaling parameter (gamma) to be negative. A negative gamma effectively inverts the weighting scheme during ties, giving disproportionate salience to lower-validity cues. This dynamically adjusts cue weights during conflict resolution and explains why subjects systematically favor options with clusters of lower-validity cues over those with isolated high-validity cues when the overall feature counts are tied.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory successfully captures the preference reversals in Experiments 5 and 6 (where validity-based tie-breaking drops below 0.5) and matches Experiment 4 perfectly. However, the rigid two-stage structure (Tallying first, then tie-breaking) causes it to fail on the earlier experiments. Specifically, it over-predicts reliance on Tallying in Experiment 2 (0.87 simulated vs 0.68 observed) and under-predicts TTB matching in Experiment 1 (0.21 simulated vs 0.57 observed). The hard cutoff of the primary Tallying stage prevents the model from capturing the probabilistic, intermediate behavior seen in Experiments 1-3.
Rationale: The current implementation relies on a hard Tallying primary stage, which over-predicts pure tallying behavior. To fix this while staying within the prescribed mechanism family, drop the two-stage (Tallying -> tie-breaker) structure and instead implement a pure Salience-Weighted Additive Model (WADD) applied to ALL trials. Compute the salience-adjusted weights (using the non-linear gamma scaling of validities) and evaluate the weighted sum for the options directly. This will naturally balance the influence of feature counts and validities across all trials, smoothing the predictions for Exps 1-3 while still allowing negative gamma to explain the preference reversals in Exps 5-6.

**Outcome of this advice:** iter 1 candidate loss=0.2441 -> iter 2 candidate loss=0.2561 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate theory replaced the two-stage base model with a static non-linear WADD, but this approach was rejected by the gate as it worsened the overall loss. While it smoothed out some predictions, it failed to capture the dynamic nature of the decision process and lost the strong performance the base model had on certain conflict trials.
Rationale: Your previous attempt to replace the two-stage model with a static non-linear WADD was rejected by the gate. The two-stage base model (iter 1) was more successful but suffered from a hard threshold that over-predicted pure tallying in Exp 2 and under-predicted TTB in Exp 1. Instead of abandoning the two-stage logic entirely, you should soften it. Implement a probabilistic gating mechanism or a soft combination: compute the tally difference and the validity-weighted difference, and combine them using a continuous function (e.g., a combination where the weight on validity increases as the absolute tally difference approaches zero). This maintains the structural success of the accepted base model on Exps 4-6 while allowing for the intermediate, probabilistic behavior observed in Exps 1-3.

**Outcome of this advice:** iter 2 candidate loss=0.2561 -> iter 3 candidate loss=0.2894 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate theory (Iteration 3) attempted to soften the boundary between Tallying and the validity-weighted tie-breaker, but this approach was rejected by the gate as it worsened the overall loss (0.2894) compared to the accepted base from Iteration 1 (0.2441). The soft combination failed to properly balance the reliance on Tallying in Experiment 2 and TTB in Experiment 1, while degrading performance on the conflict trials in Experiments 4-6.
Rationale: Since the soft combination of tallying and salience was rejected, we should return to the arbiter's suggestion of a dynamic WADD model, but avoid the static non-linear WADD from Iteration 2. Instead, implement a dynamic WADD where the cue weights are adjusted based strictly on the specific feature contrast in the current trial. Specifically, compute weights by isolating ONLY the discriminating features (where A != B), and apply the non-linear scaling (gamma) and normalization exclusively to this subset of active, discriminating cues. This trial-specific re-weighting will naturally adjust the influence of validities based on the local contrast, potentially resolving the over-prediction of Tallying in Exp 2 and under-prediction of TTB in Exp 1 without relying on a rigid two-stage tallying threshold.

**Outcome of this advice:** iter 3 candidate loss=0.2894 -> iter 4 candidate loss=0.2936 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate theory (Iteration 4) attempted a dynamic WADD model normalized over only the active discriminating features, but this was rejected by the gate (loss 0.2936 vs base 0.2441). The dynamic subset-normalization failed to outperform the original two-stage accepted base, heavily underpredicting performance in Exp 4 (0.53 simulated vs 0.85 observed) and failing to capture the correct balance in Exps 1 and 2. We have exhausted several WADD-based adjustments (static, soft-gated, and dynamic-subset) without beating the Iteration 1 base.
Rationale: Since all WADD-style variants (static, soft-gated, and active-subset) have been rejected by the gate, we should pivot to another mechanism family explicitly suggested by the arbiter: Configural feature processing. Build on the accepted Iteration 1 base by replacing the two-stage heuristic with a single-stage configural model. Specifically, compute the value of each option by taking the validity-weighted sum of its features, but pass this sum through a non-linear power function (e.g., `value = sum(w_i * f_i)^alpha`, where `alpha` is a free parameter). An `alpha < 1` introduces diminishing marginal returns for additional features, which can naturally produce tally-like behavior and preference reversals in conflict trials without requiring a hard threshold or complex dynamic re-weighting. Keep the softmax decision rule.

**Outcome of this advice:** iter 4 candidate loss=0.2936 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Configural Feature Processing: Decision-makers evaluate options holistically by computing the validity-weighted sum of their features, but this overall evidence is passed through a non-linear power function before comparison. A power parameter (alpha) less than 1 introduces diminishing marginal returns for additional features. This configural evaluation naturally produces tally-like behavior and explains preference reversals in conflict trials (where multiple lower-validity cues might collectively outweigh fewer high-validity cues due to diminishing returns on the high-validity evidence) without requiring a hard threshold or distinct decision stages.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    a, b = stim[0], stim[1]\n    \n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    # Center validities so only predictive information is summed (w >= 0 since validities in [0.5, 1.0])\n    w = val - 0.5\n    \n    alpha = float(parameters[\"alpha\"])\n    \n    # Calculate sum of weights for present features\n    sum_a = np.maximum(np.sum(a * w), 0.0)\n    sum_b = np.maximum(np.sum(b * w), 0.0)\n    \n    # Apply configural non-linear power function\n    score_a = sum_a ** alpha\n    score_b = sum_b ** alpha\n    \n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.0, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "alpha": "[0.1, 5.0]",
    "validities": "validities"
  },
  "rationale": "Since all WADD-style dynamic re-weightings and soft-gating attempts were rejected by the gate, we follow the critic's advice to pivot to a Configural Feature Processing model. By computing the validity-weighted sum of features and passing it through a non-linear power function (alpha), we introduce diminishing marginal returns for additional evidence (when alpha < 1). This single-stage mechanism naturally produces tally-like behavior and captures the preference reversals seen in conflict trials without relying on the rigid, hard-threshold two-stage logic of the Iteration 1 base."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13376,
  "candidates_token_count": 815,
  "total_token_count": 16141
}
```
