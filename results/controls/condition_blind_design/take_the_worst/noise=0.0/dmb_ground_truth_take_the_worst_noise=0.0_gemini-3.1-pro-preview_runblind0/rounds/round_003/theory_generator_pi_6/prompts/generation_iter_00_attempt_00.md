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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Propose a new theory that acts as a stronger competitor to R2L-TTB by also incorporating the right-to-left layout bias, but utilizing a different decision rule. For instance, propose a 'Right-biased Compensatory Model' (such as right-to-left tallying or a weighted additive model where subjective weights exponentially decay from right to left). This will help determine if the layout bias strictly triggers a non-compensatory heuristic (TTB) or if subjects are still integrating multiple cues but with a heavy attentional weight on the rightmost features.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.996)

**Description**
People process feature arrays from right to left, exhibiting a recency or layout bias. Instead of consulting cues in order of their objective validities, subjects use a Right-to-Left Take-The-Best (R2L-TTB) heuristic. They evaluate features starting from the last index down to the first, choosing the option that wins on the first discriminating feature encountered in this reversed order. If no feature discriminates, they guess. Response noise is incorporated via a softmax over the binary winner score and a constant lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("R2L-TTB expects a (2, n_features) stimulus.")
    
    n_features = stim.shape[1]
    # Right-to-left cue order
    cue_order = list(range(n_features - 1, -1, -1))
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for choice probability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1933 (var=0.0304) vs this=0.1300 (var=0.0248)
- Experiment 2: real=0.2067 (var=0.0540) vs this=0.3700 (var=0.0948)
- Experiment 3: real=-0.0067 (var=0.0177) vs this=-0.0033 (var=0.0233)
- Experiment 4: real=-0.7133 (var=0.0434) vs this=-0.6833 (var=0.0781)
- Experiment 5: real=0.8367 (var=0.0203) vs this=0.8417 (var=0.0184)
- Experiment 6: real=0.8200 (var=0.0509) vs this=0.8233 (var=0.0382)
- Experiment 7: real=0.8208 (var=0.0190) vs this=0.8304 (var=0.0091)
- Experiment 8: real=0.1933 (var=0.0236) vs this=0.2033 (var=0.0155)


---

### `pi_2` (overall score: 0.377)

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
- Experiment 1: real=0.1933 (var=0.0304) vs this=0.1367 (var=0.0174)
- Experiment 2: real=0.2067 (var=0.0540) vs this=0.9967 (var=0.0961)
- Experiment 3: real=-0.0067 (var=0.0177) vs this=0.0050 (var=0.0301)
- Experiment 4: real=-0.7133 (var=0.0434) vs this=-0.0233 (var=0.0300)
- Experiment 5: real=0.8367 (var=0.0203) vs this=0.1800 (var=0.0293)
- Experiment 6: real=0.8200 (var=0.0509) vs this=0.1500 (var=0.0247)
- Experiment 7: real=0.8208 (var=0.0190) vs this=0.4608 (var=0.0052)
- Experiment 8: real=0.1933 (var=0.0236) vs this=0.2467 (var=0.0132)


---

### `pi_3` (overall score: 0.237)

**Description**
People make decisions by computing a weighted sum of the features for each option, using the cue validities as the weights. This Weighted Additive Model (WADD) is fully compensatory. Instead of a translation-invariant Softmax mapping, choice probabilities are generated using a ratio-based Luce choice rule (power rule) over the options' weighted sums, along with a baseline rate of random guessing. This allows choices to be sensitive to the ratio of evidence rather than just the absolute difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Ratio-based Luce choice rule (power rule)
    scores = np.maximum(scores, 1e-9)
    v = scores ** theta
    p_core = v / np.sum(v)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- theta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1933 (var=0.0304) vs this=0.2867 (var=0.0500)
- Experiment 2: real=0.2067 (var=0.0540) vs this=1.1433 (var=0.1045)
- Experiment 3: real=-0.0067 (var=0.0177) vs this=0.1000 (var=0.0253)
- Experiment 4: real=-0.7133 (var=0.0434) vs this=0.1300 (var=0.0459)
- Experiment 5: real=0.8367 (var=0.0203) vs this=0.2183 (var=0.0261)
- Experiment 6: real=0.8200 (var=0.0509) vs this=0.1833 (var=0.0269)
- Experiment 7: real=0.8208 (var=0.0190) vs this=0.3558 (var=0.0069)
- Experiment 8: real=0.1933 (var=0.0236) vs this=0.4233 (var=0.0118)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_trial_3(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        t1 = (1, 0, 0, 1)
        t2 = (0, 1, 1, 1)
        return (a == t1 and b == t2) or (a == t2 and b == t1)
        
    mask = data.apply(is_trial_3, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    def chose_ttb_option(row):
        a = tuple(row['option_a_ratings'])
        # The TTB-favored option is the one with cue 0 == 1, i.e., (1, 0, 0, 1)
        if a == (1, 0, 0, 1):
            return 1.0 if row['response'] == 0 else 0.0
        else:
            return 1.0 if row['response'] == 1 else 0.0
            
    return float(subset.apply(chose_ttb_option, axis=1).mean())
```

**Observed (real) value:** 0.1933 (var=0.0304)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8533 (var=0.0196)
- pi_2: 0.1367 (var=0.0174)
- pi_3: 0.2867 (var=0.0500)
- pi_4: 0.1300 (var=0.0248)
- pi_5: 0.8333 (var=0.0233)

### Experiment 2
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask8 = (a_ratings == (0, 1, 0, 1)) & (b_ratings == (1, 1, 0, 0))
    mask11 = (a_ratings == (1, 1, 0, 1)) & (b_ratings == (1, 0, 1, 1))
    
    resp8 = data.loc[mask8, 'response']
    resp11 = data.loc[mask11, 'response']
    
    score8 = (resp8 == 1).mean() if len(resp8) > 0 else 0.5
    score11 = (resp11 == 0).mean() if len(resp11) > 0 else 0.5
    
    return float(score8 + score11)
```

**Observed (real) value:** 0.2067 (var=0.0540)
**Other theories' values on this metric (for reference):**
- pi_2: 0.9967 (var=0.0961)
- pi_1: 1.6333 (var=0.0578)
- pi_3: 1.1433 (var=0.1045)
- pi_4: 0.3700 (var=0.0948)
- pi_5: 1.6567 (var=0.0871)

### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify specific trials by their feature string representations
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # High WADD ratio trials (Tallying difference is exactly 1)
    # Trial 4: A=[0,0,0,1], B=[0,1,0,1] -> WADD ratio B/A = 1.35 / 0.55 = 2.45
    t4 = data[(data['a_str'] == '0001') & (data['b_str'] == '0101')]
    # Trial 8: A=[0,0,1,1], B=[0,0,0,1] -> WADD ratio A/B = 1.33 / 0.55 = 2.41
    t8 = data[(data['a_str'] == '0011') & (data['b_str'] == '0001')]
    
    # Low WADD ratio trials (Tallying difference is exactly 1)
    # Trial 1: A=[1,0,1,0], B=[1,0,1,1] -> WADD ratio B/A = 2.28 / 1.73 = 1.31
    t1 = data[(data['a_str'] == '1010') & (data['b_str'] == '1011')]
    # Trial 12: A=[1,1,0,0], B=[1,1,0,1] -> WADD ratio B/A = 2.30 / 1.75 = 1.31
    t12 = data[(data['a_str'] == '1100') & (data['b_str'] == '1101')]
    
    p_b_t4 = t4['response'].mean() if len(t4) > 0 else 0.5
    p_a_t8 = 1.0 - t8['response'].mean() if len(t8) > 0 else 0.5
    
    p_b_t1 = t1['response'].mean() if len(t1) > 0 else 0.5
    p_b_t12 = t12['response'].mean() if len(t12) > 0 else 0.5
    
    high_ratio_acc = (p_b_t4 + p_a_t8) / 2.0
    low_ratio_acc = (p_b_t1 + p_b_t12) / 2.0
    
    return float(high_ratio_acc - low_ratio_acc)
```

**Observed (real) value:** -0.0067 (var=0.0177)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1000 (var=0.0253)
- pi_2: 0.0050 (var=0.0301)
- pi_1: 0.0017 (var=0.0151)
- pi_4: -0.0033 (var=0.0233)
- pi_5: 0.0050 (var=0.0207)

### Experiment 4
**Design**
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t9_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0]) & \
              data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 0, 1])
              
    t10_mask = data['option_a_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1]) & \
               data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 0, 0])
               
    resp_9 = data[t9_mask]['response'].mean()
    resp_10 = data[t10_mask]['response'].mean()
    
    if pd.isna(resp_9) or pd.isna(resp_10):
        return 0.0
        
    return float(resp_10 - resp_9)
```

**Observed (real) value:** -0.7133 (var=0.0434)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0233 (var=0.0300)
- pi_3: 0.1300 (var=0.0459)
- pi_1: 0.7167 (var=0.0647)
- pi_4: -0.6833 (var=0.0781)
- pi_5: 0.6567 (var=0.0827)

### Experiment 5
**Design**
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t11_a, t11_b = (1, 1, 1, 0), (1, 0, 0, 1)
    t12_a, t12_b = (0, 0, 1, 1), (1, 1, 1, 0)
    
    scores = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t11_a and b == t11_b:
            scores.append(1.0 if resp == 1 else 0.0)
        elif a == t11_b and b == t11_a:
            scores.append(1.0 if resp == 0 else 0.0)
        elif a == t12_a and b == t12_b:
            scores.append(1.0 if resp == 0 else 0.0)
        elif a == t12_b and b == t12_a:
            scores.append(1.0 if resp == 1 else 0.0)
            
    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8367 (var=0.0203)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8417 (var=0.0184)
- pi_2: 0.1800 (var=0.0293)
- pi_1: 0.1567 (var=0.0202)
- pi_3: 0.2183 (var=0.0261)
- pi_5: 0.1850 (var=0.0373)

### Experiment 6
**Design**
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        if a == [0, 1, 1, 0] and b == [0, 0, 0, 1]:
            return True
        if a == [0, 0, 0, 1] and b == [0, 1, 1, 0]:
            return True
        return False
        
    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    def r2l_choice(row):
        a = list(row['option_a_ratings'])
        if a == [0, 1, 1, 0]:
            return row['response'] == 1
        else:
            return row['response'] == 0
            
    return float(subset.apply(r2l_choice, axis=1).mean())
```

**Observed (real) value:** 0.8200 (var=0.0509)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1500 (var=0.0247)
- pi_4: 0.8233 (var=0.0382)
- pi_1: 0.1767 (var=0.0327)
- pi_3: 0.1833 (var=0.0269)
- pi_5: 0.1900 (var=0.0267)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 0, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    r2l_match_count = 0
    conflict_count = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Standard TTB (Left-to-Right) winner
        l2r_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                l2r_winner = 0
                break
            elif b[j] > a[j]:
                l2r_winner = 1
                break
                
        # R2L-TTB (Right-to-Left) winner
        r2l_winner = None
        for j in range(len(a)-1, -1, -1):
            if a[j] > b[j]:
                r2l_winner = 0
                break
            elif b[j] > a[j]:
                r2l_winner = 1
                break
                
        # Only look at trials where the two heuristics disagree
        if l2r_winner is not None and r2l_winner is not None and l2r_winner != r2l_winner:
            conflict_count += 1
            if resp == r2l_winner:
                r2l_match_count += 1
                
    if conflict_count == 0:
        return 0.5
        
    return r2l_match_count / conflict_count

```

**Observed (real) value:** 0.8208 (var=0.0190)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8304 (var=0.0091)
- pi_5: 0.1829 (var=0.0147)
- pi_1: 0.1379 (var=0.0083)
- pi_2: 0.4608 (var=0.0052)
- pi_3: 0.3558 (var=0.0069)

### Experiment 8
**Design**
  A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_l2r = 0
    total_diff = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        l2r_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                l2r_pred = 0
                break
            elif b[i] > a[i]:
                l2r_pred = 1
                break
                
        r2l_pred = None
        for i in range(len(a)-1, -1, -1):
            if a[i] > b[i]:
                r2l_pred = 0
                break
            elif b[i] > a[i]:
                r2l_pred = 1
                break
                
        if l2r_pred is not None and r2l_pred is not None and l2r_pred != r2l_pred:
            total_diff += 1
            if row['response'] == l2r_pred:
                match_l2r += 1
                
    if total_diff == 0:
        return 0.5
    return match_l2r / total_diff
```

**Observed (real) value:** 0.1933 (var=0.0236)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8292 (var=0.0147)
- pi_4: 0.2033 (var=0.0155)
- pi_1: 0.8375 (var=0.0141)
- pi_2: 0.2467 (var=0.0132)
- pi_3: 0.4233 (var=0.0118)


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
  "theory": "People integrate all available features but exhibit a strong spatial attention bias, weighting features more heavily the further to the right they appear. This Right-biased Compensatory Model evaluates options using a weighted additive sum where the subjective weight of each feature grows exponentially from left to right. This allows the model to capture behavior that appears strictly non-compensatory (like Right-to-Left Take-The-Best) when the growth factor is large, while still accommodating subjects who might integrate multiple cues in a compensatory manner with a heavy rightward bias. Decisions are made by applying a softmax choice rule over these weighted sums, alongside a baseline rate of random guessing.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Right-biased WADD expects a (2, n_features) stimulus.\")\n    \n    n_features = stim.shape[1]\n    a, b = stim[0], stim[1]\n    \n    gamma = float(parameters[\"gamma\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Weights increase exponentially from left to right\n    weights = np.array([gamma ** j for j in range(n_features)])\n    \n    score_a = np.sum(a * weights)\n    score_b = np.sum(b * weights)\n    scores = np.array([score_a, score_b])\n    \n    # Softmax for choice probability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return int(np.random.choice(len(probabilities), p=probabilities))\n",
  "parameters": {
    "gamma": "[1.0, 10.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]"
  },
  "rationale": "Following the arbiter's suggestion, this theory implements a Right-biased Compensatory Model. By weighting features exponentially higher from left to right, it bridges the gap between a purely non-compensatory heuristic (R2L-TTB, which it approximates when gamma > 2) and a fully compensatory tallying model (when gamma approaches 1). This allows us to test whether the layout bias strictly triggers a sequential non-compensatory process, or if subjects are still integrating multiple cues while merely shifting their attentional weights heavily to the rightmost features."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13381,
  "candidates_token_count": 717,
  "total_token_count": 15715
}
```
