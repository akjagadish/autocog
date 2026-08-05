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
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace Theory 2 with a 'Tallying under Overload' or 'Equal Weights' theory. Instead of integrating complex cue validities (which causes the WADD model to overpredict sensitivity to score differences), subjects might completely ignore the validities and fall back to a simpler heuristic: just counting the number of positive features (1s) for each option. To capture the near-random behavior observed empirically, this simple tallying process must still be coupled with a very high lapse rate or extreme softmax noise. This provides a structured but highly simplified alternative to pure random guessing.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 1.000)

**Description**
When faced with complex multi-attribute choices without trial-by-trial feedback, subjects experience cognitive overload. Instead of systematically integrating cue validities and feature vectors, they abandon structured decision strategies and resort to random guessing. Choice behavior is driven entirely by this stochasticity, with only a potential slight bias toward one spatial position (e.g., Option A or Option B) over the other.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # Under cognitive overload, subjects ignore the state (features) and just guess.
    # The choice probability is determined only by an intrinsic side bias.
    p_b = float(parameters.get('side_bias', 0.5))
    p_a = 1.0 - p_b
    return np.array([p_a, p_b])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- side_bias: [0.3, 0.7]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5050 (var=0.0019) vs this=0.4925 (var=0.0032)
- Experiment 2: real=0.5107 (var=0.0040) vs this=0.5063 (var=0.0043)
- Experiment 3: real=0.5038 (var=0.0039) vs this=0.5231 (var=0.0205)
- Experiment 4: real=0.5018 (var=0.0029) vs this=0.4891 (var=0.0030)
- Experiment 5: real=-0.0500 (var=0.0218) vs this=0.0039 (var=0.0327)
- Experiment 6: real=0.0014 (var=0.0003) vs this=0.0010 (var=0.0005)


---

### `pi_3` (overall score: 0.748)

**Description**
People compare options by computing a compensatory overall value for each option. This is done by summing the features of each option weighted by their respective validities (Weighted Additive rule). Because empirical behavior in these experiments is highly stochastic (near random guessing), choice is subject to significant softmax noise and lapse rates. Narrowing the inverse temperature bounds forces the model to capture this high level of noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")

    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    a_score = np.sum(a * val)
    b_score = np.sum(b * val)
    scores = np.array([a_score, b_score])

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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5050 (var=0.0019) vs this=0.4608 (var=0.0041)
- Experiment 2: real=0.5107 (var=0.0040) vs this=0.5367 (var=0.0061)
- Experiment 3: real=0.5038 (var=0.0039) vs this=0.4553 (var=0.0041)
- Experiment 4: real=0.5018 (var=0.0029) vs this=0.5747 (var=0.0043)
- Experiment 5: real=-0.0500 (var=0.0218) vs this=0.2217 (var=0.0711)
- Experiment 6: real=0.0014 (var=0.0003) vs this=0.0504 (var=0.0011)


---

### `pi_2` (overall score: 0.032)

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
- Experiment 1: real=0.5050 (var=0.0019) vs this=0.3215 (var=0.0026)
- Experiment 2: real=0.5107 (var=0.0040) vs this=0.8623 (var=0.0098)
- Experiment 3: real=0.5038 (var=0.0039) vs this=0.1459 (var=0.0066)
- Experiment 4: real=0.5018 (var=0.0029) vs this=0.8267 (var=0.0091)
- Experiment 5: real=-0.0500 (var=0.0218) vs this=0.6700 (var=0.0383)
- Experiment 6: real=0.0014 (var=0.0003) vs this=0.1182 (var=0.0007)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def ttb_predict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif a[i] < b[i]:
                return 1
        return 0.5
        
    ttb_choices = data.apply(ttb_predict, axis=1)
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.5050 (var=0.0019)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8398 (var=0.0110)
- pi_2: 0.3215 (var=0.0026)
- pi_3: 0.4608 (var=0.0041)
- pi_4: 0.4925 (var=0.0032)

### Experiment 2
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Only consider trials where Tallying has a strict preference
        if a_wins > b_wins:
            tally_choice = 0
            matches.append(int(row['response'] == tally_choice))
        elif b_wins > a_wins:
            tally_choice = 1
            matches.append(int(row['response'] == tally_choice))
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.5107 (var=0.0040)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8623 (var=0.0098)
- pi_1: 0.1203 (var=0.0068)
- pi_3: 0.5367 (var=0.0061)
- pi_4: 0.5063 (var=0.0043)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    
    # TTB prediction: sign of the first non-zero difference
    abs_diff = np.abs(diff)
    first_diff_idx = np.argmax(abs_diff, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # WADD prediction: based on weighted sum
    a_score = np.dot(a_mat, val)
    b_score = np.dot(b_mat, val)
    wadd_pred = np.where(a_score > b_score, 0, 1)
    
    # Identify conflict trials where TTB and WADD make opposite predictions
    conflict = (ttb_pred != wadd_pred) & (first_diff_val != 0)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    # Calculate the proportion of choices that align with TTB on conflict trials
    matches = (responses[conflict] == ttb_pred[conflict])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5038 (var=0.0039)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8506 (var=0.0117)
- pi_3: 0.4553 (var=0.0041)
- pi_2: 0.1459 (var=0.0066)
- pi_4: 0.5231 (var=0.0205)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    score_A = A.dot(val)
    score_B = B.dot(val)
    
    wadd_choice = (score_B > score_A).astype(int)
    
    return float(np.mean(data['response'].values == wadd_choice))
```

**Observed (real) value:** 0.5018 (var=0.0029)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5747 (var=0.0043)
- pi_1: 0.1398 (var=0.0066)
- pi_2: 0.8267 (var=0.0091)
- pi_4: 0.4891 (var=0.0030)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.8, 0.7, 0.6])
    
    def score_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(b * val) - np.sum(a * val)
        
    diffs = data.apply(score_diff, axis=1)
    
    b_better = data['response'][diffs > 0.5]
    a_better = data['response'][diffs < -0.5]
    
    m_b = b_better.mean() if len(b_better) > 0 else 0.5
    m_a = a_better.mean() if len(a_better) > 0 else 0.5
    
    return float(m_b - m_a)
```

**Observed (real) value:** -0.0500 (var=0.0218)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0039 (var=0.0327)
- pi_3: 0.2217 (var=0.0711)
- pi_1: 0.4817 (var=0.0176)
- pi_2: 0.6700 (var=0.0383)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    # Calculate the difference in WADD scores between Option A and Option B
    score_diff = A.dot(validities) - B.dot(validities)
    
    # 1 if A was chosen, 0 if B was chosen
    chose_a = 1.0 - data['response'].values
    
    if np.var(score_diff) == 0:
        return 0.0
    
    # Calculate the linear slope of choosing A as a function of the score difference
    slope, _ = np.polyfit(score_diff, chose_a, 1)
    return float(slope)
```

**Observed (real) value:** 0.0014 (var=0.0003)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0504 (var=0.0011)
- pi_4: 0.0010 (var=0.0005)
- pi_1: 0.1041 (var=0.0013)
- pi_2: 0.1182 (var=0.0007)


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
  "theory": "Tallying under Overload (Equal Weights): Under cognitive overload without trial-by-trial feedback, subjects abandon complex integration of cue validities. Instead, they fall back on a highly simplified Equal Weights heuristic, merely tallying the total number of positive features (1s) for each option. Even with this simplification, the high cognitive demand leads to near-random choice behavior, which is captured by extreme softmax noise and a very high lapse rate.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    stim = np.asarray(state, dtype=float)\n    \n    # Equal Weights / Tallying: count the number of positive features (1s) for each option\n    a, b = stim[0], stim[1]\n    a_score = np.sum(a)\n    b_score = np.sum(b)\n    scores = np.array([a_score, b_score])\n    \n    beta = float(parameters['beta'])\n    epsilon = float(parameters['epsilon'])\n    \n    # Softmax with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    # Apply high lapse rate\n    n_opts = len(scores)\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probs):\n    import numpy as np\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "beta": "[0.0, 0.2]",
    "epsilon": "[0.8, 1.0]"
  },
  "rationale": "Following the arbiter's feedback, this theory replaces the Weighted Additive (WADD) approach with an Equal Weights (Tallying) heuristic. By ignoring cue validities and merely counting the number of positive features per option, the model avoids overpredicting sensitivity to score differences. To capture the empirically observed near-random behavior, the tallying process is coupled with a very high lapse rate (epsilon in [0.8, 1.0]) and extreme softmax noise (beta in [0.0, 0.2]). This provides a structured but simplified alternative to pure random guessing."
}
```
```

## Usage

```json
{
  "prompt_token_count": 8963,
  "candidates_token_count": 585,
  "total_token_count": 10758
}
```
