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

Propose a new theory of 'Extreme Spatial Bias / Key Perseveration'. The theory should state that subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy, choosing the exact same option (either always Option A or always Option B) on every single trial regardless of the stimuli. The model should have a parameter dictating the probability of a subject being an 'Always-A' versus 'Always-B' responder, but at the individual level, the policy should be deterministic.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.760)

**Description**
Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- beta: [0.0, 10.0]
- bias_A: [-20.0, 20.0]
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.4029 (var=0.0166)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5791 (var=0.0214)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.4614 (var=0.0072)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.1400 (var=0.0701)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.1700 (var=0.1160)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.0813 (var=0.0035)


---

### `pi_4` (overall score: 0.558)

**Description**
Subjects exhibit complete disengagement from the multi-attribute decision task. Rather than evaluating the options based on their features and the validities of those features, subjects ignore all stimulus information and simply guess at random on every trial, choosing Option A or Option B with equal probability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model completely ignores the state and history, 
    # reflecting total task disengagement.
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
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.5133 (var=0.0022)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.5134 (var=0.0042)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.4997 (var=0.0040)
- Experiment 4: real=0.0000 (var=0.0000) vs this=-0.0050 (var=0.0484)
- Experiment 5: real=0.0000 (var=0.0000) vs this=-0.0600 (var=0.0318)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.0186 (var=0.0001)


---

### `pi_1` (overall score: 0.165)

**Description**
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

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

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.5000 (var=0.0000) vs this=0.8575 (var=0.0070)
- Experiment 2: real=0.5000 (var=0.0000) vs this=0.1459 (var=0.0108)
- Experiment 3: real=0.5000 (var=0.0000) vs this=0.8558 (var=0.0065)
- Experiment 4: real=0.0000 (var=0.0000) vs this=-0.7200 (var=0.0474)
- Experiment 5: real=0.0000 (var=0.0000) vs this=0.0189 (var=0.0072)
- Experiment 6: real=0.2500 (var=0.0000) vs this=0.1165 (var=0.0019)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        ttb_preds.append(pred)
        
    ttb_preds = np.array(ttb_preds)
    responses = data['response'].values
    
    return float(np.mean(ttb_preds == responses))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8575 (var=0.0070)
- pi_2: 0.1094 (var=0.0037)
- pi_3: 0.4029 (var=0.0166)
- pi_4: 0.5133 (var=0.0022)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Calculate number of feature-wise wins for each option
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    responses = data['response'].values
    
    # Identify trials where Tallying has a strict preference
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    # Check if subject's response is consistent with Tallying's preference
    consistent = (tally_prefers_a & (responses == 0)) | (tally_prefers_b & (responses == 1))
    strict_trials = tally_prefers_a | tally_prefers_b
    
    if np.sum(strict_trials) == 0:
        return 0.5
        
    # Return the proportion of Tallying-consistent choices on strict trials
    return float(np.sum(consistent[strict_trials]) / np.sum(strict_trials))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8741 (var=0.0061)
- pi_1: 0.1459 (var=0.0108)
- pi_3: 0.5791 (var=0.0214)
- pi_4: 0.5134 (var=0.0042)

### Experiment 3
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # TTB predictions
    diff = a_mat - b_mat
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(4):
            if diff[i, j] > 0:
                ttb_preds[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_preds[i] = 1
                break
                
    # WADD predictions (without spatial bias)
    score_a = a_mat @ validities
    score_b = b_mat @ validities
    wadd_preds = (score_b > score_a).astype(int)
    
    # Isolate trials where TTB and WADD (unbiased) predict opposite choices
    mask = ttb_preds != wadd_preds
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values
    agreement = (responses[mask] == ttb_preds[mask]).mean()
    return float(agreement)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8558 (var=0.0065)
- pi_3: 0.4614 (var=0.0072)
- pi_2: 0.1364 (var=0.0134)
- pi_4: 0.4997 (var=0.0040)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (0, 1, 1, 1) and \
               tuple(int(x) for x in row['option_b_ratings']) == (1, 0, 0, 0)
               
    def is_t2(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (1, 0, 0, 0) and \
               tuple(int(x) for x in row['option_b_ratings']) == (0, 1, 1, 1)
               
    t1_mask = data.apply(is_t1, axis=1)
    t2_mask = data.apply(is_t2, axis=1)
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.5
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1400 (var=0.0701)
- pi_1: -0.7200 (var=0.0474)
- pi_2: 0.7567 (var=0.0366)
- pi_4: -0.0050 (var=0.0484)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where A has more positive features than B, and vice versa
    a_dom = (sum_a > sum_b)
    b_dom = (sum_a < sum_b)
    
    # Proportion of choosing A (response == 0)
    p_a_given_a_dom = (data.loc[a_dom, 'response'] == 0).mean()
    p_a_given_b_dom = (data.loc[b_dom, 'response'] == 0).mean()
    
    if np.isnan(p_a_given_a_dom):
        p_a_given_a_dom = 0.5
    if np.isnan(p_a_given_b_dom):
        p_a_given_b_dom = 0.5
        
    return float(p_a_given_a_dom - p_a_given_b_dom)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0600 (var=0.0318)
- pi_3: 0.1700 (var=0.1160)
- pi_1: 0.0189 (var=0.0072)
- pi_2: 0.7567 (var=0.0299)

### Experiment 6
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    # Convert responses to 1 if Option A was chosen, 0 otherwise
    df['is_A'] = (df['response'] == 0).astype(float)
    
    # Create a hashable string representation for each unique trial type
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + "_" + \
                       df['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option A was chosen for each trial type, per subject
    subj_trial_means = df.groupby(['subject_id', 'trial_type'])['is_A'].mean()
    
    # Calculate the squared deviation from 0.5 (random guessing baseline)
    sq_dev = (subj_trial_means - 0.5) ** 2
    
    # Average these squared deviations across trial types for each subject
    subj_scores = sq_dev.groupby('subject_id').mean()
    
    # Return the mean across all subjects
    return float(subj_scores.mean())
```

**Observed (real) value:** 0.2500 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0813 (var=0.0035)
- pi_4: 0.0186 (var=0.0001)
- pi_1: 0.1165 (var=0.0019)
- pi_2: 0.0791 (var=0.0007)


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
  "theory": "Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    # The subject adopts a deterministic response strategy, always choosing the same option.\n    pref = int(parameters[\"preferred_option\"])\n    if pref == 0:\n        return np.array([1.0, 0.0])\n    else:\n        return np.array([0.0, 1.0])",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "preferred_option": "{0, 1}"
  },
  "rationale": "Following the arbiter's suggestion, this model instantiates the 'Extreme Spatial Bias / Key Perseveration' theory. A pure random guessing model (pi_4) fails to capture the variance structure of the data, particularly in Experiment 6 where it yields a squared deviation of ~0.0 instead of 0.25. By assuming subjects are individually deterministic but perfectly split at the population level (50% Always-A, 50% Always-B), we perfectly recover the 0.5 accuracy metrics (since the experimental designs are balanced, always picking one option yields exactly 50% agreement with any balanced strategy like TTB or Tallying), the 0.0 differences in conditional probabilities, and the exact 0.25 squared deviation in Experiment 6 (since P(A) is 1.0 or 0.0 for each subject, yielding (1-0.5)^2 = 0.25 or (0-0.5)^2 = 0.25)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 9294,
  "candidates_token_count": 521,
  "total_token_count": 12456
}
```
