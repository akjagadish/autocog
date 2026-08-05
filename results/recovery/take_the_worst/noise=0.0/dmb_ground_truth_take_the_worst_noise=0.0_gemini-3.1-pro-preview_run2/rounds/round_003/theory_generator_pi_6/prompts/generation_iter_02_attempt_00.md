# generation_iter_02_attempt_00

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
- The recommendation below acts on THEORY 1 (= `pi_4`).

Propose a brand-new theory called 'Inverse Validity Weighting'. The theory should state that subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. As their primary decision mechanism (not just a tie-breaker), they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity (e.g., weight = 1 - validity). This will allow the model to strongly predict choices for options with low-validity features over high-validity features, capturing the extreme magnitudes observed in Experiments 2 and 7 better than a hard threshold or a tallying-first approach.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.946)

**Description**
Anti-Expertise Tallying: Decision-makers actively distrust or ignore high-validity 'expert' cues, perceiving them as overly salient, redundant, or manipulative. Instead of using them, subjects filter out features whose stated validity exceeds a certain personal threshold. After discarding these high-validity cues, subjects apply a standard Tallying heuristic (counting strict feature-wise wins) on the remaining lower-validity features to make their choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Anti-Expertise Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Filter out features with validity above the threshold
    mask = val <= threshold
    
    # Fallback to all features if the threshold is so low that all features are discarded
    if not np.any(mask):
        mask = np.ones_like(val, dtype=bool)
        
    a_masked = a[mask]
    b_masked = b[mask]
    
    # Tallying on the remaining features
    a_wins = float(np.sum(a_masked > b_masked))
    b_wins = float(np.sum(b_masked > a_masked))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
- threshold: [0.5, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1564 (var=0.0079) vs this=0.2147 (var=0.0067)
- Experiment 2: real=0.8943 (var=0.0064) vs this=0.8074 (var=0.0096)
- Experiment 3: real=0.1450 (var=0.0093) vs this=0.2562 (var=0.0321)
- Experiment 4: real=0.1250 (var=0.0083) vs this=0.1689 (var=0.0131)
- Experiment 5: real=0.8817 (var=0.0046) vs this=0.7096 (var=0.0215)
- Experiment 6: real=0.6781 (var=0.0038) vs this=0.8137 (var=0.0238)
- Experiment 7: real=0.6029 (var=0.0041) vs this=0.6217 (var=0.0274)
- Experiment 8: real=0.7716 (var=0.0184) vs this=0.5689 (var=0.0759)


---

### `pi_4` (overall score: 0.535)

**Description**
Skeptical Tallying: Decision-makers primarily rely on a Tallying heuristic, counting the number of features where one option strictly dominates the other. However, when the tallies are tied (or closely matched), they do not simply guess. Instead, they exhibit skepticism toward the highest-validity cues—perhaps viewing them as redundant, overly salient, or manipulated—and systematically break ties by favoring options that possess more lower-validity features. This is modeled by augmenting the tally score with a secondary component that weights features inversely to their stated validity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Skeptical Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (count of strict feature-wise wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Secondary mechanism: Tie-breaking favoring lower-validity features
    # Features are weighted by (1 - validity) so that lower validity cues provide a larger bonus.
    tie_breaker_a = np.sum(a * (1.0 - val))
    tie_breaker_b = np.sum(b * (1.0 - val))
    
    # Combine tally with the tie-breaker.
    # gamma controls the strength of the tie-breaker relative to a strict tally win.
    score_a = a_wins + gamma * tie_breaker_a
    score_b = b_wins + gamma * tie_breaker_b
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1564 (var=0.0079) vs this=0.1758 (var=0.0092)
- Experiment 2: real=0.8943 (var=0.0064) vs this=0.8652 (var=0.0101)
- Experiment 3: real=0.1450 (var=0.0093) vs this=0.3325 (var=0.0202)
- Experiment 4: real=0.1250 (var=0.0083) vs this=0.2131 (var=0.0166)
- Experiment 5: real=0.8817 (var=0.0046) vs this=0.6873 (var=0.0154)
- Experiment 6: real=0.6781 (var=0.0038) vs this=0.7891 (var=0.0184)
- Experiment 7: real=0.6029 (var=0.0041) vs this=0.8577 (var=0.0069)
- Experiment 8: real=0.7716 (var=0.0184) vs this=0.1495 (var=0.0056)


---

### `pi_2` (overall score: 0.483)

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
- Experiment 1: real=0.1564 (var=0.0079) vs this=0.2378 (var=0.0054)
- Experiment 2: real=0.8943 (var=0.0064) vs this=0.8726 (var=0.0064)
- Experiment 3: real=0.1450 (var=0.0093) vs this=0.4875 (var=0.0173)
- Experiment 4: real=0.1250 (var=0.0083) vs this=0.3792 (var=0.0038)
- Experiment 5: real=0.8817 (var=0.0046) vs this=0.4856 (var=0.0026)
- Experiment 6: real=0.6781 (var=0.0038) vs this=0.4988 (var=0.0035)
- Experiment 7: real=0.6029 (var=0.0041) vs this=0.8592 (var=0.0074)
- Experiment 8: real=0.7716 (var=0.0184) vs this=0.1553 (var=0.0103)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2144 -> ACCEPTED
- iter 2: loss=0.2809 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2144 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_pred(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            if val_b > val_a:
                return 1
        return -1

    preds = [get_ttb_pred(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    matches = sum(1 for p, r in zip(preds, data['response']) if p == r)
    valid = sum(1 for p in preds if p != -1)
    
    return float(matches) / valid if valid > 0 else 0.0
```

**Observed (real) value:** 0.1564 (var=0.0079)
**Previous candidate values (this loop):**
  - iter 1: 0.3260 (var=0.0192) (Δ vs real +0.1696)
  - iter 2 (most recent): 0.3733 (var=0.0340) (Δ vs real +0.2169)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8367 (var=0.0110)
- pi_2: 0.2378 (var=0.0054)
- pi_3: 0.3082 (var=0.0028)
- pi_4: 0.1758 (var=0.0092)
- pi_5: 0.2147 (var=0.0067)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_preds = np.where(a_wins > b_wins, 0, 1)
    actual_responses = data['response'].values
    
    matches = (tally_preds[mask] == actual_responses[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8943 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: 0.6767 (var=0.0210) (Δ vs real -0.2176)
  - iter 2 (most recent): 0.5860 (var=0.0280) (Δ vs real -0.3083)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8726 (var=0.0064)
- pi_1: 0.2512 (var=0.0055)
- pi_3: 0.8871 (var=0.0054)
- pi_4: 0.8652 (var=0.0101)
- pi_5: 0.8074 (var=0.0096)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    wadd_aligned = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Focus only on "tie" trials where Tallying predicts 50/50
        if sum_a == sum_b:
            val_a = np.sum(a * validities)
            val_b = np.sum(b * validities)
            
            if val_a > val_b:
                wadd_pref = 0
            elif val_b > val_a:
                wadd_pref = 1
            else:
                continue
                
            wadd_aligned.append(1 if row['response'] == wadd_pref else 0)
            
    if not wadd_aligned:
        return 0.5
        
    return float(np.mean(wadd_aligned))
```

**Observed (real) value:** 0.1450 (var=0.0093)
**Previous candidate values (this loop):**
  - iter 1: 0.2875 (var=0.0253) (Δ vs real +0.1425)
  - iter 2 (most recent): 0.3762 (var=0.0337) (Δ vs real +0.2312)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7738 (var=0.0340)
- pi_2: 0.4875 (var=0.0173)
- pi_1: 0.8712 (var=0.0155)
- pi_4: 0.3325 (var=0.0202)
- pi_5: 0.2562 (var=0.0321)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.99, 0.95, 0.55, 0.52, 0.5])
    
    wadd_acc = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
        
        wadd_prefers_a = (score_a > score_b)
        wadd_prefers_b = (score_b > score_a)
        
        tally_prefers_a = (a_wins > b_wins)
        tally_prefers_b = (b_wins > a_wins)
        
        # Filter for trials where Tallying does NOT agree with WADD
        # (i.e. Tallying is tied, or Tallying actively prefers the opposite)
        if (wadd_prefers_a and not tally_prefers_a) or \
           (wadd_prefers_b and not tally_prefers_b):
            
            if wadd_prefers_a:
                wadd_correct = (row['response'] == 0)
            else:
                wadd_correct = (row['response'] == 1)
                
            wadd_acc.append(float(wadd_correct))
            
    if len(wadd_acc) == 0:
        return 0.5
    return float(np.mean(wadd_acc))
```

**Observed (real) value:** 0.1250 (var=0.0083)
**Previous candidate values (this loop):**
  - iter 1: 0.2536 (var=0.0160) (Δ vs real +0.1286)
  - iter 2 (most recent): 0.3403 (var=0.0285) (Δ vs real +0.2153)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3792 (var=0.0038)
- pi_3: 0.6447 (var=0.0217)
- pi_1: 0.8542 (var=0.0155)
- pi_4: 0.2131 (var=0.0166)
- pi_5: 0.1689 (var=0.0131)

### Experiment 5
**Design**
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Return the proportion of trials where the subject chose Option A (response == 0)
    return float((data['response'] == 0).mean())

```

**Observed (real) value:** 0.8817 (var=0.0046)
**Previous candidate values (this loop):**
  - iter 1: 0.6548 (var=0.0163) (Δ vs real -0.2269)
  - iter 2 (most recent): 0.6394 (var=0.0226) (Δ vs real -0.2423)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6873 (var=0.0154)
- pi_2: 0.4856 (var=0.0026)
- pi_1: 0.1642 (var=0.0088)
- pi_3: 0.3125 (var=0.0174)
- pi_5: 0.7096 (var=0.0215)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for tie trials where both options have the same number of positive features
    tie_trials = data[sum_a == sum_b]
    
    if len(tie_trials) == 0:
        return 0.5
        
    return float(tie_trials['response'].mean())
```

**Observed (real) value:** 0.6781 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.7562 (var=0.0215) (Δ vs real +0.0781)
  - iter 2 (most recent): 0.6553 (var=0.0327) (Δ vs real -0.0228)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4988 (var=0.0035)
- pi_4: 0.7891 (var=0.0184)
- pi_1: 0.1453 (var=0.0098)
- pi_3: 0.2272 (var=0.0190)
- pi_5: 0.8137 (var=0.0238)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid = a_wins != b_wins
    if not np.any(valid):
        return 0.5
        
    tally_winner = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    match = (responses[valid] == tally_winner[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6029 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.5885 (var=0.0149) (Δ vs real -0.0144)
  - iter 2 (most recent): 0.5463 (var=0.0095) (Δ vs real -0.0567)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8577 (var=0.0069)
- pi_5: 0.6217 (var=0.0274)
- pi_1: 0.8640 (var=0.0057)
- pi_2: 0.8592 (var=0.0074)
- pi_3: 0.8792 (var=0.0077)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Trial 1
        if a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            return True
        # Trial 4
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 0, 0):
            return True
        return False
        
    mask = data.apply(is_target_trial, axis=1)
    if mask.sum() == 0:
        return 0.0
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.7716 (var=0.0184)
**Previous candidate values (this loop):**
  - iter 1: 0.5916 (var=0.0236) (Δ vs real -0.1800)
  - iter 2 (most recent): 0.5779 (var=0.0223) (Δ vs real -0.1937)
**Other theories' values on this metric (for reference):**
- pi_5: 0.5689 (var=0.0759)
- pi_4: 0.1495 (var=0.0056)
- pi_1: 0.1116 (var=0.0101)
- pi_2: 0.1553 (var=0.0103)
- pi_3: 0.1295 (var=0.0103)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Inverse Validity Weighting: Subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. Instead of relying on them, they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity. This mechanism leads subjects to consistently prefer options that are supported by lower-validity features over those supported by high-validity features, explaining the strong anti-expert choices observed in the experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Inverse Validity Weighting expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Primary mechanism: Weighted additive using inverse validities
    # Lower validity features receive higher subjective weights
    weights = (1.0 - val) ** gamma
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

`rationale`: Following the arbiter's guidance, this model instantiates Inverse Validity Weighting as the primary decision mechanism rather than a secondary tie-breaker. By assigning weights proportional to (1 - validity)^gamma, the model actively favors options with lower-validity cues. This correctly predicts the extreme magnitudes of 'anti-expert' behavior in Experiments 2, 5, 6, 7, and 8, where subjects systematically choose options with more low-validity features over fewer high-validity ones, while also keeping TTB and WADD alignment (Experiments 1, 3, 4) appropriately low.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate successfully implements the 'Inverse Validity Weighting' mechanism prescribed by the arbiter. By assigning weights proportional to (1 - validity)^gamma, the model correctly captures the anti-expert directionality observed across the experiments (e.g., low TTB/WADD alignment in Exps 1, 3, 4; high choice of low-validity options in Exps 5, 6, 8). However, the simulated values for Experiments 2, 5, and 8 under-predict the magnitude of the real human behavior, suggesting the anti-expert weighting effect needs to be even stronger.
Rationale: The candidate is in the correct mechanism family and was accepted, but the fit can be improved by allowing for more extreme inverse weighting. Expand the parameter range for `gamma` to [0.1, 10.0] to allow the model to more aggressively down-weight high-validity cues. Additionally, restrict the upper bound of the lapse rate `epsilon` to [0.0, 0.2] so that the model's choices aren't overly smoothed, enabling it to reach the higher magnitudes observed in Experiments 2, 5, and 8.

**Outcome of this advice:** iter 1 candidate loss=0.2144 -> iter 2 candidate loss=0.2809 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was rejected by the gate, as its loss (0.2809) was worse than the base iteration (0.2144). The attempt to increase the `gamma` range to [0.1, 10.0] and restrict `epsilon` actually degraded the fit, pulling predictions for Experiments 2, 5, and 8 further away from the empirical high magnitudes. The extreme non-linear transformation of the weights likely distorted the relative differences between options too severely.
Rationale: Since expanding `gamma` was rejected, we should build on the accepted base (iter 1) using a different knob to achieve the high choice magnitudes. Instead of an extreme non-linear scaling of the weights, revert the `gamma` parameter to a modest range (e.g., [0.1, 3.0]) or remove it entirely to strictly use `weights = 1.0 - val`. To allow the model to capture the extreme choice proportions in Exps 2, 5, and 8, increase the upper bound of the softmax inverse temperature `beta` to a much higher value (e.g., [0.1, 50.0] or [0.1, 100.0]). This will enable more deterministic choices based on the linear inverse-validity scores without distorting the underlying weight distribution.

**Outcome of this advice:** iter 2 candidate loss=0.2809 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Inverse Validity Weighting: Subjects actively distrust expert ratings, treating high-validity cues as manipulative or overly salient. Instead of relying on them, they use a weighted additive model where the subjective weight of each feature is inversely proportional to its stated validity (1 - validity). This mechanism leads subjects to consistently prefer options that are supported by lower-validity features over those supported by high-validity features, explaining the strong anti-expert choices observed in the experiments.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Inverse Validity Weighting expects a (2, n_features) stimulus; got shape {stim.shape}.\")\n    \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Primary mechanism: Weighted additive using inverse validities\n    # Lower validity features receive higher subjective weights\n    weights = 1.0 - val\n    \n    score_a = np.sum(a * weights)\n    score_b = np.sum(b * weights)\n    \n    scores = np.array([score_a, score_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Softmax choice rule with max-subtraction for numerical stability\n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.1, 100.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we remove the non-linear scaling parameter `gamma` to strictly use `weights = 1.0 - val`, preventing distortion of the relative differences between options. To achieve the high choice magnitudes observed in Experiments 2, 5, and 8, we increase the upper bound of the softmax inverse temperature `beta` to 100.0, allowing for more deterministic choices based on the linear inverse-validity scores without distorting the underlying weight distribution."
}
```
```

## Usage

```json
{
  "prompt_token_count": 13737,
  "candidates_token_count": 683,
  "total_token_count": 15408
}
```
