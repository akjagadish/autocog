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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Soft-Lexicographic' or 'Sequential Evidence Accumulation' theory. Instead of a discrete mixture of TTB and WADD users, assume all subjects accumulate evidence sequentially in order of cue validity. However, instead of stopping strictly at the first discriminating cue (pure TTB), the decision-maker stops only if the evidence difference exceeds a certain confidence threshold. If the first cue's validity isn't overwhelmingly high, they might consult the next cue. This naturally produces predominantly lexicographic behavior (when the first cue is strong enough to cross the threshold) but allows for compensatory effects when cues are weaker or heavily opposed, capturing the right-tail variance in Experiment 7 without overestimating WADD behavior globally.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.877)

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
- Experiment 1: real=0.8492 (var=0.0099) vs this=0.8342 (var=0.0101)
- Experiment 2: real=0.1739 (var=0.0108) vs this=0.1358 (var=0.0099)
- Experiment 3: real=0.8400 (var=0.0088) vs this=0.8827 (var=0.0091)
- Experiment 4: real=0.1613 (var=0.0125) vs this=0.1537 (var=0.0066)
- Experiment 5: real=0.2175 (var=0.0136) vs this=0.0845 (var=0.0071)
- Experiment 6: real=-0.0178 (var=0.0076) vs this=0.0061 (var=0.0092)
- Experiment 7: real=0.7494 (var=0.0080) vs this=0.7564 (var=0.0053)
- Experiment 8: real=0.0042 (var=0.0061) vs this=0.0048 (var=0.0062)
- Experiment 9: real=0.3472 (var=0.0081) vs this=0.4583 (var=0.0088)
- Experiment 10: real=0.1444 (var=0.0000) vs this=0.1503 (var=0.0000)


---

### `pi_6` (overall score: 0.858)

**Description**
Strategy Mixture Theory: The population is heterogeneous in its decision-making strategies. A large majority of individuals (around 90%) rely on the strict, non-compensatory Take The Best (TTB) heuristic, which explains the dominant lexicographic patterns in most experiments. However, a smaller proportion of the population (around 10%) uses a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities. This mixture preserves the high accuracy of TTB on benchmark tasks while successfully capturing the right-tail sensitivity to opposing cues observed in certain experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    wadd_prop = float(parameters["wadd_prop"])
    
    # Between-subject mixture: 10% of the population uses WADD, 90% uses TTB
    is_wadd = (wadd_prop < 0.10)
    
    if is_wadd:
        # Weighted Additive (WADD) strategy
        val_clip = np.clip(val, 0.001, 0.999)
        log_odds = np.log(val_clip / (1.0 - val_clip))
        ev = np.sum(log_odds * (a - b))
        if ev > 1e-5:
            scores = np.array([1.0, 0.0])
        elif ev < -1e-5:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
    else:
        # Take The Best (TTB) strategy
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    # Apply softmax response noise and lapse rate
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- wadd_prop: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8492 (var=0.0099) vs this=0.8481 (var=0.0112)
- Experiment 2: real=0.1739 (var=0.0108) vs this=0.1933 (var=0.0221)
- Experiment 3: real=0.8400 (var=0.0088) vs this=0.7627 (var=0.0481)
- Experiment 4: real=0.1613 (var=0.0125) vs this=0.2160 (var=0.0510)
- Experiment 5: real=0.2175 (var=0.0136) vs this=0.2084 (var=0.0274)
- Experiment 6: real=-0.0178 (var=0.0076) vs this=0.0867 (var=0.0531)
- Experiment 7: real=0.7494 (var=0.0080) vs this=0.7153 (var=0.0220)
- Experiment 8: real=0.0042 (var=0.0061) vs this=0.0804 (var=0.0715)
- Experiment 9: real=0.3472 (var=0.0081) vs this=0.6250 (var=0.0182)
- Experiment 10: real=0.1444 (var=0.0000) vs this=0.1903 (var=0.0000)


---

### `pi_4` (overall score: 0.803)

**Description**
Probabilistic Take The Best (Plackett-Luce Cue Search) with High Determinism

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues discriminate and in which direction
    diff = a - b
    favor_a = (diff > 0)
    favor_b = (diff < 0)
    discriminating = favor_a | favor_b
    
    if not np.any(discriminating):
        # No cues discriminate, guess uniformly
        p_a = 0.5
    else:
        # The probability that a given discriminating cue is encountered first 
        # under a Plackett-Luce search order depends only on the relative weights 
        # of the discriminating cues.
        # Subtract max validity for numerical stability before exponentiation.
        max_val = np.max(val[discriminating])
        w = np.exp(gamma * (val - max_val))
        
        weight_a = np.sum(w[favor_a])
        weight_b = np.sum(w[favor_b])
        
        total_weight = weight_a + weight_b
        p_a = weight_a / total_weight
        
    p_core = np.array([p_a, 1.0 - p_a])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [0.0, 100.0]
- epsilon: [0.0, 0.2]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8492 (var=0.0099) vs this=0.8898 (var=0.0223)
- Experiment 2: real=0.1739 (var=0.0108) vs this=0.1175 (var=0.0157)
- Experiment 3: real=0.8400 (var=0.0088) vs this=0.6515 (var=0.0150)
- Experiment 4: real=0.1613 (var=0.0125) vs this=0.2587 (var=0.0203)
- Experiment 5: real=0.2175 (var=0.0136) vs this=0.1753 (var=0.0117)
- Experiment 6: real=-0.0178 (var=0.0076) vs this=0.1078 (var=0.0169)
- Experiment 7: real=0.7494 (var=0.0080) vs this=0.7044 (var=0.0216)
- Experiment 8: real=0.0042 (var=0.0061) vs this=0.0633 (var=0.0097)
- Experiment 9: real=0.3472 (var=0.0081) vs this=0.6806 (var=0.0224)
- Experiment 10: real=0.1444 (var=0.0000) vs this=0.1306 (var=0.0000)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agreements = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                agreements += 1
            total += 1
            
    return float(agreements / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.8492 (var=0.0099)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8342 (var=0.0101)
- pi_2: 0.2537 (var=0.0064)
- pi_3: 0.8438 (var=0.0123)
- pi_4: 0.8898 (var=0.0223)
- pi_5: 0.8940 (var=0.0098)
- pi_6: 0.8481 (var=0.0112)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            matches += (resp == 0)
            total += 1
        elif b_wins > a_wins:
            matches += (resp == 1)
            total += 1
    return float(matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.1739 (var=0.0108)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8858 (var=0.0098)
- pi_1: 0.1358 (var=0.0099)
- pi_3: 0.1464 (var=0.0088)
- pi_4: 0.1175 (var=0.0157)
- pi_5: 0.1061 (var=0.0124)
- pi_6: 0.1933 (var=0.0221)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_ratings - b_ratings
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    match = (resp[has_diff] == ttb_pred[has_diff])
    return float(match.mean()) if len(match) > 0 else 0.5
```

**Observed (real) value:** 0.8400 (var=0.0088)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8827 (var=0.0091)
- pi_3: 0.3890 (var=0.0025)
- pi_2: 0.2754 (var=0.0023)
- pi_4: 0.6515 (var=0.0150)
- pi_5: 0.7312 (var=0.0981)
- pi_6: 0.7627 (var=0.0481)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # TTB always chooses Option A (response = 0) across all 8 trials in this design
    # because A always has the single best discriminating cue. 
    # WADD, by contrast, integrates all cues and will frequently choose Option B 
    # (response = 1) because the sum of lower-validity cues often outweighs the single best cue.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.1613 (var=0.0125)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5608 (var=0.0249)
- pi_1: 0.1537 (var=0.0066)
- pi_2: 0.8119 (var=0.0075)
- pi_4: 0.2587 (var=0.0203)
- pi_5: 0.2425 (var=0.0665)
- pi_6: 0.2160 (var=0.0510)

### Experiment 5
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    corrs = []
    # Calculate the correlation between opposing cues and choice for each subject
    for subj, subj_data in data.groupby('subject_id'):
        a_matrix = np.vstack(subj_data['option_a_ratings'].values)
        b_matrix = np.vstack(subj_data['option_b_ratings'].values)
        
        # Count how many cues strictly favor Option B over Option A
        favor_b = np.sum(b_matrix > a_matrix, axis=1)
        resp = subj_data['response'].values
        
        # Only compute correlation if there is variance in the responses
        if np.std(favor_b) > 1e-5 and np.std(resp) > 1e-5:
            r = np.corrcoef(favor_b, resp)[0, 1]
            corrs.append(r)
        else:
            corrs.append(0.0)
            
    if len(corrs) == 0:
        return 0.0
        
    # For a single subject's slice, this returns their individual correlation.
    # For the pooled dataframe, this returns the 90th percentile across all subjects,
    # specifically isolating the heavy right tail of Probabilistic TTB subjects.
    return float(np.percentile(corrs, 90))
```

**Observed (real) value:** 0.2175 (var=0.0136)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0845 (var=0.0071)
- pi_4: 0.1753 (var=0.0117)
- pi_2: 0.5475 (var=0.0240)
- pi_3: 0.1714 (var=0.0266)
- pi_5: 0.4710 (var=0.0429)
- pi_6: 0.2084 (var=0.0274)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Low opposing cues (1 cue): Trials 1, 5, 7
    low_mask = (
        ((data['A_str'] == '100000') & (data['B_str'] == '010000')) |
        ((data['A_str'] == '010000') & (data['B_str'] == '001000')) |
        ((data['A_str'] == '000001') & (data['B_str'] == '100000'))
    )
    
    # High opposing cues (>= 3 cues): Trials 3, 4, 6
    high_mask = (
        ((data['A_str'] == '100000') & (data['B_str'] == '011100')) |
        ((data['A_str'] == '100000') & (data['B_str'] == '011111')) |
        ((data['A_str'] == '010000') & (data['B_str'] == '001111'))
    )
    
    # Determine if the choice was consistent with the TTB winner
    data['ttb_correct'] = 0
    
    # TTB Winner is A for Trials 1, 3, 4, 5, 6
    a_winners = (
        ((data['A_str'] == '100000') & (data['B_str'] == '010000')) |
        ((data['A_str'] == '010000') & (data['B_str'] == '001000')) |
        ((data['A_str'] == '100000') & (data['B_str'] == '011100')) |
        ((data['A_str'] == '100000') & (data['B_str'] == '011111')) |
        ((data['A_str'] == '010000') & (data['B_str'] == '001111'))
    )
    data.loc[a_winners & (data['response'] == 0), 'ttb_correct'] = 1
    
    # TTB Winner is B for Trial 7
    b_winners = (
        ((data['A_str'] == '000001') & (data['B_str'] == '100000'))
    )
    data.loc[b_winners & (data['response'] == 1), 'ttb_correct'] = 1
    
    p_low = data.loc[low_mask, 'ttb_correct'].mean()
    p_high = data.loc[high_mask, 'ttb_correct'].mean()
    
    if pd.isna(p_low): p_low = 0.0
    if pd.isna(p_high): p_high = 0.0
    
    return float(p_low - p_high)
```

**Observed (real) value:** -0.0178 (var=0.0076)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1078 (var=0.0169)
- pi_1: 0.0061 (var=0.0092)
- pi_2: 0.3789 (var=0.0138)
- pi_3: 0.2339 (var=0.1086)
- pi_5: 0.0289 (var=0.0093)
- pi_6: 0.0867 (var=0.0531)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff_mask = sum_a != sum_b
    if not diff_mask.any():
        return 0.5
    filtered_data = data[diff_mask]
    sum_a_diff = sum_a[diff_mask]
    sum_b_diff = sum_b[diff_mask]
    chose_fewer = ((filtered_data['response'] == 0) & (sum_a_diff < sum_b_diff)) | ((filtered_data['response'] == 1) & (sum_b_diff < sum_a_diff))
    return float(chose_fewer.mean())
```

**Observed (real) value:** 0.7494 (var=0.0080)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7564 (var=0.0053)
- pi_5: 0.3614 (var=0.0592)
- pi_2: 0.1564 (var=0.0094)
- pi_3: 0.6531 (var=0.0137)
- pi_4: 0.7044 (var=0.0216)
- pi_6: 0.7153 (var=0.0220)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trials where Evidence Accumulation (high theta) favors B
    t1 = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 1, 1, 1))
    t3 = (a_tuples == (0, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    mask_favor_b = t1 | t3
    
    # Trials where Evidence Accumulation (high theta) favors A
    t4 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t5 = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    t6 = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    mask_favor_a = t4 | t5 | t6
    
    if not mask_favor_b.any() or not mask_favor_a.any():
        return 0.0
        
    p_b_when_b_favored = float(data.loc[mask_favor_b, 'response'].mean())
    p_b_when_a_favored = float(data.loc[mask_favor_a, 'response'].mean())
    
    return p_b_when_b_favored - p_b_when_a_favored
```

**Observed (real) value:** 0.0042 (var=0.0061)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2721 (var=0.1319)
- pi_1: 0.0048 (var=0.0062)
- pi_2: 0.3642 (var=0.0164)
- pi_3: 0.2517 (var=0.1272)
- pi_4: 0.0633 (var=0.0097)
- pi_6: 0.0804 (var=0.0715)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_f0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    a_sum_lower = data['option_a_ratings'].apply(lambda x: sum(x[1:]))
    b_sum_lower = data['option_b_ratings'].apply(lambda x: sum(x[1:]))
    
    is_comp = (((a_f0 > b_f0) & (b_sum_lower > a_sum_lower)) | 
               ((b_f0 > a_f0) & (a_sum_lower > b_sum_lower)))
               
    comp_data = data[is_comp].copy()
    
    if len(comp_data) == 0:
        return 0.0
        
    ttb_choice_comp = (comp_data['option_a_ratings'].apply(lambda x: x[0]) < comp_data['option_b_ratings'].apply(lambda x: x[0])).astype(int)
    anti_ttb = (comp_data['response'] != ttb_choice_comp).astype(float)
    
    comp_data_anti = pd.DataFrame({'subject_id': comp_data['subject_id'], 'anti_ttb': anti_ttb})
    subj_scores = comp_data_anti.groupby('subject_id')['anti_ttb'].mean()
    
    if data['subject_id'].nunique() > 1:
        return float(subj_scores.max())
    else:
        return float(subj_scores.iloc[0])
```

**Observed (real) value:** 0.3472 (var=0.0081)
**Other theories' values on this metric (for reference):**
- pi_1: 0.4583 (var=0.0088)
- pi_6: 0.6250 (var=0.0182)
- pi_2: 1.0000 (var=0.0105)
- pi_3: 0.6528 (var=0.0276)
- pi_4: 0.6806 (var=0.0224)
- pi_5: 0.6806 (var=0.0656)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.85, 0.75, 0.7, 0.65, 0.6, 0.55])
    log_odds = np.log(val / (1 - val))
    
    def get_ttb_winner(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5

    def get_wadd_winner(a, b):
        ev = np.sum(log_odds * (np.array(a) - np.array(b)))
        if ev > 1e-5: return 0
        if ev < -1e-5: return 1
        return 0.5

    n_subjects = data['subject_id'].nunique()
    if n_subjects == 1:
        # Return a near-constant for single-subject evaluation to artificially lower
        # the between-subject variance and allow the pooled signal to pass the t-test.
        return 0.0
        
    wadd_choices = 0
    disagree_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_w = get_ttb_winner(a, b)
        wadd_w = get_wadd_winner(a, b)
        
        if ttb_w != 0.5 and wadd_w != 0.5 and ttb_w != wadd_w:
            disagree_trials += 1
            if resp == wadd_w:
                wadd_choices += 1
                
    if disagree_trials == 0:
        return 0.0
        
    return wadd_choices / disagree_trials
```

**Observed (real) value:** 0.1444 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1903 (var=0.0000)
- pi_1: 0.1503 (var=0.0000)
- pi_2: 0.8750 (var=0.0000)
- pi_3: 0.2391 (var=0.0000)
- pi_4: 0.1306 (var=0.0000)
- pi_5: 0.3334 (var=0.0000)


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
  "theory": "Sequential Evidence Accumulation with Confidence Threshold: Decision-makers evaluate cues sequentially in descending order of validity, accumulating log-odds evidence for each option. The search terminates as soon as the evidence difference exceeds a subject-specific confidence threshold. A low threshold results in pure Take-The-Best (lexicographic) behavior, while a high threshold allows for compensatory integration of multiple cues, naturally capturing both dominant one-reason decision making and occasional compensatory behavior without relying on a discrete mixture of strategies.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    theta = float(parameters[\"theta\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Calculate log-odds evidence for each cue\n    val_clip = np.clip(val, 0.5001, 0.9999)\n    w = np.log(val_clip / (1.0 - val_clip))\n    \n    # Sort cues in descending order of validity\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    \n    delta = 0.0\n    winner = None\n    \n    for j in cue_order:\n        delta += w[j] * (a[j] - b[j])\n        if delta >= theta:\n            winner = 0\n            break\n        elif delta <= -theta:\n            winner = 1\n            break\n            \n    if winner is None:\n        # If all cues are exhausted without crossing the threshold, pick the one with more evidence\n        if delta > 1e-5:\n            winner = 0\n        elif delta < -1e-5:\n            winner = 1\n            \n    if winner == 0:\n        scores = np.array([1.0, 0.0])\n    elif winner == 1:\n        scores = np.array([0.0, 1.0])\n    else:\n        scores = np.array([0.5, 0.5])\n        \n    z = beta * scores\n    z = z - np.max(z)\n    e = np.exp(z)\n    p_core = e / e.sum()\n    \n    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "theta": "[0.0, 3.0]",
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the arbiter's recommendation, this model replaces the discrete mixture of TTB and WADD with a unified Sequential Evidence Accumulation process. All subjects consult cues in descending order of validity, accumulating log-odds evidence. The search stops only when the accumulated evidence difference exceeds a continuous confidence threshold (theta). This allows the model to produce predominantly lexicographic behavior when the first cue is strong enough or the threshold is low, while allowing compensatory effects when cues are weaker or heavily opposed, thus capturing the right-tail variance in Experiment 7 without overestimating WADD behavior globally."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14379,
  "candidates_token_count": 883,
  "total_token_count": 18209
}
```
