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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Weighted Additive (WADD) with Cue-Drop' or a 'Strict WADD' theory. The current theories assume Tallying is the primary driver and validities are secondary. The data from Exp 7 and Exp 3 show that subjects are highly sensitive to cue validities, often allowing one or two top-validity cues to override a larger tally of lower-validity cues. A model that computes a straightforward weighted sum of the features using the provided validities (with a softmax decision rule) would naturally capture the preference for top cues in unequal tallies (Exp 7) while also naturally breaking ties in favor of the highest validity cue (Exp 1 and 2).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.656)

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
- Experiment 1: real=0.1454 (var=0.0092) vs this=0.3196 (var=0.0022)
- Experiment 2: real=0.7971 (var=0.0103) vs this=0.8707 (var=0.0085)
- Experiment 3: real=0.1733 (var=0.0221) vs this=0.1411 (var=0.0136)
- Experiment 4: real=0.8125 (var=0.0197) vs this=0.8450 (var=0.0103)
- Experiment 5: real=0.1717 (var=0.0110) vs this=0.5008 (var=0.0051)
- Experiment 6: real=0.8554 (var=0.0133) vs this=0.8600 (var=0.0131)
- Experiment 7: real=0.6094 (var=0.0030) vs this=0.5028 (var=0.0034)
- Experiment 8: real=0.6178 (var=0.0023) vs this=0.4956 (var=0.0032)
- Experiment 9: real=0.7361 (var=0.0113) vs this=0.4979 (var=0.0043)
- Experiment 10: real=0.1525 (var=0.0073) vs this=0.5144 (var=0.0084)


---

### `pi_6` (overall score: 0.612)

**Description**
Validity-Biased Tallying: Decision-makers primarily rely on a compensatory tallying heuristic, but they do not treat all cues as perfectly equal. Instead, each winning cue contributes a base weight of 1 plus a small validity-dependent bonus based on the cue's rank. Because the validity bonus is strictly bounded, it cannot override a clear majority of cues, ensuring the model behaves like pure Tallying when one option has more wins. However, when the tallies are tied, these small, evenly spaced rank bonuses accumulate to probabilistically break the tie in favor of the option supported by higher-validity cues. This captures the slight empirical preference for more valid cues on tied trials without falling back to a deterministic, overly strong Take-The-Best strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    N = len(val)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Rank validities: lowest = 0, highest = N-1
    ranks = np.argsort(np.argsort(val)).astype(float)
    
    # Bonus is bounded such that the maximum possible sum of bonuses
    # is strictly less than gamma. Since sum(ranks) < N^2, this is guaranteed.
    weights = 1.0 + gamma * ranks / (N * N)
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(weights * a_wins)
    score_b = np.sum(weights * b_wins)
    
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 30.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 0.9]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1454 (var=0.0092) vs this=0.3875 (var=0.0048)
- Experiment 2: real=0.7971 (var=0.0103) vs this=0.8602 (var=0.0061)
- Experiment 3: real=0.1733 (var=0.0221) vs this=0.1767 (var=0.0155)
- Experiment 4: real=0.8125 (var=0.0197) vs this=0.8500 (var=0.0124)
- Experiment 5: real=0.1717 (var=0.0110) vs this=0.5758 (var=0.0050)
- Experiment 6: real=0.8554 (var=0.0133) vs this=0.8823 (var=0.0081)
- Experiment 7: real=0.6094 (var=0.0030) vs this=0.5636 (var=0.0061)
- Experiment 8: real=0.6178 (var=0.0023) vs this=0.5206 (var=0.0053)
- Experiment 9: real=0.7361 (var=0.0113) vs this=0.4547 (var=0.0044)
- Experiment 10: real=0.1525 (var=0.0073) vs this=0.5675 (var=0.0133)


---

### `pi_5` (overall score: 0.438)

**Description**
Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1454 (var=0.0092) vs this=0.4925 (var=0.0016)
- Experiment 2: real=0.7971 (var=0.0103) vs this=0.8621 (var=0.0047)
- Experiment 3: real=0.1733 (var=0.0221) vs this=0.1133 (var=0.0160)
- Experiment 4: real=0.8125 (var=0.0197) vs this=0.8444 (var=0.0134)
- Experiment 5: real=0.1717 (var=0.0110) vs this=0.6242 (var=0.0023)
- Experiment 6: real=0.8554 (var=0.0133) vs this=0.8738 (var=0.0118)
- Experiment 7: real=0.6094 (var=0.0030) vs this=0.8394 (var=0.0109)
- Experiment 8: real=0.6178 (var=0.0023) vs this=0.8386 (var=0.0099)
- Experiment 9: real=0.7361 (var=0.0113) vs this=0.8551 (var=0.0083)
- Experiment 10: real=0.1525 (var=0.0073) vs this=0.1394 (var=0.0095)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3029 -> ACCEPTED
- iter 2: loss=0.2811 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.2811 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
        
        if ttb is not None:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1454 (var=0.0092)
**Previous candidate values (this loop):**
  - iter 1: 0.3640 (var=0.0033) (Δ vs real +0.2185)
  - iter 2 (most recent): 0.4146 (var=0.0025) (Δ vs real +0.2692)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8696 (var=0.0065)
- pi_2: 0.3196 (var=0.0022)
- pi_3: 0.4487 (var=0.0024)
- pi_4: 0.3756 (var=0.0272)
- pi_5: 0.4925 (var=0.0016)
- pi_6: 0.3875 (var=0.0048)

### Experiment 2
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    
    valid = a_wins != b_wins
    if not np.any(valid):
        return 0.5
    
    pred = (b_wins[valid] > a_wins[valid]).astype(int)
    resp = data['response'].values[valid]
    
    return float(np.mean(pred == resp))
```

**Observed (real) value:** 0.7971 (var=0.0103)
**Previous candidate values (this loop):**
  - iter 1: 0.8798 (var=0.0065) (Δ vs real +0.0826)
  - iter 2 (most recent): 0.8762 (var=0.0061) (Δ vs real +0.0790)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8707 (var=0.0085)
- pi_1: 0.1590 (var=0.0097)
- pi_3: 0.8033 (var=0.0073)
- pi_4: 0.8731 (var=0.0191)
- pi_5: 0.8621 (var=0.0047)
- pi_6: 0.8602 (var=0.0061)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    wadd_consistent_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.sum(a * val)
        wadd_b = np.sum(b * val)
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # Check if WADD and Tallying make strictly opposite predictions
        if (wadd_a > wadd_b and tally_a < tally_b) or (wadd_a < wadd_b and tally_a > tally_b):
            wadd_pref = 0 if wadd_a > wadd_b else 1
            if row['response'] == wadd_pref:
                wadd_consistent_choices.append(1)
            else:
                wadd_consistent_choices.append(0)
                
    if len(wadd_consistent_choices) == 0:
        return 0.5
    return float(np.mean(wadd_consistent_choices))
```

**Observed (real) value:** 0.1733 (var=0.0221)
**Previous candidate values (this loop):**
  - iter 1: 0.1344 (var=0.0108) (Δ vs real -0.0389)
  - iter 2 (most recent): 0.1489 (var=0.0152) (Δ vs real -0.0244)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6844 (var=0.0230)
- pi_2: 0.1411 (var=0.0136)
- pi_1: 0.8633 (var=0.0160)
- pi_4: 0.1600 (var=0.0444)
- pi_5: 0.1133 (var=0.0160)
- pi_6: 0.1767 (var=0.0155)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying has a strict preference
    # (i.e., one option has more positive ratings than the other)
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with the higher sum.
    # Response is 0 for A, 1 for B.
    # If sum_a < sum_b, Tallying prefers B (1).
    # If sum_a > sum_b, Tallying prefers A (0).
    tallying_choice = (sum_a < sum_b).astype(int)
    
    # Calculate the proportion of responses matching the Tallying prediction
    matches = (data.loc[mask, 'response'] == tallying_choice.loc[mask])
    
    return float(matches.mean())
```

**Observed (real) value:** 0.8125 (var=0.0197)
**Previous candidate values (this loop):**
  - iter 1: 0.8700 (var=0.0075) (Δ vs real +0.0575)
  - iter 2 (most recent): 0.8531 (var=0.0115) (Δ vs real +0.0406)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8450 (var=0.0103)
- pi_3: 0.2462 (var=0.0221)
- pi_1: 0.1631 (var=0.0138)
- pi_4: 0.8444 (var=0.0501)
- pi_5: 0.8444 (var=0.0134)
- pi_6: 0.8500 (var=0.0124)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_top5 = a[:5]
        b_top5 = b[:5]
        
        a_wins = np.sum(a_top5 > b_top5)
        b_wins = np.sum(b_top5 > a_top5)
        
        if a_wins > b_wins:
            if row['response'] == 0:
                match_count += 1
            total_count += 1
        elif b_wins > a_wins:
            if row['response'] == 1:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
    return float(match_count / total_count)
```

**Observed (real) value:** 0.1717 (var=0.0110)
**Previous candidate values (this loop):**
  - iter 1: 0.6033 (var=0.0117) (Δ vs real +0.4317)
  - iter 2 (most recent): 0.5942 (var=0.0150) (Δ vs real +0.4225)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7113 (var=0.0562)
- pi_2: 0.5008 (var=0.0051)
- pi_1: 0.6154 (var=0.0052)
- pi_3: 0.7250 (var=0.0058)
- pi_5: 0.6242 (var=0.0023)
- pi_6: 0.5758 (var=0.0050)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    correct_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials with a strong tally difference (>= 2)
        if abs(a_wins - b_wins) >= 2:
            total_count += 1
            if a_wins > b_wins and row['response'] == 0:
                correct_count += 1
            elif b_wins > a_wins and row['response'] == 1:
                correct_count += 1
                
    if total_count == 0:
        return 0.0
    return float(correct_count / total_count)
```

**Observed (real) value:** 0.8554 (var=0.0133)
**Previous candidate values (this loop):**
  - iter 1: 0.8585 (var=0.0116) (Δ vs real +0.0031)
  - iter 2 (most recent): 0.8908 (var=0.0100) (Δ vs real +0.0354)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8600 (var=0.0131)
- pi_4: 0.9754 (var=0.0009)
- pi_1: 0.5131 (var=0.0041)
- pi_3: 0.8785 (var=0.0125)
- pi_5: 0.8738 (var=0.0118)
- pi_6: 0.8823 (var=0.0081)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    correct = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins == b_wins:
            ttb_pred = None
            for idx in range(len(a)):
                if a[idx] > b[idx]:
                    ttb_pred = 0
                    break
                elif b[idx] > a[idx]:
                    ttb_pred = 1
                    break
            if ttb_pred is not None:
                if row['response'] == ttb_pred:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.5
```

**Observed (real) value:** 0.6094 (var=0.0030)
**Previous candidate values (this loop):**
  - iter 1: 0.5650 (var=0.0058) (Δ vs real -0.0444)
  - iter 2 (most recent): 0.6108 (var=0.0173) (Δ vs real +0.0014)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8394 (var=0.0109)
- pi_2: 0.5028 (var=0.0034)
- pi_1: 0.8544 (var=0.0110)
- pi_3: 0.6428 (var=0.0063)
- pi_4: 0.4542 (var=0.0103)
- pi_6: 0.5636 (var=0.0061)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    tied_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tied_trials += 1
            ttb_choice = -1
            # Validities are monotonically decreasing with index
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            
            if row['response'] == ttb_choice:
                matches += 1
                
    if tied_trials == 0:
        return 0.5
    return float(matches / tied_trials)
```

**Observed (real) value:** 0.6178 (var=0.0023)
**Previous candidate values (this loop):**
  - iter 1: 0.5228 (var=0.0036) (Δ vs real -0.0950)
  - iter 2 (most recent): 0.5478 (var=0.0093) (Δ vs real -0.0700)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4956 (var=0.0032)
- pi_5: 0.8386 (var=0.0099)
- pi_1: 0.8364 (var=0.0104)
- pi_3: 0.5556 (var=0.0031)
- pi_4: 0.4419 (var=0.0090)
- pi_6: 0.5206 (var=0.0053)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tied_mask = (a_wins == b_wins)
    
    if np.sum(tied_mask) == 0:
        return 0.5
        
    # Response is 0 if subject chose A, 1 if subject chose B
    # We calculate the proportion of times A was chosen on tied trials
    return float(np.mean(data['response'].values[tied_mask] == 0))
```

**Observed (real) value:** 0.7361 (var=0.0113)
**Previous candidate values (this loop):**
  - iter 1: 0.4439 (var=0.0060) (Δ vs real -0.2923)
  - iter 2 (most recent): 0.4646 (var=0.0523) (Δ vs real -0.2716)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8551 (var=0.0083)
- pi_6: 0.4547 (var=0.0044)
- pi_1: 0.8519 (var=0.0109)
- pi_2: 0.4979 (var=0.0043)
- pi_3: 0.3853 (var=0.0086)
- pi_4: 0.3772 (var=0.0260)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_advocated = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Advocated favors B (1), Competing favors A (0)
        if a == (1, 0, 0, 0, 1, 1) and b == (0, 1, 1, 1, 0, 0):
            if resp == 1:
                match_advocated += 1
            total += 1
        # Trial 2: Advocated favors A (0), Competing favors B (1)
        elif a == (0, 1, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1, 1):
            if resp == 0:
                match_advocated += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(match_advocated / total)
```

**Observed (real) value:** 0.1525 (var=0.0073)
**Previous candidate values (this loop):**
  - iter 1: 0.6188 (var=0.0192) (Δ vs real +0.4663)
  - iter 2 (most recent): 0.5669 (var=0.0643) (Δ vs real +0.4144)
**Other theories' values on this metric (for reference):**
- pi_6: 0.5675 (var=0.0133)
- pi_5: 0.1394 (var=0.0095)
- pi_1: 0.1425 (var=0.0118)
- pi_2: 0.5144 (var=0.0084)
- pi_3: 0.7775 (var=0.0206)
- pi_4: 0.7100 (var=0.0565)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Soft Lexicographic Tallying with Cue-Drop: Decision-makers primarily rely on a compensatory Tallying heuristic, counting the number of winning features for each option. However, to resolve ties, they incorporate cue validities as a secondary, strictly bounded probabilistic tie-breaker, but they exhibit 'Cue-Drop' by only considering the top K most valid cues. By dropping lower-ranked cues from the tie-breaker, the model ensures that a single highly valid cue correctly outweighs a larger number of middling cues when tallies are tied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    k = int(parameters["k"])
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    # Cue-Drop: only consider top K validities for the WADD tie-breaker
    k_actual = min(k, len(val))
    top_k_idx = np.argsort(-val)[:k_actual]
    val_dropped = np.zeros_like(val)
    val_dropped[top_k_idx] = val[top_k_idx]
    
    # Calculate WADD scores based on dropped validities
    wadd_a = np.sum(val_dropped * a_wins)
    wadd_b = np.sum(val_dropped * b_wins)
    
    # Normalize WADD so the maximum possible value is 1.0
    # Then scale by gamma (which is < 1.0) to ensure it never overrides a tally difference of 1
    max_wadd = np.sum(val_dropped)
    if max_wadd == 0:
        max_wadd = 1.0
        
    bonus_a = gamma * (wadd_a / max_wadd)
    bonus_b = gamma * (wadd_b / max_wadd)
    
    score_a = tally_a + bonus_a
    score_b = tally_b + bonus_b
    
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 0.99]
- k: {1, 2, 3, 4, 5}
- validities: validities

`rationale`: I am implementing the critic's 'Cue-Drop' suggestion, but integrating it into the Soft Lexicographic Tallying model rather than using a Strict WADD model. A Strict WADD model (even with Cue-Drop) would catastrophically fail Exp 3 and 4, because subjects overwhelmingly choose the Tallying-favored option (~83% in Exp 3) even when WADD strongly disagrees. By keeping Tallying as the primary compensatory mechanism and applying Cue-Drop to the secondary WADD tie-breaker (zeroing out cues beyond the top K), we correctly favor the single top cue over a large number of middling cues on tied trials. This resolves the failures on Exp 9 and 10 without breaking the fundamental Tallying dominance on unequal tallies.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate model drifted from the arbiter's prescribed mechanism family. Instead of implementing the requested 'Weighted Additive (WADD) with Cue-Drop' or 'Strict WADD', the candidate implemented 'Soft Lexicographic Tallying', which uses a full WADD score as a bounded tie-breaker. This leads to failures on Exp 9 and Exp 10. In Exp 10, the tied tallies pit a single top cue (index 0) plus weak cues against several strong middling cues (indices 1, 2, 3). The full WADD score favors the middling cues, leading the model to predict the wrong choice (~62% for B), whereas humans overwhelmingly choose the option with the top cue (~85% for A, metric = 0.15).
Rationale: You must faithfully implement the arbiter's recommendation: a WADD model with Cue-Drop. Your current model fails on Exp 9 and 10 because a full WADD sum overvalues a large number of middling cues compared to a single top cue. To fix this and remain within the prescribed WADD family, implement 'Cue-Drop': compute the WADD score but ignore (zero out) cues beyond the top K validities (where K could be a free parameter, e.g., K in [1, 5]). By dropping lower-ranked cues, the WADD score will correctly favor the option with the highest validity cues, solving the failures in Exp 9 and 10 while adhering to the arbiter's mechanism family.

**Outcome of this advice:** iter 1 candidate loss=0.3029 -> iter 2 candidate loss=0.2811 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate model explicitly rejected the arbiter's recommendation to use WADD (with or without Cue-Drop) as the primary decision mechanism, instead retaining Tallying as the primary driver and relegating Cue-Drop WADD to a tie-breaker. As a result, the model continues to fail on Exp 9 and Exp 10, where cue validities override simple tallies. In Exp 10, the observed metric is 0.1525, but the model predicts 0.5669. In Exp 9, the observed metric is 0.7361, but the model predicts 0.4646.
Rationale: You must faithfully implement the mechanism family prescribed by the arbiter: WADD with Cue-Drop must be the PRIMARY driver of the decision, not a bounded tie-breaker for Tallying. Calculate the WADD score for each option using only the top K validities (zeroing out the rest) and feed these scores directly into the softmax choice rule. Do not compute or add a Tallying score. To address your concerns about Exp 3 and 4, a WADD model with Cue-Drop can still capture Tallying-like behavior if K is tuned appropriately (e.g., K=3 or 4) alongside the softmax temperature. Please return to the prescribed family.

**Outcome of this advice:** iter 2 candidate loss=0.2811 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Expected a (2, n_features) stimulus; got shape {stim.shape}.\")\n        \n    a, b = stim[0], stim[1]\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    k = int(parameters[\"k\"])\n    \n    a_wins = (a > b).astype(float)\n    b_wins = (b > a).astype(float)\n    \n    k_actual = min(k, len(val))\n    top_k_idx = np.argsort(-val)[:k_actual]\n    val_dropped = np.zeros_like(val)\n    val_dropped[top_k_idx] = val[top_k_idx]\n    \n    score_a = np.sum(val_dropped * a_wins)\n    score_b = np.sum(val_dropped * b_wins)\n    \n    scores = np.array([score_a, score_b])\n    \n    z = beta * (scores - np.max(scores))\n    e = np.exp(z)\n    p_core = e / np.sum(e)\n    \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "beta": "[0.1, 50.0]",
    "epsilon": "[0.0, 0.5]",
    "k": "{1, 2, 3, 4, 5, 6, 7}",
    "validities": "validities"
  },
  "rationale": "I am adopting the requested WADD with Cue-Drop structure as the primary decision mechanism. The Tallying baseline has been removed.",
  "theory": "Decision-makers use a Weighted Additive (WADD) strategy but exhibit Cue-Drop, meaning they only integrate the top K most valid cues and ignore the rest."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16417,
  "candidates_token_count": 633,
  "total_token_count": 17050
}
```
