# generation_iter_02_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Heuristic Decision Making domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
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
- rating_max: Upper bound (inclusive) of each rating value (LLM-proposed; >= 1). Ratings are integers in [0, rating_max].

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
  - `"option_a_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option A on this trial.
  - `"option_b_ratings"`: List of n_features integer expert ratings (each in [0, rating_max]) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` integer expert ratings in [0, rating_max]. The per-feature validities, n_features, and rating_max are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. All three — `n_features`, `validities`, and `rating_max` — are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Propose a 'Heuristic Strategy Mixture' or 'Soft Take-The-Best' theory. A mixture theory would posit that decision-makers probabilistically sample between a Pure Tallying strategy and a strict Take-The-Best (lexicographic) strategy on each trial (or across subjects). Alternatively, a Soft TTB theory would use rank-based exponential weights rather than the explicit validities, allowing the most valid cue to have strong but not absolute dominance. This will naturally capture the intermediate means observed across experiments better than Weighted Tallying, which uses the linear validities.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.584)

**Description**
Pure Tallying Heuristic: Decision-makers evaluate options by completely ignoring the provided validities or weights of the cues. Instead, they simply count the number of features on which one option is strictly better than the other. The option with the higher total count of winning features is chosen. This unweighted counting mechanism explains the robust preference for options with more winning features, even if those features are of lower validity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins; completely ignore validities and magnitudes
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    scores = np.array([a_wins, b_wins], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7250 (var=0.0281) vs this=0.8526 (var=0.0098)
- Experiment 2: real=0.2600 (var=0.1924) vs this=0.1600 (var=0.0177)
- Experiment 3: real=0.7440 (var=0.0257) vs this=0.8423 (var=0.0117)
- Experiment 4: real=0.3680 (var=0.0278) vs this=0.2895 (var=0.0063)
- Experiment 5: real=0.4200 (var=0.2436) vs this=0.1900 (var=0.0167)
- Experiment 6: real=0.2800 (var=0.2016) vs this=0.1926 (var=0.0315)
- Experiment 7: real=0.3400 (var=0.0733) vs this=0.3008 (var=0.0070)
- Experiment 8: real=0.6080 (var=0.0943) vs this=0.7491 (var=0.0121)
- Experiment 9: real=0.0400 (var=0.9584) vs this=0.3695 (var=0.0253)
- Experiment 10: real=0.5400 (var=0.2484) vs this=0.1500 (var=0.0158)


---

### `pi_4` (overall score: 0.395)

**Description**
Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7250 (var=0.0281) vs this=0.8592 (var=0.0096)
- Experiment 2: real=0.2600 (var=0.1924) vs this=0.1537 (var=0.0208)
- Experiment 3: real=0.7440 (var=0.0257) vs this=0.8658 (var=0.0049)
- Experiment 4: real=0.3680 (var=0.0278) vs this=0.2812 (var=0.0030)
- Experiment 5: real=0.4200 (var=0.2436) vs this=0.7405 (var=0.0148)
- Experiment 6: real=0.2800 (var=0.2016) vs this=0.7411 (var=0.0266)
- Experiment 7: real=0.3400 (var=0.0733) vs this=0.1760 (var=0.0121)
- Experiment 8: real=0.6080 (var=0.0943) vs this=0.7714 (var=0.0087)
- Experiment 9: real=0.0400 (var=0.9584) vs this=-0.3305 (var=0.0274)
- Experiment 10: real=0.5400 (var=0.2484) vs this=0.6350 (var=0.0226)


---

### `pi_2` (overall score: 0.351)

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

    w = np.asarray(parameters["validities"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"validities length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ w

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
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.7250 (var=0.0281) vs this=0.3600 (var=0.0729)
- Experiment 2: real=0.2600 (var=0.1924) vs this=0.3600 (var=0.2304)
- Experiment 3: real=0.7440 (var=0.0257) vs this=0.2880 (var=0.0387)
- Experiment 4: real=0.3680 (var=0.0278) vs this=0.8480 (var=0.0233)
- Experiment 5: real=0.4200 (var=0.2436) vs this=0.4000 (var=0.1900)
- Experiment 6: real=0.2800 (var=0.2016) vs this=0.3600 (var=0.2304)
- Experiment 7: real=0.3400 (var=0.0733) vs this=0.2667 (var=0.0500)
- Experiment 8: real=0.6080 (var=0.0943) vs this=0.7840 (var=0.0445)
- Experiment 9: real=0.0400 (var=0.9584) vs this=0.3000 (var=0.7300)
- Experiment 10: real=0.5400 (var=0.2484) vs this=0.1800 (var=0.1476)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3570 -> ACCEPTED
- iter 2: loss=0.2140 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.2140 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[5, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[2, 5, 2, 2]  B=[3, 1, 3, 3]
  A=[4, 4, 1, 5]  B=[5, 5, 2, 0]
  A=[0, 0, 5, 5]  B=[1, 1, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        if a[0] == 1:
            matches.append(r == 0)
        elif a[0] == 5:
            matches.append(r == 1)
        elif a[0] == 2:
            matches.append(r == 1)
        elif a[0] == 4:
            matches.append(r == 1)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7250 (var=0.0281)
**Previous candidate values (this loop):**
  - iter 1: 0.6729 (var=0.0314) (Δ vs real -0.0521)
  - iter 2 (most recent): 0.6876 (var=0.0201) (Δ vs real -0.0374)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.3600 (var=0.0729)
- pi_3: 0.1121 (var=0.0066)
- pi_4: 0.8592 (var=0.0096)
- pi_5: 0.6097 (var=0.0228)
- pi_6: 0.8526 (var=0.0098)

### Experiment 2
**Design**
  A=[5, 5, 5, 5]  B=[0, 6, 6, 6]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 5, 5, 4]
  A=[8, 4, 4, 4]  B=[5, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[0, 4, 4, 4], B=[10, 0, 0, 0]
    # WADD prefers B (response=1) because 10*0.9 = 9.0 > 4*(0.8+0.6+0.5) = 7.6
    # Tallying prefers A (response=0) because A wins on 3 features vs B's 1
    
    # Identify trial 2 by checking the first element of option A
    is_trial_2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 4, 4, 4))
    trial_data = data[is_trial_2]
    
    if len(trial_data) == 0:
        return 0.5
        
    return float(trial_data['response'].mean())
```

**Observed (real) value:** 0.2600 (var=0.1924)
**Previous candidate values (this loop):**
  - iter 1: 0.4989 (var=0.1079) (Δ vs real +0.2389)
  - iter 2 (most recent): 0.5253 (var=0.0496) (Δ vs real +0.2653)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3600 (var=0.2304)
- pi_1: 0.0000 (var=0.0000)
- pi_3: 0.1421 (var=0.0130)
- pi_4: 0.1537 (var=0.0208)
- pi_5: 0.6358 (var=0.1058)
- pi_6: 0.1600 (var=0.0177)

### Experiment 3
**Design**
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 10, 0, 0]  B=[0, 0, 1, 1]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 8]
  A=[9, 9, 0, 0]  B=[10, 0, 1, 1]

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
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            matches.append(1 if row['response'] == tally_pref else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7440 (var=0.0257)
**Previous candidate values (this loop):**
  - iter 1: 0.7585 (var=0.0269) (Δ vs real +0.0145)
  - iter 2 (most recent): 0.6935 (var=0.0091) (Δ vs real -0.0505)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.2258 (var=0.0057)
- pi_2: 0.2880 (var=0.0387)
- pi_4: 0.8658 (var=0.0049)
- pi_5: 0.6803 (var=0.0220)
- pi_6: 0.8423 (var=0.0117)

### Experiment 4
**Design**
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[10, 10, 0, 0]  B=[0, 0, 2, 2]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[2, 2, 0, 0]  B=[0, 0, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    eqw_predicts_a = a_sums > b_sums
    eqw_predicts_b = b_sums > a_sums
    
    match_a = eqw_predicts_a & (data['response'] == 0)
    match_b = eqw_predicts_b & (data['response'] == 1)
    
    valid_trials = eqw_predicts_a | eqw_predicts_b
    if not valid_trials.any():
        return 0.5
        
    return float((match_a | match_b).sum() / valid_trials.sum())
```

**Observed (real) value:** 0.3680 (var=0.0278)
**Previous candidate values (this loop):**
  - iter 1: 0.3713 (var=0.0064) (Δ vs real +0.0033)
  - iter 2 (most recent): 0.3250 (var=0.0050) (Δ vs real -0.0430)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8640 (var=0.0069)
- pi_1: 0.2000 (var=0.0000)
- pi_2: 0.8480 (var=0.0233)
- pi_4: 0.2812 (var=0.0030)
- pi_5: 0.3635 (var=0.0081)
- pi_6: 0.2895 (var=0.0063)

### Experiment 5
**Design**
  A=[1, 1, 4, 4, 4]  B=[4, 4, 1, 1, 1]
  A=[5, 5, 2, 2, 2]  B=[2, 2, 4, 4, 4]
  A=[4, 2, 4, 2, 3]  B=[2, 4, 2, 4, 3]
  A=[2, 3, 4, 4, 2]  B=[4, 3, 2, 2, 4]
  A=[4, 2, 3, 3, 3]  B=[2, 4, 3, 3, 3]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Trial 1: A=[1, 1, 4, 4, 4], B=[4, 4, 1, 1, 1]
    # Tallying prefers A (3 vs 2). Weighted Tallying prefers B (1.7 vs 1.5).
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 4)
    
    # Trial 2: A=[5, 5, 2, 2, 2], B=[2, 2, 4, 4, 4]
    # Tallying prefers B (3 vs 2). Weighted Tallying prefers A (1.7 vs 1.5).
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 2)
    
    # Proportion of times choosing B (response == 1) in Trial 1
    t1_data = data.loc[t1_mask, 'response']
    val1 = t1_data.mean() if len(t1_data) > 0 else 0.5
    
    # Proportion of times choosing A (response == 0) in Trial 2
    t2_data = data.loc[t2_mask, 'response']
    val2 = (1.0 - t2_data.mean()) if len(t2_data) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)
```

**Observed (real) value:** 0.4200 (var=0.2436)
**Previous candidate values (this loop):**
  - iter 1: 0.7211 (var=0.0670) (Δ vs real +0.3011)
  - iter 2 (most recent): 0.4837 (var=0.0515) (Δ vs real +0.0637)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0000)
- pi_4: 0.7405 (var=0.0148)
- pi_2: 0.4000 (var=0.1900)
- pi_3: 0.3284 (var=0.0060)
- pi_5: 0.8221 (var=0.0280)
- pi_6: 0.1900 (var=0.0167)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where A=[1, 1, 0, 0, 0] and B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.5
    trial_1_data = data[is_trial_1]
    # Calculate the proportion of times Option A was chosen (response == 0)
    prob_a = (trial_1_data['response'] == 0).mean()
    return float(prob_a)
```

**Observed (real) value:** 0.2800 (var=0.2016)
**Previous candidate values (this loop):**
  - iter 1: 0.7463 (var=0.0619) (Δ vs real +0.4663)
  - iter 2 (most recent): 0.5579 (var=0.0465) (Δ vs real +0.2779)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7411 (var=0.0266)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.3600 (var=0.2304)
- pi_3: 0.1463 (var=0.0153)
- pi_5: 0.7274 (var=0.0701)
- pi_6: 0.1926 (var=0.0315)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In all designed trials, Option A has fewer wins but on higher-ranked cues,
    # whereas Option B has more wins but on slightly lower-validity cues.
    # Weighted Tallying treats the clustered validities as nearly linear,
    # so the sum of multiple lower-ranked wins (Option B) easily outweighs Option A.
    # Soft TTB uses a power-law decay on rank, allowing the top-ranked cue(s) of Option A to dominate.
    # Thus, Soft TTB predicts choosing A (response == 0) much more frequently than Weighted Tallying.
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.3400 (var=0.0733)
**Previous candidate values (this loop):**
  - iter 1: 0.6827 (var=0.0440) (Δ vs real +0.3427)
  - iter 2 (most recent): 0.5652 (var=0.0407) (Δ vs real +0.2252)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7006 (var=0.0367)
- pi_4: 0.1760 (var=0.0121)
- pi_1: 0.3333 (var=0.0000)
- pi_2: 0.2667 (var=0.0500)
- pi_3: 0.2644 (var=0.0056)
- pi_6: 0.3008 (var=0.0070)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract the first feature (highest validity) for options A and B
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Weighted Tallying favors the combination of multiple lower-ranked features
    # over the single highest-ranked feature in this specific experimental design.
    # Therefore, WT predicts choosing the option that LOSES on feature 1.
    # If A > B on feature 1, B loses on feature 1, so WT predicts 1 (Option B).
    # If A < B on feature 1, A loses on feature 1, so WT predicts 0 (Option A).
    wt_predicted_response = (a_f1 > b_f1).astype(int)
    
    # Calculate the proportion of trials where the subject's choice aligns with WT
    match = (data['response'] == wt_predicted_response)
    return float(match.mean())
```

**Observed (real) value:** 0.6080 (var=0.0943)
**Previous candidate values (this loop):**
  - iter 1: 0.3467 (var=0.0567) (Δ vs real -0.2613)
  - iter 2 (most recent): 0.4438 (var=0.0391) (Δ vs real -0.1642)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7714 (var=0.0087)
- pi_5: 0.2817 (var=0.0536)
- pi_1: 0.8000 (var=0.0000)
- pi_2: 0.7840 (var=0.0445)
- pi_3: 0.8017 (var=0.0047)
- pi_6: 0.7491 (var=0.0121)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def match_t1(row):
        a, b = row['option_a_ratings'], row['option_b_ratings']
        return sum(a) == 2 and sum(b) == 3
        
    def match_t2(row):
        a, b = row['option_a_ratings'], row['option_b_ratings']
        return sum(a) == 2 and sum(b) == 2 and a[0] == 1 and b[2] == 1
        
    is_t1 = data.apply(match_t1, axis=1)
    is_t2 = data.apply(match_t2, axis=1)
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    if len(t1_data) == 0 or len(t2_data) == 0:
        return 0.0
        
    p_b_t1 = t1_data['response'].mean()
    p_a_t2 = 1.0 - t2_data['response'].mean()
    
    return float(p_b_t1 - p_a_t2)

```

**Observed (real) value:** 0.0400 (var=0.9584)
**Previous candidate values (this loop):**
  - iter 1: -0.5653 (var=0.1244) (Δ vs real -0.6053)
  - iter 2 (most recent): -0.2474 (var=0.1088) (Δ vs real -0.2874)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3695 (var=0.0253)
- pi_4: -0.3305 (var=0.0274)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.3000 (var=0.7300)
- pi_3: 0.3621 (var=0.0331)
- pi_5: -0.5453 (var=0.1090)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where one of the options has the two highest-validity features (index 0 and 1) 
    # and nothing else. This corresponds to Trial 1: [1, 1, 0, 0, 0, 0] vs [0, 0, 0, 1, 1, 1].
    a_target = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)
    b_target = data['option_b_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1)
    
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Check if the subject chose the target option
    target_chosen = (a_target & chose_a) | (b_target & chose_b)
    valid_trials = a_target | b_target
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(target_chosen.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5400 (var=0.2484)
**Previous candidate values (this loop):**
  - iter 1: 0.7950 (var=0.0528) (Δ vs real +0.2550)
  - iter 2 (most recent): 0.5088 (var=0.0839) (Δ vs real -0.0312)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6350 (var=0.0226)
- pi_6: 0.1500 (var=0.0158)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.1800 (var=0.1476)
- pi_3: 0.1688 (var=0.0183)
- pi_5: 0.8025 (var=0.0315)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Heuristic Strategy Mixture: Decision-makers probabilistically sample between a Pure Tallying strategy (unweighted sum of strict feature wins) and a strict Take-The-Best strategy (relying solely on the highest-validity cue that differentiates the options). This mixture approach captures the heterogeneity of choice patterns across subjects and trials better than a single compromised weighting scheme.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Pure Tallying Strategy: unweighted count of strict wins
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    scores_tally = np.array([a_wins, b_wins], dtype=float)
    
    # 2. Strict Take-The-Best Strategy: lexicographic choice based on highest-validity differentiating cue
    sorted_idx = np.argsort(-v)
    a_ttb = 0.0
    b_ttb = 0.0
    for idx in sorted_idx:
        if a[idx] > b[idx]:
            a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            b_ttb = 1.0
            break
    scores_ttb = np.array([a_ttb, b_ttb], dtype=float)
    
    beta = float(parameters["beta"])
    
    # Tallying probabilities
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # TTB probabilities
    z_ttb = beta * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_mixed = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Blend with uniform lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mixed)
    p_final = (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- w_tally: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the critic's feedback, this version implements the 'Heuristic Strategy Mixture' model. Rather than forcing a single 'soft' weighting scheme that failed to capture empirical patterns in Experiments 6-9, this theory posits that choices reflect a probabilistic mixture of two distinct, well-established heuristics: Pure Tallying (which completely ignores validities and counts wins) and strict Take-The-Best (which lexicographically decides based solely on the highest-validity differentiating cue). The parameter `w_tally` blends the choice probabilities generated by these two independent mechanisms, allowing the model to naturally capture the intermediate and heterogeneous choice frequencies observed in the data.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory implemented Soft Take-The-Best using rank-based exponential weights. While this was ACCEPTED by the gate (loss improved to 0.3570), the fit remains quite poor on several key experiments. In Experiment 6, observed is 0.28 while the candidate predicts 0.74. In Experiment 7, observed is 0.34 while the candidate predicts 0.68. In Experiment 8, observed is 0.61 while the candidate predicts 0.35. In Experiment 9, observed is 0.04 while the candidate predicts -0.56. The rank-based exponential weighting forces a single compromise strategy that fails to capture the empirical patterns where behavior seems to reflect a blend of distinct strategies rather than a single 'soft' weighting scheme.
Rationale: While you successfully implemented one of the arbiter's options (Soft TTB), it is failing to capture the intermediate means across Experiments 6-9. Please switch to the arbiter's other suggested option within this mechanism family: the 'Heuristic Strategy Mixture'. Implement a model that probabilistically samples between a Pure Tallying strategy (unweighted sum of strict wins) and a strict Take-The-Best strategy (relying solely on the highest-validity cue that differentiates the options). Use a mixing parameter 'w_tally' to blend the choice probabilities from the two distinct strategies. This mixture approach will likely capture the heterogeneity and intermediate choice probabilities much better than rank-based exponential weights.

**Outcome of this advice:** iter 1 candidate loss=0.3570 -> iter 2 candidate loss=0.2140 -> the gate marked it ACCEPTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The 'Heuristic Strategy Mixture' model was highly successful, dropping the aggregate loss from 0.3570 to 0.2140 and being accepted by the gate. By probabilistically blending Pure Tallying and strict Take-The-Best, the candidate closely matches empirical data on Experiments 1, 3, 4, 5, and 10. However, there are still notable discrepancies in Experiments 6, 7, 8, and 9. One likely reason is that the model currently applies the exact same inverse-temperature parameter (`beta`) to both the Tallying scores and the TTB scores. Because Tallying scores are counts of wins (differences can be up to the number of features, e.g., 3 or 4) while TTB scores are binary flags (difference is always exactly 1), a single `beta` forces Tallying to be systematically more deterministic than TTB. This artificially constrains the mixture.
Rationale: The mixture approach is definitively the right mechanism family and yielded a large improvement. To fix the remaining misfit on Experiments 6-9, keep the exact same mixture logic but decouple the temperatures. Introduce two separate parameters: `beta_tally` (applied to the tallying scores) and `beta_ttb` (applied to the TTB scores), both with ranges like [0.1, 20.0]. This will allow the model to independently calibrate the determinism of each heuristic before mixing their probabilities.

**Outcome of this advice:** iter 2 candidate loss=0.2140 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Heuristic Strategy Mixture with Independent Noise: Decision-makers probabilistically sample between a Pure Tallying strategy and a strict Take-The-Best strategy. Because these heuristics operate on fundamentally different scales (Tallying produces counts of wins, while TTB produces binary indicators), the determinism (inverse temperature) of each strategy is calibrated independently before their choice probabilities are mixed.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    a, b = stim[0], stim[1]\n    v = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # 1. Pure Tallying Strategy: unweighted count of strict wins\n    a_wins = np.sum(a > b)\n    b_wins = np.sum(b > a)\n    scores_tally = np.array([a_wins, b_wins], dtype=float)\n    \n    # 2. Strict Take-The-Best Strategy: lexicographic choice based on highest-validity differentiating cue\n    sorted_idx = np.argsort(-v)\n    a_ttb = 0.0\n    b_ttb = 0.0\n    for idx in sorted_idx:\n        if a[idx] > b[idx]:\n            a_ttb = 1.0\n            break\n        elif b[idx] > a[idx]:\n            b_ttb = 1.0\n            break\n    scores_ttb = np.array([a_ttb, b_ttb], dtype=float)\n    \n    beta_tally = float(parameters[\"beta_tally\"])\n    beta_ttb = float(parameters[\"beta_ttb\"])\n    \n    # Tallying probabilities\n    z_tally = beta_tally * (scores_tally - np.max(scores_tally))\n    e_tally = np.exp(z_tally)\n    p_tally = e_tally / np.sum(e_tally)\n    \n    # TTB probabilities\n    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))\n    e_ttb = np.exp(z_ttb)\n    p_ttb = e_ttb / np.sum(e_ttb)\n    \n    # Mixture of the two strategies\n    w_tally = float(parameters[\"w_tally\"])\n    p_mixed = w_tally * p_tally + (1.0 - w_tally) * p_ttb\n    \n    # Blend with uniform lapse rate\n    epsilon = float(parameters[\"epsilon\"])\n    n_opts = len(p_mixed)\n    p_final = (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)\n    \n    return p_final",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "w_tally": "[0.0, 1.0]",
    "beta_tally": "[0.1, 20.0]",
    "beta_ttb": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's feedback, this minimal edit introduces independent inverse temperatures (`beta_tally` and `beta_ttb`) for the Pure Tallying and strict Take-The-Best components of the mixture model. Because Tallying scores are counts (e.g., differences of 2 or 3) while TTB scores are binary (differences of exactly 1), a shared `beta` forces Tallying to be systematically more deterministic. By decoupling them, the model can independently calibrate the noise level of each heuristic before blending their predictions via `w_tally`."
}
```
```

## Usage

```json
{
  "prompt_token_count": 15183,
  "candidates_token_count": 1003,
  "total_token_count": 16663
}
```
