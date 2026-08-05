# generation_iter_05_attempt_00

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
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Propose a 'Strategy Selection' or 'Mixed Heuristic' theory. This theory should posit that decision makers do not use a single monolithic rule. Instead, a proportion of the population (or a probability on each trial) uses the non-compensatory Take The Best heuristic, while others (or on other trials) use a simple compensatory rule like Tallying (Equal Weight) or WADD. This would naturally explain the intermediate metrics in Experiments 1 and 4, while allowing the TTB component to drive the flat response observed in Experiment 2's specific compensatory setups. Alternatively, propose a threshold-based model where people only integrate lower-validity features if the difference on the primary feature is below a certain subjective threshold.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.802)

**Description**
Take The Best (TTB) is a non-compensatory lexicographic heuristic. Decision makers rank features according to their validity. To choose between two options, they compare them on the most valid feature. If one option has a higher value on this feature, it is chosen immediately, and all remaining features are ignored. If the options are tied on this feature, the decision maker moves to the next most valid feature, and so on. If the options tie on all features, the decision maker guesses randomly. Response noise is modeled via a simple lapse rate (epsilon) where the subject makes a random choice instead of following the TTB rule. The lapse rate can be high, reflecting significant guessing in the empirical data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Order features by validity, descending
    order = np.argsort(validities)[::-1]
    
    # Find the first discriminating feature
    chosen = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # Tie on all features
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- epsilon: [0.0, 1.0]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.3117 (var=0.0269)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.7400 (var=0.0299)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.7942 (var=0.0242)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=-0.0025 (var=0.0123)


---

### `pi_2` (overall score: 0.529)

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

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.3258 (var=0.0439)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.5833 (var=0.0931)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.3625 (var=0.0141)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=0.3296 (var=0.0398)


---

### `pi_1` (overall score: 0.000)

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
- Experiment 1: real=0.3883 (var=0.0126) vs this=0.8325 (var=0.0101)
- Experiment 2: real=0.5633 (var=0.0246) vs this=0.1467 (var=0.0174)
- Experiment 3: real=0.5971 (var=0.0020) vs this=0.2048 (var=0.0064)
- Experiment 4: real=-0.0133 (var=0.0090) vs this=0.3579 (var=0.0194)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1696 -> ACCEPTED
- iter 2: loss=0.2375 -> REJECTED
- iter 3: loss=0.2130 -> REJECTED
- iter 4: loss=0.3776 -> REJECTED
- iter 5: loss=0.1977 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1696 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Calculate the sum of features for options A and B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Isolate trials 1 and 2 where Tallying and WADD make opposite predictions.
    # These are uniquely identified by the total number of features across both options being 5.
    mask = (a_sums + b_sums) == 5
    if not mask.any():
        return 0.5
    
    # Tallying predicts the option with more features (lower sum means response 1, i.e., B)
    tally_predictions = (a_sums < b_sums).astype(int)
    
    # Calculate the proportion of choices on these trials that align with Tallying
    matches = (data.loc[mask, 'response'] == tally_predictions.loc[mask]).mean()
    return float(matches)

```

**Observed (real) value:** 0.3883 (var=0.0126)
**Previous candidate values (this loop):**
  - iter 1: 0.4100 (var=0.0441) (Δ vs real +0.0217)
  - iter 2: 0.2883 (var=0.0169) (Δ vs real -0.1000)
  - iter 3: 0.4325 (var=0.0515) (Δ vs real +0.0442)
  - iter 4: 0.1525 (var=0.0118) (Δ vs real -0.2358)
  - iter 5 (most recent): 0.4933 (var=0.0431) (Δ vs real +0.1050)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8325 (var=0.0101)
- pi_2: 0.3258 (var=0.0439)
- pi_3: 0.3117 (var=0.0269)

### Experiment 2
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    mask = (a_str == '11000') & (b_str == '00111')
    if not mask.any():
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5633 (var=0.0246)
**Previous candidate values (this loop):**
  - iter 1: 0.5650 (var=0.0620) (Δ vs real +0.0017)
  - iter 2: 0.7183 (var=0.0264) (Δ vs real +0.1550)
  - iter 3: 0.4550 (var=0.0706) (Δ vs real -0.1083)
  - iter 4: 0.8050 (var=0.0257) (Δ vs real +0.2417)
  - iter 5 (most recent): 0.5417 (var=0.0728) (Δ vs real -0.0217)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5833 (var=0.0931)
- pi_1: 0.1467 (var=0.0174)
- pi_3: 0.7400 (var=0.0299)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0.5
        
    preds = [ttb_predict(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    resps = data['response'].values
    
    agreements = []
    for p, r in zip(preds, resps):
        if p == 0.5:
            agreements.append(0.5)
        else:
            agreements.append(1.0 if p == r else 0.0)
            
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5971 (var=0.0020)
**Previous candidate values (this loop):**
  - iter 1: 0.5481 (var=0.0291) (Δ vs real -0.0490)
  - iter 2: 0.7200 (var=0.0076) (Δ vs real +0.1229)
  - iter 3: 0.5429 (var=0.0184) (Δ vs real -0.0542)
  - iter 4: 0.5821 (var=0.0460) (Δ vs real -0.0150)
  - iter 5 (most recent): 0.5452 (var=0.0419) (Δ vs real -0.0519)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7942 (var=0.0242)
- pi_2: 0.3625 (var=0.0141)
- pi_1: 0.2048 (var=0.0064)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    a_sum = data['option_a_ratings'].apply(sum)
    diff = b_sum - a_sum
    
    high_diff = data[diff >= 2]['response']
    low_diff = data[diff <= 0]['response']
    
    if len(high_diff) == 0 or len(low_diff) == 0:
        return 0.0
        
    return float(high_diff.mean() - low_diff.mean())
```

**Observed (real) value:** -0.0133 (var=0.0090)
**Previous candidate values (this loop):**
  - iter 1: 0.1912 (var=0.0196) (Δ vs real +0.2046)
  - iter 2: 0.0967 (var=0.0129) (Δ vs real +0.1100)
  - iter 3: 0.1696 (var=0.0191) (Δ vs real +0.1829)
  - iter 4: 0.2729 (var=0.0337) (Δ vs real +0.2862)
  - iter 5 (most recent): 0.1592 (var=0.0373) (Δ vs real +0.1725)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3296 (var=0.0398)
- pi_3: -0.0025 (var=0.0123)
- pi_1: 0.3579 (var=0.0194)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Strategy Selection (Mixed Heuristic): Decision makers do not rely on a single monolithic rule for binary choice. Instead, they use a mixture of strategies, probabilistically selecting between a non-compensatory lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). This allows the model to capture behavior that sometimes relies heavily on the most valid feature and sometimes integrates across multiple features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(validities)[::-1]
    chosen_ttb = -1
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            chosen_ttb = 0
            break
        elif stim[1, idx] > stim[0, idx]:
            chosen_ttb = 1
            break
            
    if chosen_ttb == 0:
        p_ttb_core = np.array([1.0, 0.0])
    elif chosen_ttb == 1:
        p_ttb_core = np.array([0.0, 1.0])
    else:
        p_ttb_core = np.array([0.5, 0.5])
        
    # Tallying prediction
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_tally_core = e / e.sum()
    
    # Strategy mixture
    p_ttb = float(parameters["p_ttb"])
    p_mixed = p_ttb * p_ttb_core + (1.0 - p_ttb) * p_tally_core
    
    # Response noise (lapse rate)
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- p_ttb: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's suggestion, this theory posits that individuals use a Strategy Selection or Mixed Heuristic approach. Rather than exclusively using TTB or Tallying, decision makers mix between a non-compensatory rule (TTB) and a simple compensatory rule (Tallying). This probabilistic mixture naturally explains the intermediate metrics in Experiments 1 and 3, while preserving the flat response in Experiment 4 and capturing the specific compensatory setups in Experiment 2. It unifies the strengths of both heuristics.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The mixed heuristic (Strategy Selection) model provides a very strong fit for Experiments 1 and 2, accurately capturing the intermediate reliance on Tallying and TTB. However, in Experiment 4, the observed metric is -0.0133 (indicating flat sensitivity to the sum difference), while the candidate predicts 0.1912. This suggests that the compensatory (Tallying) component is still exerting too much influence when the difference in total features is large, pulling the model away from the observed flat response.
Rationale: The candidate was accepted and is our best base yet, but the fit on Experiment 4 can be improved. To reduce the over-sensitivity to compensatory differences in Experiment 4 without breaking Experiments 1 and 2, consider adjusting the mixture mechanism. Instead of a fixed probabilistic coin-flip on every trial, you might implement the arbiter's alternative suggestion: a threshold-based or contingent mixture. For example, decision makers might only use Tallying if the most valid feature is tied (or if the difference in validities is small). Alternatively, simply shift the prior range of `p_ttb` higher (e.g., [0.5, 1.0]) to suppress the Tallying influence, or make `p_ttb` an inverse function of the difference in the primary feature.

**Outcome of this advice:** iter 1 candidate loss=0.1696 -> iter 2 candidate loss=0.2375 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The attempt to fix Experiment 4 by forcing a high prior on `p_ttb` (range [0.6, 1.0]) was rejected by the accept gate because it severely degraded the fit on Experiments 1, 2, and 3. In Experiment 1, the match dropped to 0.288 (real = 0.388); in Experiment 2, it overshot to 0.718 (real = 0.563). This indicates that a simple fixed probabilistic mixture heavily skewed towards TTB is too rigid to capture the nuanced behavior across these experiments.
Rationale: Since shifting the uniform `p_ttb` range higher was rejected, we must abandon the fixed high-TTB prior approach. Instead, implement the arbiter's alternative suggestion: a contingent or threshold-based strategy selection. Specifically, build a model where the strategy depends on the stimulus state. For example, decision makers might use TTB when the most valid feature discriminates between the options (difference != 0), but switch to Tallying (or a mixture heavily favoring Tallying) when the most valid feature is tied. This allows TTB to dominate when a clear primary cue exists (explaining the flat response in Exp 4) while allowing compensatory integration to emerge when the primary cue is uninformative.

**Outcome of this advice:** iter 2 candidate loss=0.2375 -> iter 3 candidate loss=0.2130 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The contingent strategy selection model was rejected by the gate. While it attempted to isolate Experiment 4's flat response by conditioning on whether the primary cue discriminates, this added complexity degraded the previously excellent fits on Experiments 1 and 2 (e.g., Exp 2 match dropped from 0.565 to 0.455) without actually solving the over-sensitivity in Experiment 4 (metric still at 0.1696 vs real -0.0133). The simple fixed probabilistic mixture from Iteration 1 remains our best base model.
Rationale: Since the contingent strategy was rejected, we should revert to the simple fixed-probability mixture of TTB and a compensatory rule that gave our best loss of 0.1696 in Iteration 1. However, to address the lingering issue in Experiment 4, change the compensatory component from Tallying (equal weights) to WADD (weighted additive, using the provided validities). Because WADD naturally scales the influence of secondary features by their validities, it may allow the model to fit the varying compensatory sensitivity across experiments better than equal-weight Tallying, even with a single fixed mixture probability. Implement a simple mixture of TTB and WADD, keeping parameter ranges broad (e.g., p_ttb in [0.0, 1.0], beta in [0.1, 20.0], epsilon in [0.0, 0.5]).

**Outcome of this advice:** iter 3 candidate loss=0.2130 -> iter 4 candidate loss=0.3776 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The attempt to replace Tallying with WADD (Iteration 4) was rejected by the accept gate, as it significantly worsened the fit across the board (aggregate loss increased to 0.3776). WADD caused severe over-sensitivity in Experiment 4 (0.2729 vs real -0.0133) and broke the previously good fits on Experiments 1 and 2. The simple fixed-probability mixture of TTB and equal-weight Tallying from Iteration 1 remains our best base model.
Rationale: Since WADD (Iter 4), contingent strategy (Iter 3), and forced high-TTB priors (Iter 2) were all rejected, we must revert to the Iteration 1 base: a simple fixed-probability mixture of TTB and Tallying. To address the over-sensitivity in Experiment 4 without breaking the other experiments, keep the TTB + Tallying mixture but focus on the Tallying component's softmax temperature. Widen the `beta` parameter range for Tallying to include much lower values (e.g., [0.01, 10.0]). A lower beta allows the Tallying component to be less extreme when feature sums are large, which can pull the Experiment 4 metric closer to 0 while maintaining the correct directional influence in Experiments 1 and 2.

**Outcome of this advice:** iter 4 candidate loss=0.3776 -> iter 5 candidate loss=0.1977 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The attempt to fix Experiment 4 by lowering the beta parameter for the Tallying component (Iteration 5) was rejected by the gate. While it slightly reduced the over-sensitivity in Experiment 4 (0.1592 vs real -0.0133), it degraded the fit on Experiment 1 (0.4933 vs real 0.3883). The simple fixed-probability mixture of TTB and Tallying from Iteration 1 remains our best-performing base model, but it still struggles to reconcile the strong compensatory behavior in Experiment 1 with the flat, TTB-dominated response in Experiment 4.
Rationale: Since adjusting the temperature (Iter 5), switching to WADD (Iter 4), and using a strict primary-feature-tie contingent strategy (Iter 3) were all rejected, we need a softer way to modulate the strategy mixture across experiments. Revert to the TTB + Tallying mixture from Iteration 1, but instead of a single fixed `p_ttb` parameter, make the probability of using TTB contingent on the *validity* of the primary discriminating feature. For example, `p_ttb = base_p_ttb * validity_of_first_discriminating_cue`. This allows the model to heavily rely on TTB in experiments where the leading cue is very strong (flattening Exp 4), while naturally falling back toward Tallying in experiments where the primary cues are weaker or less decisive. Keep `beta` in a moderate range (e.g., [0.1, 10.0]) to ensure Tallying remains well-calibrated when used.

**Outcome of this advice:** iter 5 candidate loss=0.1977 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Strategy Selection (Validity-Contingent Mixed Heuristic): Decision makers probabilistically mix between a non-compensatory lexicographic heuristic (Take The Best) and a simple compensatory heuristic (Tallying). However, the probability of using TTB is not fixed; it is contingent on the validity of the primary discriminating feature. When the leading cue is highly valid, decision makers are more likely to rely on TTB (leading to flat sensitivity to secondary features), but when primary cues are weaker, they naturally fall back toward Tallying.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n    \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    # Take The Best (TTB) prediction\n    order = np.argsort(validities)[::-1]\n    chosen_ttb = -1\n    discrim_validity = 0.0\n    for idx in order:\n        if stim[0, idx] > stim[1, idx]:\n            chosen_ttb = 0\n            discrim_validity = validities[idx]\n            break\n        elif stim[1, idx] > stim[0, idx]:\n            chosen_ttb = 1\n            discrim_validity = validities[idx]\n            break\n            \n    if chosen_ttb == 0:\n        p_ttb_core = np.array([1.0, 0.0])\n    elif chosen_ttb == 1:\n        p_ttb_core = np.array([0.0, 1.0])\n    else:\n        p_ttb_core = np.array([0.5, 0.5])\n        \n    # Tallying prediction\n    a_wins = float(np.sum(stim[0] > stim[1]))\n    b_wins = float(np.sum(stim[1] > stim[0]))\n    scores = np.array([a_wins, b_wins])\n    \n    beta = float(parameters[\"beta\"])\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_tally_core = e / e.sum()\n    \n    # Strategy mixture (contingent on validity of the first discriminating cue)\n    base_p_ttb = float(parameters[\"base_p_ttb\"])\n    p_ttb = base_p_ttb * discrim_validity\n    p_mixed = p_ttb * p_ttb_core + (1.0 - p_ttb) * p_tally_core\n    \n    # Response noise (lapse rate)\n    epsilon = float(parameters[\"epsilon\"])\n    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])\n    \n    return p_final\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "base_p_ttb": "[0.0, 1.0]",
    "beta": "[0.1, 10.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's suggestion, we apply a minimal edit to the Iteration 1 base model. We replace the fixed `p_ttb` parameter with `base_p_ttb * discrim_validity`, making the probability of using TTB contingent on how valid the primary discriminating feature actually is. This allows the model to heavily rely on TTB when the leading cue is strong (which helps explain the flat response in Experiment 4), while shifting back toward Tallying when primary cues are weaker or less decisive (maintaining the good fits in Experiments 1 and 2). We also restrict `beta` to a moderate range [0.1, 10.0] to keep the Tallying component stable."
}
```
```

## Usage

```json
{
  "prompt_token_count": 11750,
  "candidates_token_count": 1052,
  "total_token_count": 14511
}
```
