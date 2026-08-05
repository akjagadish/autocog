# generation_iter_03_attempt_00

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

Propose a 'Heuristic Toolbox' or 'Take-The-Best (TTB) Mixture' theory. The current theories fail to account for trials where subjects might rely on a single discriminating cue (lexicographic strategy) rather than compensatory integration or simple counting. The new theory should posit that individuals probability-mix between Take-The-Best (which sequentially searches validities and stops at the first discriminating cue), Tallying, and WADD. This 3-way mixture or a TTB+Tallying mixture can better capture the empirical data where choices sometimes strongly deviate from compensatory WADD predictions even when Tallying is tied.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.874)

**Description**
Strategy Mixture Theory: Decision-makers are heterogeneous in their strategies. A proportion of choices are made using a compensatory Weighted Additive (WADD) strategy that integrates cue validities and subjective weights, while the remainder rely on a simpler Tallying (Equal Weight) heuristic that merely counts the number of positive features. Behavior on any given trial is a probabilistic mixture of these two strategies, with the mixture proportion varying across individuals. This naturally explains why aggregate behavior falls between the pure predictions of WADD and Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_mix * p_wadd + (1.0 - w_mix) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3511 (var=0.0061) vs this=0.2527 (var=0.0083)
- Experiment 2: real=0.6604 (var=0.0042) vs this=0.7085 (var=0.0149)
- Experiment 3: real=0.7067 (var=0.0162) vs this=0.6433 (var=0.0614)
- Experiment 4: real=0.3050 (var=0.0157) vs this=0.3400 (var=0.0725)
- Experiment 5: real=0.4267 (var=0.0718) vs this=0.4117 (var=0.1343)
- Experiment 6: real=0.5993 (var=0.0045) vs this=0.5469 (var=0.0305)
- Experiment 7: real=0.1467 (var=0.0502) vs this=0.2033 (var=0.0412)
- Experiment 8: real=0.5383 (var=0.0087) vs this=0.3942 (var=0.0220)


---

### `pi_3` (overall score: 0.686)

**Description**
Tallying (Equal Weight) Theory: People make choices by simply counting the number of positive features for each option, completely ignoring cue validities. The option with the highest total count of positive features is chosen. This compensatory heuristic trades off the cognitive effort of weighting cues against accuracy, assuming all cues are equally important. Noise is introduced via a softmax over the feature counts and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: Simply sum the features for each option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3511 (var=0.0061) vs this=0.1813 (var=0.0063)
- Experiment 2: real=0.6604 (var=0.0042) vs this=0.8125 (var=0.0084)
- Experiment 3: real=0.7067 (var=0.0162) vs this=0.8667 (var=0.0111)
- Experiment 4: real=0.3050 (var=0.0157) vs this=0.1338 (var=0.0122)
- Experiment 5: real=0.4267 (var=0.0718) vs this=0.6850 (var=0.0654)
- Experiment 6: real=0.5993 (var=0.0045) vs this=0.7993 (var=0.0143)
- Experiment 7: real=0.1467 (var=0.0502) vs this=0.3383 (var=0.0490)
- Experiment 8: real=0.5383 (var=0.0087) vs this=0.4917 (var=0.0090)


---

### `pi_2` (overall score: 0.666)

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
- Experiment 1: real=0.3511 (var=0.0061) vs this=0.3147 (var=0.0155)
- Experiment 2: real=0.6604 (var=0.0042) vs this=0.6256 (var=0.0201)
- Experiment 3: real=0.7067 (var=0.0162) vs this=0.4200 (var=0.0778)
- Experiment 4: real=0.3050 (var=0.0157) vs this=0.5613 (var=0.1096)
- Experiment 5: real=0.4267 (var=0.0718) vs this=0.0633 (var=0.2332)
- Experiment 6: real=0.5993 (var=0.0045) vs this=0.3771 (var=0.0658)
- Experiment 7: real=0.1467 (var=0.0502) vs this=0.1233 (var=0.0306)
- Experiment 8: real=0.5383 (var=0.0087) vs this=0.3425 (var=0.0638)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3643 -> ACCEPTED
- iter 2: loss=0.4272 -> REJECTED
- iter 3: loss=0.4257 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.3643 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] != b[j]:
                ttb_pred = 0 if a[j] > b[j] else 1
                if row['response'] == ttb_pred:
                    ttb_matches += 1
                total += 1
                break
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3511 (var=0.0061)
**Previous candidate values (this loop):**
  - iter 1: 0.5411 (var=0.0383) (Δ vs real +0.1900)
  - iter 2: 0.5389 (var=0.0303) (Δ vs real +0.1878)
  - iter 3 (most recent): 0.5489 (var=0.0393) (Δ vs real +0.1978)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8409 (var=0.0110)
- pi_2: 0.3147 (var=0.0155)
- pi_3: 0.1813 (var=0.0063)
- pi_4: 0.2527 (var=0.0083)
- pi_5: 0.1776 (var=0.0075)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.6604 (var=0.0042)
**Previous candidate values (this loop):**
  - iter 1: 0.3937 (var=0.0376) (Δ vs real -0.2667)
  - iter 2: 0.4396 (var=0.0376) (Δ vs real -0.2208)
  - iter 3 (most recent): 0.4098 (var=0.0388) (Δ vs real -0.2506)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6256 (var=0.0201)
- pi_1: 0.1429 (var=0.0087)
- pi_3: 0.8125 (var=0.0084)
- pi_4: 0.7085 (var=0.0149)
- pi_5: 0.7583 (var=0.0100)

### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify the critical dissociation trial (Trial 1 in the design)
    # Option A has more positive features (3 vs 2), favoring Tallying.
    # Option B has the two most valid features, favoring WADD.
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                 
    if is_trial_1.sum() == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A (response == 0)
    return float(np.mean(data.loc[is_trial_1, 'response'] == 0))
```

**Observed (real) value:** 0.7067 (var=0.0162)
**Previous candidate values (this loop):**
  - iter 1: 0.4450 (var=0.0527) (Δ vs real -0.2617)
  - iter 2: 0.3067 (var=0.0326) (Δ vs real -0.4000)
  - iter 3 (most recent): 0.3433 (var=0.0524) (Δ vs real -0.3633)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8667 (var=0.0111)
- pi_2: 0.4200 (var=0.0778)
- pi_1: 0.1383 (var=0.0182)
- pi_4: 0.6433 (var=0.0614)
- pi_5: 0.6400 (var=0.0632)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_wadd_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        # Trial 1: WADD favors the option with fewer, but higher-validity features.
        # Tallying strictly favors the option with more features.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            return 0 == row['response']
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            return 1 == row['response']
        return np.nan

    choices = data.apply(get_wadd_choice, axis=1).dropna()
    
    if len(choices) == 0:
        return 0.5
        
    return float(choices.mean())
```

**Observed (real) value:** 0.3050 (var=0.0157)
**Previous candidate values (this loop):**
  - iter 1: 0.5787 (var=0.0626) (Δ vs real +0.2737)
  - iter 2: 0.6500 (var=0.0414) (Δ vs real +0.3450)
  - iter 3 (most recent): 0.6400 (var=0.0509) (Δ vs real +0.3350)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5613 (var=0.1096)
- pi_3: 0.1338 (var=0.0122)
- pi_1: 0.8788 (var=0.0135)
- pi_4: 0.3400 (var=0.0725)
- pi_5: 0.3162 (var=0.0451)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1: WADD prefers A (1.8 vs 1.7), but Tallying prefers B (3 vs 2).
    t1_mask = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    
    # Trial 7: WADD prefers A (2.05 vs 1.45), and Tallying prefers A (3 vs 2).
    t7_mask = (data['option_a_ratings'].apply(tuple) == (0, 1, 1, 1, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (1, 0, 0, 0, 1))
    
    if t1_mask.sum() == 0 or t7_mask.sum() == 0:
        return 0.0
        
    # Difference in choice rate for B between the conflict trial and the agreement trial.
    # Subtracting the baseline noise/lapse rate controls for subject-specific epsilon variance.
    return float(data.loc[t1_mask, 'response'].mean() - data.loc[t7_mask, 'response'].mean())
```

**Observed (real) value:** 0.4267 (var=0.0718)
**Previous candidate values (this loop):**
  - iter 1: -0.2067 (var=0.1517) (Δ vs real -0.6333)
  - iter 2: -0.1683 (var=0.1812) (Δ vs real -0.5950)
  - iter 3 (most recent): -0.2300 (var=0.2110) (Δ vs real -0.6567)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4117 (var=0.1343)
- pi_2: 0.0633 (var=0.2332)
- pi_1: -0.7133 (var=0.0442)
- pi_3: 0.6850 (var=0.0654)
- pi_5: 0.5533 (var=0.0816)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a unique string identifier for each trial type based on the feature vectors
    t_a = data['option_a_ratings'].apply(tuple).astype(str)
    t_b = data['option_b_ratings'].apply(tuple).astype(str)
    df = data.assign(trial_id=t_a + "_" + t_b)
    
    def subj_metric(subj_df):
        # Calculate the mean response (proportion of B choices) for each of the 8 trial types
        t_means = subj_df.groupby('trial_id')['response'].mean()
        overall_mean = t_means.mean()
        # Calculate Mean Absolute Deviation (MAD) across the trial types
        mad = (t_means - overall_mean).abs().mean()
        # The metric combines the overall bias towards B and the consistency across trial types
        return float(overall_mean - mad)
        
    if df['subject_id'].nunique() > 1:
        return float(df.groupby('subject_id').apply(subj_metric).mean())
    else:
        return float(subj_metric(df))
```

**Observed (real) value:** 0.5993 (var=0.0045)
**Previous candidate values (this loop):**
  - iter 1: 0.3496 (var=0.0293) (Δ vs real -0.2496)
  - iter 2: 0.2089 (var=0.0254) (Δ vs real -0.3904)
  - iter 3 (most recent): 0.2983 (var=0.0351) (Δ vs real -0.3010)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3771 (var=0.0658)
- pi_4: 0.5469 (var=0.0305)
- pi_1: 0.0654 (var=0.0044)
- pi_3: 0.7993 (var=0.0143)
- pi_5: 0.6774 (var=0.0299)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1, 0))
    t3_mask = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0, 0))
    p_a_t2 = 1.0 - data[t2_mask]['response'].mean()
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    return float(p_a_t3 - p_a_t2)
```

**Observed (real) value:** 0.1467 (var=0.0502)
**Previous candidate values (this loop):**
  - iter 1: 0.1417 (var=0.0328) (Δ vs real -0.0050)
  - iter 2: 0.1250 (var=0.0434) (Δ vs real -0.0217)
  - iter 3 (most recent): 0.0650 (var=0.0367) (Δ vs real -0.0817)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2033 (var=0.0412)
- pi_5: 0.4783 (var=0.0877)
- pi_1: 0.0150 (var=0.0163)
- pi_2: 0.1233 (var=0.0306)
- pi_3: 0.3383 (var=0.0490)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Trial 7: A=[0, 1, 1, 0, 0], B=[0, 0, 0, 1, 1]
    # Trial 8: A=[1, 1, 0, 1, 0], B=[0, 0, 1, 1, 1]
    # In these trials, Tally is tied (2v2 and 3v3). 
    # Non-linear scaling strictly preserves the ordinality of validities: A's transformed validities 
    # always sum to a higher value than B's, regardless of gamma (except at exactly gamma=0 where they tie).
    # Thus, Non-linear scaling almost invariably chooses A (response=0).
    # Strategy Mixture uses WADD with uniformly random subjective weights per feature. Because the weights 
    # are random, WADD will sometimes incorrectly prefer B, inflating the choice proportion of B.
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return (a == (0, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1)) or \
               (a == (1, 1, 0, 1, 0) and b == (0, 0, 1, 1, 1))
               
    mask = data.apply(is_target_trial, axis=1)
    if not mask.any():
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())

```

**Observed (real) value:** 0.5383 (var=0.0087)
**Previous candidate values (this loop):**
  - iter 1: 0.2658 (var=0.0187) (Δ vs real -0.2725)
  - iter 2: 0.2308 (var=0.0170) (Δ vs real -0.3075)
  - iter 3 (most recent): 0.2542 (var=0.0197) (Δ vs real -0.2842)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2383 (var=0.0202)
- pi_4: 0.3942 (var=0.0220)
- pi_1: 0.1483 (var=0.0112)
- pi_2: 0.3425 (var=0.0638)
- pi_3: 0.4917 (var=0.0090)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Heuristic Toolbox (3-Way Mixture): Decision-makers are equipped with a repertoire of strategies and probabilistically switch between them. Specifically, individuals mix between a non-compensatory Take-The-Best (TTB) heuristic, an equal-weight Tallying heuristic, and a compensatory Weighted Additive (WADD) strategy. TTB sequentially searches cues in order of validity and decides based on the first discriminating cue. Tallying counts positive features, and WADD integrates features with subjective validity weights. This mixture captures instances where subjects rely on a single dominant cue, as well as instances where they integrate multiple cues equally or proportionally to their importance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    w_tally_rel = float(parameters["w_tally_rel"])
    
    # Calculate absolute mixture weights
    w_tally = (1.0 - w_ttb) * w_tally_rel
    w_wadd = 1.0 - w_ttb - w_tally
    
    # 1. Take-The-Best (TTB) strategy component
    p_ttb = np.array([0.5, 0.5])
    # Sort indices by validity descending
    order = np.argsort(validities)[::-1]
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            if stim[0, idx] > stim[1, idx]:
                p_ttb = np.array([1.0, 0.0])
            else:
                p_ttb = np.array([0.0, 1.0])
            break
            
    # 2. Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # 3. WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of all three strategies
    p_core = w_ttb * p_ttb + w_tally * p_tally + w_wadd * p_wadd
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_ttb: [0.0, 1.0]
- w_tally_rel: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

`rationale`: Following the arbiter's feedback, this model implements a 'Heuristic Toolbox' comprising Take-The-Best (TTB), Tallying, and WADD. By introducing TTB into the mixture, the model accounts for trials where subjects exhibit lexicographic decision-making, stopping their search at the first discriminating cue based on validity. This 3-way mixture resolves failures in purely compensatory or tallying-based models by allowing individuals to probabilistically switch to a fast-and-frugal one-reason heuristic when cues strongly conflict, thereby better capturing empirical deviations from both WADD and Tallying.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory successfully implemented the 3-way mixture of Take-The-Best (TTB), Tallying, and WADD. However, the simulation results show a significant degradation in fit across several experiments (e.g., Experiments 2, 3, 4, 5, and 6) compared to the reference theories. The primary issue lies in the WADD component: by multiplying the objective validities by a vector of uniformly distributed random weights ('w' bounded between 0 and 1), the model effectively scrambles the cue validities for each subject. This subjective weighting acts as unstructured noise rather than a meaningful compensatory integration of validities, severely hampering the model's ability to capture choices where subjects systematically rely on WADD-like compensatory strategies.
Rationale: The WADD component is currently miscalibrated due to the random 'weights' vector multiplying the validities, which introduces excessive noise and destroys the ordinality of the cues. Regenerate the model by removing the random 'weights' array. Instead, WADD should directly use the provided 'validities' (e.g., `stim @ validities`), or optionally apply a single non-linear scaling parameter `gamma` (e.g., `validities ** gamma`) to capture subjective sensitivity. Keep the 3-way mixture (TTB, Tallying, WADD) intact, but fix the WADD score computation so that it faithfully represents compensatory integration.

**Outcome of this advice:** iter 1 candidate loss=0.3643 -> iter 2 candidate loss=0.4272 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate from iteration 2 attempted to fix the WADD component by replacing random subjective weights with a non-linear scaling parameter (gamma). However, this candidate was rejected by the gate, as its loss (0.4272) was worse than the accepted base from iteration 1 (0.3643). The degradation in fit across experiments like 3, 4, and 8 suggests that either the model needs the individual variance provided by the subjective weights in the base model, or there is a deeper miscalibration in how the strategies are mixed. A key issue in both iterations is the use of a single, shared inverse temperature parameter (`beta`) for both the Tallying and WADD softmax functions. Tallying scores are counts (integers), whereas WADD scores are sums of fractional validities, meaning they operate on fundamentally different scales. A shared beta forces one of the components to be either excessively deterministic or overly uniform.
Rationale: The previous suggestion to replace subjective weights with a gamma scaling parameter was rejected by the accept gate, so we will build upon the iter 1 base. The primary issue is the scale mismatch between Tallying and WADD scores. Regenerate the model by introducing separate inverse temperature parameters for the two compensatory strategies (e.g., `beta_tally` and `beta_wadd`). This will allow the softmax functions to independently calibrate the choice determinism for each strategy's natural scale, while preserving the 3-way mixture structure prescribed by the arbiter.

**Outcome of this advice:** iter 2 candidate loss=0.4272 -> iter 3 candidate loss=0.4257 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The recent attempt to introduce separate inverse temperatures for Tallying and WADD was rejected by the accept gate, as was the earlier attempt to replace random subjective weights with a non-linear scaling parameter. The persistent failures across these iterations suggest that the WADD component, particularly its interaction with the other heuristics, is introducing excessive noise or complexity that harms the model's fit across experiments like 3, 4, and 8.
Rationale: Since tweaking the WADD component's parameters and temperatures has repeatedly failed to beat the baseline, we should pivot to the alternative structure explicitly permitted by the arbiter: a 2-way TTB + Tallying mixture. Regenerate the model by removing the WADD strategy and its associated 'weights' array entirely. Implement a simple probabilistic mixture between Take-The-Best and Tallying (using a single mixing weight `w_ttb`), applying a standard softmax with temperature for the Tallying component and a lapse rate. This will drastically simplify the parameter space and eliminate the miscalibrated WADD predictions.

**Outcome of this advice:** iter 3 candidate loss=0.4257 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Heuristic Toolbox (TTB + Tallying Mixture): Decision-makers are equipped with a repertoire of fast-and-frugal heuristics rather than complex compensatory algorithms. Specifically, individuals probabilistically switch between a non-compensatory Take-The-Best (TTB) heuristic and an equal-weight Tallying heuristic. TTB sequentially searches cues in order of validity and decides based on the first discriminating cue, capturing lexicographic decision-making. Tallying simply counts the number of positive features, capturing instances where cues are integrated equally. This 2-way mixture accounts for choices where subjects either rely on a single dominant cue or fall back to simple feature counting, without the excessive noise introduced by subjective compensatory weighting.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(f\"Stimulus must be (2, n_features); got {stim.shape}\")\n        \n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    w_ttb = float(parameters[\"w_ttb\"])\n    \n    # 1. Take-The-Best (TTB) strategy component\n    p_ttb = np.array([0.5, 0.5])\n    # Sort indices by validity descending\n    order = np.argsort(validities)[::-1]\n    for idx in order:\n        if stim[0, idx] != stim[1, idx]:\n            if stim[0, idx] > stim[1, idx]:\n                p_ttb = np.array([1.0, 0.0])\n            else:\n                p_ttb = np.array([0.0, 1.0])\n            break\n            \n    # 2. Tallying strategy component\n    tally_scores = stim.sum(axis=1)\n    z_tally = beta * (tally_scores - tally_scores.max())\n    e_tally = np.exp(z_tally)\n    p_tally = e_tally / e_tally.sum()\n    \n    # Mixture of TTB and Tallying\n    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally\n    \n    # Apply lapse rate\n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()  # Ensure valid probabilities\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "w_ttb": "[0.0, 1.0]",
    "validities": "validities"
  },
  "rationale": "Following the critic's recommendation, we simplify the Heuristic Toolbox model to a 2-way probabilistic mixture between Take-The-Best (TTB) and Tallying. Previous attempts to fit a 3-way mixture (including WADD) suffered from miscalibration and excessive noise due to subjective weighting and scale mismatches. By restricting the mixture to TTB and Tallying, we drastically simplify the parameter space and eliminate the problematic WADD component. This aligns with the arbiter's original suggestion that a TTB+Tally mixture can better capture empirical deviations from compensatory predictions, specifically modeling instances where subjects either rely on a single dominant cue or fall back to simple feature counting."
}
```
```

## Usage

```json
{
  "prompt_token_count": 14852,
  "candidates_token_count": 947,
  "total_token_count": 16875
}
```
