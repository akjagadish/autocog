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
- THEORY 1 = `pi_3_1`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

The Mixture of Heuristics model is fundamentally flawed because the data strongly refutes any use of validity-based heuristics like Take-The-Best; subjects consistently guess 50/50 when feature counts are tied. Replace Theory 2 with a brand-new theory that also ignores cue validities but posits a different underlying cognitive mechanism than simple Tallying. For example, propose a 'Cancellation and Focus' or 'Similarity Heuristic' where subjects first eliminate features shared by both options and only tally the remaining unique features. This would provide a psychologically distinct alternative to Theory 1 while maintaining the necessary property of validity-independence.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3_1` (overall score: 0.989)

**Description**
Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3458 (var=0.0045) vs this=0.3287 (var=0.0195)
- Experiment 2: real=0.3217 (var=0.0048) vs this=0.3088 (var=0.0206)
- Experiment 3: real=0.6333 (var=0.0139) vs this=0.6467 (var=0.0382)
- Experiment 4: real=0.3550 (var=0.0190) vs this=0.3425 (var=0.0269)
- Experiment 5: real=0.5100 (var=0.0037) vs this=0.5012 (var=0.0062)
- Experiment 6: real=0.5000 (var=0.0051) vs this=0.4929 (var=0.0044)
- Experiment 7: real=0.5167 (var=0.0043) vs this=0.5083 (var=0.0088)
- Experiment 8: real=0.5044 (var=0.0081) vs this=0.4950 (var=0.0090)


---

### `pi_3` (overall score: 0.870)

**Description**
Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.01, 20.0]
- epsilon: [0.0, 1.0]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.3458 (var=0.0045) vs this=0.2477 (var=0.0261)
- Experiment 2: real=0.3217 (var=0.0048) vs this=0.2842 (var=0.0239)
- Experiment 3: real=0.6333 (var=0.0139) vs this=0.7700 (var=0.0377)
- Experiment 4: real=0.3550 (var=0.0190) vs this=0.2975 (var=0.0360)
- Experiment 5: real=0.5100 (var=0.0037) vs this=0.4988 (var=0.0071)
- Experiment 6: real=0.5000 (var=0.0051) vs this=0.5058 (var=0.0050)
- Experiment 7: real=0.5167 (var=0.0043) vs this=0.4800 (var=0.0081)
- Experiment 8: real=0.5044 (var=0.0081) vs this=0.5033 (var=0.0054)


---

### `pi_2` (overall score: 0.741)

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
- Experiment 1: real=0.3458 (var=0.0045) vs this=0.2794 (var=0.0178)
- Experiment 2: real=0.3217 (var=0.0048) vs this=0.3296 (var=0.0486)
- Experiment 3: real=0.6333 (var=0.0139) vs this=0.4617 (var=0.0964)
- Experiment 4: real=0.3550 (var=0.0190) vs this=0.6162 (var=0.0858)
- Experiment 5: real=0.5100 (var=0.0037) vs this=0.6275 (var=0.0558)
- Experiment 6: real=0.5000 (var=0.0051) vs this=0.5062 (var=0.0188)
- Experiment 7: real=0.5167 (var=0.0043) vs this=0.5439 (var=0.0532)
- Experiment 8: real=0.5044 (var=0.0081) vs this=0.5289 (var=0.0421)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    diff = a_ratings - b_ratings
    # By using exponentially decreasing weights, the sign of the dot product 
    # is completely determined by the first non-zero difference, 
    # exactly mimicking Take The Best's lexicographic choice rule.
    weights = np.array([10000, 1000, 100, 10, 1])
    ttb_scores = diff @ weights
    ttb_choices = np.where(ttb_scores > 0, 0, 1)
    valid_trials = ttb_scores != 0
    if not np.any(valid_trials):
        return 0.5
    matches = (data['response'].values[valid_trials] == ttb_choices[valid_trials])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3458 (var=0.0045)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8390 (var=0.0099)
- pi_2: 0.2794 (var=0.0178)
- pi_3: 0.2477 (var=0.0261)
- pi_4: 0.3965 (var=0.0186)
- pi_3_1: 0.3287 (var=0.0195)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    count = 0
    
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
            sum_a = sum(a)
            sum_b = sum(b)
            # Isolate compensatory trials: the TTB winner has a strictly smaller sum of features
            if (ttb_winner == 0 and sum_a < sum_b) or (ttb_winner == 1 and sum_b < sum_a):
                if resp == ttb_winner:
                    matches += 1
                count += 1
                
    return matches / count if count > 0 else 0.5
```

**Observed (real) value:** 0.3217 (var=0.0048)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3296 (var=0.0486)
- pi_1: 0.8675 (var=0.0103)
- pi_3: 0.2842 (var=0.0239)
- pi_4: 0.4108 (var=0.0268)
- pi_3_1: 0.3088 (var=0.0206)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 1, 0, 0, 0)
    t1_b = (0, 0, 1, 1, 1)
    
    mask1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    mask2 = (a_tuples == t1_b) & (b_tuples == t1_a)
    
    chosen_t1_b = (mask1 & (data['response'] == 1)).sum() + (mask2 & (data['response'] == 0)).sum()
    total = mask1.sum() + mask2.sum()
    
    if total == 0:
        return 0.5
    return float(chosen_t1_b / total)
```

**Observed (real) value:** 0.6333 (var=0.0139)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7700 (var=0.0377)
- pi_2: 0.4617 (var=0.0964)
- pi_1: 0.1967 (var=0.0219)
- pi_4: 0.6000 (var=0.0361)
- pi_3_1: 0.6467 (var=0.0382)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # We focus on the 'opposition' trial where the two models make opposite predictions.
    # The target option is (1, 1, 0, 0, 0), which has fewer positive features (2 vs 3) 
    # but higher total validity (0.95+0.85=1.80 vs 0.60+0.55+0.50=1.65).
    # WADD prefers the target option; Tallying prefers the alternative (0, 0, 1, 1, 1).
    
    a_is_target = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    b_is_target = data['option_b_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 1, 0, 0, 0))
    
    # Response 0 means A was chosen, 1 means B was chosen.
    chose_target = (a_is_target & (data['response'] == 0)) | (b_is_target & (data['response'] == 1))
    relevant = a_is_target | b_is_target
    
    if relevant.sum() == 0:
        return 0.5
        
    return float(chose_target.sum() / relevant.sum())
```

**Observed (real) value:** 0.3550 (var=0.0190)
**Other theories' values on this metric (for reference):**
- pi_2: 0.6162 (var=0.0858)
- pi_3: 0.2975 (var=0.0360)
- pi_1: 0.8638 (var=0.0146)
- pi_4: 0.3588 (var=0.0270)
- pi_3_1: 0.3425 (var=0.0269)

### Experiment 5
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Focus on 'indifference' trials for Tallying
        if sum(a) == sum(b):
            # Cue 0 is the most valid cue in this design
            if a[0] > b[0]:
                ttb_choice = 0
            elif b[0] > a[0]:
                ttb_choice = 1
            else:
                continue
                
            if row['response'] == ttb_choice:
                ttb_aligned.append(1)
            else:
                ttb_aligned.append(0)
                
    if len(ttb_aligned) == 0:
        return 0.5
        
    return float(np.mean(ttb_aligned))
```

**Observed (real) value:** 0.5100 (var=0.0037)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4988 (var=0.0071)
- pi_4: 0.6475 (var=0.0132)
- pi_1: 0.8344 (var=0.0134)
- pi_2: 0.6275 (var=0.0558)
- pi_3_1: 0.5012 (var=0.0062)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying predicts indifference (equal sums)
    indiff_mask = sum_a == sum_b
    if not indiff_mask.any():
        return 0.5
        
    indiff_data = data[indiff_mask]
    
    # Compute the proportion of times Option A was chosen on these trials
    # Response == 0 means Option A was chosen
    p_a = (indiff_data['response'] == 0).mean()
    
    return float(p_a)
```

**Observed (real) value:** 0.5000 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6758 (var=0.0087)
- pi_3: 0.5058 (var=0.0050)
- pi_1: 0.8550 (var=0.0112)
- pi_2: 0.5062 (var=0.0188)
- pi_3_1: 0.4929 (var=0.0044)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for idx, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        if sum(a) == sum(b):
            ttb_winner = None
            for j in range(len(a)):
                if a[j] > b[j]:
                    ttb_winner = 0
                    break
                elif b[j] > a[j]:
                    ttb_winner = 1
                    break
                    
            if ttb_winner is not None:
                ttb_match.append(1 if row['response'] == ttb_winner else 0)
                
    if len(ttb_match) == 0:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.5167 (var=0.0043)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.5083 (var=0.0088)
- pi_4: 0.6672 (var=0.0126)
- pi_1: 0.8456 (var=0.0136)
- pi_2: 0.5439 (var=0.0532)
- pi_3: 0.4800 (var=0.0081)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Isolate 'indifference' trials where both options have the same number of positive features
    indiff_mask = sum_a == sum_b
    if not indiff_mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen on these trials
    # Option A is favored by the Take-The-Best heuristic on all these trials.
    return float((data.loc[indiff_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5044 (var=0.0081)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6633 (var=0.0104)
- pi_3_1: 0.4950 (var=0.0090)
- pi_1: 0.8517 (var=0.0073)
- pi_2: 0.5289 (var=0.0421)
- pi_3: 0.5033 (var=0.0054)


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
  "theory": "Cancellation and Focus (Ratio Rule)",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    \n    stim = np.asarray(stimulus, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be shape (2, n_features).\")\n        \n    # Cancellation: eliminate features shared by both options.\n    # Focus only on unique positive features.\n    unique_a = np.sum((stim[0] > stim[1]).astype(float))\n    unique_b = np.sum((stim[1] > stim[0]).astype(float))\n    \n    scores = np.array([unique_a, unique_b])\n    \n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    if unique_a == 0 and unique_b == 0:\n        p_core = np.array([0.5, 0.5])\n    else:\n        # Ratio rule over unique features to provide a distinct alternative to Tallying's softmax.\n        s_beta = np.zeros_like(scores)\n        for i in range(len(scores)):\n            if scores[i] > 0:\n                s_beta[i] = scores[i] ** beta\n            else:\n                s_beta[i] = 0.0 if beta > 0 else 1.0\n        p_core = s_beta / np.sum(s_beta)\n        \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities /= probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)\n",
  "parameters": {
    "beta": "[0.0, 5.0]",
    "epsilon": "[0.0, 1.0]"
  },
  "rationale": "Following the arbiter's guidance, this model implements the 'Cancellation and Focus' heuristic. Subjects first simplify the decision by eliminating (canceling) any features that are shared between the two options. They then focus only on the remaining unique positive features. For binary features, taking the difference of unique features is mathematically identical to taking the difference of total features (Tallying). To ensure this theory is genuinely psychologically and mathematically distinct from Theory 1, choice probabilities are determined by a ratio rule (Luce's choice) over the unique features, scaled by a sensitivity parameter beta. This completely ignores cue validities and perfectly captures the empirical evidence that subjects guess 50/50 when feature counts are tied (since tied total counts imply tied unique counts)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 10900,
  "candidates_token_count": 696,
  "total_token_count": 15005
}
```
