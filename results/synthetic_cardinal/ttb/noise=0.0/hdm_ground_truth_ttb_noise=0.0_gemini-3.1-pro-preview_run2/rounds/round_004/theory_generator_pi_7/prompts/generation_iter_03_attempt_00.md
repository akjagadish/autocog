# generation_iter_03_attempt_00

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Propose a 'Take-The-Best' (TTB) or Lexicographic heuristic. Unlike the compensatory WADD model or the tallying models that count all wins, TTB is strictly non-compensatory. It searches through features in descending order of their explicit validities and stops at the first feature that discriminates between the two options (where one has a strictly higher rating than the other), choosing the option that wins on that highest-validity discriminating feature. This provides a distinct, ordinal, non-compensatory mechanism to contrast with Theory 1.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.929)

**Description**
Mixture of Tallying and Weighted Tallying: Decision makers do not universally ignore explicit validities (as in pure Tallying) nor do they perfectly weight every feature by its numerical validity (as in pure Weighted Tallying). Instead, the population consists of a mixture of strategies, or individuals use a blended strategy. Some individuals rely on a simple count of winning features (Tallying), while others incorporate the provided validities to weight those wins. By modeling choice as a convex combination of the Tallying score and the Weighted Tallying score, the model can capture intermediate levels of consensus and account for why human behavior often falls between the predictions of these two pure heuristics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying scores: simple count of strictly winning features
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    
    # Weighted Tallying scores: sum of validities for winning features
    wt_a = np.sum(validities[a > b])
    wt_b = np.sum(validities[b > a])
    
    # Blend the two strategies
    w = float(parameters["w_tally"])
    score_a = w * tally_a + (1.0 - w) * wt_a
    score_b = w * tally_b + (1.0 - w) * wt_b
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the blended scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8000 (var=0.0667) vs this=0.8514 (var=0.0108)
- Experiment 2: real=-0.4000 (var=0.8400) vs this=-0.6950 (var=0.0527)
- Experiment 3: real=0.7467 (var=0.0780) vs this=0.8372 (var=0.0082)
- Experiment 4: real=0.3500 (var=0.0469) vs this=0.2575 (var=0.0042)
- Experiment 5: real=0.7600 (var=0.1824) vs this=0.6783 (var=0.0731)
- Experiment 6: real=0.5000 (var=0.2500) vs this=0.2658 (var=0.0478)
- Experiment 7: real=0.6867 (var=0.1529) vs this=0.7256 (var=0.0291)
- Experiment 8: real=0.6200 (var=0.2356) vs this=0.7800 (var=0.0209)
- Experiment 9: real=0.6000 (var=0.6400) vs this=0.6692 (var=0.0611)
- Experiment 10: real=0.4000 (var=0.6800) vs this=0.6053 (var=0.0811)


---

### `pi_4` (overall score: 0.475)

**Description**
Weighted Tallying Heuristic: People make decisions by integrating across multiple features, but rather than treating all cues equally (as in simple Tallying), they weight each feature-wise win by the explicitly provided expert validity. The score for each option is the sum of the validities for the features where that option has a strictly higher rating than the other. The option with the highest validity-weighted tally is chosen. This captures both the tendency to integrate across features and the sensitivity to explicit validities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Weighted Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sum validities for features where an option strictly wins
    score_a = np.sum(validities[a > b])
    score_b = np.sum(validities[b > a])
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

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
- Experiment 1: real=0.8000 (var=0.0667) vs this=0.8631 (var=0.0088)
- Experiment 2: real=-0.4000 (var=0.8400) vs this=-0.7633 (var=0.0345)
- Experiment 3: real=0.7467 (var=0.0780) vs this=0.8867 (var=0.0066)
- Experiment 4: real=0.3500 (var=0.0469) vs this=0.2539 (var=0.0038)
- Experiment 5: real=0.7600 (var=0.1824) vs this=0.1858 (var=0.0200)
- Experiment 6: real=0.5000 (var=0.2500) vs this=0.7342 (var=0.0269)
- Experiment 7: real=0.6867 (var=0.1529) vs this=0.3144 (var=0.0099)
- Experiment 8: real=0.6200 (var=0.2356) vs this=0.4926 (var=0.0117)
- Experiment 9: real=0.6000 (var=0.6400) vs this=0.6558 (var=0.0445)
- Experiment 10: real=0.4000 (var=0.6800) vs this=0.5947 (var=0.0642)


---

### `pi_1` (overall score: 0.329)

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
    return int(np.argmax(probabilities))


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.8000 (var=0.0667) vs this=1.0000 (var=0.0000)
- Experiment 2: real=-0.4000 (var=0.8400) vs this=-1.0000 (var=0.0000)
- Experiment 3: real=0.7467 (var=0.0780) vs this=1.0000 (var=0.0000)
- Experiment 4: real=0.3500 (var=0.0469) vs this=0.1667 (var=0.0000)
- Experiment 5: real=0.7600 (var=0.1824) vs this=1.0000 (var=0.0000)
- Experiment 6: real=0.5000 (var=0.2500) vs this=0.0000 (var=0.0000)
- Experiment 7: real=0.6867 (var=0.1529) vs this=1.0000 (var=0.0000)
- Experiment 8: real=0.6200 (var=0.2356) vs this=1.0000 (var=0.0000)
- Experiment 9: real=0.6000 (var=0.6400) vs this=1.0000 (var=0.0000)
- Experiment 10: real=0.4000 (var=0.6800) vs this=1.0000 (var=0.0000)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4627 -> ACCEPTED
- iter 2: loss=0.4979 -> REJECTED
- iter 3: loss=0.4676 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.4627 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[5, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[2, 8, 8, 8]  B=[9, 7, 7, 7]
  A=[10, 4, 4, 4]  B=[5, 5, 5, 5]
  A=[9, 3, 3, 3]  B=[2, 4, 4, 4]
  A=[6, 6, 6, 2]  B=[5, 5, 5, 9]
  A=[3, 3, 3, 10]  B=[4, 4, 4, 1]
  A=[8, 8, 2, 2]  B=[7, 7, 4, 4]
  A=[2, 2, 8, 8]  B=[4, 4, 7, 7]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    mask = tally_a != tally_b
    if not np.any(mask):
        return 0.5
        
    preds = np.where(tally_a > tally_b, 0, 1)
    responses = data['response'].values
    
    accuracy = np.mean(preds[mask] == responses[mask])
    return float(accuracy)

```

**Observed (real) value:** 0.8000 (var=0.0667)
**Previous candidate values (this loop):**
  - iter 1: 0.3672 (var=0.0017) (Δ vs real -0.4328)
  - iter 2: 0.2036 (var=0.0143) (Δ vs real -0.5964)
  - iter 3 (most recent): 0.3664 (var=0.0030) (Δ vs real -0.4336)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.2233 (var=0.0673)
- pi_3: 0.1189 (var=0.0057)
- pi_4: 0.8631 (var=0.0088)
- pi_5: 0.8514 (var=0.0108)
- pi_6: 0.1297 (var=0.0087)

### Experiment 2
**Design**
  A=[0, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[1, 5, 5, 5]  B=[5, 4, 4, 4]
  A=[5, 4, 1, 1]  B=[1, 1, 5, 5]
  A=[5, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[4, 1, 1, 1]  B=[0, 2, 2, 2]
  A=[1, 2, 3, 5]  B=[2, 3, 4, 0]
  A=[5, 1, 1, 1]  B=[1, 2, 2, 2]
  A=[2, 5, 1, 1]  B=[3, 1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 4 (A=[5,0,0,0], sum=5) and trial 1 (A=[0,4,4,4], sum=12)
    a_sum = data['option_a_ratings'].apply(sum)
    chose_a = (data['response'] == 0).astype(float)
    
    mask_t4 = (a_sum == 5)
    mask_t1 = (a_sum == 12)
    
    p_a_t4 = chose_a[mask_t4].mean() if mask_t4.any() else 0.5
    p_a_t1 = chose_a[mask_t1].mean() if mask_t1.any() else 0.5
    
    return float(p_a_t4 - p_a_t1)
```

**Observed (real) value:** -0.4000 (var=0.8400)
**Previous candidate values (this loop):**
  - iter 1: 0.7517 (var=0.0326) (Δ vs real +1.1517)
  - iter 2: 0.7883 (var=0.0370) (Δ vs real +1.1883)
  - iter 3 (most recent): 0.7333 (var=0.0389) (Δ vs real +1.1333)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5600 (var=0.6864)
- pi_1: -1.0000 (var=0.0000)
- pi_3: 0.7383 (var=0.0522)
- pi_4: -0.7633 (var=0.0345)
- pi_5: -0.6950 (var=0.0527)
- pi_6: 0.7300 (var=0.0293)

### Experiment 3
**Design**
  A=[2, 2, 2, 0]  B=[0, 0, 0, 10]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 8]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[9, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[5, 5, 0, 0]  B=[0, 0, 5, 6]
  A=[6, 6, 0, 0]  B=[0, 0, 5, 5]
  A=[0, 0, 0, 10]  B=[2, 2, 2, 2]
  A=[2, 2, 2, 0]  B=[0, 0, 0, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(A > B, axis=1)
    b_wins = np.sum(B > A, axis=1)
    
    a_sum = np.sum(A, axis=1)
    b_sum = np.sum(B, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    ew_pref = np.where(a_sum > b_sum, 0, np.where(b_sum > a_sum, 1, -1))
    
    discordant = (tally_pref != -1) & (ew_pref != -1) & (tally_pref != ew_pref)
    
    if not np.any(discordant):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[discordant] == tally_pref[discordant])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7467 (var=0.0780)
**Previous candidate values (this loop):**
  - iter 1: 0.6261 (var=0.0020) (Δ vs real -0.1206)
  - iter 2: 0.4356 (var=0.0425) (Δ vs real -0.3111)
  - iter 3 (most recent): 0.5883 (var=0.0041) (Δ vs real -0.1583)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: 0.1489 (var=0.0087)
- pi_2: 0.2600 (var=0.1057)
- pi_4: 0.8867 (var=0.0066)
- pi_5: 0.8372 (var=0.0082)
- pi_6: 0.1750 (var=0.0052)

### Experiment 4
**Design**
  A=[10, 2, 2, 2]  B=[0, 3, 3, 3]
  A=[4, 4, 4, 0]  B=[3, 3, 3, 10]
  A=[10, 8, 2, 1]  B=[0, 1, 3, 4]
  A=[1, 1, 0, 0]  B=[0, 0, 10, 10]
  A=[6, 6, 6, 0]  B=[5, 5, 5, 3]
  A=[10, 2, 2, 2]  B=[1, 5, 5, 5]
  A=[0, 0, 0, 10]  B=[1, 1, 1, 0]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filtered = sum_a[mask]
    sum_b_filtered = sum_b[mask]
    
    eqw_pred = (sum_a_filtered < sum_b_filtered).astype(int)
    matches = (filtered_data['response'] == eqw_pred).mean()
    
    return float(matches)
```

**Observed (real) value:** 0.3500 (var=0.0469)
**Previous candidate values (this loop):**
  - iter 1: 0.3717 (var=0.0024) (Δ vs real +0.0217)
  - iter 2: 0.7108 (var=0.0574) (Δ vs real +0.3608)
  - iter 3 (most recent): 0.4453 (var=0.0078) (Δ vs real +0.0953)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8767 (var=0.0074)
- pi_1: 0.1667 (var=0.0000)
- pi_2: 0.9567 (var=0.0109)
- pi_4: 0.2539 (var=0.0038)
- pi_5: 0.2575 (var=0.0042)
- pi_6: 0.8567 (var=0.0063)

### Experiment 5
**Design**
  A=[0, 0, 2, 2, 2]  B=[2, 2, 0, 0, 0]
  A=[2, 2, 0, 0, 0]  B=[0, 0, 2, 2, 2]
  A=[2, 0, 2, 0, 0]  B=[0, 2, 0, 2, 0]
  A=[0, 2, 0, 2, 0]  B=[2, 0, 2, 0, 0]
  A=[2, 0, 0, 0, 0]  B=[0, 0, 2, 2, 0]
  A=[0, 0, 2, 2, 0]  B=[2, 0, 0, 0, 0]
  A=[2, 0, 2, 2, 0]  B=[0, 2, 0, 0, 2]
  A=[0, 2, 0, 0, 2]  B=[2, 0, 2, 2, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 2, 2, 2) and b == (2, 2, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        elif a == (2, 2, 0, 0, 0) and b == (0, 0, 2, 2, 2):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.7600 (var=0.1824)
**Previous candidate values (this loop):**
  - iter 1: 0.1300 (var=0.0089) (Δ vs real -0.6300)
  - iter 2: 0.2850 (var=0.0539) (Δ vs real -0.4750)
  - iter 3 (most recent): 0.1367 (var=0.0131) (Δ vs real -0.6233)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_4: 0.1858 (var=0.0200)
- pi_2: 0.7600 (var=0.1824)
- pi_3: 0.8308 (var=0.0257)
- pi_5: 0.6783 (var=0.0731)
- pi_6: 0.2125 (var=0.0159)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([1.0, 0.8, 0.6, 0.5, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    wt_a = np.sum(validities * (A > B), axis=1)
    wt_b = np.sum(validities * (B > A), axis=1)
    wt_sign = np.sign(wt_a - wt_b)
    
    tally_a = np.sum(A > B, axis=1)
    tally_b = np.sum(B > A, axis=1)
    tally_sign = np.sign(tally_a - tally_b)
    
    conflict_mask = (wt_sign != 0) & (tally_sign != 0) & (wt_sign != tally_sign)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    chosen_sign = np.where(responses == 0, 1, -1)
    
    match = (chosen_sign[conflict_mask] == wt_sign[conflict_mask])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5000 (var=0.2500)
**Previous candidate values (this loop):**
  - iter 1: 0.8808 (var=0.0090) (Δ vs real +0.3808)
  - iter 2: 0.6258 (var=0.0438) (Δ vs real +0.1258)
  - iter 3 (most recent): 0.8058 (var=0.0218) (Δ vs real +0.3058)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7342 (var=0.0269)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.2400 (var=0.1824)
- pi_3: 0.1550 (var=0.0153)
- pi_5: 0.2658 (var=0.0478)
- pi_6: 0.7725 (var=0.0174)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    validities = np.array([1.0, 0.8, 0.6, 0.5, 0.5])
    
    a_wins = a_ratings > b_ratings
    b_wins = b_ratings > a_ratings
    
    tally_a = a_wins.sum(axis=1)
    tally_b = b_wins.sum(axis=1)
    
    wt_a = (a_wins * validities).sum(axis=1)
    wt_b = (b_wins * validities).sum(axis=1)
    
    # Identify trials where Tallying strictly prefers A but Weighted Tallying does not
    condition_a = (tally_a > tally_b) & (wt_a <= wt_b)
    # Identify trials where Tallying strictly prefers B but Weighted Tallying does not
    condition_b = (tally_b > tally_a) & (wt_b <= wt_a)
    
    mask = condition_a | condition_b
    if not np.any(mask):
        return 0.0
        
    responses = data['response'].values
    
    # 0 means chose A, 1 means chose B
    match_a = condition_a & (responses == 0)
    match_b = condition_b & (responses == 1)
    
    return float(np.mean((match_a | match_b)[mask]))
```

**Observed (real) value:** 0.6867 (var=0.1529)
**Previous candidate values (this loop):**
  - iter 1: 0.1172 (var=0.0099) (Δ vs real -0.5694)
  - iter 2: 0.3889 (var=0.0396) (Δ vs real -0.2978)
  - iter 3 (most recent): 0.2278 (var=0.0212) (Δ vs real -0.4589)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7256 (var=0.0291)
- pi_4: 0.3144 (var=0.0099)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.8333 (var=0.1167)
- pi_3: 0.8889 (var=0.0073)
- pi_6: 0.3311 (var=0.0084)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where WT predicts a perfect tie (score A = 1.0, score B = 0.5 + 0.5 = 1.0)
    # but Tallying prefers B (1 win vs 2 wins).
    mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))) & \
           (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1)))
    if mask.sum() == 0:
        return 0.5
    # Return the proportion of times B was chosen
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.6200 (var=0.2356)
**Previous candidate values (this loop):**
  - iter 1: 0.1189 (var=0.0085) (Δ vs real -0.5011)
  - iter 2: 0.3453 (var=0.0463) (Δ vs real -0.2747)
  - iter 3 (most recent): 0.2242 (var=0.0219) (Δ vs real -0.3958)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4926 (var=0.0117)
- pi_5: 0.7800 (var=0.0209)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.7800 (var=0.1716)
- pi_3: 0.8632 (var=0.0133)
- pi_6: 0.4863 (var=0.0123)

### Experiment 9
**Design**
  A=[2, 5, 5, 5]  B=[10, 4, 4, 4]
  A=[6, 6, 6, 6]  B=[5, 5, 5, 10]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 8, 0, 0]  B=[7, 7, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: x[0] == 2)
    is_t3 = data['option_a_ratings'].apply(lambda x: x[0] == 10)
    
    p_a_t1 = (data.loc[is_t1, 'response'] == 0).mean()
    p_a_t3 = (data.loc[is_t3, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    
    return float(p_a_t1 - p_a_t3)
```

**Observed (real) value:** 0.6000 (var=0.6400)
**Previous candidate values (this loop):**
  - iter 1: -0.7167 (var=0.0287) (Δ vs real -1.3167)
  - iter 2: -0.7325 (var=0.0322) (Δ vs real -1.3325)
  - iter 3 (most recent): -0.7492 (var=0.0573) (Δ vs real -1.3492)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6692 (var=0.0611)
- pi_6: -0.7733 (var=0.0228)
- pi_1: 1.0000 (var=0.0000)
- pi_2: -0.6800 (var=0.3776)
- pi_3: -0.7492 (var=0.0191)
- pi_4: 0.6558 (var=0.0445)

### Experiment 10
**Design**
  A=[10, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 10, 10, 10]  B=[1, 1, 0, 0, 0]
  A=[2, 2, 2, 2, 2]  B=[0, 10, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    is_t1 = (a0 == 10) & (a1 == 0)
    is_t2 = (a0 == 1) & (a1 == 1)
    is_t5 = (a0 == 2) & (a1 == 2)
    
    resp_t1 = data.loc[is_t1, 'response'].mean()
    resp_t2 = data.loc[is_t2, 'response'].mean()
    resp_t5 = data.loc[is_t5, 'response'].mean()
    
    # Handle potential empty slices gracefully
    if pd.isna(resp_t1): resp_t1 = 0.5
    if pd.isna(resp_t2): resp_t2 = 0.5
    if pd.isna(resp_t5): resp_t5 = 0.5
    
    return float(resp_t1 - resp_t2 - resp_t5)
```

**Observed (real) value:** 0.4000 (var=0.6800)
**Previous candidate values (this loop):**
  - iter 1: -0.1189 (var=0.0176) (Δ vs real -0.5189)
  - iter 2: -0.8211 (var=0.4831) (Δ vs real -1.2211)
  - iter 3 (most recent): -0.2811 (var=0.0650) (Δ vs real -0.6811)
**Other theories' values on this metric (for reference):**
- pi_6: -1.6168 (var=0.0505)
- pi_5: 0.6053 (var=0.0811)
- pi_1: 1.0000 (var=0.0000)
- pi_2: -1.2800 (var=0.8016)
- pi_3: -1.6347 (var=0.0834)
- pi_4: 0.5947 (var=0.0642)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Take-The-Best (TTB) Heuristic: Decision makers use a strictly non-compensatory lexicographic strategy. They do not integrate information across all features; rather, they evaluate features one by one in descending order of their explicit validities. The search stops at the first feature that discriminates between the two options (i.e., where one option has a strictly higher rating than the other). The option that is superior on this single highest-validity discriminating feature is chosen, ignoring all remaining features. If no feature discriminates, the decision maker guesses. Response noise is captured via an epsilon-greedy parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    # We use stable sort to handle ties deterministically
    sorted_idx = np.argsort(-validities, kind='stable')
    
    score_a, score_b = 0.0, 0.0
    
    # Search for the first discriminating feature
    for idx in sorted_idx:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    # If no feature discriminates, guess uniformly
    if score_a == 0.0 and score_b == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`: Following the arbiter's guidance, this model replaces compensatory integration with a strictly non-compensatory lexicographic process. The Take-The-Best (TTB) heuristic captures decision-making where cognitive bounds or strong cue validities induce serial search. Instead of summing wins or weighted values, the model sorts features by validity and halts at the first one that breaks the tie. By doing so, it provides an ordinal, fast-and-frugal alternative mechanism to WADD and Tallying.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate implements a strictly non-compensatory Take-The-Best (TTB) heuristic, faithfully following the arbiter's recommendation. It successfully establishes a new running-best loss (0.4627), meaning it provides a useful theoretical anchor. However, the model struggles significantly on experiments where cardinal rating magnitudes matter (e.g., Experiment 2 and Experiment 9). In these experiments, a strictly ordinal comparison (is A > B?) on the highest-validity cue ignores cases where the difference on that cue is negligible but the difference on subsequent cues is massive.
Rationale: To improve the fit while strictly remaining within the prescribed Lexicographic/TTB family, introduce a 'Just Noticeable Difference' (JND) or 'threshold' parameter. Instead of a feature discriminating if `a[idx] > b[idx]`, it should only discriminate if `a[idx] - b[idx] > threshold`. If the difference is within the threshold, the cue is treated as a tie, and the decision maker moves to the next highest-validity cue. This allows the lexicographic search to bypass features where the options are practically tied, better capturing human sensitivity to cardinal magnitudes without resorting to compensatory integration. Add a `threshold` parameter (e.g., [0.0, 3.0]) to the parameters list.

**Outcome of this advice:** iter 1 candidate loss=0.4627 -> iter 2 candidate loss=0.4979 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate incorporated a Just Noticeable Difference (JND) threshold into the Take-The-Best heuristic. However, this approach was rejected by the gate (loss increased from 0.4627 to 0.4979), likely because a hard threshold completely discards small differences on high-validity cues, harming accuracy in experiments where those small differences are still meaningful. We must revert to the pure lexicographic search order but find another way to incorporate sensitivity to cardinal magnitudes.
Rationale: Since the hard JND threshold was rejected, try a probabilistic decision rule based on the magnitude of the *first discriminating cue*. Keep the strict lexicographic search (evaluating features in descending order of explicit validities). When the first discriminating feature is found (i.e., a[idx] != b[idx]), do not make a deterministic choice. Instead, use a logistic function (softmax) on the difference to determine the choice probability: `p_a = 1 / (1 + exp(-beta * (a[idx] - b[idx])))`. If no feature discriminates, return 0.5. Add a `beta` parameter (e.g., [0.0, 5.0]) to scale the difference. This strictly preserves the non-compensatory TTB search while allowing the magnitude of the deciding cue to smoothly modulate confidence, which should improve fit on cardinal-dependent experiments without the brittleness of a hard threshold.

**Outcome of this advice:** iter 2 candidate loss=0.4979 -> iter 3 candidate loss=0.4676 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate attempted to soften the Take-The-Best (TTB) heuristic by applying a softmax over the magnitude of the first discriminating cue. This was rejected by the gate (loss increased to 0.4676 compared to the base 0.4627). Modulating confidence by the deciding cue's magnitude or using a JND threshold both failed to improve the aggregate fit. The core issue remains that in experiments like 2 and 9, human choices frequently contradict the single highest-validity cue, which a strict deterministic search order can never capture, regardless of how confidence is scaled after the cue is found.
Rationale: Since modifying the evaluation of the deciding cue (JND, magnitude softmax) was rejected, we should instead modify the *search order* while strictly keeping the non-compensatory decision rule. Implement a 'Stochastic Take-The-Best' model: instead of sorting cues deterministically by validity, sample the search order on each trial. Cues should be sampled without replacement, where the probability of selecting a cue next is proportional to a softmax over the remaining cues' validities, controlled by a temperature parameter `tau` (e.g., [0.01, 5.0]). Once a cue is selected, if it discriminates, the model deterministically chooses the option with the higher value on that cue (subject to a standard lapse rate `epsilon`). This preserves the fast-and-frugal, single-cue stopping rule prescribed by the arbiter, but allows occasional reliance on lower-validity cues, improving fit on experiments where humans don't always follow the strict validity ranking.

**Outcome of this advice:** iter 3 candidate loss=0.4676 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Stochastic Take-The-Best (Plackett-Luce Search): Decision makers employ a non-compensatory lexicographic strategy, but their search order through cues is probabilistic rather than strictly deterministic. Cues are sampled sequentially without replacement, with selection probabilities proportional to a softmax over their explicit validities. The search terminates at the first cue that discriminates between the options. By the properties of the Plackett-Luce model, the probability that a specific discriminating cue is the one that drives the choice is exactly its relative softmax weight among the subset of all discriminating cues, seamlessly interpolating between strict Take-The-Best (at low temperatures) and uniform Tallying (at high temperatures).",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\"Stimulus must be (2, n_features)\")\n        \n    a, b = stim[0], stim[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    tau = float(parameters[\"tau\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    discrim_a = a > b\n    discrim_b = b > a\n    discrim_any = discrim_a | discrim_b\n    \n    if not np.any(discrim_any):\n        p_core = np.array([0.5, 0.5])\n    else:\n        # Under Plackett-Luce sampling without replacement, the probability that \n        # a specific discriminating feature is drawn *before* any other \n        # discriminating feature is exactly its relative weight among the \n        # discriminating features. Non-discriminating features skipped along the \n        # way do not affect this relative probability.\n        v_discrim = validities[discrim_any]\n        z = v_discrim / tau\n        z = z - np.max(z)  # for numerical stability\n        w = np.exp(z)\n        w = w / np.sum(w)\n        \n        w_all = np.zeros_like(validities)\n        w_all[discrim_any] = w\n        \n        score_a = np.sum(w_all[discrim_a])\n        score_b = np.sum(w_all[discrim_b])\n        \n        p_core = np.array([score_a, score_b])\n        \n    n_opts = p_core.shape[0]\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=np.float64)\n    probs /= probs.sum()\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "tau": "[0.01, 5.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities"
  },
  "rationale": "Following the critic's advice, we implement a Stochastic Take-The-Best heuristic where the search order is sampled probabilistically. Instead of a deterministic sort by validity, features are selected sequentially proportional to a softmax over their validities (temperature `tau`). The decision is made by the *first* discriminating feature encountered. Mathematically, this is a Plackett-Luce model over the search order. A known property of Luce's Choice Axiom is that the probability of drawing item i before item j is independent of the presence of other non-discriminating items. Thus, the probability that a specific discriminating cue is the deciding one is exactly its softmax weight normalized *only* over the subset of discriminating cues. This elegantly and efficiently implements the requested mechanism without requiring noisy Monte Carlo simulations, interpolating smoothly between pure TTB (tau -> 0) and Tallying (tau -> inf)."
}
```
```

## Usage

```json
{
  "prompt_token_count": 16327,
  "candidates_token_count": 965,
  "total_token_count": 21085
}
```
