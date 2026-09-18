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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Propose a new theory: 'Validity-Weighted Accumulation' or 'Soft Threshold Decision Rule.' This theory should: (1) Accumulate evidence across features weighted by their validities (not equal weights as in Tallying, not binary hierarchy as in TTB); (2) Use a soft decision threshold rather than a hard stopping rule—features are consulted in descending validity order, but each feature contributes to a running accumulation score rather than triggering a deterministic stop; (3) The accumulation rule might be: for each feature in validity order, add its contribution (scaled by validity) to a running score for each option; (4) At each step, compute a soft threshold or decision boundary: if the accumulated evidence for one option exceeds a threshold (possibly relative to the remaining features' potential contribution), choose that option; otherwise, continue to the next feature; (5) Alternatively, accumulate all features weighted by validity and choose based on the weighted sum (similar to a linear model or weighted Tallying). This hybrid approach would explain why subjects in Experiment 1 do not rigidly follow the top cue (observed 0.38 vs. TTB's 0.82) while still respecting validity structure enough to outperform pure Tallying in cases where validity differences are large. It would also explain the intermediate performance in Experiment 2 (observed 0.72, between TTB's 0.14 and Tallying's 0.88): subjects integrate the high-validity feature's signal but do not ignore the consensus of lower-validity features entirely. The new theory should retain the softmax noise and lapse parameters from both existing theories but replace the core decision logic with a validity-weighted accumulation process.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 1.000)

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
- Experiment 1: real=0.3842 (var=0.0029) vs this=0.3202 (var=0.0034)
- Experiment 2: real=0.7204 (var=0.0038) vs this=0.8753 (var=0.0054)


---

### `pi_1` (overall score: 0.000)

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
- Experiment 1: real=0.3842 (var=0.0029) vs this=0.8248 (var=0.0097)
- Experiment 2: real=0.7204 (var=0.0038) vs this=0.1371 (var=0.0100)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1420 -> ACCEPTED
- iter 2: loss=0.8311 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1420 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Metric: Proportion of trials where subject chooses the option that wins on the HIGHEST-VALIDITY feature.
    # TTB predicts subjects will almost always choose based on the top cue (validity 0.9),
    # so they should choose the option that wins on feature 0 with high probability.
    # Tallying treats all features equally, so it has no reason to preferentially choose
    # based on feature 0 alone — it will sometimes choose the option that loses on feature 0
    # but wins on the lower-validity features (e.g., 3 vs 1 on the remaining features).
    # This metric isolates the signature of TTB's cue hierarchy.
    
    def choose_based_on_top_cue(row):
        a_ratings = np.array(row['option_a_ratings'])
        b_ratings = np.array(row['option_b_ratings'])
        # Top cue is feature 0 (validity 0.9)
        if a_ratings[0] > b_ratings[0]:
            # Top cue favors A
            return row['response'] == 0
        elif b_ratings[0] > a_ratings[0]:
            # Top cue favors B
            return row['response'] == 1
        else:
            # Top cue ties; this trial does not constrain the metric
            # (neither theory predicts a strong preference)
            return np.nan
    
    # Apply to all trials
    matches = data.apply(choose_based_on_top_cue, axis=1)
    # Count trials where response matches top-cue prediction, excluding ties
    valid_matches = matches[~matches.isna()]
    if len(valid_matches) == 0:
        return np.nan
    return float(valid_matches.sum()) / float(len(valid_matches))
```

**Observed (real) value:** 0.3842 (var=0.0029)
**Previous candidate values (this loop):**
  - iter 1: 0.5096 (var=0.0450) (Δ vs real +0.1254)
  - iter 2 (most recent): 0.8475 (var=0.0101) (Δ vs real +0.4633)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8248 (var=0.0097)
- pi_2: 0.3202 (var=0.0034)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Metric: Proportion of trials where subject chose the option with MORE feature wins (tally-consistent choice)
    # TTB will often choose the option favored by the highest-validity feature (0.95),
    # even when that option loses on the remaining features (1 win vs. 3+ wins).
    # Tallying will choose the option with more feature wins.
    # This metric measures the rate at which subjects exhibit tally-consistent behavior.
    
    def compute_tally_winner(row):
        # Returns 0 if A wins more features, 1 if B wins more, None if tie
        a_ratings = np.asarray(row['option_a_ratings'], dtype=float)
        b_ratings = np.asarray(row['option_b_ratings'], dtype=float)
        a_wins = float(np.sum(a_ratings > b_ratings))
        b_wins = float(np.sum(b_ratings > a_ratings))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        else:
            return None
    
    # Compute tally winner for each trial
    data_copy = data.copy()
    data_copy['tally_winner'] = data_copy.apply(compute_tally_winner, axis=1)
    
    # Filter out tie trials (where tally provides no basis for preference)
    non_tie_trials = data_copy[data_copy['tally_winner'].notna()]
    
    if len(non_tie_trials) == 0:
        return 0.5  # No informative trials
    
    # Count how often subject's response matches the tally winner
    tally_consistent = (non_tie_trials['response'] == non_tie_trials['tally_winner']).sum()
    
    # Return proportion of tally-consistent choices
    return float(tally_consistent) / len(non_tie_trials)

```

**Observed (real) value:** 0.7204 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.6942 (var=0.0580) (Δ vs real -0.0262)
  - iter 2 (most recent): 0.1420 (var=0.0100) (Δ vs real -0.5784)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8753 (var=0.0054)
- pi_1: 0.1371 (var=0.0100)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Validity-Weighted Accumulation with Soft Threshold. Subjects accumulate evidence for each option by processing features in descending order of validity. Each feature contributes to a running score weighted by its validity. At each step, the model computes whether the accumulated evidence for one option exceeds a soft threshold (defined relative to the maximum possible remaining evidence). If the threshold is exceeded, the option is chosen; otherwise, the next feature is consulted. This hybrid approach respects validity structure (high-validity features have stronger influence) while allowing lower-validity features to collectively override a weak lead from higher-validity features—explaining why subjects in Experiment 1 choose based on the top cue only 38% of the time (not TTB's 82%) and why Experiment 2 shows intermediate tally-consistency (72%, between TTB's 14% and Tallying's 88%).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from scipy.special import softmax
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-Weighted Accumulation expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    n_features = stim.shape[1]
    a, b = stim[0], stim[1]
    
    # Extract parameters
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    
    threshold_param = float(parameters["threshold_param"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Compute maximum possible evidence (sum of all validities)
    max_evidence = np.sum(val)
    
    # Accumulate evidence feature by feature
    score_a = 0.0
    score_b = 0.0
    
    for idx, j in enumerate(cue_order):
        # Add validity-weighted contribution for this feature
        if a[j] > b[j]:
            score_a += val[j]
        elif b[j] > a[j]:
            score_b += val[j]
        # If a[j] == b[j], neither option gains evidence
        
        # Compute remaining maximum evidence (features not yet processed)
        remaining_features = cue_order[idx + 1:]
        max_remaining = np.sum(val[remaining_features])
        
        # Soft threshold: if one option's lead exceeds threshold_param times
        # the maximum remaining evidence, stop and decide
        lead = abs(score_a - score_b)
        threshold = threshold_param * max_remaining
        
        if lead > threshold:
            # Decision threshold exceeded; choose based on current scores
            if score_a > score_b:
                scores = np.array([1.0, 0.0])
            else:
                scores = np.array([0.0, 1.0])
            
            # Apply softmax noise and lapse
            z = beta * (scores - scores.max())
            e = np.exp(z)
            p_core = e / e.sum()
            return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    
    # If all features processed without exceeding threshold, choose based on final scores
    if score_a > score_b:
        scores = np.array([1.0, 0.0])
    elif score_b > score_a:
        scores = np.array([0.0, 1.0])
    else:
        # Tie: no preference
        return np.ones(2) / 2.0
    
    # Apply softmax noise and lapse
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- threshold_param: [0.1, 2.0]

`rationale`: This theory bridges TTB and Tallying by introducing a soft stopping rule based on accumulated validity-weighted evidence. The key innovation is the `threshold_param`, which controls when a decision is triggered relative to remaining evidence. When `threshold_param` is high (e.g., 2.0), subjects require a very large lead before stopping, forcing them to consult more features and thus behave more like Tallying (weighted sum of all features). When `threshold_param` is low (e.g., 0.1), subjects stop quickly after the first discriminating high-validity feature, approximating TTB. The arbiter's feedback identified two mechanistic failures: (1) TTB's rigid one-reason rule fails on Experiment 1 (predicted 0.82, observed 0.38), and (2) Tallying's equal weighting fails on Experiment 2 (predicted 0.88, observed 0.72). Validity-Weighted Accumulation overcomes both: it respects validity structure (high-validity features dominate early) but allows lower-validity features to collectively accumulate and override weak leads—explaining intermediate behavior. In Experiment 1, with validity [0.9, 0.7, 0.7, 0.7], the top feature (0.9) can establish a large lead, but the three lower features (each 0.7) collectively have total weight 2.1, which can overcome that lead if threshold_param permits continued accumulation. In Experiment 2, with validity [0.95, 0.7, 0.7, 0.7, 0.7], the top feature is even more dominant, but lower-validity features (total 2.8) are numerous enough to override it if the threshold allows. The soft threshold mechanism is more cognitively plausible than TTB's all-or-nothing stopping rule, allowing gradual confidence accumulation rather than deterministic stopping. Combined with softmax noise and lapse parameters (retained from both prior theories), this model should capture the intermediate, validity-respecting but non-deterministic behavior observed across both experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate theory (Validity-Weighted Accumulation with Soft Threshold) has been ACCEPTED by the gate (loss=0.1420), indicating it improves over the prior base. Examining the per-experiment fit:

**Experiment 1 (top-cue consistency):**
- Observed: 0.3842 (subjects choose based on top cue ~38% of the time)
- Candidate: 0.5096 (predicts ~51% top-cue consistency)
- Direction of miss: The candidate OVERSHOOTS the observed value by ~13 percentage points, predicting subjects rely more heavily on the top cue than they actually do.
- Reference: pi_1 (TTB) predicts 0.8248 (too rigid), pi_2 predicts 0.3202 (slightly too low). The candidate lands between TTB and the observed value, but still biased toward the top cue.

**Experiment 2 (tally consistency):**
- Observed: 0.7204 (subjects choose tally-consistent option ~72% of the time)
- Candidate: 0.6942 (predicts ~69% tally consistency)
- Direction of miss: The candidate UNDERSHOOTS by ~2.6 percentage points, predicting slightly less tally-consistent behavior than observed.
- Reference: pi_2 (Tallying) predicts 0.8753 (too high), pi_1 (TTB) predicts 0.1371 (too low). The candidate is close to observed, slightly conservative.

**Variance analysis:**
- Experiment 1: candidate var=0.0450 vs. observed var=0.0029 (candidate shows 15× higher between-subject variance). This suggests the model's behavior is less stable across subjects than real data—parameter heterogeneity or threshold_param sensitivity is inflating variability.
- Experiment 2: candidate var=0.0580 vs. observed var=0.0038 (candidate shows 15× higher variance). Same issue: the model is less consistent across subjects.

**Mechanistic assessment:**
The soft threshold mechanism is conceptually sound: it allows early stopping when a high-validity feature builds a sufficient lead (approximating TTB when threshold_param is low) and continued accumulation when the lead is weak (approximating Tallying when threshold_param is high). However, the implementation appears to be tuning toward a middle ground that:
1. Still overweights the top cue in Experiment 1 (candidate 0.51 vs. observed 0.38), suggesting threshold_param may be calibrated too low on average, causing early stopping even when lower-validity features should override.
2. Slightly underweights lower-validity features in Experiment 2 (candidate 0.69 vs. observed 0.72), suggesting the same threshold_param is not permitting enough accumulation.
3. Shows high cross-subject variance, indicating that the threshold_param range [0.1, 2.0] or its interaction with beta and epsilon is producing heterogeneous behavior that does not match the empirical consistency.
Rationale: Although the candidate was accepted by the gate and represents an improvement over the prior base, it exhibits two correctable issues within the prescribed mechanism family:

1. **Experiment 1 overfitting to top cue:** The candidate predicts 0.5096 (top-cue consistency) vs. observed 0.3842. The soft threshold is still triggering too early—the model is stopping after the top feature's lead exceeds the threshold too often. The threshold_param range [0.1, 2.0] may need adjustment: specifically, the lower bound (0.1) may be too low, causing premature stopping. Consider raising the minimum threshold_param to [0.3, 2.0] or higher, or introduce a multiplicative factor that makes the threshold relative to the *cumulative* remaining evidence more conservative.

2. **High between-subject variance:** Both experiments show candidate variance ~15× higher than observed (0.045 vs. 0.003 in Exp 1; 0.058 vs. 0.004 in Exp 2). This suggests parameter heterogeneity or sensitivity. The threshold_param may be too loosely bounded or interacting poorly with beta. Consider: (a) tightening the threshold_param range to a narrower, more principled window (e.g., [0.4, 1.2] or [0.5, 1.5]); (b) examining whether beta's range [0.1, 20.0] is too wide and creating parameter drift; or (c) adding a constraint that threshold_param and beta covary (e.g., high threshold_param pairs with lower beta to reduce variability).

3. **Mechanism fidelity:** The core soft-threshold logic is sound and stays within the prescribed mechanism family. However, the parameterization is not yet calibrated to produce the observed level of consistency. The proposer should:
   - Reduce the threshold_param minimum (e.g., raise to 0.3–0.5) to reduce top-cue reliance in Experiment 1.
   - Narrow the threshold_param range overall to reduce cross-subject variance.
   - Consider whether the threshold should scale by the *number* of remaining features (not just their sum of validities) to avoid giving too much weight to many weak features vs. one strong feature.

The theory is on the right track; minor re-calibration of the threshold_param range and possibly beta should resolve both the Experiment 1 overfitting and the variance issue.

**Outcome of this advice:** iter 1 candidate loss=0.1420 -> iter 2 candidate loss=0.8311 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate (iter 2, loss=0.8311) has been REJECTED by the gate, performing substantially worse than the accepted iter 1 baseline (loss=0.1420). Examining per-experiment fit reveals a dramatic reversal of the intended direction:

**Experiment 1 (top-cue consistency):**
- Observed: 0.3842
- Iter 1 (accepted): 0.5096 (overshot by +0.1254)
- Iter 2 (current, rejected): 0.8475 (massively overshot by +0.4633)
- The current candidate now predicts subjects follow the top cue 85% of the time—worse than TTB's 0.8248 and in the opposite direction from the observed 38%. This is a catastrophic regression.

**Experiment 2 (tally consistency):**
- Observed: 0.7204
- Iter 1 (accepted): 0.6942 (undershot by -0.0262)
- Iter 2 (current, rejected): 0.1420 (massively undershot by -0.5784)
- The current candidate now predicts only 14% tally consistency, matching TTB's 0.1371 and abandoning the validity-weighted accumulation logic entirely.

**Mechanism diagnosis:**
The changes made in iter 2 (narrowing threshold_param to [0.4, 1.2] and normalizing by n_remaining features) had the *opposite* effect from the intended one. Rather than delaying early stopping and allowing lower-validity features more influence, the per-feature normalization threshold = threshold_param * (max_remaining_evidence / (n_remaining + 1.0)) appears to have created a much *higher* effective threshold. For example, in Experiment 1 with 3 remaining features after the top cue (sum validities ~2.1), the normalized threshold becomes 0.4 * (2.1 / 4) ≈ 0.21—but this is actually *lower* per remaining feature. However, the narrower parameter range [0.4, 1.2] forces threshold_param to stay high, and the interaction with the normalization divisor is now making the model stop *earlier*, not later. The result is that the model is now behaving like a rigid TTB model in Experiment 1 and like a pure TTB model in Experiment 2.

**Variance:**
Variance in iter 2 is lower (0.0101 in Exp 1, 0.0100 in Exp 2), but this is because the model is now so constrained by the narrow parameter range and aggressive threshold that it produces nearly deterministic behavior—not because the mechanism is well-calibrated. This is a false improvement masking a fundamental mechanism failure.

**Root cause:**
The proposer's rationale claimed that dividing by (n_remaining + 1) would "normalize by feature count" and "prevent many weak features from overwhelming a strong lead." In fact, this normalization *increased* the effective threshold relative to per-feature contribution, causing the model to stop even more aggressively after the top feature, not less. The narrowed parameter range [0.4, 1.2] compounded this by preventing threshold_param from dropping low enough to permit accumulation.
Rationale: The iter 2 candidate is a clear regression from the accepted iter 1 baseline and must be regenerated. The proposer's edit to the threshold computation and parameter bounds moved in the wrong direction—the per-feature normalization and narrower range made the model *more* TTB-like, not less.

**What went wrong:**
My prior critique (iter 1) identified two issues: (1) Experiment 1 overfitting (0.5096 vs. 0.3842), and (2) high variance. The proposer attempted to fix (1) by raising threshold_param's lower bound from 0.1 to 0.4 and normalizing the threshold by (n_remaining + 1). However, this normalization had the *opposite* effect: it made the threshold per-feature smaller relative to the accumulated score, which should have delayed stopping, but the narrower overall parameter range prevented the model from tuning low enough to compensate. The result is a model that now stops even earlier and more rigidly than iter 1.

**Direction for regeneration (staying within prescribed mechanism family):**
The core validity-weighted accumulation mechanism is sound, but the parameterization needs a different fix:

1. **Revert the per-feature normalization:** The original threshold = threshold_param * max_remaining_evidence was closer to correct. The issue in iter 1 was not that the threshold formula was wrong, but that threshold_param's *range* was poorly calibrated. Remove the (n_remaining + 1.0) divisor and return to the simpler accumulation-based threshold.

2. **Adjust threshold_param range more carefully:** Instead of narrowing to [0.4, 1.2], try [0.2, 0.8] or [0.15, 0.6]. The goal is to allow the model to tune lower when needed (Experiment 1, where many weak features should override a strong lead) while still respecting validity structure. The iter 1 range [0.1, 2.0] was too wide; the iter 2 range [0.4, 1.2] was too narrow and too high. A middle ground like [0.2, 0.8] should allow more flexibility.

3. **Alternatively, introduce a feature-count penalty directly:** Rather than dividing by (n_remaining + 1), consider a different approach: add a small penalty term to the threshold that *decreases* (not increases) as more features accumulate. For example, threshold = threshold_param * max_remaining_evidence - 0.05 * n_remaining. This would allow lower-validity features to collectively contribute without creating an arbitrary per-feature normalization.

4. **Do not narrow beta or epsilon further:** The variance issue in iter 1 (15× higher than observed) was real but is not solved by making the model deterministic. Instead, focus on the threshold_param tuning to allow the model to find a more natural middle ground that produces less variance naturally, not by constraint.

5. **Validate against iter 1 trajectory:** The proposer should verify that the new threshold_param range and formula produce predictions closer to iter 1's 0.5096 (Exp 1) and 0.6942 (Exp 2) as a sanity check before exploring further.

**Why this stays in the prescribed family:**
Validity-weighted accumulation with a soft threshold is the arbiter's mandated mechanism. The fix is a re-calibration of the threshold formula and parameter bounds, not a switch to a different decision logic. The core loop (process features in validity order, accumulate weighted evidence, check soft threshold) remains intact.

**Outcome of this advice:** iter 2 candidate loss=0.8311 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Validity-Weighted Accumulation with Adaptive Soft Threshold. Subjects accumulate evidence for each option by processing features in descending order of validity. Each feature contributes to a running score weighted by its validity. A soft stopping rule checks at each step whether the accumulated evidence for one option exceeds a threshold proportional to the maximum remaining evidence. The threshold parameter controls the stopping criterion: lower values cause earlier stopping (more TTB-like), higher values permit continued accumulation (more Tallying-like). Critically, the threshold is applied directly to the absolute lead (not normalized by feature count), allowing the model to naturally balance high-validity features against the collective weight of multiple lower-validity features. This mechanism explains why subjects in Experiment 1 choose based on the top cue only ~38% of the time (not TTB's 82%) while still respecting validity structure, and why Experiment 2 shows ~72% tally consistency (intermediate between TTB's 14% and Tallying's 88%).",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n    \n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Validity-Weighted Accumulation expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    \n    n_features = stim.shape[1]\n    a, b = stim[0], stim[1]\n    \n    # Extract parameters\n    val = np.asarray(parameters[\"validities\"], dtype=float)\n    if val.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {val.shape[0]} != n_features {n_features}.\"\n        )\n    \n    threshold_param = float(parameters[\"threshold_param\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n    \n    # Sort features by descending validity\n    cue_order = np.argsort(-val, kind=\"stable\").tolist()\n    \n    # Compute maximum possible evidence (sum of all validities)\n    max_evidence = np.sum(val)\n    \n    # Accumulate evidence feature by feature\n    score_a = 0.0\n    score_b = 0.0\n    \n    for idx, j in enumerate(cue_order):\n        # Add validity-weighted contribution for this feature\n        if a[j] > b[j]:\n            score_a += val[j]\n        elif b[j] > a[j]:\n            score_b += val[j]\n        # If a[j] == b[j], neither option gains evidence\n        \n        # Compute remaining maximum evidence (features not yet processed)\n        remaining_features = cue_order[idx + 1:]\n        max_remaining = np.sum(val[remaining_features])\n        \n        # Soft threshold: if one option's lead exceeds threshold_param times\n        # the maximum remaining evidence, stop and decide.\n        # NOTE: threshold is applied directly to the absolute lead, not normalized by feature count.\n        lead = abs(score_a - score_b)\n        threshold = threshold_param * max_remaining\n        \n        if lead > threshold:\n            # Decision threshold exceeded; choose based on current scores\n            if score_a > score_b:\n                scores = np.array([1.0, 0.0])\n            else:\n                scores = np.array([0.0, 1.0])\n            \n            # Apply softmax noise and lapse\n            z = beta * (scores - scores.max())\n            e = np.exp(z)\n            p_core = e / e.sum()\n            return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)\n    \n    # If all features processed without exceeding threshold, choose based on final scores\n    if score_a > score_b:\n        scores = np.array([1.0, 0.0])\n    elif score_b > score_a:\n        scores = np.array([0.0, 1.0])\n    else:\n        # Tie: no preference\n        return np.ones(2) / 2.0\n    \n    # Apply softmax noise and lapse\n    z = beta * (scores - scores.max())\n    e = np.exp(z)\n    p_core = e / e.sum()\n    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)",
  "policy": "def policy(probabilities):\n    import numpy as np\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "beta": "[0.1, 20.0]",
    "epsilon": "[0.0, 0.5]",
    "validities": "validities",
    "threshold_param": "[0.2, 0.8]"
  },
  "rationale": "MINIMAL-DIFF EDIT: The only change from the PREVIOUS CANDIDATE (iter 1, loss=0.1420, which was ACCEPTED) is the threshold_param range: from [0.1, 2.0] to [0.2, 0.8]. This directly addresses the iter 2 failure and the iter 1 feedback.\n\n**Why this edit succeeds where iter 2 failed:**\n\nIter 2 attempted to narrow the range to [0.4, 1.2] AND introduced a per-feature normalization (dividing by n_remaining + 1.0). This dual change caused a catastrophic regression: the normalization made the effective threshold *higher* per remaining feature, causing the model to stop even more rigidly after the top feature, and the narrow lower bound [0.4] prevented compensation. Result: Exp 1 jumped from 0.5096 to 0.8475 (now worse than TTB), and Exp 2 collapsed to 0.1420 (pure TTB).\n\nThis edit REVERTS the per-feature normalization (keeping the original, simpler threshold formula: threshold = threshold_param * max_remaining_evidence) and applies a more modest adjustment to the parameter range: [0.2, 0.8]. This range is:\n- **Lower minimum (0.2 vs. 0.1)**: Slightly tighter than iter 1, reducing parameter drift and cross-subject variance (addressing the 15× variance inflation in iter 1).\n- **Lower maximum (0.8 vs. 2.0)**: Prevents threshold_param from being so high that early stopping is avoided entirely, keeping the mechanism grounded in validity structure.\n- **Still permissive enough**: The range [0.2, 0.8] allows the model to tune across a meaningful space. At threshold_param = 0.2, the model requires the lead to exceed only 20% of remaining evidence before stopping (favoring early stopping, TTB-like). At threshold_param = 0.8, the model requires the lead to exceed 80% of remaining evidence (favoring continued accumulation, Tallying-like).\n\n**Predicted effect:**\n- **Experiment 1**: With threshold_param tuned in [0.2, 0.8], the model should produce predictions closer to the observed 0.3842 (top-cue consistency) than iter 1's 0.5096. The lower maximum (0.8 vs. 2.0) reduces the model's tendency to stop after the top feature, allowing lower-validity features [0.7, 0.7, 0.7] (total weight 2.1) to collectively override a weak lead from the top feature [0.9]. This is the intended behavior of validity-weighted accumulation.\n- **Experiment 2**: With the narrower range, the model should remain close to iter 1's 0.6942 (tally consistency) while reducing variance. The lower maximum prevents the model from accumulating so much that it ignores the high-validity top feature entirely.\n- **Variance**: The tighter range [0.2, 0.8] should reduce the 15× variance inflation observed in iter 1 by constraining parameter heterogeneity, without making the model deterministic (as iter 2's narrow [0.4, 1.2] range did).\n\n**Why not a larger rewrite:**\nThe core mechanism (validity-weighted accumulation with soft threshold) is sound and was ACCEPTED in iter 1. The problem in iter 2 was not the mechanism but the parameterization: the per-feature normalization was a misdirected fix that broke the threshold logic, and the narrow range [0.4, 1.2] was too conservative. Reverting the normalization and applying a modest range adjustment ([0.2, 0.8]) is the minimal edit that preserves the mechanism while addressing both the Experiment 1 overfitting and the variance issue without the catastrophic regression seen in iter 2.\n\n**Mechanism integrity:**\nThe threshold formula remains threshold = threshold_param * max_remaining_evidence (simple, interpretable, and validated in iter 1). The accumulation loop and softmax noise remain unchanged. Only the sampling range for threshold_param is adjusted, which is a parameter-space edit, not a mechanism-space edit."
}
```
```

## Usage

```json
{
  "input_tokens": 12250,
  "output_tokens": 2327
}
```
