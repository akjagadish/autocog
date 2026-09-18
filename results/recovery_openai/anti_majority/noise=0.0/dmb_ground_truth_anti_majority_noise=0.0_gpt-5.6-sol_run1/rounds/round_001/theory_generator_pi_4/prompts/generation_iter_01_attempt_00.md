# generation_iter_01_attempt_00

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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace pi_3 with a new stochastic priority-search theory rather than another threshold-plus-integration hybrid. A strong competitor to fixed-order TTB would posit that communicated validities create subject-specific, noisy cue priorities: subjects form persistent subjective validity representations, and trial-level attention or retrieval noise can perturb which cue is inspected first. The decision then stops at the first discriminating inspected cue, without any fallback that sums the remaining cues. Stable heterogeneity in validity sensitivity, priority noise, response temperature, and lapse rate can account for the elevated between-subject variance in Experiment 2, while the noncompensatory stopping rule predicts the observed absence of aggregate-cue reversals and tally-margin gradients. Unlike pi_1, this theory should predict systematic departures from the stated validity order when validity differences are small, but near-deterministic ordering when differences are large. It should be tested with designs that orthogonally vary adjacent validity gaps, display order, and repetition while holding cue conflict constant. This would preserve the successful lexicographic signature across the present experiments while creating a meaningful new contrast with pi_1, rather than retaining pi_3's empirically unsupported compensatory fallback.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 1.000)

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
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0736)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.2400)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.1670 (var=0.0108)


---

### `pi_3` (overall score: 0.320)

**Description**
Confidence-Capped Cascade Integration proposes that people translate each communicated cue validity into an absolute diagnosticity signal and inspect cues from most to least valid. A discriminating cue controls the decision immediately when its diagnosticity exceeds a subject-specific confidence threshold. Earlier weak cues do not force premature stopping: search continues until a sufficiently diagnostic cue is encountered. If no discriminating cue reaches threshold, the decision maker switches to compensatory integration, summing all discriminating cues with weights determined by their stated validities. Thus, the theory produces one-cue dominance when reliable cues are available but graded multi-cue sensitivity when every cue is weak. Subject heterogeneity arises from differences in confidence threshold, sensitivity to stated validity, integration weighting, response temperature, and lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Confidence-Capped Cascade Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got shape {validities.shape}."
        )

    threshold = float(parameters["confidence_threshold"])
    sensitivity = float(parameters["validity_sensitivity"])
    integration_exponent = float(parameters["integration_exponent"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # A validity is the communicated probability that a cue points toward
    # the superior option. Its log odds are therefore an absolute measure
    # of the cue's diagnosticity. Clipping handles validity exactly equal
    # to 0.5 or 1 without numerical singularities.
    v = np.clip(validities, 0.5, 1.0 - 1e-9)
    log_diagnosticity = np.log(v / (1.0 - v))
    subjective_confidence = sensitivity * log_diagnosticity

    # Stable sorting preserves the displayed feature order when two cues
    # have equal communicated validity.
    cue_order = np.argsort(-validities, kind="stable")
    differences = stim[0] - stim[1]

    decision_evidence = None
    for j in cue_order:
        if differences[j] == 0.0:
            continue
        if subjective_confidence[j] >= threshold:
            # The first sufficiently diagnostic discriminating cue caps
            # search. Positive evidence favors A and negative favors B.
            decision_evidence = (
                np.sign(differences[j]) * subjective_confidence[j]
            )
            break

    if decision_evidence is None:
        # When no cue is individually convincing, combine all available
        # weak evidence. The exponent controls how sharply integration
        # privileges validity differences.
        weights = np.power(
            np.maximum(log_diagnosticity, 1e-12), integration_exponent
        )
        decision_evidence = float(
            np.sum(weights * np.sign(differences))
        )

    # Symmetric utilities ensure exact indifference when integrated evidence
    # is zero. Positive evidence favors action 0 (A), negative favors 1 (B).
    logits = beta * np.array(
        [0.5 * decision_evidence, -0.5 * decision_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- confidence_threshold: [0.05, 0.2]
- validity_sensitivity: [0.75, 1.5]
- integration_exponent: [0.5, 2.5]
- beta: [4.0, 16.0]
- epsilon: [0.0, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0900)
- Experiment 3: real=0.0000 (var=0.0000) vs this=1.0000 (var=0.1056)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.5913 (var=0.0137)


---

### `pi_2` (overall score: 0.000)

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
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.0000 (var=0.0196)
- Experiment 2: real=0.0000 (var=0.0000) vs this=1.0000 (var=0.1204)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.8492 (var=0.0103)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0059 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0059 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify whether choices are predominantly controlled by the highest-ranked
    # discriminating cue on trials where at least three net lower-cue votes point
    # in the opposite direction.
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        nz = np.flatnonzero(d != 0)
        if nz.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        first = int(nz[0])
        ttb_sign = 1.0 if d[first] > 0 else -1.0  # +1 means A, -1 means B
        tally_margin = float(np.sum(np.sign(d)))  # positive means tally favors A

        # Retain only strong conflicts: tally favors the other option by at
        # least three net cue votes. This selects patterns 2, 3, 5, and 6 and
        # their A/B reversals in the specified design.
        if ttb_sign * tally_margin < 0 and abs(tally_margin) >= 3.0:
            ttb_response = 0 if ttb_sign > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

    if len(agreements) == 0:
        return 0.5

    # A categorical dominance index: 1 if the top discriminating cue controls
    # a strict majority of these high-conflict choices, otherwise 0.
    return float(np.mean(agreements) > 0.5)

```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0196)
- pi_3: 1.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """AIC model-selection indicator for graded tally-margin sensitivity."""
    if data is None or len(data) == 0:
        return 0.0

    margins = []
    successes = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signed_margin = int(np.sum(a > b) - np.sum(b > a))
        if signed_margin == 0:
            continue

        winner = 0 if signed_margin > 0 else 1
        margins.append(abs(signed_margin))
        successes.append(float(int(row["response"]) == winner))

    if len(successes) == 0:
        return 0.0

    margins = np.asarray(margins, dtype=int)
    successes = np.asarray(successes, dtype=float)
    levels = np.arange(1, 7, dtype=float)
    n_by_margin = np.asarray([np.sum(margins == m) for m in range(1, 7)], dtype=float)
    k_by_margin = np.asarray(
        [np.sum(successes[margins == m]) for m in range(1, 7)], dtype=float
    )
    keep = n_by_margin > 0
    levels = levels[keep]
    n_by_margin = n_by_margin[keep]
    k_by_margin = k_by_margin[keep]

    if len(levels) < 2:
        return 0.0

    # Best margin-invariant winner probability: the observable implication
    # of TTB for this design. Its parameterization permits probabilities
    # arbitrarily close to the full interval [0.5, 1].
    n = float(np.sum(n_by_margin))
    k = float(np.sum(k_by_margin))
    q_const = float(np.clip(k / n, 0.5 + 1e-12, 1.0 - 1e-12))
    ll_ttb = k * np.log(q_const) + (n - k) * np.log1p(-q_const)

    # Maximize the exact Tallying likelihood over its stated beta and lapse
    # ranges. The grid is denser where sigmoid(beta*m) changes most rapidly.
    beta = np.concatenate(
        [np.linspace(0.1, 3.0, 241), np.linspace(3.1, 20.0, 170)]
    )
    epsilon = np.linspace(0.0, 0.5, 251)

    core = 1.0 / (1.0 + np.exp(-beta[:, None] * levels[None, :]))
    q = (
        (1.0 - epsilon[None, :, None]) * core[:, None, :]
        + epsilon[None, :, None] * 0.5
    )
    q = np.clip(q, 1e-12, 1.0 - 1e-12)

    ll = np.sum(
        k_by_margin[None, None, :] * np.log(q)
        + (n_by_margin - k_by_margin)[None, None, :] * np.log1p(-q),
        axis=2,
    )
    ll_tally = float(np.max(ll))

    # Tallying has one additional identifiable probability-shape parameter.
    # Thus AIC favors it exactly when the likelihood-ratio deviance exceeds 2.
    deviance = 2.0 * (ll_tally - ll_ttb)
    return float(deviance > 2.0)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0000 (var=0.0900) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0736)
- pi_3: 0.0000 (var=0.0900)

### Experiment 3
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return 1 when first-discriminating-cue consistency is at least 90%."""
    if len(data) == 0:
        return float("nan")

    follows_first_cue = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        expected_response = None
        for j in range(min(a.size, b.size)):
            if a[j] > b[j]:
                expected_response = 0
                break
            if b[j] > a[j]:
                expected_response = 1
                break

        if expected_response is not None:
            follows_first_cue.append(int(int(response) == expected_response))

    if len(follows_first_cue) == 0:
        return float("nan")

    consistency = float(np.mean(follows_first_cue))
    return float(consistency >= 0.90)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0000 (var=0.2496) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.2400)
- pi_3: 1.0000 (var=0.1056)
- pi_2: 0.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Proportion of choices following the net direction of all weak cues on
    # trials where that direction opposes Expert 2. These are conflict levels
    # M=2,...,6; the M=1 cancellation trials are omitted.
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        directions = np.sign(a - b)

        # Feature 0 always ties. Features 1 onward are the equally valid weak
        # cues that the advocated theory integrates.
        weak_net = float(np.sum(directions[1:]))
        expert2_direction = float(directions[1])

        # Retain only genuine reversal trials: the weak-cue aggregate points
        # opposite to the first discriminating cue used by TTB.
        if weak_net == 0.0 or expert2_direction == 0.0:
            continue
        if np.sign(weak_net) == np.sign(expert2_direction):
            continue

        integrated_winner = 0 if weak_net > 0.0 else 1
        scores.append(float(int(row["response"]) == integrated_winner))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1620 (var=0.0198)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1455 (var=0.0116) (Δ vs real -0.0165)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5913 (var=0.0137)
- pi_1: 0.1670 (var=0.0108)
- pi_2: 0.8492 (var=0.0103)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Noisy Priority Race Search proposes that communicated cue validities establish persistent but imperfect subjective cue priorities rather than a universally fixed ordering. Each subject transforms validity log-odds into priority, adds a stable cue-specific representation bias, and has a serial-position prior favoring earlier displayed cues. On each trial, independent retrieval fluctuations perturb these priorities. Cues are inspected in the resulting stochastic order, tied cues are skipped, and search stops at the first discriminating cue. Only that cue determines the intended response; evidence from uninspected or lower-priority cues is never integrated. Small validity gaps can therefore produce stable individual departures and trial-level reversals of the stated ordering, whereas large validity gaps produce nearly deterministic lexicographic search. Response execution is separately subject to temperature-controlled errors and occasional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Noisy Priority Race Search expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    persistent_bias = np.asarray(parameters["priority_bias"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )
    if persistent_bias.ndim != 1 or persistent_bias.size != n_features:
        raise ValueError(
            f"priority_bias must have length {n_features}; got {persistent_bias.shape}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    serial_priority = float(parameters["serial_priority"])
    retrieval_noise = float(parameters["retrieval_noise"])
    beta = float(parameters["response_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    # Persistent subjective priority representation. Validity log odds encode
    # communicated diagnosticity. The serial term supplies a weak display-order
    # prior and resolves equal-validity cues without assuming exact equality in
    # their subjective representations.
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-9)
    validity_signal = np.log(v / (1.0 - v))
    serial_signal = (n_features - 1) - np.arange(n_features, dtype=np.float64)
    priority = (
        validity_sensitivity * validity_signal
        + serial_priority * serial_signal
        + persistent_bias
    )

    differences = stim[0] - stim[1]
    discriminating = np.flatnonzero(differences != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # If retrieval shocks are iid Gumbel, the probability that a particular
    # discriminating cue is reached first is a softmax over its priority. Cues
    # that tie on the current trial may be inspected but cannot terminate
    # search, so they do not enter the race among stopping cues.
    race_logits = priority[discriminating] / max(retrieval_noise, 1e-12)
    race_logits = race_logits - np.max(race_logits)
    first_prob = np.exp(race_logits)
    first_prob /= first_prob.sum()

    # Marginalize over which single cue wins the priority race. This is not
    # evidence integration: in every latent search realization exactly one
    # first discriminating cue determines the intended choice.
    favors_a = differences[discriminating] > 0.0
    intended_a = float(np.sum(first_prob[favors_a]))

    # Conditional on a stopping cue, response temperature permits execution
    # errors of equal size in either direction.
    if beta >= 0.0:
        follow_probability = 1.0 / (1.0 + np.exp(-beta))
    else:
        exp_beta = np.exp(beta)
        follow_probability = exp_beta / (1.0 + exp_beta)

    core_a = (
        intended_a * follow_probability
        + (1.0 - intended_a) * (1.0 - follow_probability)
    )
    p_a = (1.0 - lapse_rate) * core_a + lapse_rate * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))

    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- n_features: n_features
- validities: validities
- priority_bias: [(-0.08, 0.08)] * n_features
- validity_sensitivity: [1.0, 5.0]
- serial_priority: [0.15, 0.45]
- retrieval_noise: [0.02, 0.08]
- response_temperature: [0.4, 6.0]
- lapse_rate: [0.0, 0.2]

`rationale`: The model removes pi_3's unsupported threshold and compensatory fallback. Every non-tie decision is generated by a genuinely noncompensatory stopping rule, so additional lower cues cannot reverse a choice after a cue has terminated search. Strong validity and serial-position signals preserve the lexicographic dominance needed in Experiments 1 and 4 and avoid a tally-margin gradient in Experiment 2. Moderate response noise makes the 90% consistency criterion in Experiment 3 difficult to cross while retaining majority control by the leading cue. Unlike fixed-order TTB, the ordering is probabilistic and subject-specific: persistent representation biases can overturn small validity gaps, retrieval noise can occasionally alter inspection order within a subject, and large validity differences overwhelm both sources of variation. The serial prior also predicts display-order effects, especially for equal or nearly equal validities. Because parameter values persist across a subject's run, the mechanism naturally produces stable between-subject heterogeneity without learning from nonexistent outcome feedback.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed stochastic priority-search family and reproduces all four pooled qualitative signatures. It matches Experiments 1–3 exactly at the aggregate level and is close in Experiment 4 (0.1455 versus 0.1620), while greatly improving on the compensatory pi_3 result there. The remaining problem is between-subject fit. In Experiment 3, the observed indicator is uniformly zero across subjects, whereas the candidate variance of 0.2496 implies that many simulated subjects exceed the 90% first-cue-consistency threshold even though the pooled indicator is zero. Experiment 2 shows a smaller version of the same issue: the pooled indicator matches, but candidate variance is 0.0900 rather than the observed 0.0000. Thus broad parameter ranges are generating near-deterministic subjects not present in these data.
Rationale: Keep the pure first-discriminating-cue priority race, but tighten its stochasticity calibration. In particular, eliminate the low-noise/high-priority corner created by retrieval_noise as low as 0.02 combined with high validity sensitivity, serial priority, and response temperature. A modestly higher retrieval-noise floor and/or narrower upper bounds on priority strength should prevent simulated subjects from crossing 90% consistency in Experiment 3. Prefer adjusting retrieval-to-priority ratios over adding substantial generic response error: slightly more priority competition should also raise Experiment 4's reversal rate toward 0.162, whereas excessive lapses could unnecessarily weaken the perfect Experiment 1 dominance result. Recalibrate narrowly around the current accepted model, preserving the aggregate matches and noncompensatory stopping rule while reducing categorical between-subject variance in Experiments 2 and especially 3.

**Outcome of this advice:** iter 1 candidate loss=0.0059 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
{
  "theory": "Noisy Priority Race Search proposes that communicated cue validities establish persistent but imperfect subjective cue priorities rather than a universally fixed ordering. Each subject transforms validity log-odds into priority, adds a stable cue-specific representation bias, and has a serial-position prior favoring earlier displayed cues. On each trial, independent retrieval fluctuations perturb these priorities. Cues are inspected in the resulting stochastic order, tied cues are skipped, and search stops at the first discriminating cue. Only that cue determines the intended response; evidence from uninspected or lower-priority cues is never integrated. Small validity gaps can therefore produce stable individual departures and trial-level reversals of the stated ordering, whereas large validity gaps produce nearly deterministic lexicographic search. Response execution is separately subject to temperature-controlled errors and occasional lapses.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Noisy Priority Race Search expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"Stimulus has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    persistent_bias = np.asarray(parameters[\"priority_bias\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities must have length {n_features}; got {validities.shape}.\"\n        )\n    if persistent_bias.ndim != 1 or persistent_bias.size != n_features:\n        raise ValueError(\n            f\"priority_bias must have length {n_features}; got {persistent_bias.shape}.\"\n        )\n\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    serial_priority = float(parameters[\"serial_priority\"])\n    retrieval_noise = float(parameters[\"retrieval_noise\"])\n    beta = float(parameters[\"response_temperature\"])\n    lapse_rate = float(parameters[\"lapse_rate\"])\n\n    # Persistent subjective priority representation. Validity log odds encode\n    # communicated diagnosticity. The serial term supplies a weak display-order\n    # prior and resolves equal-validity cues without assuming exact equality in\n    # their subjective representations.\n    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-9)\n    validity_signal = np.log(v / (1.0 - v))\n    serial_signal = (n_features - 1) - np.arange(n_features, dtype=np.float64)\n    priority = (\n        validity_sensitivity * validity_signal\n        + serial_priority * serial_signal\n        + persistent_bias\n    )\n\n    differences = stim[0] - stim[1]\n    discriminating = np.flatnonzero(differences != 0.0)\n    if discriminating.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # If retrieval shocks are iid Gumbel, the probability that a particular\n    # discriminating cue is reached first is a softmax over its priority. Cues\n    # that tie on the current trial may be inspected but cannot terminate\n    # search, so they do not enter the race among stopping cues.\n    race_logits = priority[discriminating] / max(retrieval_noise, 1e-12)\n    race_logits = race_logits - np.max(race_logits)\n    first_prob = np.exp(race_logits)\n    first_prob /= first_prob.sum()\n\n    # Marginalize over which single cue wins the priority race. This is not\n    # evidence integration: in every latent search realization exactly one\n    # first discriminating cue determines the intended choice.\n    favors_a = differences[discriminating] > 0.0\n    intended_a = float(np.sum(first_prob[favors_a]))\n\n    # Conditional on a stopping cue, response temperature permits execution\n    # errors of equal size in either direction.\n    if beta >= 0.0:\n        follow_probability = 1.0 / (1.0 + np.exp(-beta))\n    else:\n        exp_beta = np.exp(beta)\n        follow_probability = exp_beta / (1.0 + exp_beta)\n\n    core_a = (\n        intended_a * follow_probability\n        + (1.0 - intended_a) * (1.0 - follow_probability)\n    )\n    p_a = (1.0 - lapse_rate) * core_a + lapse_rate * 0.5\n    p_a = float(np.clip(p_a, 0.0, 1.0))\n\n    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs /= probs.sum()\n    return probs",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "priority_bias": "[(-0.08, 0.08)] * n_features",
    "validity_sensitivity": "[1.0, 5.0]",
    "serial_priority": "[0.15, 0.45]",
    "retrieval_noise": "[0.04, 0.08]",
    "response_temperature": "[1.6, 2.0]",
    "lapse_rate": "[0.0, 0.05]"
  },
  "rationale": "This is a minimal calibration edit that leaves the pure first-discriminating-cue priority race unchanged. Raising the retrieval-noise floor from 0.02 to 0.04 removes the most deterministic priority-race corner while preserving strong lexicographic dominance. Narrowing response temperature to 1.6–2.0 prevents near-errorless subjects from crossing Experiment 3's 90% consistency threshold, while still yielding roughly 83–88% execution fidelity—comfortably above the majority criterion in Experiment 1. The lapse range is reduced to 0–0.05 so this calibration reflects priority competition and bounded execution fidelity rather than broad generic guessing. Together, these changes should reduce the categorical between-subject variance in Experiments 2 and 3 and modestly increase Experiment 4's reversal rate toward 0.162 without introducing any compensatory aggregation."
}
```

## Usage

```json
{
  "input_tokens": 12550,
  "output_tokens": 2997
}
```
