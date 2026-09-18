# generation_iter_07_attempt_00

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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_4`).

Replace pi_4 with a Conflict-Density-Adaptive Evidence theory. The central claim should be that subjects use a task-invariant, stimulus-contingent control rule rather than one fixed search architecture: when only a small number of cues discriminate, they serially inspect cues in a stable accessibility hierarchy dominated by display position and stop at the first discriminating cue; when many cues discriminate, the perceptual display supports parallel comparison and choices are based on a noisy, weakly position-weighted tally of cue support. The transition should depend smoothly on the proportion or count of discriminating cues, not on experiment identity or feedback. Instructed validity should be represented as a weak modifier or tie-breaker rather than a mechanism capable of promoting a late cue by itself. Subject-level parameters should govern the sparse-to-dense transition, positional weighting, tally precision, and lapse rate, with constrained heterogeneity to reflect the low observed variances. The new theory must predict: strong earlier-cue choices for adjacent two-cue conflicts in Experiment 1; little distance-dependent change among sparse two-cue conflicts in Experiment 6; near invariance across the sparse conflict-count manipulation in Experiment 3; reversal by the five-cue coalition in the dense eight-cue trials of Experiment 2; low endorsement of isolated cue 8 against three earlier cues in Experiment 5; and approximately chance agreement with validity-ranked lexicographic choice in Experiment 4. This replacement creates a substantive competitor to pi_3 by attributing its successful coalition effect to density-triggered aggregation while avoiding pi_3's erroneous graded distance effect in sparse conflicts.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.497)

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
- Experiment 1: real=0.0283 (var=0.0097) vs this=0.3208 (var=0.0247)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=0.0079 (var=0.0152)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0031 (var=0.0005)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0125 (var=0.0195)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0680 (var=0.0025)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0000 (var=0.0000)


---

### `pi_3` (overall score: 0.497)

**Description**
Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- accessibility_weights: [(0.48, 0.52)] * n_features
- position_strength: [0.02, 0.15]
- validity_strength: [3.0, 5.0]
- mapping_fidelity: [0.8, 1.0]
- complexity_power: [3.0, 8.0]
- attention_temperature: [0.30, 0.40]
- epsilon: [0.02, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=-0.0025 (var=0.0152)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=-0.0179 (var=0.0142)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0564 (var=0.0042)
- Experiment 4: real=0.0042 (var=0.0061) vs this=0.2100 (var=0.0120)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0571 (var=0.0041)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0112 (var=0.0004)


---

### `pi_4` (overall score: 0.289)

**Description**
Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. The strength of that adjustment is expressed as a narrowly distributed ratio to positional anchoring, preventing independent parameter draws from causing unstable population-wide cue-order reversals. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Accessibility-Anchored Fixed Lexicographic Search.
    # Parameters are fixed for the subject, so the resulting hierarchy is
    # stable across all trials. History is intentionally ignored because no
    # outcome feedback is available to support cue-order learning.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(
        parameters["accessibility_weights"], dtype=float
    )

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != "
            f"n_features {n_features}."
        )

    # Position imposes a linear cost for moving attention to later cues.
    # Raw validity differences are deliberately not converted to ranks:
    # communicated validity is only a weak local adjustment and therefore
    # cannot receive an extra boost when its ordering matches the display.
    position_strength = float(parameters["position_strength"])
    validity_position_ratio = float(parameters["validity_position_ratio"])
    validity_weight = validity_position_ratio * position_strength
    positions = np.arange(n_features, dtype=float)
    centered_validity = validities - float(np.mean(validities))

    priority = (
        -position_strength * positions
        + accessibility
        + validity_weight * centered_validity
    )

    # Stable sorting supplies a deterministic tie rule favoring earlier
    # positions. Since all components are subject-level constants, this order
    # is fixed rather than redrawn on every trial.
    cue_order = np.argsort(-priority, kind="stable")

    a, b = stim[0], stim[1]
    discriminating_order = [int(j) for j in cue_order if a[j] != b[j]]
    if len(discriminating_order) == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def cue_winner(j):
        return 0 if a[j] > b[j] else 1

    first_winner = cue_winner(discriminating_order[0])

    # A rare search-control slip skips the first discriminating cue. This is
    # a tightly constrained perturbation of a fixed hierarchy, not a newly
    # sampled Plackett-Luce ordering.
    order_perturbation = float(parameters["order_perturbation"])
    order_perturbation = float(np.clip(order_perturbation, 0.0, 1.0))
    if len(discriminating_order) >= 2:
        second_winner = cue_winner(discriminating_order[1])
    else:
        second_winner = first_winner

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - order_perturbation
    p_selected[second_winner] += order_perturbation

    # Conditional on the selected cue, response precision controls accidental
    # reversal of its recommendation. The stable logistic form avoids an
    # unconstrained trial-wise attentional softmax.
    response_precision = float(parameters["response_precision"])
    if response_precision >= 0.0:
        response_reliability = 1.0 / (1.0 + np.exp(-response_precision))
    else:
        e = np.exp(response_precision)
        response_reliability = e / (1.0 + e)

    p_core = (
        response_reliability * p_selected
        + (1.0 - response_reliability) * p_selected[::-1]
    )

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- accessibility_weights: [(0.0, 0.03)] * n_features
- position_strength: [0.12, 0.16]
- validity_position_ratio: [3.80, 3.90]
- order_perturbation: [0.0, 0.03]
- response_precision: [1.45, 1.85]
- epsilon: [0.01, 0.04]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=-0.0108 (var=0.0114)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=-0.0963 (var=0.0198)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0084 (var=0.0002)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0004 (var=0.0094)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.9652 (var=0.0010)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.9796 (var=0.0006)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0588 -> ACCEPTED
- iter 2: loss=0.0530 -> ACCEPTED
- iter 3: loss=0.1837 -> REJECTED
- iter 4: loss=0.0683 -> REJECTED
- iter 5: loss=0.0422 -> ACCEPTED
- iter 6: loss=0.0416 -> ACCEPTED
- iter 7: loss=0.0338 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.0338 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code each response as adherence (1) versus non-adherence (0) to the
    # highest-validity cue, and recover the number of lower cues opposing it.
    adherence = []
    conflict_q = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        if a[0] > b[0]:
            top_cue_choice = 0
            q = int(np.sum(b[1:] > a[1:]))
        elif b[0] > a[0]:
            top_cue_choice = 1
            q = int(np.sum(a[1:] > b[1:]))
        else:
            continue

        adherence.append(float(response == top_cue_choice))
        conflict_q.append(q)

    adherence = np.asarray(adherence, dtype=float)
    conflict_q = np.asarray(conflict_q, dtype=int)

    low_conflict = conflict_q == 1
    high_conflict = conflict_q >= 4
    if not np.any(low_conflict) or not np.any(high_conflict):
        return float("nan")

    return float(np.mean(adherence[low_conflict]) -
                 np.mean(adherence[high_conflict]))

```

**Observed (real) value:** 0.0283 (var=0.0097)
**Previous candidate values (this loop):**
  - iter 1: 0.0200 (var=0.0135) (Δ vs real -0.0083)
  - iter 2: -0.0100 (var=0.0188) (Δ vs real -0.0383)
  - iter 3: 0.4383 (var=0.0169) (Δ vs real +0.4100)
  - iter 4: 0.1063 (var=0.0169) (Δ vs real +0.0779)
  - iter 5: 0.0463 (var=0.0187) (Δ vs real +0.0179)
  - iter 6: 0.0183 (var=0.0127) (Δ vs real -0.0100)
  - iter 7 (most recent): 0.0300 (var=0.0152) (Δ vs real +0.0017)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0088 (var=0.0057)
- pi_2: 0.3208 (var=0.0247)
- pi_3: -0.0025 (var=0.0152)
- pi_4: -0.0108 (var=0.0114)

### Experiment 2
**Design**
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    validities = np.asarray([0.67, 0.95, 0.58, 0.82, 0.53, 0.74, 0.61, 0.88], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    signed_agreement = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break

        if winner is not None and pd.notna(row["response"]):
            response = int(row["response"])
            signed_agreement.append(1.0 if response == winner else -1.0)

    return float(np.mean(signed_agreement)) if signed_agreement else float("nan")
```

**Observed (real) value:** -0.0050 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: -0.2067 (var=0.0058) (Δ vs real -0.2017)
  - iter 2: -0.1304 (var=0.0072) (Δ vs real -0.1254)
  - iter 3: -0.0292 (var=0.0101) (Δ vs real -0.0242)
  - iter 4: -0.0708 (var=0.0132) (Δ vs real -0.0658)
  - iter 5: -0.1317 (var=0.0117) (Δ vs real -0.1267)
  - iter 6: -0.1212 (var=0.0083) (Δ vs real -0.1162)
  - iter 7 (most recent): -0.0950 (var=0.0102) (Δ vs real -0.0900)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0079 (var=0.0152)
- pi_1: 0.7113 (var=0.0406)
- pi_3: -0.0179 (var=0.0142)
- pi_4: -0.0963 (var=0.0198)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)

        # Feature 8 (index 7, equivalently the final feature here) is the
        # globally most valid cue in this experimental design.
        j = 7
        if a.size <= j or b.size <= j or a[j] == b[j]:
            continue

        cue8_winner = 0 if a[j] > b[j] else 1
        agreements.append(float(int(row['response']) == cue8_winner))

    if len(agreements) == 0:
        return float('nan')

    cue8_choice_rate = float(np.mean(agreements))

    # Soft classification around a preregistered separating rate. This is
    # smooth (unlike a hard threshold), remains defined per subject, and
    # compresses rates well inside either theory's predicted region.
    z = float(np.clip(20.0 * (cue8_choice_rate - 0.40), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0089 (var=0.0080)
**Previous candidate values (this loop):**
  - iter 1: 0.0067 (var=0.0001) (Δ vs real -0.0022)
  - iter 2: 0.0069 (var=0.0002) (Δ vs real -0.0020)
  - iter 3: 0.0075 (var=0.0001) (Δ vs real -0.0014)
  - iter 4: 0.0075 (var=0.0001) (Δ vs real -0.0014)
  - iter 5: 0.0197 (var=0.0006) (Δ vs real +0.0107)
  - iter 6: 0.0275 (var=0.0008) (Δ vs real +0.0185)
  - iter 7 (most recent): 0.0124 (var=0.0003) (Δ vs real +0.0035)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9999 (var=0.0064)
- pi_3: 0.0564 (var=0.0042)
- pi_2: 0.0031 (var=0.0005)
- pi_4: 0.0084 (var=0.0002)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast choices supported by the higher-validity cue on adjacent
    # conflicts versus conflicts separated by four or seven validity ranks.
    high_choice = {1: [], 4: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = np.flatnonzero(a != b)
        if diff.size != 2:
            continue

        high_idx = int(np.min(diff))  # Validities decrease with feature index.
        low_idx = int(np.max(diff))
        distance = low_idx - high_idx
        if distance not in high_choice:
            continue

        high_option = 0 if a[high_idx] > b[high_idx] else 1
        high_choice[distance].append(
            1.0 if int(row["response"]) == high_option else 0.0
        )

    if any(len(high_choice[d]) == 0 for d in (1, 4, 7)):
        return float("nan")

    adjacent = float(np.mean(high_choice[1]))
    well_separated = 0.5 * (
        float(np.mean(high_choice[4])) + float(np.mean(high_choice[7]))
    )
    return float(well_separated - adjacent)

```

**Observed (real) value:** 0.0042 (var=0.0061)
**Previous candidate values (this loop):**
  - iter 1: -0.0192 (var=0.0059) (Δ vs real -0.0233)
  - iter 2: -0.0054 (var=0.0185) (Δ vs real -0.0096)
  - iter 3: -0.0342 (var=0.0152) (Δ vs real -0.0383)
  - iter 4: -0.0142 (var=0.0163) (Δ vs real -0.0183)
  - iter 5: 0.0088 (var=0.0188) (Δ vs real +0.0046)
  - iter 6: 0.0154 (var=0.0171) (Δ vs real +0.0112)
  - iter 7 (most recent): -0.0113 (var=0.0139) (Δ vs real -0.0154)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2100 (var=0.0120)
- pi_1: -0.0075 (var=0.0062)
- pi_2: -0.0125 (var=0.0195)
- pi_4: -0.0004 (var=0.0094)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Soft prevalence of subjects reliably choosing the later conflicting cue."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        later_cue_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            differing = np.flatnonzero(a != b)
            if differing.size == 0:
                continue

            # In this design, the later of the two discriminating cues is the
            # locally high-validity cue.
            later = int(np.max(differing))
            later_winner = 0 if a[later] > b[later] else 1
            later_cue_choices.append(float(int(row["response"]) == later_winner))

        if len(later_cue_choices) == 0:
            continue

        rate = float(np.mean(later_cue_choices))
        # Smoothly score whether the subject shows a reliable (>65%) rather
        # than near-chance preference for the validity-promoted later cue.
        z = float(np.clip(20.0 * (rate - 0.65), -60.0, 60.0))
        scores.append(1.0 / (1.0 + np.exp(-z)))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0033 (var=0.0001)
**Previous candidate values (this loop):**
  - iter 1: 0.0001 (var=0.0000) (Δ vs real -0.0033)
  - iter 2: 0.0111 (var=0.0001) (Δ vs real +0.0078)
  - iter 3: 0.0027 (var=0.0000) (Δ vs real -0.0007)
  - iter 4: 0.0129 (var=0.0001) (Δ vs real +0.0096)
  - iter 5: 0.0197 (var=0.0006) (Δ vs real +0.0164)
  - iter 6: 0.0133 (var=0.0001) (Δ vs real +0.0100)
  - iter 7 (most recent): 0.0158 (var=0.0003) (Δ vs real +0.0125)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9652 (var=0.0010)
- pi_3: 0.0571 (var=0.0041)
- pi_1: 0.9496 (var=0.0122)
- pi_2: 0.0680 (var=0.0025)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    target_choices = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 2 or b.size < 2:
            continue

        # Features 1 and 2 are aligned in this design. Define the target as
        # the option jointly recommended by these two high-validity cues.
        direction = float(np.sum(a[:2] - b[:2]))
        if direction > 0:
            target_response = 0
        elif direction < 0:
            target_response = 1
        else:
            continue

        response = int(row["response"])
        target_choices.append(float(response == target_response))

    if len(target_choices) == 0:
        return float("nan")

    target_rate = float(np.mean(target_choices))

    # Smoothly classify rates around the otherwise-empty region separating
    # the advocated prediction (roughly .38-.48) from the competitor's
    # prediction (roughly .80-.86). Saturation also limits binomial and
    # parameter-driven between-subject variation away from the boundary.
    z = float(np.clip(20.0 * (target_rate - 0.64), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0005 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 5: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real -0.0004)
  - iter 7 (most recent): 0.0000 (var=0.0000) (Δ vs real -0.0004)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0112 (var=0.0004)
- pi_4: 0.9796 (var=0.0006)
- pi_1: 0.9857 (var=0.0302)
- pi_2: 0.0000 (var=0.0000)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Attention
    # then continues through the residual hierarchy. Conditional on a miss,
    # residual accessibility decays shallowly by ordinal rank. A separate
    # terminal-gap penalty limits access to cues lying very far from the
    # leading cue without reinstating a steep gradient over early/middle cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- transition_count: [7.25, 7.35]
- transition_steepness: [7.0, 8.5]
- position_bias: [0.90, 1.05]
- validity_modifier: [0.01, 0.04]
- serial_second_look: [0.36, 0.40]
- residual_rank_decay: [0.08, 0.12]
- terminal_gap_penalty: [4.0, 5.0]
- serial_precision: [1.65, 1.90]
- positional_discount: [0.06, 0.14]
- tally_precision: [1.05, 1.35]
- epsilon: [0.01, 0.04]

`rationale`: This is a minimal change to the accepted iteration-6 model. The absolute-position residual softmax is replaced by the critic-recommended rank-normalized residual decay, and one narrowly distributed terminal-gap penalty is added. Rank normalization makes the fixed second-look mass substantially more available to early and middle residual cues in Experiment 2 instead of recreating the initial positional ordering. The explicit hinge penalty applies only beyond a normalized gap of 0.55, preventing that redistribution from disproportionately promoting a remote cue 8 in Experiment 3. The transition midpoint, density gate, serial second-look probability, validity modifier, response precision, tally process, and lapse mechanism are unchanged. Experiment 1 remains protected because all residual cues recommend the same coalition option, while Experiments 4 and 5 remain exactly unchanged because a softmax over their single residual cue assigns it probability one regardless of either new accessibility term. Narrow ranges preserve constrained subject heterogeneity.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate captures five of the six experiments reasonably well and is faithful to the prescribed conflict-density-adaptive family. Experiment 1 is close (0.0200 versus 0.0283), Experiment 3 is close (0.0067 versus 0.0089), and the saturated low scores in Experiments 5 and 6 correctly express rejection of the late isolated cue and low endorsement of the first-two-cue target. Experiment 4 has only a modest excess negative distance contrast (-0.0192 versus 0.0042). The decisive failure is Experiment 2: agreement with the validity-ranked lexicographic choice is systematically negative (-0.2067), whereas humans are essentially at chance (-0.0050). Its variance is appropriately low, so this is a consistent directional bias rather than simulation noise.
Rationale: Retain the serial-parallel conflict-density mechanism, but recalibrate the sparse serial component responsible for Experiment 2. With transition_count around 7.3 and very high steepness, the five- and six-discrimination displays in Experiment 2 are treated almost purely serially. The deterministic, nearly exact positional ordering then creates systematic anti-agreement with the validity-ranked winner. Simply lowering serial_precision globally is ill-advised because it would push isolated cue-8 choice toward chance and damage the strong fits in Experiments 3 and 5. Instead, add a small attention/accessibility temperature or constrained subject-stable perturbations to serial cue priorities, while preserving an earlier-position trend and keeping instructed validity weak. This should remove the rigid anti-validity alignment in Experiment 2 and also bring Experiment 4's small negative distance contrast toward zero without abandoning first-discriminating-cue search. If the transition is adjusted, make only a modest smoothing of its steepness rather than a large downward shift in its midpoint, since allowing extensive tallying at five to seven conflicts risks destroying the near-invariance already captured in Experiment 1. The primary target is to raise Experiment 2 by about 0.20 toward zero while preserving the current fits elsewhere.

**Outcome of this advice:** iter 1 candidate loss=0.0588 -> iter 2 candidate loss=0.0530 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-density-adaptive serial–parallel family and improved the aggregate loss. The accepted second-look modification moved Experiment 2 in the correct direction, from -0.2067 to -0.1304, and improved Experiment 4 from -0.0192 to -0.0054. Experiments 3 and 6 remain excellent, and Experiment 4 is now close to its near-zero target. Nevertheless, Experiment 2 still shows a substantial, consistent anti-agreement with the validity-ranked choice: -0.1304 versus the observed -0.0050. Experiment 1 also deteriorated from 0.0200 to -0.0100, whereas the observed contrast is positive at 0.0283, and Experiment 5 moved above its low target (0.0111 versus 0.0033). The elevated variances in Experiments 1 and 4 further indicate more response-level inconsistency than observed.
Rationale: Continue tuning within the same mechanism family, but do not rely on a further increase in serial_second_look. The previous second-look advice was accepted and helped Experiment 2, yet extending it is likely to worsen Experiments 1 and 5 because both depend on reliable early-cue control in sparse displays. The principal remaining miscalibration is the density gate: transition_count around 7.3 with steepness 7–8.5 makes the five- and six-discriminating-cue trials in Experiment 2 almost purely serial, despite being the displays for which the prescribed theory requires substantial parallel aggregation. Lower the transition midpoint into the intermediate conflict-count region, or broaden the transition enough that k=5–6 receives meaningful tally weight while k=2–4 remains predominantly serial. This should move Experiment 2 toward chance agreement and may also restore Experiment 1’s positive low-versus-high conflict contrast by allowing its k>=4 coalition trials progressively more aggregation. Preserve near-pure serial control at k=2 for Experiments 4 and 5 and near-pure tallying at k=8 for Experiment 6. Once the gate carries the Experiment 2 correction, reduce rather than increase serial_second_look modestly to recover the low Experiment 5 score and stronger sparse early-cue adherence. Keep validity weighting weak and avoid changing tally precision aggressively, since Experiments 3 and 6 already fit well.

**Outcome of this advice:** iter 2 candidate loss=0.0530 -> iter 3 candidate loss=0.1837 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is not adequate and was correctly rejected by the accept gate. It improves Experiment 2 substantially, moving validity-ranked agreement from the accepted base’s -0.1304 to -0.0292 against an observed -0.0050, and it preserves excellent fits in Experiments 3, 5, and 6. However, that improvement is purchased by a catastrophic failure in Experiment 1: the low-minus-high conflict contrast rises to 0.4383, compared with 0.0283 in humans and -0.0100 in the accepted base. Lowering the transition midpoint to about 5.45 makes the k=5–7 coalition trials in Experiment 1 strongly tally-controlled, so their many opposing cues overwhelm the early cue, whereas the k=2 trials remain serial. This creates precisely the large conflict-count effect that the data rule out. Experiment 4 also deteriorates from -0.0054 to -0.0342 versus 0.0042 observed. Thus the theory remains faithful to the prescribed family, but its density transition and dense-mode response gain are badly miscalibrated.
Rationale: The previous recommendation to lower and broaden the density transition was rejected by the gate, so do not repeat that large midpoint shift. Build on the unchanged iteration-2 base and preserve its predominantly serial treatment of Experiment 1’s k=5–7 displays, its second-look range, and its good Experiment 4 fit. Try a different in-family knob aimed at Experiment 2’s rigid anti-agreement: introduce a constrained conflict-density-dependent temperature or stochastic softening of serial cue selection, increasing smoothly with the number of discriminating cues while leaving k=2 essentially unchanged. This can weaken the deterministic positional anti-alignment on Experiment 2’s intermediate-density displays without forcing them into a high-gain tally that reverses the lopsided coalitions in Experiment 1. Alternatively, if any parallel contribution is introduced at k=5–6, normalize the signed tally by conflict count and substantially lower its effective precision there, so it moves balanced displays toward chance rather than strongly reversing one-versus-many displays. Keep full dense aggregation concentrated near k=8 to preserve Experiment 6. The next candidate should prioritize retaining Experiment 1 near zero and Experiment 4 near zero while raising Experiment 2 from -0.1304; the current near-perfect Experiment 2 fit is not worth the 0.41 overshoot in Experiment 1.

**Outcome of this advice:** iter 3 candidate loss=0.1837 -> iter 4 candidate loss=0.0683 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains within the prescribed conflict-density-adaptive serial–parallel family, but it is not an improvement over the accepted iteration-2 base and was correctly rejected by the gate. The new density softening partially repairs Experiment 2, moving agreement from -0.1304 to -0.0708 against -0.0050 observed, but it induces an excessive conflict-count effect in Experiment 1: 0.1063 versus 0.0283 observed and -0.0100 in the accepted base. It also worsens Experiment 4 (-0.0142 versus 0.0042) and Experiment 5 (0.0129 versus 0.0033), while Experiments 3 and 6 remain excellent. Thus, the remaining central failure is Experiment 2’s systematic anti-agreement, but uniform density-dependent softening is too blunt: because it varies strongly across k, it necessarily separates Experiment 1’s low- and high-conflict conditions. Between-subject variances in Experiments 1, 2, and 4 also remain higher than observed.
Rationale: Build on the unchanged iteration-2 base and remove the rejected serial-density-softening addition. The prior large downward transition shift was rejected, and the current density-dependent softening was also rejected, so neither direction should be repeated. Instead, revise the already-helpful accessibility-failure process without increasing its overall probability: when the hierarchy-leading discriminating cue is unavailable, do not always select exactly the second discriminating cue. Distribute the existing serial_second_look mass over the remaining discriminating cues using a constrained position-decaying softmax, or model cascading cue availability until an accessible cue is found. This remains a serial positional hierarchy with weak validity and preserves k=2 behavior exactly. It should also leave Experiment 1 nearly invariant across conflict count because all cues after cue 1 support the same opposing option, while reducing Experiment 2’s rigid anti-agreement because its later discriminating cues support mixed options. Experiment 3 should remain low because its three early cues jointly oppose cue 8, and Experiments 4–5 should be protected because their two-cue trials reduce to the accepted first-versus-second process. Keep the transition near the accepted high midpoint and full tallying near k=8. Use narrow ranges for the new residual-position decay and existing accessibility rate to reduce the excessive variances rather than adding another broad source of subject heterogeneity.

**Outcome of this advice:** iter 4 candidate loss=0.0683 -> iter 5 candidate loss=0.0422 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed conflict-density-adaptive serial–parallel family and is the best iteration so far. The accepted residual-accessibility change yields good point estimates in Experiments 1 (0.0463 versus 0.0283), 3 (0.0197 versus 0.0089), 4 (0.0088 versus 0.0042), and 6 (approximately 0 versus 0.0005). It therefore captures near-invariance across conflict count, negligible sparse distance effects, rejection of isolated late cues, and dense coalition behavior. However, it still fails a central prescribed prediction in Experiment 2: agreement with validity-ranked lexicographic choice remains systematically negative (-0.1317 versus -0.0050), essentially unchanged from iteration 2. The residual redistribution improved aggregate loss mainly by repairing Experiments 1 and 4, not by solving the anti-agreement it was intended to address. Experiment 5 is also somewhat too high (0.0197 versus 0.0033), and variances in Experiments 1, 2, and 4 remain above the observed values.
Rationale: Retain the accepted architecture and its high transition midpoint; the previously attempted lower transition and generic density-dependent softening were both rejected and should not be revisited. Also do not increase serial_second_look, because that risks worsening sparse early-cue control. Instead, tune the accepted residual-selection process itself. Its current absolute-position decay of 0.25–0.30 remains strong enough that accessibility failures mostly reproduce the same early positional anti-alignment in Experiment 2. Try a modestly flatter or rank-normalized residual softmax, with narrow heterogeneity, so the fixed second-look mass can reach a broader mixture of residual cues on the five- and six-conflict displays. Experiment 1 should remain protected because all residual cues support the same option, while Experiments 4 and 5 remain exactly two-cue cases. Because excessive flattening could increase cue-8 endorsement in Experiment 3, constrain the change and, if necessary, retain a small extra penalty for very large positional gaps rather than restoring a globally steep decay. A weak bounded validity term may be added to residual accessibility as a tie-breaker, but it must remain too small to promote cue 8 against several earlier cues. The primary acceptance target is a material movement of Experiment 2 toward zero while preserving Experiment 1 near zero and keeping Experiments 3 and 5 in their low-endorsal region.

**Outcome of this advice:** iter 5 candidate loss=0.0422 -> iter 6 candidate loss=0.0416 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-density-adaptive serial–parallel family and achieves the best aggregate loss so far. It captures Experiment 1’s near-zero conflict-count contrast (0.0183 versus 0.0283), Experiment 4’s negligible distance effect (0.0154 versus 0.0042), low endorsement of the isolated late cue in Experiment 5 (0.0133 versus 0.0033), and the saturated dense-display result in Experiment 6. Experiment 3 remains qualitatively correct but has drifted upward (0.0275 versus 0.0089). The decisive unresolved failure is Experiment 2: validity-ranked agreement remains consistently negative at -0.1212 rather than approximately zero (-0.0050). The flatter residual gradient moved this metric only slightly from -0.1317, while worsening Experiment 3, showing that globally flattening the absolute-position decay is reaching a poor tradeoff. The low Experiment 2 variance indicates a systematic mechanism bias rather than noise.
Rationale: Retain the accepted architecture, high transition midpoint, fixed second-look probability, and weak validity influence. The previously rejected lower transition and generic density-dependent softening should not be revisited, and the newly accepted flattening direction should not simply be extended toward zero decay: it produced only a small Experiment 2 gain while increasing cue-8 endorsement in Experiment 3. Instead, change the shape—not merely the strength—of residual accessibility. Use a rank-normalized residual decay that spreads second-look mass across the early and middle residual cues on Experiment 2’s mixed-support displays, while retaining a small explicit terminal-gap penalty for cues separated by a very large positional gap. This should prevent cue 8 from receiving the excess residual mass responsible for Experiment 3’s deterioration without reverting to the steep absolute-position gradient that generated Experiment 2’s anti-agreement. Keep the overall accessibility-failure mass unchanged so Experiment 1 remains protected and two-cue Experiments 4–5 remain structurally unchanged. Use narrow parameter ranges to preserve the observed low variance. Acceptance should require a material reduction of Experiment 2’s roughly -0.12 bias without further increasing Experiment 3.

**Outcome of this advice:** iter 6 candidate loss=0.0416 -> iter 7 candidate loss=0.0338 -> the gate marked it ACCEPTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted rank-normalized residual rule with a terminal-gap penalty materially improves the model and remains faithful to the prescribed conflict-density-adaptive family. Experiment 1 is now essentially exact (0.0300 versus 0.0283), Experiment 3 is close again (0.0124 versus 0.0089), and Experiment 6 is correctly saturated near zero. Experiment 4 shows only a modest residual distance contrast (-0.0113 versus 0.0042), while Experiment 5 remains qualitatively correct but overestimates reliable later-cue choice (0.0158 versus 0.0033). The main unresolved failure is still Experiment 2: agreement with validity-ranked choice is systematically negative (-0.0950) rather than approximately at chance (-0.0050). Although the latest accepted change improved this metric from -0.1212, a sizable directional bias remains, and its variance is also about twice the observed variance. Thus the candidate captures most cross-experiment invariances but not the prescribed near-chance result on Experiment 2’s relatively dense mixed-support displays.
Rationale: Build on the accepted rank-normalized-plus-terminal-gap architecture; the latest advice was successful and should not be reversed. Do not lower the transition midpoint or reintroduce generic density-dependent softening, since both previously failed by producing excessive conflict-count effects in Experiment 1. Instead, make a small stimulus-contingent adjustment within the residual-accessibility process: let the terminal-gap penalty or residual-rank decay weaken smoothly as the number of residual discriminating cues increases. Keep the total serial_second_look mass fixed. This would preserve the strong terminal suppression at k=4 needed to keep cue 8 low in Experiment 3, while allowing the same second-look mass to spread more broadly across Experiment 2’s larger mixed-support residual sets. Experiment 1 should be protected because all residual cues favor the same coalition option, and the two-cue Experiments 4 and 5 are unaffected because they contain only one residual cue. Use a narrow range for this count modulation and retain the high dense-mode transition, weak validity modifier, and existing precision parameters. The immediate target is to move Experiment 2 materially closer to zero without sacrificing the now-excellent fits in Experiments 1 and 3.

**Outcome of this advice:** iter 7 candidate loss=0.0338 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. A terminal-gap penalty selectively reduces access to residual cues separated from the leading cue by a very large positional gap. As the residual conflict set becomes larger, both restrictions relax smoothly because a denser set supports broader attentional sampling even within serial control. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    # Conflict-Density Adaptive Serial-Parallel Control.\n    # History is intentionally unused: without outcome feedback, the stable\n    # accessibility hierarchy and stimulus-triggered control rule do not learn\n    # across trials.\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.shape != (n_features,):\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    diff = a - b\n    discriminating = np.flatnonzero(diff != 0.0)\n    k = int(discriminating.size)\n    if k == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Put validity on a bounded scale. Its influence remains weak in both\n    # modes and therefore cannot by itself overcome a full position step.\n    v_range = float(np.max(validities) - np.min(validities))\n    if np.isfinite(v_range) and v_range > 0.0:\n        v_scaled = (validities - np.mean(validities)) / v_range\n    else:\n        v_scaled = np.zeros(n_features, dtype=np.float64)\n\n    validity_modifier = float(parameters[\"validity_modifier\"])\n\n    # Sparse mode: stable, position-dominated serial hierarchy. The validity\n    # adjustment is much smaller than the priority difference between adjacent\n    # display positions, so it serves only as a local tie-breaker.\n    position_bias = float(parameters[\"position_bias\"])\n    positions = np.arange(n_features, dtype=np.float64)\n    serial_priority = -position_bias * positions + validity_modifier * v_scaled\n    serial_order = np.argsort(-serial_priority, kind=\"stable\")\n\n    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]\n    first_discriminating = discriminating_order[0]\n    first_winner = 0 if diff[first_discriminating] > 0.0 else 1\n\n    # The hierarchy itself remains stable, but a constrained accessibility\n    # failure sometimes omits its leading discriminating member. Attention\n    # then continues through the residual hierarchy. Conditional on a miss,\n    # residual accessibility decays shallowly by ordinal rank. A separate\n    # terminal-gap penalty limits access to very remote cues. Both constraints\n    # relax smoothly when the residual conflict set is large enough to support\n    # broader sampling, while remaining unchanged for k <= 4.\n    serial_second_look = float(np.clip(\n        parameters[\"serial_second_look\"], 0.0, 1.0\n    ))\n    p_selected = np.zeros(2, dtype=np.float64)\n    p_selected[first_winner] += 1.0 - serial_second_look\n\n    if len(discriminating_order) == 1:\n        p_selected[first_winner] += serial_second_look\n    else:\n        residual = np.asarray(discriminating_order[1:], dtype=int)\n        residual_rank_decay = float(parameters[\"residual_rank_decay\"])\n        terminal_gap_penalty = float(parameters[\"terminal_gap_penalty\"])\n        residual_count_relief = float(parameters[\"residual_count_relief\"])\n\n        # Preserve the accepted k=4 terminal suppression used for isolated\n        # late cues, but broaden residual access smoothly at k=5 and above.\n        count_relaxation = np.exp(\n            -residual_count_relief * max(float(k) - 4.0, 0.0)\n        )\n        residual_rank_decay *= count_relaxation\n        terminal_gap_penalty *= count_relaxation\n\n        if residual.size > 1:\n            residual_rank = np.arange(\n                residual.size, dtype=np.float64\n            ) / float(residual.size - 1)\n        else:\n            residual_rank = np.zeros(1, dtype=np.float64)\n\n        if n_features > 1:\n            normalized_gap = (\n                positions[residual] - positions[first_discriminating]\n            ) / float(n_features - 1)\n        else:\n            normalized_gap = np.zeros(residual.size, dtype=np.float64)\n\n        large_gap = np.maximum(normalized_gap - 0.55, 0.0)\n        residual_logits = (\n            -residual_rank_decay * residual_rank\n            -terminal_gap_penalty * large_gap\n        )\n        residual_logits -= np.max(residual_logits)\n        residual_probs = np.exp(residual_logits)\n        residual_probs /= residual_probs.sum()\n\n        for j, cue_prob in zip(residual, residual_probs):\n            residual_winner = 0 if diff[j] > 0.0 else 1\n            p_selected[residual_winner] += serial_second_look * cue_prob\n\n    serial_precision = float(parameters[\"serial_precision\"])\n    if serial_precision >= 0.0:\n        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))\n    else:\n        ex = np.exp(serial_precision)\n        serial_reliability = ex / (1.0 + ex)\n\n    p_serial = (\n        serial_reliability * p_selected\n        + (1.0 - serial_reliability) * p_selected[::-1]\n    )\n\n    # Dense mode: all discriminating cues contribute in parallel. Position\n    # produces only a shallow gradient here, and validity is a weak multiplier\n    # rather than a lexicographic ordering mechanism.\n    if n_features > 1:\n        normalized_position = positions / float(n_features - 1)\n    else:\n        normalized_position = np.zeros(1, dtype=np.float64)\n\n    positional_discount = float(parameters[\"positional_discount\"])\n    tally_weights = np.exp(-positional_discount * normalized_position)\n    tally_weights *= 1.0 + validity_modifier * v_scaled\n    tally_weights = np.clip(tally_weights, 1e-8, None)\n\n    signed_tally = float(np.dot(tally_weights, diff))\n    tally_precision = float(parameters[\"tally_precision\"])\n    tally_logits = np.array(\n        [tally_precision * signed_tally, 0.0], dtype=np.float64\n    )\n    tally_logits -= np.max(tally_logits)\n    p_tally = np.exp(tally_logits)\n    p_tally /= p_tally.sum()\n\n    # A smooth capacity transition makes the architecture contingent on the\n    # current conflict count. The narrow parameter ranges encode constrained\n    # subject heterogeneity while preserving a common control regime.\n    transition_count = float(parameters[\"transition_count\"])\n    transition_steepness = float(parameters[\"transition_steepness\"])\n    transition_argument = float(np.clip(\n        transition_steepness * (float(k) - transition_count), -60.0, 60.0\n    ))\n    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))\n\n    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally\n\n    epsilon = float(np.clip(parameters[\"epsilon\"], 0.0, 1.0))\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "transition_count": "[7.25, 7.35]",
    "transition_steepness": "[7.0, 8.5]",
    "position_bias": "[0.90, 1.05]",
    "validity_modifier": "[0.01, 0.04]",
    "serial_second_look": "[0.36, 0.40]",
    "residual_rank_decay": "[0.08, 0.12]",
    "terminal_gap_penalty": "[4.0, 5.0]",
    "residual_count_relief": "[0.65, 0.80]",
    "serial_precision": "[1.65, 1.90]",
    "positional_discount": "[0.06, 0.14]",
    "tally_precision": "[1.05, 1.35]",
    "epsilon": "[0.01, 0.04]"
  },
  "rationale": "This is a minimal edit to the accepted iteration-7 model. It adds one narrowly distributed residual_count_relief parameter and uses it to attenuate the existing residual-rank and terminal-gap penalties only when the total number of discriminating cues exceeds four. Consequently, Experiment 3's k=4 displays retain exactly the accepted terminal suppression of remote cue 8, while Experiment 2's k=5–6 mixed-support displays distribute the unchanged second-look mass more broadly across residual cues and should move validity-ranked agreement toward chance. Experiment 1 is protected because all residual cues favor the same coalition option, regardless of how second-look mass is distributed. Experiments 4 and 5 are unchanged because their single residual cue receives probability one, and the high transition midpoint and dense tally process remain unchanged for Experiment 6. The narrow relief range limits added subject heterogeneity."
}
```

## Usage

```json
{
  "input_tokens": 21535,
  "output_tokens": 2951
}
```
