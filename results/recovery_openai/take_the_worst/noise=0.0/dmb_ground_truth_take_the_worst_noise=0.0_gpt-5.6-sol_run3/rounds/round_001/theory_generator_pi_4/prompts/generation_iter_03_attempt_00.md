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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 with an Accessibility-Anchored Fixed Lexicographic theory. Subjects should maintain a stable cue hierarchy determined primarily by display position, perceptual salience, and long-run accessibility rather than sorting cues by communicated validity. On each trial they inspect cues in this stable order and stop at the first discriminating cue; occasional order perturbations and response lapses may occur, but communicated validities should exert only a weak subject-level adjustment and should not be amplified merely because validity rank happens to align with position. This differs from pi_1 by denying that instructed validity determines the search order, and from pi_3 by rejecting mapping-simplicity gating and substantial trial-wise Plackett-Luce order variability. Its concentrated fixed-order search predicts near-zero cue-8 choice in Experiment 1, no rank-distance gradient in Experiment 2, little or no effect of the number of later conflicting cues in Experiment 3, and near-zero agreement with a non-position-aligned validity hierarchy in Experiment 4. The new theory should include subject-specific but cross-task-stable accessibility weights, strong positional anchoring, low order noise, a small lapse rate, and an optional weak validity-instruction weight. Future discriminating experiments should cross display position and validity while separately estimating absolute early-cue adherence, order stability across repeated trials, and transfer after cue positions are rearranged.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_2` (overall score: 0.293)

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


---

### `pi_3` (overall score: 0.292)

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
- Experiment 1: real=0.0283 (var=0.0097) vs this=0.0088 (var=0.0057)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=0.7113 (var=0.0406)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.9999 (var=0.0064)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0075 (var=0.0062)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0497 -> ACCEPTED
- iter 2: loss=0.0560 -> REJECTED
- iter 3: loss=0.0780 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0497 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.0046 (var=0.0135) (Δ vs real -0.0238)
  - iter 2: -0.0354 (var=0.0100) (Δ vs real -0.0638)
  - iter 3 (most recent): -0.0062 (var=0.0178) (Δ vs real -0.0346)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0088 (var=0.0057)
- pi_2: 0.3208 (var=0.0247)
- pi_3: -0.0025 (var=0.0152)

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
  - iter 1: -0.0979 (var=0.0165) (Δ vs real -0.0929)
  - iter 2: -0.0075 (var=0.0050) (Δ vs real -0.0025)
  - iter 3 (most recent): -0.1204 (var=0.0151) (Δ vs real -0.1154)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0079 (var=0.0152)
- pi_1: 0.7113 (var=0.0406)
- pi_3: -0.0179 (var=0.0142)

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
  - iter 1: 0.0118 (var=0.0003) (Δ vs real +0.0028)
  - iter 2: 0.0084 (var=0.0002) (Δ vs real -0.0005)
  - iter 3 (most recent): 0.0089 (var=0.0002) (Δ vs real -0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9999 (var=0.0064)
- pi_3: 0.0564 (var=0.0042)
- pi_2: 0.0031 (var=0.0005)

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
  - iter 1: 0.0058 (var=0.0097) (Δ vs real +0.0017)
  - iter 2: 0.0133 (var=0.0092) (Δ vs real +0.0092)
  - iter 3 (most recent): -0.0108 (var=0.0083) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2100 (var=0.0120)
- pi_1: -0.0075 (var=0.0062)
- pi_2: -0.0125 (var=0.0195)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

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
    validity_weight = float(parameters["validity_weight"])
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
- validity_weight: [0.45, 0.60]
- order_perturbation: [0.0, 0.03]
- response_precision: [1.45, 1.85]
- epsilon: [0.01, 0.04]

`rationale`: The model replaces validity-ranked search with a concentrated, subject-stable accessibility hierarchy. Positional costs accumulate with display distance, whereas validity enters through bounded raw differences rather than validity ranks or a mapping-simplicity gate. Consequently, validity can sometimes affect the order of neighboring cues but cannot elevate a distant cue such as cue 8 over several early discriminating cues. Subject-specific accessibility offsets represent stable salience differences without creating substantial trial-wise order variability. In Experiment 1, the early cue is selected regardless of how many later cues oppose it, producing essentially no conflict-count gradient. In Experiment 2, the unusually valid second-position cue can locally compete with the first position, but the model does not generate general validity-ranked search or a rank-distance gradient. In Experiment 3, the late cue remains behind multiple early opposing cues; only response imprecision and small lapses produce cue-8 choices, placing the smooth classification metric near zero. In Experiment 4, position and validity distance are not converted into stronger evidence: the fixed first-discriminating rule and uniformly rare skip errors predict little difference between adjacent and widely separated conflicts. Unlike the previous stochastic-accessibility theory, no Plackett-Luce hierarchy is resampled each trial, and unlike Take The Best, communicated validity is not the primary determinant of search order.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed accessibility-anchored fixed-lexicographic family and captures its main cross-experiment signatures. Experiment 3 is very well fit: the simulated cue-8 classification is 0.0118 versus 0.0089 observed, showing that a late high-validity cue is not promoted over early cues. Experiment 4 is also nearly exact (0.0058 versus 0.0042), correctly eliminating a validity-rank-distance gradient. Experiment 1 has the correct near-zero qualitative pattern, although it somewhat underpredicts the small observed conflict effect (0.0046 versus 0.0283). The principal failure is Experiment 2: agreement with the validity-ranked cue is too negative (-0.0979 versus -0.0050), and its between-subject variance is excessive (0.0165 versus 0.0051). Thus the model's positional hierarchy is systematically opposing the highest-validity discriminating cue in this stimulus set more than humans do.
Rationale: Because this candidate was accepted and already has low aggregate loss, retain its mechanism and make a targeted calibration rather than redesigning it. The most promising in-family adjustment is to slightly increase the validity-to-position ratio so the exceptionally valid cue at position 2 in Experiment 2 can compete more evenly with position 1. This can be done with a modestly higher validity weight or slightly lower position strength, while preserving a steep cumulative positional barrier that prevents cue 8 from rising in Experiment 3. Narrowing the parameter ranges and reducing accessibility-offset dispersion should also reduce Experiment 2's excessive between-subject variance. Avoid globally lowering response precision: that could move Experiment 2 toward zero but would increase late-cue choices toward chance and likely spoil the excellent Experiment 3 fit. The next version should preserve the near-zero effects in Experiments 1, 3, and 4 while moving Experiment 2 upward by roughly 0.09 toward zero.

**Outcome of this advice:** iter 1 candidate loss=0.0497 -> iter 2 candidate loss=0.0560 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed accessibility-anchored fixed-lexicographic family, and Experiment 3 is essentially exact (0.0084 versus 0.0089), confirming that the late high-validity cue remains inaccessible. Experiment 2 also improved dramatically to -0.0075 versus -0.0050, with nearly exact between-subject variance. However, that gain came from a tightly calibrated validity-to-position ratio that appears to cross a discrete ordering threshold. It worsened Experiment 1 from 0.0046 to -0.0354, reversing the observed small positive conflict effect of 0.0283, and worsened Experiment 4 from 0.0058 to 0.0133. Consequently, aggregate loss increased from 0.0497 to 0.0560 and the candidate was rejected. The problem is not the mechanism family but the brittle population-wide cue-order reversal induced by the stronger validity adjustment.
Rationale: Do not repeat the rejected strategy of further increasing and tightly constraining the validity-to-position ratio. Build on the unchanged accepted base and restore its position strength, validity weight, and accessibility dispersion. Instead, try a modest increase in the existing search-slip/order-perturbation parameter, preferably narrowly centered around roughly 0.05 rather than 0–0.03. In Experiment 2, occasionally consulting the second discriminating cue can soften the accepted model's excessive negative validity agreement without permanently promoting cue 2 for nearly every subject. This knob should be comparatively safe elsewhere: in Experiment 1 the second discriminating cue opposes the first across conflict levels, so a fixed skip probability mainly changes absolute adherence rather than its conflict-count contrast; in Experiment 3 the early discriminating cues generally agree against cue 8, so skipping one should not elevate cue 8; and in Experiment 4 it should affect the two-cue conflicts relatively uniformly across distances. Keep response precision unchanged, because global response noise risks destroying the excellent Experiment 3 fit. This is a more faithful and less threshold-sensitive route to improving Experiment 2 than the rejected validity-weight increase.

**Outcome of this advice:** iter 2 candidate loss=0.0560 -> iter 3 candidate loss=0.0780 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is faithful to the accessibility-anchored fixed-lexicographic family, but it is not an improvement: the gate rejected it at loss 0.0780 versus the accepted base’s 0.0497. Experiment 3 remains essentially exact (0.0089 observed and simulated), so the strong positional barrier against cue 8 is working. The global increase in order perturbation, however, failed in its intended target. Experiment 2 moved farther from zero, from -0.0979 in the accepted base to -0.1204 rather than toward the observed -0.0050; its variance also remains much too high (0.0151 versus 0.0051). Experiment 1 retains a roughly null qualitative effect but has the wrong sign and an enlarged variance (-0.0062, var 0.0178, versus 0.0283, var 0.0097). Experiment 4 also deteriorated from the accepted base’s near-exact 0.0058 to -0.0108 versus 0.0042. Thus an indiscriminate skip to the second discriminating cue is not a benign noise source: across these stimuli it often selects a cue that agrees even less with the validity benchmark and introduces unwanted contrasts.
Rationale: The previous recommendation to raise the global skip probability was rejected by the gate and should not be repeated. Likewise, do not return to the earlier rejected global increase in the validity-to-position ratio. Build instead on the unchanged accepted base and make order perturbation local and priority-gap-sensitive: permit a rare swap only when the first two discriminating cues are nearly tied in accessibility priority, and preferably only when the second cue has the weak validity advantage that would plausibly resolve that local tie. For example, use a small bounded perturbation proportional to exp(-priority_gap/tau), with a narrow tau, rather than a constant skip probability on every trial. This remains low-noise fixed lexicographic search, not a Plackett–Luce hierarchy: position still determines the stable order, while communicated validity only affects genuinely local competition. It specifically targets Experiment 2’s conflict between the early positional cue and the highly valid second-position cue, while avoiding skips when cue 2 is already first or when the next cue is less valid—the cases that evidently drove the current change in the wrong direction. The narrow gap condition should leave Experiment 1’s validity-aligned first cue, Experiment 3’s distant cue 8, and Experiment 4’s rank-distance contrast largely unchanged. Keep response precision and the cumulative positional slope at their accepted-base settings, and constrain the new local perturbation tightly to avoid increasing the already excessive Experiment 2 subject variance.

**Outcome of this advice:** iter 3 candidate loss=0.0780 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips consult the next discriminating cue only when it is almost tied in accessibility priority and has greater communicated validity; bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.",
  "predict": "def predict(parameters, state, history):\n    # Accessibility-Anchored Fixed Lexicographic Search.\n    # Parameters are fixed for the subject, so the resulting hierarchy is\n    # stable across all trials. History is intentionally ignored because no\n    # outcome feedback is available to support cue-order learning.\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    accessibility = np.asarray(\n        parameters[\"accessibility_weights\"], dtype=float\n    )\n\n    if validities.shape != (n_features,):\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n    if accessibility.shape != (n_features,):\n        raise ValueError(\n            f\"accessibility_weights length {accessibility.size} != \"\n            f\"n_features {n_features}.\"\n        )\n\n    # Position imposes a linear cost for moving attention to later cues.\n    # Raw validity differences are deliberately not converted to ranks:\n    # communicated validity is only a weak local adjustment and therefore\n    # cannot receive an extra boost when its ordering matches the display.\n    position_strength = float(parameters[\"position_strength\"])\n    validity_weight = float(parameters[\"validity_weight\"])\n    positions = np.arange(n_features, dtype=float)\n    centered_validity = validities - float(np.mean(validities))\n\n    priority = (\n        -position_strength * positions\n        + accessibility\n        + validity_weight * centered_validity\n    )\n\n    # Stable sorting supplies a deterministic tie rule favoring earlier\n    # positions. Since all components are subject-level constants, this order\n    # is fixed rather than redrawn on every trial.\n    cue_order = np.argsort(-priority, kind=\"stable\")\n\n    a, b = stim[0], stim[1]\n    discriminating_order = [int(j) for j in cue_order if a[j] != b[j]]\n    if len(discriminating_order) == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def cue_winner(j):\n        return 0 if a[j] > b[j] else 1\n\n    first_cue = discriminating_order[0]\n    first_winner = cue_winner(first_cue)\n\n    # A rare search-control slip is permitted only for a genuinely local\n    # priority tie and only when the second cue has a validity advantage.\n    # This smooths a local ordering threshold without resampling the hierarchy.\n    local_perturbation = 0.0\n    if len(discriminating_order) >= 2:\n        second_cue = discriminating_order[1]\n        second_winner = cue_winner(second_cue)\n        if validities[second_cue] > validities[first_cue]:\n            priority_gap = max(0.0, float(priority[first_cue] - priority[second_cue]))\n            priority_gap_tau = float(parameters[\"priority_gap_tau\"])\n            order_perturbation = float(parameters[\"order_perturbation\"])\n            local_perturbation = order_perturbation * np.exp(\n                -priority_gap / priority_gap_tau\n            )\n            local_perturbation = float(np.clip(local_perturbation, 0.0, 1.0))\n    else:\n        second_winner = first_winner\n\n    p_selected = np.zeros(2, dtype=np.float64)\n    p_selected[first_winner] += 1.0 - local_perturbation\n    p_selected[second_winner] += local_perturbation\n\n    # Conditional on the selected cue, response precision controls accidental\n    # reversal of its recommendation. The stable logistic form avoids an\n    # unconstrained trial-wise attentional softmax.\n    response_precision = float(parameters[\"response_precision\"])\n    if response_precision >= 0.0:\n        response_reliability = 1.0 / (1.0 + np.exp(-response_precision))\n    else:\n        e = np.exp(response_precision)\n        response_reliability = e / (1.0 + e)\n\n    p_core = (\n        response_reliability * p_selected\n        + (1.0 - response_reliability) * p_selected[::-1]\n    )\n\n    epsilon = float(np.clip(parameters[\"epsilon\"], 0.0, 1.0))\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "accessibility_weights": "[(0.0, 0.03)] * n_features",
    "position_strength": "[0.12, 0.16]",
    "validity_weight": "[0.45, 0.60]",
    "order_perturbation": "[0.08, 0.12]",
    "priority_gap_tau": "[0.015, 0.03]",
    "response_precision": "[1.45, 1.85]",
    "epsilon": "[0.01, 0.04]"
  },
  "rationale": "This is a minimal edit of the accepted candidate. The fixed subject-level hierarchy, positional slope, validity weight, accessibility dispersion, response precision, and lapse rate are unchanged. Only the indiscriminate skip mechanism is replaced by a local, priority-gap-sensitive perturbation. A second discriminating cue can be consulted only if it has higher instructed validity, and the probability falls exponentially with its priority deficit. This targets Experiment 2's near-tie between the positionally accessible first cue and the unusually valid second-position cue while avoiding the harmful global skips observed in the rejected iteration. The perturbation is effectively absent in Experiment 1 and Experiment 4 when the first cue also has the validity advantage, and it cannot promote cue 8 across the large positional barrier in Experiment 3. Because it probabilistically smooths only a narrow local ordering boundary, it may also reduce the excessive consequences of discrete subject-level hierarchy reversals without introducing Plackett-Luce trial-wise order variability."
}
```

## Usage

```json
{
  "input_tokens": 15041,
  "output_tokens": 2839
}
```
