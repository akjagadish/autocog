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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace pi_5 with a new competitor centered on validity-guided competitive retrieval rather than independent accessibility failures. Communicated validities should establish a generally lexicographic hierarchy, preserving the top-cue dominance seen in Experiments 3, 4, 6, and 7. However, simultaneously active, similar, and closely ranked cues should compete during encoding and retrieval. Same-direction high-validity cues should therefore be capable of producing source confusion, crowding, or mutual inhibition rather than serving as independent monotonic backups. In Experiment 2, adding several adjacent terminal cues would reduce reliable identification or use of the designated 95% cue and increase fallback to the salient early opposing cues, yielding the observed negative redundancy gradient. Absolute late position alone should have weak or no obligatory cost, allowing the near-zero or slightly reversed position effect in Experiment 1. The theory should also include compression or uncertainty in represented validity differences, plus display-based cue competition, so that larger advertised validity gaps need not monotonically increase following of the higher-validity cue and the negative contrast in Experiment 8 can be recovered. Stable subject-level parameters should govern validity compression, interference susceptibility, primacy/recency, and execution noise to accommodate the elevated observed between-subject variance. This is a genuinely different theory from both strict TTB and pi_5: cue interactions alter retrieval fidelity, whereas pi_5 assumes independent accessibility and guaranteed monotonic benefit from redundant supporting cues. It should make preregisterable predictions separating pure position, number of neighboring cues, validity similarity, and cue-direction redundancy.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.924)

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
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.1450 (var=0.0101)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=0.0113 (var=0.0129)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=-0.0015 (var=0.0020)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=0.0169 (var=0.0052)


---

### `pi_5` (overall score: 0.510)

**Description**
Accessibility-Gated Validity Cascade: Communicated validities create a stable, descending cue hierarchy rather than a noisy priority race. On each trial, however, a cue can control choice only if its identity, stated validity, and displayed value are successfully accessible. Accessibility varies stably across people and cues and declines with list length and absolute display position, with a modest accelerating cost for cues near the end of long displays. Search follows the advertised-validity hierarchy among accessible cues and stops at the first accessible cue that discriminates. Lower-ranked evidence is never integrated. If no discriminating cue is accessible, the decision maker guesses. Retrieval failures are distinct from response execution errors: after forming an intended choice, the person can accidentally reverse it or lapse to a random response.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Gated Validity Cascade expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    cue_offsets = np.asarray(
        parameters["cue_accessibility_offsets"], dtype=np.float64
    )
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )
    if cue_offsets.ndim != 1 or cue_offsets.size != n_features:
        raise ValueError(
            f"cue_accessibility_offsets must have length {n_features}; "
            f"got shape {cue_offsets.shape}."
        )

    baseline = float(parameters["baseline_accessibility"])
    list_gradient = float(parameters["list_length_gradient"])
    position_gradient = float(parameters["position_gradient"])
    late_position_cost = float(parameters["late_position_cost"])
    execution_flip = float(parameters["execution_flip"])
    lapse_rate = float(parameters["lapse_rate"])

    # Absolute zero-based display position. The quadratic component is a
    # restrained terminal-display cost: it has little effect on early cues
    # but makes cues near the end of long lists appreciably harder to encode
    # and retrieve. Cue offsets represent stable identity/complexity effects.
    positions = np.arange(n_features, dtype=np.float64)
    access_logits = (
        baseline
        - list_gradient * max(n_features - 1, 0)
        - position_gradient * positions
        - late_position_cost * positions * positions
        + cue_offsets
    )

    # Numerically stable logistic transformation.
    access_prob = np.empty(n_features, dtype=np.float64)
    positive = access_logits >= 0.0
    access_prob[positive] = 1.0 / (1.0 + np.exp(-access_logits[positive]))
    exp_x = np.exp(access_logits[~positive])
    access_prob[~positive] = exp_x / (1.0 + exp_x)
    access_prob = np.clip(access_prob, 1e-9, 1.0 - 1e-9)

    differences = stim[0] - stim[1]
    discriminating = differences != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Advertised validity fixes the search hierarchy. Stable sorting means
    # equal-validity cues retain their displayed order. Accessibility changes
    # whether a cue participates, never its priority relative to another cue.
    cue_order = np.argsort(-validities, kind="stable")

    # Marginalize exactly over independent trial-level accessibility events.
    # reach_probability is the probability that every earlier discriminating
    # cue in the validity hierarchy was inaccessible. Accessible tying cues do
    # not terminate search and therefore leave this quantity unchanged.
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue

        stop_probability = reach_probability * access_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability

        reach_probability *= 1.0 - access_prob[j]

    # If every discriminating cue is inaccessible, no evidence is available.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability

    intended_total = intended_a + intended_b
    if not np.isfinite(intended_total) or intended_total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= intended_total

    # Execution reversal is separate from retrieval failure. A final lapse
    # replaces the executed response with a random choice.
    executed_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
    p_a = (1.0 - lapse_rate) * executed_a + lapse_rate * 0.5
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
- cue_accessibility_offsets: [(-0.15, 0.15)] * n_features
- baseline_accessibility: [3.2, 3.8]
- list_length_gradient: [0.035, 0.055]
- position_gradient: [0.015, 0.035]
- late_position_cost: [0.003, 0.0055]
- execution_flip: [0.09, 0.13]
- lapse_rate: [0.0, 0.02]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0564)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.1204)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.1505 (var=0.0020)
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.1588 (var=0.0035)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=-0.1363 (var=0.0200)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=0.2664 (var=0.0022)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=0.3306 (var=0.0175)


---

### `pi_3` (overall score: 0.448)

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
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.0369 (var=0.0012)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=0.1400 (var=0.0120)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=0.0028 (var=0.0006)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=-0.0113 (var=0.0033)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0733 -> ACCEPTED
- iter 2: loss=0.0576 -> ACCEPTED
- iter 3: loss=0.0380 -> ACCEPTED
- iter 4: loss=0.0574 -> REJECTED
- iter 5: loss=0.0383 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0380 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 5 (most recent): 1.0000 (var=0.0736) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0196)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)

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
  - iter 1: 0.0000 (var=0.0736) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0564) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0736) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 5 (most recent): 0.0000 (var=0.0384) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0736)
- pi_3: 0.0000 (var=0.0900)
- pi_4: 0.0000 (var=0.1056)
- pi_5: 0.0000 (var=0.0564)

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
  - iter 1: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 5 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.2400)
- pi_3: 1.0000 (var=0.1056)
- pi_2: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0564)
- pi_5: 0.0000 (var=0.1204)

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
  - iter 1: 0.1727 (var=0.0019) (Δ vs real +0.0107)
  - iter 2: 0.1560 (var=0.0026) (Δ vs real -0.0060)
  - iter 3: 0.1613 (var=0.0022) (Δ vs real -0.0008)
  - iter 4: 0.1600 (var=0.0021) (Δ vs real -0.0020)
  - iter 5 (most recent): 0.1623 (var=0.0025) (Δ vs real +0.0003)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5913 (var=0.0137)
- pi_1: 0.1670 (var=0.0108)
- pi_2: 0.8492 (var=0.0103)
- pi_4: 0.1683 (var=0.0038)
- pi_5: 0.1505 (var=0.0020)

### Experiment 5
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which the 70%-cue majority overrides the opposing 72% cue
    on the two strongest conflict levels (five or six active 70% cues)."""
    matches = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 9 or b.size < 9:
            continue

        # Features 3--8 (zero-based indices 2:8) are the six 70% cues.
        d70 = a[2:8] - b[2:8]
        active = d70[d70 != 0]
        if active.size < 5:
            continue

        # Retain only designed conflicts in which the active 70% cues agree
        # and the 72% cue recommends the opposite option.
        direction = np.sign(active[0])
        if not np.all(np.sign(active) == direction):
            continue
        d72 = a[8] - b[8]
        if d72 == 0 or np.sign(d72) == direction:
            continue

        majority_choice = 0 if direction > 0 else 1
        response = int(row["response"])
        if response in (0, 1):
            matches.append(float(response == majority_choice))

    if len(matches) == 0:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1613 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.2119 (var=0.0051) (Δ vs real +0.0506)
  - iter 2: 0.2062 (var=0.0065) (Δ vs real +0.0450)
  - iter 3: 0.1825 (var=0.0052) (Δ vs real +0.0212)
  - iter 4: 0.1281 (var=0.0036) (Δ vs real -0.0331)
  - iter 5 (most recent): 0.1794 (var=0.0051) (Δ vs real +0.0181)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1450 (var=0.0101)
- pi_4: 0.8588 (var=0.0039)
- pi_2: 0.8450 (var=0.0101)
- pi_3: 0.0369 (var=0.0012)
- pi_5: 0.1588 (var=0.0035)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868,
        0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901,
        0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177
    ], dtype=float)
    log_odds = np.log(validities / (1.0 - validities))

    gaps = []
    follows_higher = []

    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 2:
            continue

        j0, j1 = int(discriminating[0]), int(discriminating[1])
        if validities[j0] == validities[j1]:
            continue
        high = j0 if validities[j0] > validities[j1] else j1
        low = j1 if high == j0 else j0

        high_favors_a = diff[high] > 0
        chose_a = int(response) == 0
        follows_higher.append(float(chose_a == high_favors_a))
        gaps.append(float(log_odds[high] - log_odds[low]))

    gaps = np.asarray(gaps, dtype=float)
    follows_higher = np.asarray(follows_higher, dtype=float)
    if gaps.size == 0:
        return float('nan')

    unique_gaps = np.unique(np.round(gaps, 10))
    if unique_gaps.size < 2:
        return float('nan')

    smallest = unique_gaps[0]
    largest = unique_gaps[-1]
    low_mask = np.isclose(gaps, smallest, atol=1e-9, rtol=0.0)
    high_mask = np.isclose(gaps, largest, atol=1e-9, rtol=0.0)

    return float(
        np.mean(follows_higher[high_mask])
        - np.mean(follows_higher[low_mask])
    )
```

**Observed (real) value:** -0.1275 (var=0.0523)
**Previous candidate values (this loop):**
  - iter 1: -0.0737 (var=0.0215) (Δ vs real +0.0538)
  - iter 2: -0.0437 (var=0.0272) (Δ vs real +0.0838)
  - iter 3: -0.0600 (var=0.0342) (Δ vs real +0.0675)
  - iter 4: -0.2137 (var=0.0292) (Δ vs real -0.0862)
  - iter 5 (most recent): -0.1800 (var=0.0296) (Δ vs real -0.0525)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7075 (var=0.0151)
- pi_1: 0.0113 (var=0.0129)
- pi_2: -0.0112 (var=0.0229)
- pi_3: 0.1400 (var=0.0120)
- pi_5: -0.1363 (var=0.0200)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure whether choices favoring the decisive high-validity cue decline
    # as that cue moves toward the end of the display.
    positions = np.array([2, 7, 13, 19, 25, 31], dtype=int)
    # Approximate centered accessibility-cascade profile. The weights sum to
    # zero, so any position-invariant choice reliability contributes zero.
    weights = np.array([0.27, 0.24, 0.16, -0.02, -0.25, -0.40], dtype=float)

    weighted_choices = []
    used_weights = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # In this design, exactly one of these six positions contains the
        # higher-validity target cue that opposes the two early cues.
        target_matches = [j for j in positions if j < diff.size and diff[j] != 0]
        if len(target_matches) != 1:
            continue

        target_pos = target_matches[0]
        k = int(np.where(positions == target_pos)[0][0])
        w = weights[k]

        response = int(row['response'])
        target_choice = 1.0 if ((diff[target_pos] > 0 and response == 0) or
                                (diff[target_pos] < 0 and response == 1)) else 0.0
        weighted_choices.append(w * target_choice)
        used_weights.append(abs(w))

    if len(weighted_choices) == 0:
        return float('nan')

    # Absolute-weight normalization gives an interpretable bounded contrast
    # while preserving the matched-filter ordering of the six conditions.
    denom = float(np.sum(used_weights))
    if denom <= 0.0:
        return float('nan')
    return float(np.sum(weighted_choices) / denom)

```

**Observed (real) value:** -0.0543 (var=0.0193)
**Previous candidate values (this loop):**
  - iter 1: 0.0200 (var=0.0030) (Δ vs real +0.0743)
  - iter 2: 0.0244 (var=0.0031) (Δ vs real +0.0788)
  - iter 3: -0.0090 (var=0.0043) (Δ vs real +0.0453)
  - iter 4: -0.0223 (var=0.0029) (Δ vs real +0.0320)
  - iter 5 (most recent): 0.0169 (var=0.0051) (Δ vs real +0.0712)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0015 (var=0.0020)
- pi_5: 0.2664 (var=0.0022)
- pi_2: -0.0100 (var=0.0017)
- pi_3: 0.0028 (var=0.0006)
- pi_4: 0.2945 (var=0.0059)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure choice of the option favored by the highest-validity cue,
    # separately for sparse versus highly redundant terminal-cue conditions.
    high_redundancy = []
    low_redundancy = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 24 or b.size < 24:
            continue

        diff = a - b
        # Features 19--24 (zero-based indices 18--23) are the nested set
        # of high-validity cues favoring the same option.
        redundancy = int(np.count_nonzero(diff[18:24]))

        # Feature 24 has validity .95 and is TTB's decisive cue on every
        # experimental trial. Code whether the response follows that cue.
        if diff[23] > 0:
            followed_top_cue = float(int(row["response"]) == 0)
        elif diff[23] < 0:
            followed_top_cue = float(int(row["response"]) == 1)
        else:
            continue

        if redundancy in (1, 2):
            low_redundancy.append(followed_top_cue)
        elif redundancy in (5, 6):
            high_redundancy.append(followed_top_cue)

    if len(low_redundancy) == 0 or len(high_redundancy) == 0:
        return float("nan")

    return float(np.mean(high_redundancy) - np.mean(low_redundancy))

```

**Observed (real) value:** -0.2675 (var=0.0382)
**Previous candidate values (this loop):**
  - iter 1: -0.0775 (var=0.0156) (Δ vs real +0.1900)
  - iter 2: -0.3313 (var=0.0543) (Δ vs real -0.0638)
  - iter 3: -0.3263 (var=0.0581) (Δ vs real -0.0588)
  - iter 4: -0.0794 (var=0.0152) (Δ vs real +0.1881)
  - iter 5 (most recent): -0.2506 (var=0.0553) (Δ vs real +0.0169)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3306 (var=0.0175)
- pi_1: 0.0169 (var=0.0052)
- pi_2: 0.5481 (var=0.0232)
- pi_3: -0.0113 (var=0.0033)
- pi_4: 0.0087 (var=0.0075)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, and response reversal.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive Source-Binding Cascade expects shape (2, n_features); got {stim.shape}."
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

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["baseline_binding"])
    binding_uncertainty = float(parameters["binding_uncertainty"])
    uncertainty_center = float(parameters["uncertainty_center"])
    uncertainty_width = float(parameters["uncertainty_width"])
    interference = float(parameters["interference_strength"])
    intermediate_competition = float(parameters["intermediate_competition"])
    crowd_exponent = float(parameters["crowding_exponent"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    neighborhood_width = float(parameters["neighborhood_width"])
    primacy = float(parameters["primacy_strength"])
    recency = float(parameters["recency_strength"])
    execution_flip = float(parameters["execution_flip"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    discriminating = directions != 0.0
    if not np.any(discriminating):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Compress communicated validity onto a bounded subjective diagnosticity
    # scale. Ranking remains based on the communicated values, while distances
    # used in source competition are subjective and compressed.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)

    positions = np.arange(n_features, dtype=np.float64)
    serial_scale = max(0.08 * max(n_features - 1, 1), 1.0)
    primacy_profile = np.exp(-positions / serial_scale)
    recency_profile = np.exp(-(n_features - 1.0 - positions) / serial_scale)

    # Binding uncertainty is concentrated around a subject-specific region of
    # moderately diagnostic labels. This permits the .8177 region to be more
    # confusable than near-equal .55 labels without changing validity rank.
    uncertainty_scale = max(uncertainty_width, 1e-9)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_scale
        )
    )

    # Compute display-contingent competitive crowding. Only cues that actually
    # discriminate on this trial enter the competition. Same-direction cues
    # inhibit one another most when they are nearby and have similar compressed
    # validities. Consequently, redundant cues can jointly suppress retrieval.
    crowding = np.zeros(n_features, dtype=np.float64)
    active = np.flatnonzero(discriminating)
    sim_scale = max(similarity_bandwidth, 1e-9)
    pos_scale = max(neighborhood_width, 1e-9)

    for j in active:
        competitors = active[active != j]
        if competitors.size == 0:
            continue
        same_direction = directions[competitors] == directions[j]
        competitors = competitors[same_direction]
        if competitors.size == 0:
            continue

        position_similarity = np.exp(
            -np.abs(positions[competitors] - positions[j]) / pos_scale
        )
        validity_similarity = np.exp(
            -np.abs(
                subjective_strength[competitors] - subjective_strength[j]
            ) / sim_scale
        )
        crowding[j] = float(np.sum(position_similarity * validity_similarity))

    # Moderately diagnostic sources are especially vulnerable when they must
    # be distinguished from a cluster of similar sources. This selectively
    # strengthens competition among grouped 70% cues while allowing a slightly
    # shallower global density penalty for very high-validity terminal clusters.
    competition_cost = (
        interference
        * np.power(crowding, crowd_exponent)
        * (1.0 + intermediate_competition * uncertainty_profile)
    )
    binding_logits = (
        baseline
        - binding_uncertainty * uncertainty_profile
        - competition_cost
        + primacy * primacy_profile
        + recency * recency_profile
    )

    # Stable logistic transformation to cue-binding probabilities.
    binding_prob = np.empty(n_features, dtype=np.float64)
    nonnegative = binding_logits >= 0.0
    binding_prob[nonnegative] = 1.0 / (
        1.0 + np.exp(-binding_logits[nonnegative])
    )
    exp_logits = np.exp(binding_logits[~nonnegative])
    binding_prob[~nonnegative] = exp_logits / (1.0 + exp_logits)
    binding_prob = np.clip(binding_prob, 1e-9, 1.0 - 1e-9)

    # Validity-guided cascade: the first successfully bound discriminating cue
    # controls the intended response. Stable sorting uses display order only to
    # resolve exact validity ties. We marginalize over residual binding outcomes
    # after the display-level competitive costs have been computed.
    cue_order = np.argsort(-validities, kind="stable")
    reach_probability = 1.0
    intended_a = 0.0
    intended_b = 0.0

    for j in cue_order:
        if not discriminating[j]:
            continue
        stop_probability = reach_probability * binding_prob[j]
        if differences[j] > 0.0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability
        reach_probability *= 1.0 - binding_prob[j]

    # If no discriminating source is successfully bound, the person guesses.
    intended_a += 0.5 * reach_probability
    intended_b += 0.5 * reach_probability
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        intended_a = 0.5
    else:
        intended_a /= total

    # A separate execution process can reverse the intended binary response.
    p_a = (
        intended_a * (1.0 - execution_flip)
        + (1.0 - intended_a) * execution_flip
    )
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
- validity_compression: [0.85, 1.15]
- baseline_binding: [2.4, 3.6]
- binding_uncertainty: [2.8, 4.5]
- uncertainty_center: [0.52, 0.64]
- uncertainty_width: [0.14, 0.22]
- interference_strength: [1.6, 2.6]
- intermediate_competition: [0.8, 1.5]
- crowding_exponent: [2.0, 2.5]
- similarity_bandwidth: [0.16, 0.4]
- neighborhood_width: [1.5, 3.5]
- primacy_strength: [0.0, 0.08]
- recency_strength: [1.4, 2.4]
- execution_flip: [0.08, 0.18]

`rationale`: This is a targeted edit of the accepted cascade rather than a change of theory family. First, the broad quadratic uncertainty profile is replaced by a Gaussian profile with subject-specific center and width. It can selectively impair the .8177 region while leaving near-equal .55/.5525 labels relatively easy to bind, addressing Experiment 6 without merely increasing global uncertainty. Second, uncertainty now amplifies competition only when a cue is actually crowded. This separately penalizes the tightly grouped 70% cues in Experiment 5, allowing the global crowding exponent and interference ranges to be reduced slightly so the Experiment 8 redundancy effect should retreat from its overshoot. Third, the recency profile is made more terminally localized and its coefficient is increased. Isolated late targets should therefore receive a reliable advantage in Experiment 7, while dense terminal clusters remain suppressed by superlinear competition. The wider center, width, and recency ranges introduce stable heterogeneity specifically in the mechanisms whose between-subject variance was underestimated.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed validity-guided competitive-retrieval family and captures the qualitative cross-experiment pattern substantially better than independent accessibility: Experiments 1–3 are classified correctly, Experiment 4 is close (0.1727 versus 0.1620), and the signs in Experiments 6 and 8 are correctly negative. However, its central competitive-interference effect is much too weak. In Experiment 8 the predicted redundancy cost is only -0.0775, versus -0.2675 observed, recovering less than one third of the empirical contrast. Experiment 5 likewise permits too many overrides by the crowded 70% cluster (0.2119 versus 0.1613), and Experiment 6 underestimates the negative validity-gap contrast (-0.0737 versus -0.1275). Experiment 7 also has the wrong sign: the model predicts an early-position advantage of 0.0200, whereas humans show a modest late-position advantage of -0.0543. Finally, between-subject variance is considerably too low in the continuous metrics, especially Experiments 4, 6, 7, and 8, despite the theory's stated emphasis on stable heterogeneity.
Rationale: Keep the Competitive Source-Binding Cascade, but strengthen and sharpen its existing components. Increase the effective same-direction cluster penalty—through a higher interference range, steeper crowding exponent, or a less saturating competition transformation—so that moving from one or two to five or six adjacent similar cues causes a much larger collapse in top-cue retrieval. This should improve the defining Experiment 8 gradient and likely reduce the excessive clustered-majority choice in Experiment 5 while nudging Experiment 4 toward its observed value. Avoid raising baseline binding to compensate, because that would erase the needed redundancy cost. Separately, strengthen recency relative to primacy enough to reverse Experiment 7's small positive positional contrast, while ensuring that stronger cluster interference dominates the recency boost in Experiment 8. The nonmonotonic binding-uncertainty/compression component should also be modestly intensified or reshaped to enlarge Experiment 6's negative gap effect without changing cue rank. Wider subject-level ranges for interference susceptibility, validity compression/uncertainty, and recency are warranted to recover the much larger observed variances. This was an accepted and promising first instantiation, but the arbiter's key diagnostic—strong negative redundancy under dense terminal competition—remains quantitatively underfit.

**Outcome of this advice:** iter 1 candidate loss=0.0733 -> iter 2 candidate loss=0.0576 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed validity-guided competitive-retrieval family and the accepted revision materially improved aggregate loss. It preserves the correct categorical results in Experiments 1–3 and closely fits Experiment 4 (0.1560 versus 0.1620). The stronger superlinear competition also recovered the defining negative redundancy effect in Experiment 8, although it now overshoots it (-0.3313 versus -0.2675) and produces somewhat excessive variance (0.0543 versus 0.0382). Important systematic misses remain. Experiment 5 still predicts too many overrides by the crowded 70% cluster (0.2062 versus 0.1613). Experiment 6 worsened despite the attempted uncertainty adjustment: the negative validity-gap effect is now only -0.0437 versus -0.1275 observed. Experiment 7 retains the wrong sign (0.0244 versus -0.0543), so the current positional implementation still predicts an early rather than late advantage. Between-subject variance also remains much too low in Experiments 4 and 7 and about half the observed level in Experiment 6.
Rationale: Continue tuning the Competitive Source-Binding Cascade, but do not simply repeat the last global increases. The accepted advice successfully established strong dense-cluster interference, yet the current exponent/interference combination moved Experiment 8 from substantial underfit to moderate overshoot. Use an intermediate crowding slope or slightly lower dense-cluster penalty rather than returning to the weak iteration-1 setting. Preserve the low Experiment-5 majority rate through a separate in-family adjustment, such as stronger binding uncertainty for the intermediate, tightly grouped 70% cues, rather than relying entirely on the same density exponent that controls Experiment 8. For Experiment 6, further global intensification of uncertainty is not indicated because the previous change made the contrast less negative. Instead reshape or recenter the nonmonotonic uncertainty profile so that the .8177 cue lies nearer maximal binding uncertainty while near-equal .55/.5525 cues remain relatively well bound; a narrower compression range centered near the value accomplishing this is preferable to another broad increase. For Experiment 7, merely increasing the current recency coefficient also failed. Inspect and revise the serial-position normalization/profile within the same mechanism—for example, use a centered or more terminally localized recency boost—so late isolated targets receive reliably greater binding than early targets without altering validity order. Then widen subject-level variation specifically in this effective positional parameter and in the reshaped uncertainty parameter, rather than indiscriminately widening all ranges. This should address the wrong signs and underestimated variances while retaining the now-successful competitive-retrieval account of Experiment 8.

**Outcome of this advice:** iter 2 candidate loss=0.0576 -> iter 3 candidate loss=0.0380 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed validity-guided competitive-retrieval family, and the latest accepted revision substantially improved aggregate fit. Experiments 1–3 retain the correct pooled categorical outcomes, Experiment 4 is essentially exact (0.1613 versus 0.1620), Experiment 5 improved materially (0.1825 versus 0.1613), and the revised recency profile finally corrected Experiment 7’s sign. Nevertheless, several systematic quantitative misses remain. Experiment 6 still underproduces the negative validity-gap contrast (-0.0600 versus -0.1275), indicating that the selective binding trough around the .8177 cue is not deep or narrow enough. Experiment 7’s late advantage remains too weak (-0.0090 versus -0.0543). Experiment 8 still overstates dense-cluster interference (-0.3263 versus -0.2675) and has excessive between-subject variance (0.0581 versus 0.0382). Conversely, variance remains markedly too low in Experiments 4 and 7 and somewhat low in Experiment 6. Thus, the central qualitative mechanism is successful, but its global cluster penalty, validity-specific uncertainty, and positional kernel are not yet quantitatively balanced across experiments.
Rationale: Continue from this accepted Competitive Source-Binding Cascade and make a smaller, targeted in-family refinement. The previous advice was accepted, so extend rather than reverse its successful separation of global crowding from intermediate-validity vulnerability. First, slightly reduce or narrow subject variation in the global interference/crowding-exponent combination to bring Experiment 8’s mean and variance back from their overshoot. Preserve—and modestly strengthen—the separate uncertainty-amplified competition term around the clustered 70%–72% cues so Experiment 5 can move from 0.1825 toward 0.1613 without again making all dense clusters excessively costly. Second, sharpen the Gaussian binding-uncertainty trough around the compressed strength corresponding to .8177: use a narrower width and/or somewhat greater peak depth, while keeping .55/.5525 strengths outside the trough. This is supported by the accepted improvement from -0.0437 to -0.0600, but a larger selective effect is still needed for Experiment 6. Third, strengthen the effective late-position advantage without returning to a broad serial-position gradient. The current kernel mainly benefits the final position; modestly broaden the terminal recency window so positions near 25 as well as 31 in Experiment 7 receive appreciable support, or normalize position by display length and tune a terminal kernel over the final display fraction. Keep this positional adjustment independent of cue density so crowding can continue to dominate in Experiment 8. Widen heterogeneity specifically in the sharpened uncertainty and effective terminal-recency parameters, while narrowing heterogeneity in global interference; indiscriminate widening would worsen Experiment 8 and the spurious categorical variance in Experiment 1.

**Outcome of this advice:** iter 3 candidate loss=0.0380 -> iter 4 candidate loss=0.0574 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed validity-guided competitive-retrieval family, but this particular refinement is a clear regression and was correctly rejected by the accept gate (loss 0.0574 versus the accepted base’s 0.0380). Experiments 1–4 remain well fit at the pooled level, and Experiment 7 improves modestly from -0.0090 to -0.0223 against -0.0543 observed. However, the changes badly destabilize the three competition/uncertainty diagnostics. Experiment 8 loses most of the defining redundancy effect, moving from -0.3263 in the accepted base to -0.0794 rather than toward -0.2675. Experiment 5 crosses past its target, falling from 0.1825 to 0.1281 versus 0.1613 observed. Experiment 6 likewise crosses from underproducing the negative contrast (-0.0600) to substantially overproducing it (-0.2137) versus -0.1275. This is an overshoot pattern, not evidence against the mechanism: the reduced global competition plus stronger terminal recency made dense terminal redundancy far too benign, while the sharpened validity-specific and intermediate penalties became too strong. Between-subject variance also remains too low in Experiments 4, 6, and 7, while the apparent variance improvement in Experiment 8 merely accompanies loss of its mean effect.
Rationale: Build again from the unchanged iteration-3 accepted base, not from this rejected instantiation. My previous recommendation to simultaneously narrow/deepen the uncertainty trough, strengthen intermediate competition, reduce global interference, and broaden recency was rejected; do not repeat that bundled move. Instead, preserve iteration 3’s global interference, crowding exponent, similarity bandwidth, and neighborhood width, because together they successfully generate the strong negative Experiment-8 gradient. Also retain its intermediate-validity term initially, since the rejected strengthening pushed Experiment 5 past the target. Make conservative, isolated adjustments. For Experiment 6, choose an uncertainty-trough depth/width between iterations 3 and 4—the two versions flank the observed value—and avoid changing its center, width, and amplitude together. For Experiment 7, test a small position-profile change alone: modestly extend the iteration-3 terminal kernel toward position 25 while keeping its peak coefficient near the accepted setting, rather than combining a much stronger recency coefficient with weakened crowding. A density-independent recency contribution should be capped or normalized so it cannot rescue a large same-direction terminal cluster in Experiment 8. Tune one module at a time and retain changes only if Experiment 8 stays near -0.27. Subject heterogeneity should likewise be adjusted locally: modestly widen only the effective recency parameter to address Experiment 7’s low variance, while leaving global interference dispersion near the accepted base. The appropriate next step is therefore a smaller interpolation between the accepted and rejected settings, especially for Experiment 6, rather than another directional increase or decrease across all three modules.

**Outcome of this advice:** iter 4 candidate loss=0.0574 -> iter 5 candidate loss=0.0383 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed competitive source-binding family and retains a strong cross-experiment qualitative fit. Experiments 1–3 have the correct pooled classifications, Experiment 4 is essentially exact (0.1623 versus 0.1620), Experiment 5 is only modestly high (0.1794 versus 0.1613), and Experiment 8 is now close (-0.2506 versus -0.2675). However, the current depth-only adjustment was rejected by the gate, with loss 0.0383 slightly worse than the iteration-3 base at 0.0380. It also moved Experiment 6 past the target: the accepted base underproduced the negative gap effect at -0.0600, but this candidate overproduces it at -0.1800 versus -0.1275. Thus, binding-uncertainty depth is highly sensitive and the proposed interpolation did not yield the intended midpoint. Experiment 7 is again the clearest unresolved failure: its predicted contrast has the wrong sign (0.0169 versus -0.0543), indicating insufficient late-target benefit. Between-subject variance also remains much too low in Experiments 4 and 7, while Experiment 8 variance is somewhat excessive.
Rationale: Return programmatically to the unchanged iteration-3 accepted base and do not repeat the depth increase: the attempted uncertainty-depth interpolation was rejected and overshot Experiment 6. Leave the uncertainty center, width, and depth at the accepted settings for the next test. Instead, make the previously unisolated positional adjustment alone. Modestly broaden the terminal recency kernel so that isolated targets around positions 25 and 31 in Experiment 7 receive more support, rather than increasing only the final-position peak. Keep the accepted global interference, crowding exponent, similarity bandwidth, neighborhood width, and intermediate-validity term fixed. A small broadening may also soften Experiment 8’s accepted-base overshoot by supporting multiple terminal cues, but it must be capped so it does not erase dense-cluster inhibition; validate that Experiment 8 remains near -0.27. This one-module test is preferable to another uncertainty change because Experiments 6 and 8 have already shown unstable responses to bundled or depth-based revisions. If heterogeneity is adjusted, widen only the effective recency-kernel range, which directly targets Experiment 7’s severely underestimated variance, while leaving global competition dispersion unchanged.

**Outcome of this advice:** iter 5 candidate loss=0.0383 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Competitive Source-Binding Cascade. Communicated validities establish an ordinal, usually lexicographic cue hierarchy, but successful use of a cue requires binding its validity label, display identity, and current recommendation into one retrievable representation. Binding fidelity is determined jointly rather than by independent cue accessibility. In particular, nearby discriminating cues with similar compressed validities and the same recommendation form a competitive cluster. Members of such a cluster mutually inhibit one another and can become collectively difficult to identify, so adding redundant support can reduce rather than increase use of the highest-validity cue. When the leading cue is not retrieved, search falls through the validity hierarchy, often allowing isolated, salient opposing cues to control choice. Validities are represented nonlinearly, with moderately diagnostic labels especially vulnerable to source-binding uncertainty. Display position contributes modest primacy and a terminally localized recency boost but no obligatory late-position cost. Thus, pure terminal position can be beneficial while dense terminal redundancy remains harmful because competition overwhelms recency. Stable individual differences govern validity compression, the location and width of maximal binding uncertainty, competitive interference, serial-position biases, and response reversal.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Competitive Source-Binding Cascade expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"Stimulus has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities must have length {n_features}; got shape {validities.shape}.\"\n        )\n\n    compression = float(parameters[\"validity_compression\"])\n    baseline = float(parameters[\"baseline_binding\"])\n    binding_uncertainty = float(parameters[\"binding_uncertainty\"])\n    uncertainty_center = float(parameters[\"uncertainty_center\"])\n    uncertainty_width = float(parameters[\"uncertainty_width\"])\n    interference = float(parameters[\"interference_strength\"])\n    intermediate_competition = float(parameters[\"intermediate_competition\"])\n    crowd_exponent = float(parameters[\"crowding_exponent\"])\n    similarity_bandwidth = float(parameters[\"similarity_bandwidth\"])\n    neighborhood_width = float(parameters[\"neighborhood_width\"])\n    primacy = float(parameters[\"primacy_strength\"])\n    recency = float(parameters[\"recency_strength\"])\n    execution_flip = float(parameters[\"execution_flip\"])\n\n    differences = stim[0] - stim[1]\n    directions = np.sign(differences)\n    discriminating = directions != 0.0\n    if not np.any(discriminating):\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Compress communicated validity onto a bounded subjective diagnosticity\n    # scale. Ranking remains based on the communicated values, while distances\n    # used in source competition are subjective and compressed.\n    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)\n    subjective_strength = np.power(objective_strength, compression)\n\n    positions = np.arange(n_features, dtype=np.float64)\n    serial_scale = max(0.12 * max(n_features - 1, 1), 1.0)\n    primacy_profile = np.exp(-positions / serial_scale)\n    recency_profile = np.exp(-(n_features - 1.0 - positions) / serial_scale)\n\n    # Binding uncertainty is concentrated around a subject-specific region of\n    # moderately diagnostic labels. This permits the .8177 region to be more\n    # confusable than near-equal .55 labels without changing validity rank.\n    uncertainty_scale = max(uncertainty_width, 1e-9)\n    uncertainty_profile = np.exp(\n        -0.5 * np.square(\n            (subjective_strength - uncertainty_center) / uncertainty_scale\n        )\n    )\n\n    # Compute display-contingent competitive crowding. Only cues that actually\n    # discriminate on this trial enter the competition. Same-direction cues\n    # inhibit one another most when they are nearby and have similar compressed\n    # validities. Consequently, redundant cues can jointly suppress retrieval.\n    crowding = np.zeros(n_features, dtype=np.float64)\n    active = np.flatnonzero(discriminating)\n    sim_scale = max(similarity_bandwidth, 1e-9)\n    pos_scale = max(neighborhood_width, 1e-9)\n\n    for j in active:\n        competitors = active[active != j]\n        if competitors.size == 0:\n            continue\n        same_direction = directions[competitors] == directions[j]\n        competitors = competitors[same_direction]\n        if competitors.size == 0:\n            continue\n\n        position_similarity = np.exp(\n            -np.abs(positions[competitors] - positions[j]) / pos_scale\n        )\n        validity_similarity = np.exp(\n            -np.abs(\n                subjective_strength[competitors] - subjective_strength[j]\n            ) / sim_scale\n        )\n        crowding[j] = float(np.sum(position_similarity * validity_similarity))\n\n    # Moderately diagnostic sources are especially vulnerable when they must\n    # be distinguished from a cluster of similar sources. This selectively\n    # strengthens competition among grouped 70% cues while allowing a slightly\n    # shallower global density penalty for very high-validity terminal clusters.\n    competition_cost = (\n        interference\n        * np.power(crowding, crowd_exponent)\n        * (1.0 + intermediate_competition * uncertainty_profile)\n    )\n    binding_logits = (\n        baseline\n        - binding_uncertainty * uncertainty_profile\n        - competition_cost\n        + primacy * primacy_profile\n        + recency * recency_profile\n    )\n\n    # Stable logistic transformation to cue-binding probabilities.\n    binding_prob = np.empty(n_features, dtype=np.float64)\n    nonnegative = binding_logits >= 0.0\n    binding_prob[nonnegative] = 1.0 / (\n        1.0 + np.exp(-binding_logits[nonnegative])\n    )\n    exp_logits = np.exp(binding_logits[~nonnegative])\n    binding_prob[~nonnegative] = exp_logits / (1.0 + exp_logits)\n    binding_prob = np.clip(binding_prob, 1e-9, 1.0 - 1e-9)\n\n    # Validity-guided cascade: the first successfully bound discriminating cue\n    # controls the intended response. Stable sorting uses display order only to\n    # resolve exact validity ties. We marginalize over residual binding outcomes\n    # after the display-level competitive costs have been computed.\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    reach_probability = 1.0\n    intended_a = 0.0\n    intended_b = 0.0\n\n    for j in cue_order:\n        if not discriminating[j]:\n            continue\n        stop_probability = reach_probability * binding_prob[j]\n        if differences[j] > 0.0:\n            intended_a += stop_probability\n        else:\n            intended_b += stop_probability\n        reach_probability *= 1.0 - binding_prob[j]\n\n    # If no discriminating source is successfully bound, the person guesses.\n    intended_a += 0.5 * reach_probability\n    intended_b += 0.5 * reach_probability\n    total = intended_a + intended_b\n    if not np.isfinite(total) or total <= 0.0:\n        intended_a = 0.5\n    else:\n        intended_a /= total\n\n    # A separate execution process can reverse the intended binary response.\n    p_a = (\n        intended_a * (1.0 - execution_flip)\n        + (1.0 - intended_a) * execution_flip\n    )\n    p_a = float(np.clip(p_a, 0.0, 1.0))\n    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs /= probs.sum()\n    return probs",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "validity_compression": "[0.85, 1.15]",
    "baseline_binding": "[2.4, 3.6]",
    "binding_uncertainty": "[2.8, 4.5]",
    "uncertainty_center": "[0.52, 0.64]",
    "uncertainty_width": "[0.14, 0.22]",
    "interference_strength": "[1.6, 2.6]",
    "intermediate_competition": "[0.8, 1.5]",
    "crowding_exponent": "[2.0, 2.5]",
    "similarity_bandwidth": "[0.16, 0.4]",
    "neighborhood_width": "[1.5, 3.5]",
    "primacy_strength": "[0.0, 0.08]",
    "recency_strength": "[1.4, 2.4]",
    "execution_flip": "[0.08, 0.18]"
  },
  "rationale": "This is a one-line positional-kernel edit to the accepted iteration-3 model. The terminal serial scale is increased from 0.08 to 0.12 of display length, so the recency benefit extends beyond only the final cue and reaches isolated targets around position 25 as well as position 31 in Experiment 7. This should strengthen the currently too-small late-target advantage without changing the uncertainty module that proved highly sensitive in Experiment 6. All global crowding, intermediate-validity competition, and subject-level parameter ranges remain exactly at their accepted settings, preserving the strong negative redundancy effect in Experiment 8 and the close fits in Experiments 1–5. Because recency enters independently of density while superlinear crowding remains unchanged, dense terminal clusters should remain inhibited even though isolated late cues become easier to bind."
}
```

## Usage

```json
{
  "input_tokens": 26938,
  "output_tokens": 2860
}
```
