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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace pi_5 with a new graded conflict-compression theory. The new theory should posit competition between an advertised-rank route and an equal-direction aggregation route, but route accessibility should depend jointly on (a) slowly learned block density, (b) the current and recent normalized tally margin, and (c) aggregate validity support on each side. Dense blocks should make cue counting or compression chronically accessible, preserving pi_4's success in Experiment 1, while a sufficiently large and salient majority margin should recruit aggregation even in a sparse block, explaining Experiment 8. Small anti-leader margins should normally leave rank-based processing dominant, explaining Experiments 2 and 5. Validity dominance may continuously bias evidence but must not operate as pi_5's near-veto safeguard, because a perfectly valid leader was frequently rejected in Experiment 1. Recurrence of cue identities should have little or no independent effect without outcome feedback; at most, repetition may reduce the processing cost of an already useful tally representation. The model should preserve continuous tally magnitude rather than sign-normalizing it, allow subject-level thresholds and sensitivities to explain heterogeneous intermediate behavior, and include slow contextual persistence without making density the sole control variable. Critical future tests should orthogonally manipulate density, tally margin, aggregate validity imbalance, and coalition recurrence. The new theory should predict strong effects of density and margin, an interaction between them, graded validity effects, and little recurrence-specific acquisition when density, margin, and validity support are matched.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.534)

**Description**
Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.0091 (var=0.0018)
- Experiment 2: real=0.3433 (var=0.0123) vs this=0.2442 (var=0.0221)
- Experiment 3: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.5411 (var=0.0106)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.6725 (var=0.0026)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.1183 (var=0.0028)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.6617 (var=0.0029)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.1520 (var=0.0100)


---

### `pi_3` (overall score: 0.423)

**Description**
Rank-Budget Arbitration theory proposes that people initially represent advertised cue validities as an ordered hierarchy and try to make a lexicographic choice using the highest-ranked discriminating cue. Reliable access to that hierarchy is capacity-limited, however. The probability that rank retrieval fails increases smoothly with the observable representational load imposed by the number of available cues and the number of cues that must be compared on the current trial. When rank retrieval succeeds, choice approximates Take The Best and lower-ranked tally direction has little influence. When retrieval fails, cue ranks are compressed into a coarse common category, and the person pools the directions of all discriminating cues with approximately equal weight. Strategy arbitration occurs anew on every trial rather than assigning a person permanently to one heuristic. Individual capacity determines the transition location, gate steepness determines how abrupt it is, evidence sensitivity controls response reliability, and lapses produce occasional uninformed choices. Thus, lexicographic and tally-like rules are limiting regimes of a single adaptive representational mechanism.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Rank-Budget Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    diff = stim[0] - stim[1]
    directions = np.sign(diff)
    discriminating = np.flatnonzero(directions != 0)

    # Successful rank retrieval: use only the most valid discriminating cue.
    if discriminating.size == 0:
        lexical_delta = 0.0
    else:
        cue_order = np.argsort(-validities, kind="stable")
        lexical_delta = 0.0
        for cue in cue_order:
            if directions[cue] != 0:
                lexical_delta = float(directions[cue])
                break

    # Failed rank retrieval: cue ranks are compressed and directions are pooled.
    # The sign transform prevents larger tallies from receiving mechanically
    # greater response precision than lexicographic evidence.
    pooled_delta = float(np.sign(np.sum(directions)))

    # Besides cue-set size, simultaneous discriminations add comparison load.
    comparison_fraction = float(discriminating.size) / float(max(n_features, 1))
    observable_load = float(n_features) + 0.25 * comparison_fraction

    capacity = float(parameters["capacity"])
    steepness = float(parameters["gate_steepness"])
    gate_logit = steepness * (observable_load - capacity)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    pooling_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    sensitivity = float(parameters["evidence_sensitivity"])

    def binary_softmax(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    p_lexical = binary_softmax(lexical_delta)
    p_pooled = binary_softmax(pooled_delta)
    p_core = ((1.0 - pooling_probability) * p_lexical
              + pooling_probability * p_pooled)

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- capacity: [9.7, 9.9]
- gate_steepness: [2.0, 2.4]
- evidence_sensitivity: [1.0, 5.0]
- lapse: [0.0, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.0019 (var=0.0011)
- Experiment 2: real=0.3433 (var=0.0123) vs this=0.3675 (var=0.0173)
- Experiment 3: real=1.0000 (var=0.0000) vs this=-1.0000 (var=0.0196)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.5706 (var=0.0125)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.2905 (var=0.0033)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.6185 (var=0.0035)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.8793 (var=0.0044)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.1195 (var=0.0100)


---

### `pi_2` (overall score: 0.157)

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
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.3604 (var=0.0118)
- Experiment 2: real=0.3433 (var=0.0123) vs this=0.7633 (var=0.0349)
- Experiment 3: real=1.0000 (var=0.0000) vs this=-1.0000 (var=0.0784)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.7294 (var=0.0327)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.8314 (var=0.0161)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.8581 (var=0.0081)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.8537 (var=0.0083)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.8414 (var=0.0147)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3486 -> ACCEPTED
- iter 2: loss=0.2749 -> ACCEPTED
- iter 3: loss=0.2571 -> ACCEPTED
- iter 4: loss=0.2474 -> ACCEPTED
- iter 5: loss=0.2453 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.2453 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as agreement with Take-The-Best (TTB), then
    # estimate the within-data slope of that agreement on the direction
    # of the equal-weight tally relative to the TTB winner.
    x = []
    y = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b

        # Features are ordered from highest to lowest advertised validity.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_winner = 0 if diff[first] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        tally_margin_for_ttb = (
            a_wins - b_wins if ttb_winner == 0 else b_wins - a_wins
        )

        # -1: tally opposes TTB; 0: tally ties; +1: tally supports TTB.
        x.append(float(np.sign(tally_margin_for_ttb)))
        y.append(float(int(response) == ttb_winner))

    if len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")

    return float(np.dot(xc, y - np.mean(y)) / denom)
```

**Observed (real) value:** 0.0011 (var=0.0014)
**Previous candidate values (this loop):**
  - iter 1: 0.0663 (var=0.0021) (Δ vs real +0.0653)
  - iter 2: 0.2057 (var=0.0024) (Δ vs real +0.2046)
  - iter 3: 0.2898 (var=0.0029) (Δ vs real +0.2887)
  - iter 4: 0.2934 (var=0.0025) (Δ vs real +0.2923)
  - iter 5 (most recent): 0.3159 (var=0.0024) (Δ vs real +0.3148)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)
- pi_4: 0.0091 (var=0.0018)
- pi_5: 0.3640 (var=0.0028)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    contrasts = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        cue_directions = np.sign(a - b)

        # +1 denotes an A winner and -1 a B winner.
        tally_winner = float(np.sign(cue_directions.sum()))

        discriminating = np.flatnonzero(cue_directions != 0)
        if tally_winner == 0 or discriminating.size == 0:
            continue
        ttb_winner = float(cue_directions[discriminating[0]])

        # Retain only trials on which the theories make opposite predictions.
        if tally_winner != ttb_winner:
            chosen_side = 1.0 if int(row["response"]) == 0 else -1.0
            contrasts.append(chosen_side * tally_winner)

    if len(contrasts) == 0:
        return float("nan")
    return float(np.mean(contrasts))
```

**Observed (real) value:** 0.3433 (var=0.0123)
**Previous candidate values (this loop):**
  - iter 1: -0.5783 (var=0.0232) (Δ vs real -0.9217)
  - iter 2: -0.3367 (var=0.0329) (Δ vs real -0.6800)
  - iter 3: -0.1558 (var=0.0419) (Δ vs real -0.4992)
  - iter 4: -0.2217 (var=0.0316) (Δ vs real -0.5650)
  - iter 5 (most recent): -0.3292 (var=0.0238) (Δ vs real -0.6725)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)
- pi_4: 0.2442 (var=0.0221)
- pi_5: 0.3233 (var=0.0234)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Polarity of preference for the highest-validity discriminating cue."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are supplied in descending validity order.
        first = int(discriminating[0])
        lexical_winner = 0 if diff[first] > 0 else 1
        aligned.append(float(int(row["response"]) == lexical_winner))

    if len(aligned) == 0:
        return 0.0

    rate = float(np.mean(aligned))
    if rate > 0.5:
        return 1.0
    if rate < 0.5:
        return -1.0
    return 0.0
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0784) (Δ vs real +0.0000)
  - iter 5 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)
- pi_4: 1.0000 (var=0.0000)
- pi_5: -1.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for a, b, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        cue_directions = np.sign(a_arr - b_arr)
        pooled_direction = float(np.sign(cue_directions.sum()))

        # Tie-tally trials carry no pooled directional prediction.
        if pooled_direction == 0.0:
            continue

        # Positive choice direction denotes choosing A; negative denotes B.
        choice_direction = 1.0 if int(response) == 0 else -1.0
        scores.append(choice_direction * pooled_direction)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.5200 (var=0.0305)
**Previous candidate values (this loop):**
  - iter 1: 0.6700 (var=0.0047) (Δ vs real +0.1500)
  - iter 2: 0.7339 (var=0.0045) (Δ vs real +0.2139)
  - iter 3: 0.6594 (var=0.0060) (Δ vs real +0.1394)
  - iter 4: 0.6172 (var=0.0078) (Δ vs real +0.0972)
  - iter 5 (most recent): 0.5933 (var=0.0080) (Δ vs real +0.0733)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)
- pi_4: 0.5411 (var=0.0106)
- pi_5: 0.4717 (var=0.0088)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    d = data.copy()
    # Trial position is reconstructed separately for each subject because rows
    # are supplied in within-subject trial order.
    d["trial_position"] = d.groupby("subject_id", sort=False).cumcount()

    a_sum = d["option_a_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )
    b_sum = d["option_b_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )

    # All designed trials have an unambiguous cue-majority winner. Determine
    # it from the displayed ratings so the score remains invariant to A/B
    # reversal and does not depend on memorized trial labels.
    majority_response = np.where(b_sum > a_sum, 1, 0)
    d["majority_choice"] = (
        d["response"].to_numpy(dtype=int) == majority_response
    ).astype(float)

    # Omit the first eight observations, during which the advocated model is
    # acquiring the dense-block context. The competitor is stationary, while
    # the advocated model should predominantly use compressed pooling after
    # this burn-in.
    late = d[d["trial_position"] >= 8]
    if len(late) == 0:
        late = d

    # Averaging subject-level rates makes the pooled-data definition agree
    # with the quantity recomputed on each individual subject.
    subject_rates = late.groupby("subject_id", sort=False)["majority_choice"].mean()
    if len(subject_rates) == 0:
        return float("nan")
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.5773 (var=0.0012)
**Previous candidate values (this loop):**
  - iter 1: 0.3025 (var=0.0091) (Δ vs real -0.2748)
  - iter 2: 0.4989 (var=0.0034) (Δ vs real -0.0784)
  - iter 3: 0.5075 (var=0.0029) (Δ vs real -0.0698)
  - iter 4: 0.5250 (var=0.0033) (Δ vs real -0.0523)
  - iter 5 (most recent): 0.5530 (var=0.0021) (Δ vs real -0.0243)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6725 (var=0.0026)
- pi_3: 0.2905 (var=0.0033)
- pi_1: 0.1370 (var=0.0071)
- pi_2: 0.8314 (var=0.0161)
- pi_5: 0.4523 (var=0.0026)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the equal-weight cue-majority winner."""
    majority_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
        b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
        if a.size != b.size or a.size == 0:
            continue

        majority_direction = float(np.sum(np.sign(a - b)))
        if majority_direction == 0.0:
            continue

        majority_response = 0 if majority_direction > 0.0 else 1
        response = int(row["response"])
        if response in (0, 1):
            majority_choices.append(float(response == majority_response))

    if len(majority_choices) == 0:
        return float("nan")
    return float(np.mean(majority_choices))
```

**Observed (real) value:** 0.8300 (var=0.0061)
**Previous candidate values (this loop):**
  - iter 1: 0.1512 (var=0.0019) (Δ vs real -0.6787)
  - iter 2: 0.4292 (var=0.0022) (Δ vs real -0.4008)
  - iter 3: 0.5217 (var=0.0031) (Δ vs real -0.3083)
  - iter 4: 0.5875 (var=0.0054) (Δ vs real -0.2425)
  - iter 5 (most recent): 0.5619 (var=0.0054) (Δ vs real -0.2681)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0035)
- pi_4: 0.1183 (var=0.0028)
- pi_1: 0.1504 (var=0.0063)
- pi_2: 0.8581 (var=0.0081)
- pi_5: 0.8013 (var=0.0034)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    # Omit only the brief initialization period in which Ecological Encoding
    # Hysteresis is still updating its estimate of the block ecology.
    if 'subject_id' in data.columns:
        serial_position = data.groupby('subject_id', sort=False).cumcount().to_numpy()
    else:
        serial_position = np.arange(len(data))
    late = serial_position >= 4

    scores = []
    for keep, a_cell, b_cell, response in zip(
        late,
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        if not keep or pd.isna(response):
            continue
        a = np.asarray(a_cell, dtype=float).reshape(-1)
        b = np.asarray(b_cell, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size:
            continue
        difference = float(np.sum(a) - np.sum(b))
        if difference == 0.0 or not np.isfinite(difference):
            continue
        majority_response = 0 if difference > 0.0 else 1
        scores.append(float(int(response) == majority_response))

    if len(scores) == 0:
        return float('nan')
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.6404 (var=0.0014)
**Previous candidate values (this loop):**
  - iter 1: 0.3011 (var=0.0082) (Δ vs real -0.3393)
  - iter 2: 0.4715 (var=0.0083) (Δ vs real -0.1689)
  - iter 3: 0.5030 (var=0.0033) (Δ vs real -0.1374)
  - iter 4: 0.5163 (var=0.0046) (Δ vs real -0.1241)
  - iter 5 (most recent): 0.5243 (var=0.0028) (Δ vs real -0.1161)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6617 (var=0.0029)
- pi_5: 0.0711 (var=0.0013)
- pi_1: 0.1522 (var=0.0140)
- pi_2: 0.8537 (var=0.0083)
- pi_3: 0.8793 (var=0.0044)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Acquisition-weighted capture by the recurring anti-leader coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        coalition_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
            b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
            if a.size == 0 or b.size == 0 or a[0] == b[0]:
                continue

            # Expert 1 is the highest-validity cue. The coalition winner is
            # the option opposed to the option favored by that expert.
            leader_response = 0 if a[0] > b[0] else 1
            response = int(row["response"])
            coalition_choices.append(float(response != leader_response))

        n = len(coalition_choices)
        if n == 0:
            continue

        choices = np.asarray(coalition_choices, dtype=float)
        early_n = min(8, n)
        late_n = min(32, n)
        early_rate = float(np.mean(choices[:early_n]))
        late_rate = float(np.mean(choices[-late_n:]))

        # Late coalition capture plus a modest acquisition bonus.
        scores.append(late_rate + 0.25 * (late_rate - early_rate))

    if not scores:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1631 (var=0.0139)
**Previous candidate values (this loop):**
  - iter 1: 0.1359 (var=0.0069) (Δ vs real -0.0272)
  - iter 2: 0.1789 (var=0.0072) (Δ vs real +0.0158)
  - iter 3: 0.2114 (var=0.0133) (Δ vs real +0.0483)
  - iter 4: 0.2608 (var=0.0137) (Δ vs real +0.0977)
  - iter 5 (most recent): 0.1713 (var=0.0066) (Δ vs real +0.0081)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9923 (var=0.0035)
- pi_4: 0.1520 (var=0.0100)
- pi_1: 0.1422 (var=0.0186)
- pi_2: 0.8414 (var=0.0147)
- pi_3: 0.1195 (var=0.0100)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Contextual Margin Arbitration theory proposes that people maintain two competing representations of binary-feature comparisons. The advertised-rank route retrieves the most valid discriminating expert and favors its option. The conflict-compression route represents all discriminating experts as a directional tally while retaining the tally's graded normalized magnitude. Accessibility of compression is jointly controlled by a slowly persistent estimate of block density, the current normalized majority margin, a recency-weighted trace of recent margins, and the aggregate advertised-validity support for the current majority. Dense blocks make compression chronically accessible, whereas sparse blocks require a clearly suprathreshold majority to overcome rank processing. Density and margin provide compensatory routes to compression, so their overlap does not produce a superadditive accessibility boost. Aggregate validity shifts route accessibility and evidence continuously, but its opposition is released differently by chronic density and momentary margin: dense context strongly reduces validity opposition, whereas margin-driven release is weaker and retains a nonzero residual. Thus, even a salient majority does not automatically erase a strongly dominant advertised expert. Repetition of particular cue identities has no independent learning effect because choices receive no correctness feedback. Stable individual differences in contextual thresholds, margin thresholds, gating sensitivities, evidence precision, and lapses generate heterogeneous intermediate behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contextual Margin Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Diagnosticity weights express advertised reliability, but clipping keeps
    # even a nominally perfect expert from becoming a logical veto.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.minimum(diagnosticity, 3.0)

    memory_span = float(parameters["context_memory_span"])
    alpha = 1.0 / max(memory_span, 1.0)
    density_trace = float(parameters["density_prior"])
    margin_trace = float(parameters["margin_prior"])

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Density and margin traces are learned from displays alone. Cue identities
    # and previous responses do not enter because there is no outcome feedback.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        d_i = np.sign(a_i - b_i)
        k_i = int(np.count_nonzero(d_i))
        density_i = float(k_i) / float(max(n_features, 1))
        margin_i = (
            abs(float(np.sum(d_i))) / float(k_i) if k_i > 0 else 0.0
        )
        density_trace += alpha * (density_i - density_trace)
        margin_trace += alpha * (margin_i - margin_trace)

    directions = np.sign(stim[0] - stim[1])
    discriminating = np.flatnonzero(directions != 0.0)
    n_discriminating = int(discriminating.size)
    current_density = float(n_discriminating) / float(max(n_features, 1))
    raw_tally = float(np.sum(directions))
    current_margin = (
        abs(raw_tally) / float(n_discriminating)
        if n_discriminating > 0 else 0.0
    )

    # The current display is incorporated weakly into the persistent context.
    context_density = density_trace + alpha * (current_density - density_trace)
    recent_margin = margin_trace + alpha * (current_margin - margin_trace)

    # Advertised-rank evidence comes from the most valid discriminating cue.
    lexical_delta = 0.0
    if n_discriminating > 0:
        cue_order = np.argsort(-validities, kind="stable")
        for cue in cue_order:
            if directions[cue] != 0.0:
                lexical_delta = float(directions[cue])
                break

    # Signed aggregate validity support lies in [-1, 1]. Positive values favor
    # A and negative values favor B. The bounded scale prevents a perfect cue
    # from functioning as a safeguard or veto.
    if n_discriminating > 0:
        active_weights = diagnosticity[discriminating]
        denom = float(np.sum(active_weights))
        if denom > 1e-12:
            validity_direction = float(
                np.dot(active_weights, directions[discriminating]) / denom
            )
        else:
            validity_direction = 0.0
    else:
        validity_direction = 0.0

    tally_sign = float(np.sign(raw_tally))
    validity_for_majority = tally_sign * validity_direction

    # Density remains an asymmetric contextual bonus. A sharper, individually
    # varying soft hinge makes margins below threshold contribute negligibly.
    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )
    margin_offset = current_margin - float(parameters["margin_threshold"])
    recent_offset = recent_margin - float(parameters["margin_threshold"])
    smoothness = float(parameters["margin_smoothness"])
    margin_excess = float(np.logaddexp(0.0, smoothness * margin_offset) / smoothness)
    recent_excess = float(np.logaddexp(0.0, smoothness * recent_offset) / smoothness)

    # Opposing advertised-validity support is released strongly by a genuinely
    # dense context but only weakly by momentary margin salience. A nonzero
    # attenuation floor preserves graded residual validity opposition without
    # allowing it to become a categorical veto.
    positive_validity = max(validity_for_majority, 0.0)
    negative_validity = min(validity_for_majority, 0.0)
    release_floor = float(parameters["validity_release_floor"])
    release_exponent = (
        float(parameters["density_validity_release"]) * density_excess
        + float(parameters["margin_validity_release"]) * margin_excess
    )
    negative_attenuation = release_floor + (1.0 - release_floor) * np.exp(
        -release_exponent
    )
    effective_validity_access = (
        positive_validity + negative_validity * negative_attenuation
    )

    # Density and current margin are compensatory routes to compression. The
    # overlap correction makes their combination approximately max/noisy-OR
    # like instead of granting a superadditive bonus when both are salient.
    density_drive = (
        float(parameters["density_sensitivity"]) * density_excess
    )
    margin_drive = (
        float(parameters["margin_sensitivity"]) * margin_excess
    )
    overlap_correction = (
        float(parameters["density_margin_interaction"])
        * min(density_drive, margin_drive)
    )

    gate_logit = (
        float(parameters["base_compression_access"])
        + density_drive
        + margin_drive
        - overlap_correction
        + float(parameters["recent_margin_sensitivity"]) * recent_excess
        + float(parameters["validity_access_bias"]) * effective_validity_access
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    validity_evidence_gain = float(parameters["validity_evidence_gain"])
    rank_delta = lexical_delta + validity_evidence_gain * validity_direction

    # Carry the same salience-dependent release into compressed evidence. Once
    # a majority representation is active, opposing validity remains a graded
    # residual bias instead of re-entering as a full anti-majority safeguard.
    if tally_sign != 0.0:
        compressed_validity_direction = (
            tally_sign * effective_validity_access
        )
    else:
        compressed_validity_direction = validity_direction

    # Preserve continuous tally magnitude on a feature-count-invariant scale.
    # This avoids treating the same normalized coalition margin as stronger
    # merely because an experiment displays more discriminating experts.
    compressed_delta = (
        tally_sign
        * current_margin
        * float(parameters["tally_evidence_scale"])
        + validity_evidence_gain * compressed_validity_direction
    )

    rank_sensitivity = float(parameters["rank_sensitivity"])
    tally_sensitivity = float(parameters["tally_sensitivity"])

    def binary_probs(delta, sensitivity):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_rank = binary_probs(rank_delta, rank_sensitivity)
    p_compressed = binary_probs(compressed_delta, tally_sensitivity)
    p_core = (
        (1.0 - compression_probability) * p_rank
        + compression_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
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
- n_features: n_features
- validities: validities
- context_memory_span: [14.0, 26.0]
- density_prior: [0.48, 0.58]
- margin_prior: [0.10, 0.22]
- density_threshold: [0.70, 0.73]
- margin_threshold: [0.24, 0.28]
- margin_smoothness: [45.0, 70.0]
- base_compression_access: [-2.8, -2.0]
- density_sensitivity: [30.0, 40.0]
- margin_sensitivity: [30.0, 42.0]
- recent_margin_sensitivity: [0.5, 1.6]
- density_margin_interaction: [0.75, 1.0]
- validity_access_bias: [2.0, 4.0]
- density_validity_release: [10.0, 16.0]
- margin_validity_release: [1.0, 3.0]
- validity_release_floor: [0.20, 0.40]
- validity_evidence_gain: [0.15, 0.65]
- tally_evidence_scale: [3.5, 4.5]
- rank_sensitivity: [1.8, 3.2]
- tally_sensitivity: [0.70, 0.90]
- lapse: [0.0, 0.08]

`rationale`: This is a minimal revision of the accepted candidate. The compensatory overlap correction, normalized graded tally, contextual traces, and response equations are retained. The fixed softplus smoothness is replaced by a sharply bounded subject-level margin_smoothness, while the margin threshold is moved slightly upward; a 0.20 margin therefore contributes almost no accessibility in Experiment 8, but the larger margins in Experiment 6 remain strongly suprathreshold. The density threshold is also raised modestly while above-threshold density sensitivity is increased, suppressing medium-density Experiment 1 contexts but strengthening chronic compression in the genuinely dense blocks of Experiments 2, 5, and 7. Finally, the single validity-release rate is split into strong density-specific release and weaker margin-specific release with a nonzero floor. Dense blocks can consequently overcome opposing advertised validity, whereas a momentarily large margin leaves a graded residual when the advertised leader is highly dominant. This targets lower coalition effects in Experiments 1 and 8 and higher majority choice in Experiments 2 and 5–7 without globally changing tally precision or introducing recurrence learning.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed graded arbitration family and improves substantially over the old pi_5 pattern in Experiments 1, 3, and 8. It correctly preserves lexical dominance in Experiment 3 and predicts little recurrence-specific coalition acquisition in Experiment 8. However, it is not experiment-invariant. It is strongly too rank-dominant in Experiment 2 (-0.578 versus 0.343), Experiment 5 (0.303 versus 0.577), Experiment 6 (0.151 versus 0.830), and Experiment 7 (0.301 versus 0.640). These are directional, not merely quantitative, failures in Experiment 2 and very large underpredictions elsewhere. Conversely, it modestly overpredicts tally alignment in Experiment 4 (0.670 versus 0.520) and produces too much tally-dependent slope in Experiment 1 (0.066 versus 0.001). The central problem is the gate geometry: signed linear density_excess imposes a severe penalty in sparse displays, so even conspicuous normalized margins cannot recruit compression in Experiment 6. Meanwhile, small margins are penalized even in maximally dense blocks, preventing the chronic compression required in Experiments 5 and 7. Thus density and margin currently act more like conjunctive requirements than compensatory routes to accessibility.
Rationale: Keep the advertised-rank versus graded conflict-compression theory, but recalibrate the accessibility gate. Replace the large signed density penalty with an asymmetric density bonus, such as sensitivity times max(context_density - threshold, 0), or otherwise cap the below-threshold penalty. Likewise, avoid strongly penalizing margins below threshold; use a hinge or nonlinear salience term so that dense context alone can sustain compression in Experiments 5 and 7, while a sufficiently large margin can overcome sparse density in Experiment 6. Increase the leverage of above-threshold current margin relative to recent-margin persistence, while retaining little or no cue-identity recurrence effect. Preserve low compression for the moderate-margin, subthreshold-density displays of Experiment 8 through the base gate and graded validity opposition rather than a density veto. Because broader compression would otherwise worsen Experiment 4 and the residual slope in Experiment 1, jointly lower or narrow tally_sensitivity so route access can rise without making every compressed choice nearly deterministic; raw tally magnitude should still be preserved. The next fit should specifically target higher compression probability in dense small-margin trials and sparse large-margin trials, not a global increase in compression. This should raise Experiments 2, 5, 6, and 7 while limiting the existing overprediction in Experiments 1 and 4.

**Outcome of this advice:** iter 1 candidate loss=0.3486 -> iter 2 candidate loss=0.2749 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision improved aggregate loss and moved Experiments 2, 5, 6, and 7 in the intended direction, confirming that the previous signed, conjunctive gate was a problem. It remains faithful to the prescribed graded rank-versus-compression family, and it fits Experiment 3 exactly and Experiment 8 closely. Nevertheless, it is not yet experiment-invariant. Experiment 2 remains directionally wrong (-0.337 versus 0.343), Experiment 6 still severely underpredicts majority choice (0.429 versus 0.830), and Experiment 7 remains low (0.472 versus 0.640). Experiment 5 is now reasonably close but still low (0.499 versus 0.577). At the same time, the revision substantially worsened Experiment 1's tally-dependent slope (0.206 versus 0.001) and Experiment 4's tally alignment (0.734 versus 0.520). Thus another global increase in compression would be counterproductive. The likely remaining problem is that validity opposition is still an additive gate penalty and can function as a de facto veto precisely on anti-leader trials, despite the verbal theory denying a veto. In addition, using unscaled raw tally as compressed evidence makes evidence strength depend strongly on feature count and can double-count margin through both the gate and the choice stage, contributing to excessive tally effects in Experiments 1 and 4.
Rationale: Keep the current mechanism family and build on the accepted positive-part gate, since that previous change was beneficial. The next revision should not simply raise density_sensitivity, margin_sensitivity, or baseline access globally. Instead, make negative validity support lose influence smoothly when either contextual density or majority margin is high: for example, bound the negative validity-access contribution and attenuate it as density_excess or margin salience grows, while retaining its full graded effect for moderate-density, small-margin cases such as Experiment 8. This should release compression in dense Experiments 2 and 7 and in the large-margin trials of Experiment 6 without destroying the good Experiment 8 fit. Replace the hard margin cliff with a smooth hinge or softplus near the threshold so Experiment 2's near-threshold margins are not treated like negligible margins. Finally, normalize compressed tally evidence for the number of discriminating cues, or use a monotone sublinear transform such as signed normalized margin times a calibrated scale; this preserves continuous magnitude while reducing feature-count-driven overconfidence and the excessive effects in Experiments 1 and 4. Retune tally sensitivity upward only as needed after that rescaling. The target is selective accessibility plus calibrated evidence: raise Experiments 2, 6, and 7, preserve Experiment 8, and bring Experiments 1 and 4 back down.

**Outcome of this advice:** iter 2 candidate loss=0.2749 -> iter 3 candidate loss=0.2571 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed graded rank-versus-compression family, and both prior revisions were accepted, so the positive-part accessibility gate, smooth margin term, validity release, and normalized tally representation are productive foundations. The current revision further improves Experiments 2, 5, 6, and 7 while preserving the exact qualitative result in Experiment 3. Experiment 8 is still reasonably close, although it has drifted upward from 0.179 to 0.211 against 0.163. The fit is nevertheless not experiment-invariant. Experiment 2 remains directionally wrong (-0.156 versus 0.343), Experiment 6 remains severely too rank-dominant (0.522 versus 0.830), and Experiments 5 and 7 still underpredict majority choice (0.508 versus 0.577 and 0.503 versus 0.640). Most importantly, Experiment 1 has deteriorated monotonically across accepted iterations: its tally-dependent slope is now 0.290 versus essentially zero in humans. Experiment 4 also remains too tally-aligned (0.659 versus 0.520). Thus the model simultaneously underuses compression in the diagnostic dense-small-margin and sparse-large-margin conditions, yet overexpresses tally evidence when density and margin are jointly salient. The positive density-by-margin product is likely central: it makes the two intended alternative routes to compression superadditive, even though the empirical pattern calls for compensatory, approximately OR-like recruitment. A second issue is that validity opposition is released in the accessibility gate but remains fully present in compressed_delta, allowing advertised validity to continue suppressing anti-leader majorities after compression has ostensibly won arbitration.
Rationale: Build on the accepted changes rather than reversing them, but revise the interaction geometry. Replace the strictly positive density_margin_interaction with a saturating or compensatory combination of density and margin—such as a noisy-OR, capped sum, log-sum-exp/max-like combination, or a small negative overlap correction. Density alone should recruit compression in Experiments 2, 5, and 7, and margin alone should recruit it in Experiment 6, without an extra boost when both are already salient. This should reduce the excessive tally effects in Experiments 1 and 4 while permitting modest upward retuning of the separate density and margin routes. Also propagate the existing salience-dependent validity release into compressed evidence: once compression is strongly accessible, opposing validity should remain a graded residual bias rather than again entering compressed_delta at full validity_evidence_gain. Keep that attenuation weak in moderate-density, small-margin displays to avoid further increasing Experiment 8. Finally, modestly narrow the effective product of tally_evidence_scale and tally_sensitivity if Experiment 1 and Experiment 4 remain high; do not globally increase baseline compression. The target is selective improvement: flip Experiment 2 positive, substantially raise Experiment 6 and moderately raise Experiments 5 and 7, while bringing Experiment 1 sharply toward zero and Experiment 4 toward 0.52 without losing Experiment 3 or the near-fit in Experiment 8.

**Outcome of this advice:** iter 3 candidate loss=0.2571 -> iter 4 candidate loss=0.2474 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current revision was accepted and modestly improves aggregate fit, so the compensatory overlap correction and released compressed-validity term are useful. In particular, Experiment 4 moved toward the data (0.617 versus 0.520), and Experiments 5–7 also improved. The candidate remains faithful to the prescribed rank-versus-graded-compression family and preserves the pooled qualitative result in Experiment 3. It is nevertheless not experiment-invariant. Experiment 1 is the largest structural failure: the predicted tally-dependent TTB-agreement slope is 0.293 versus essentially zero, and this error has increased monotonically from 0.066 across accepted iterations. Experiment 2 is still directionally wrong (-0.222 versus +0.343). Majority choice also remains too low in Experiments 6 (0.588 versus 0.830) and 7 (0.516 versus 0.640), with a smaller shortfall in Experiment 5 (0.525 versus 0.577). Meanwhile Experiment 8 has deteriorated to 0.261 versus 0.163. This pattern indicates insufficient discrimination around the accessibility thresholds: the softplus margin term gives appreciable compression drive even just below threshold, helping the fixed 0.20 anti-leader margin in Experiment 8 and likely contributing to Experiment 1, while compression remains insufficient in genuinely dense blocks and on conspicuously large-margin trials. The current validity release also treats density and margin as interchangeable sources of attenuation, which may release a highly dominant opposing leader too readily in medium-density, high-margin Experiment 1 displays.
Rationale: Keep the accepted compensatory gate geometry rather than revisiting the interaction overhaul; that change lowered loss and improved Experiment 4. The next revision should instead increase contrast between near-threshold and clearly suprathreshold accessibility. Make the margin hinge substantially sharper or parameterize its smoothness, so a margin near 0.20 contributes almost no drive while large margins in Experiment 6 still contribute strongly. Likewise, use a steeper, more selective density transition: suppress medium-density Experiment 1 contexts but strengthen access in the genuinely dense blocks of Experiments 2, 5, and 7. This can be accomplished by modestly raising the density threshold while increasing above-threshold density sensitivity, rather than raising baseline compression. Separate density-driven from margin-driven validity release instead of using their unweighted sum. Dense context may strongly attenuate opposing validity, helping Experiments 2 and 7, whereas margin alone should leave a larger graded residual when aggregate validity opposition is extreme. A nonzero attenuation floor or weaker margin-specific release would reduce the excessive Experiment 1 slope and Experiment 8 coalition capture without restoring a categorical veto. Do not globally lower tally precision yet, because Experiment 6 still needs substantially stronger majority choice. The selective targets are: sharply reduce Experiments 1 and 8, flip Experiment 2 positive, and raise Experiments 5–7 while retaining the improved Experiment 4 fit and lexical polarity in Experiment 3.

**Outcome of this advice:** iter 4 candidate loss=0.2474 -> iter 5 candidate loss=0.2453 -> the gate marked it ACCEPTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed graded rank-versus-compression family, and the latest revision was accepted, but the improvement in aggregate loss is very small. The sharper margin transition successfully repaired Experiment 8 (0.171 versus 0.163), while Experiment 5 is now close (0.553 versus 0.577) and Experiment 4 continues to improve (0.593 versus 0.520). Experiment 3 is also exactly correct. However, the revision worsened several central tests: Experiment 1 rose further to 0.316 versus 0.001, Experiment 2 moved backward to -0.329 versus +0.343, and Experiment 6 fell to 0.562 versus 0.830. Experiment 7 remains low at 0.524 versus 0.640. Thus the model is still strongly rank-dominant in dense Experiments 2 and 7 and on the salient-majority trials of Experiment 6, while paradoxically producing far too much tally-dependent behavior in Experiment 1. The latest threshold sharpening should not simply be repeated: although accepted overall, it mainly fixed Experiment 8 and modestly improved Experiments 4 and 5 while aggravating Experiments 1, 2, and 6. A likely remaining problem is that the model discards useful information about the extremity of validity imbalance. Diagnosticity is hard-capped at 3 and then reduced to a normalized signed average, so a uniquely near-perfect leader is not adequately distinguished from ordinary aggregate opposition. Accessibility is consequently being separated mostly by density and margin thresholds, which cannot discriminate the high-margin trials of Experiments 1 and 6 in the required direction.
Rationale: Keep the compensatory arbitration architecture, normalized graded tally, contextual persistence, and absence of recurrence learning. Try a different in-family calibration rather than further sharpening both thresholds. First, represent aggregate validity imbalance with a smooth graded dominance statistic that preserves extreme advertised-validity differences; replace the hard diagnosticity cap with a smooth saturation or compute a separate bounded side-support imbalance before normalization. Use that statistic to make the residual opposition after margin-driven release increase continuously with the extremity of anti-majority validity support. This must remain a graded bias, not a veto, but it should distinguish Experiment 1's near-perfect opposing leader from less extreme large-margin conflicts such as Experiment 6. Second, partially undo the latest upward density-threshold move—choose an intermediate threshold or a smooth density hinge centered between the iteration-4 and iteration-5 settings. The last move backward on density coincided with the large deterioration in Experiment 2; dense blocks in Experiments 2, 5, and 7 need more reliable chronic compression without raising global baseline access. Third, consider a mildly convex continuous transform of normalized tally magnitude in the compressed evidence equation, with its scale recalibrated: this can strengthen conspicuously large margins in Experiment 6 while leaving the small margins in Experiments 5, 7, and 8 dependent mainly on chronic density. Do not globally increase tally sensitivity or margin accessibility, which would further worsen Experiment 1 and Experiment 4. The next revision should primarily recover Experiment 2, raise Experiments 6 and 7, and sharply reduce Experiment 1 while preserving the newly accurate Experiment 8 and near-fit in Experiment 5.

**Outcome of this advice:** iter 5 candidate loss=0.2453 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Contextual Margin Arbitration theory proposes that people maintain two competing representations of binary-feature comparisons. The advertised-rank route retrieves the most valid discriminating expert and favors its option. The conflict-compression route represents all discriminating experts as a directional tally while retaining the tally's graded normalized magnitude. Accessibility of compression is jointly controlled by a slowly persistent estimate of block density, the current normalized majority margin, a recency-weighted trace of recent margins, and the aggregate advertised-validity support for the current majority. Dense blocks make compression chronically accessible, whereas sparse blocks require a clearly suprathreshold majority to overcome rank processing. Density and margin provide compensatory routes to compression, so their overlap does not produce a superadditive accessibility boost. Aggregate validity shifts route accessibility and evidence continuously. A smoothly saturating diagnosticity representation preserves unusually extreme advertised-validity dominance without turning it into a categorical veto: dense context strongly reduces validity opposition, whereas margin-driven release is weaker and leaves a residual that grows continuously with the extremity of anti-majority support. Compression uses a mildly convex transform of normalized tally magnitude, making conspicuously large majorities more persuasive than small anti-leader coalitions while retaining continuous evidence. Repetition of particular cue identities has no independent learning effect because choices receive no correctness feedback. Stable individual differences in contextual thresholds, margin thresholds, gating sensitivities, evidence precision, and lapses generate heterogeneous intermediate behavior.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Contextual Margin Arbitration expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"Stimulus has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=float).reshape(-1)\n    if validities.size != n_features:\n        raise ValueError(\"validities must contain one value per feature.\")\n\n    # Diagnosticity weights express advertised reliability. Smooth saturation\n    # preserves the special extremity of a near-perfect expert while ensuring\n    # that no finite advertised validity becomes a logical veto.\n    v = np.clip(validities, 0.500001, 0.999999)\n    raw_diagnosticity = np.log(v / (1.0 - v))\n    validity_saturation = float(parameters[\"validity_saturation\"])\n    diagnosticity = validity_saturation * np.tanh(\n        raw_diagnosticity / validity_saturation\n    )\n\n    memory_span = float(parameters[\"context_memory_span\"])\n    alpha = 1.0 / max(memory_span, 1.0)\n    density_trace = float(parameters[\"density_prior\"])\n    margin_trace = float(parameters[\"margin_prior\"])\n\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n\n    # Density and margin traces are learned from displays alone. Cue identities\n    # and previous responses do not enter because there is no outcome feedback.\n    for i in range(n_past):\n        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)\n        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)\n        if a_i.size != n_features or b_i.size != n_features:\n            continue\n        d_i = np.sign(a_i - b_i)\n        k_i = int(np.count_nonzero(d_i))\n        density_i = float(k_i) / float(max(n_features, 1))\n        margin_i = (\n            abs(float(np.sum(d_i))) / float(k_i) if k_i > 0 else 0.0\n        )\n        density_trace += alpha * (density_i - density_trace)\n        margin_trace += alpha * (margin_i - margin_trace)\n\n    directions = np.sign(stim[0] - stim[1])\n    discriminating = np.flatnonzero(directions != 0.0)\n    n_discriminating = int(discriminating.size)\n    current_density = float(n_discriminating) / float(max(n_features, 1))\n    raw_tally = float(np.sum(directions))\n    current_margin = (\n        abs(raw_tally) / float(n_discriminating)\n        if n_discriminating > 0 else 0.0\n    )\n\n    # The current display is incorporated weakly into the persistent context.\n    context_density = density_trace + alpha * (current_density - density_trace)\n    recent_margin = margin_trace + alpha * (current_margin - margin_trace)\n\n    # Advertised-rank evidence comes from the most valid discriminating cue.\n    lexical_delta = 0.0\n    if n_discriminating > 0:\n        cue_order = np.argsort(-validities, kind=\"stable\")\n        for cue in cue_order:\n            if directions[cue] != 0.0:\n                lexical_delta = float(directions[cue])\n                break\n\n    # Signed aggregate validity support lies in [-1, 1]. Positive values favor\n    # A and negative values favor B. Smoothly saturated weights retain graded\n    # information about exceptionally dominant advertised experts.\n    if n_discriminating > 0:\n        active_weights = diagnosticity[discriminating]\n        denom = float(np.sum(active_weights))\n        if denom > 1e-12:\n            validity_direction = float(\n                np.dot(active_weights, directions[discriminating]) / denom\n            )\n        else:\n            validity_direction = 0.0\n    else:\n        validity_direction = 0.0\n\n    tally_sign = float(np.sign(raw_tally))\n    validity_for_majority = tally_sign * validity_direction\n\n    # Density remains an asymmetric contextual bonus. A sharper, individually\n    # varying soft hinge makes margins below threshold contribute negligibly.\n    density_excess = max(\n        context_density - float(parameters[\"density_threshold\"]), 0.0\n    )\n    margin_offset = current_margin - float(parameters[\"margin_threshold\"])\n    recent_offset = recent_margin - float(parameters[\"margin_threshold\"])\n    smoothness = float(parameters[\"margin_smoothness\"])\n    margin_excess = float(np.logaddexp(0.0, smoothness * margin_offset) / smoothness)\n    recent_excess = float(np.logaddexp(0.0, smoothness * recent_offset) / smoothness)\n\n    # Opposing advertised-validity support is released strongly by a genuinely\n    # dense context but only weakly by momentary margin salience. A nonzero\n    # attenuation floor preserves graded residual validity opposition without\n    # allowing it to become a categorical veto.\n    positive_validity = max(validity_for_majority, 0.0)\n    negative_validity = min(validity_for_majority, 0.0)\n    release_floor = float(parameters[\"validity_release_floor\"])\n    release_exponent = (\n        float(parameters[\"density_validity_release\"]) * density_excess\n        + float(parameters[\"margin_validity_release\"]) * margin_excess\n    )\n    negative_attenuation = release_floor + (1.0 - release_floor) * np.exp(\n        -release_exponent\n    )\n    effective_validity_access = (\n        positive_validity + negative_validity * negative_attenuation\n    )\n\n    # Density and current margin are compensatory routes to compression. The\n    # overlap correction makes their combination approximately max/noisy-OR\n    # like instead of granting a superadditive bonus when both are salient.\n    density_drive = (\n        float(parameters[\"density_sensitivity\"]) * density_excess\n    )\n    margin_drive = (\n        float(parameters[\"margin_sensitivity\"]) * margin_excess\n    )\n    overlap_correction = (\n        float(parameters[\"density_margin_interaction\"])\n        * min(density_drive, margin_drive)\n    )\n\n    gate_logit = (\n        float(parameters[\"base_compression_access\"])\n        + density_drive\n        + margin_drive\n        - overlap_correction\n        + float(parameters[\"recent_margin_sensitivity\"]) * recent_excess\n        + float(parameters[\"validity_access_bias\"]) * effective_validity_access\n    )\n    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))\n    compression_probability = 1.0 / (1.0 + np.exp(-gate_logit))\n\n    validity_evidence_gain = float(parameters[\"validity_evidence_gain\"])\n    rank_delta = lexical_delta + validity_evidence_gain * validity_direction\n\n    # Carry the same salience-dependent release into compressed evidence. Once\n    # a majority representation is active, opposing validity remains a graded\n    # residual bias instead of re-entering as a full anti-majority safeguard.\n    if tally_sign != 0.0:\n        compressed_validity_direction = (\n            tally_sign * effective_validity_access\n        )\n    else:\n        compressed_validity_direction = validity_direction\n\n    # Preserve continuous normalized tally magnitude. Mild convexity separates\n    # conspicuously large majorities from small anti-leader margins without\n    # introducing dependence on the experiment's number of active experts.\n    transformed_margin = current_margin ** float(\n        parameters[\"tally_margin_exponent\"]\n    )\n    compressed_delta = (\n        tally_sign\n        * transformed_margin\n        * float(parameters[\"tally_evidence_scale\"])\n        + validity_evidence_gain * compressed_validity_direction\n    )\n\n    rank_sensitivity = float(parameters[\"rank_sensitivity\"])\n    tally_sensitivity = float(parameters[\"tally_sensitivity\"])\n\n    def binary_probs(delta, sensitivity):\n        logits = np.array([\n            0.5 * sensitivity * delta,\n            -0.5 * sensitivity * delta\n        ], dtype=float)\n        logits = logits - np.max(logits)\n        p = np.exp(logits)\n        p /= p.sum()\n        return p\n\n    p_rank = binary_probs(rank_delta, rank_sensitivity)\n    p_compressed = binary_probs(compressed_delta, tally_sensitivity)\n    p_core = (\n        (1.0 - compression_probability) * p_rank\n        + compression_probability * p_compressed\n    )\n\n    lapse = float(parameters[\"lapse\"])\n    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "validity_saturation": "[5.0, 7.0]",
    "context_memory_span": "[14.0, 26.0]",
    "density_prior": "[0.48, 0.58]",
    "margin_prior": "[0.10, 0.22]",
    "density_threshold": "[0.68, 0.71]",
    "margin_threshold": "[0.24, 0.28]",
    "margin_smoothness": "[45.0, 70.0]",
    "base_compression_access": "[-2.8, -2.0]",
    "density_sensitivity": "[30.0, 40.0]",
    "margin_sensitivity": "[30.0, 42.0]",
    "recent_margin_sensitivity": "[0.5, 1.6]",
    "density_margin_interaction": "[0.75, 1.0]",
    "validity_access_bias": "[2.0, 4.0]",
    "density_validity_release": "[10.0, 16.0]",
    "margin_validity_release": "[1.0, 3.0]",
    "validity_release_floor": "[0.20, 0.40]",
    "validity_evidence_gain": "[0.15, 0.65]",
    "tally_margin_exponent": "[1.05, 1.15]",
    "tally_evidence_scale": "[4.8, 5.4]",
    "rank_sensitivity": "[1.8, 3.2]",
    "tally_sensitivity": "[0.70, 0.90]",
    "lapse": "[0.0, 0.08]"
  },
  "rationale": "This is a minimal in-family revision of the accepted candidate. The hard diagnosticity cap at 3 is replaced by smooth tanh saturation with a higher subject-level scale. Consequently, a uniquely near-perfect anti-majority leader produces stronger but still bounded residual opposition than an ordinary validity imbalance. This specifically supplies the missing distinction between Experiment 1 and the less validity-extreme large-margin conflicts in Experiment 6 without introducing pi_5's categorical safeguard. The density threshold is moved only modestly downward to an intermediate range, partially undoing the latest change so dense Experiments 2, 5, and 7 can recover chronic compression while leaving the baseline gate unchanged. Finally, normalized tally magnitude receives a very mild convex transform with a recalibrated scale. This preserves small-margin evidence at approximately its previous level but selectively strengthens conspicuous majorities in Experiment 6. All contextual traces, compensatory gate geometry, validity release, and the absence of cue-identity learning are otherwise unchanged, protecting the improved fits in Experiments 4, 5, and 8."
}
```

## Usage

```json
{
  "input_tokens": 26418,
  "output_tokens": 4377
}
```
