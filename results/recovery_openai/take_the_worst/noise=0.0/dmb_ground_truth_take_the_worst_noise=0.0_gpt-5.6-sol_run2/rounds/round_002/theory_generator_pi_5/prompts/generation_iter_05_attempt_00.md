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
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace pi_3 with a Structure-Learning Coalition Arbitration theory. The new theory should propose that subjects learn how discriminating cues are organized into recurring coalitions across a block, rather than treating representational load as a function only of cue count and current discrimination density. A stable, repeatedly encountered coalition of lower-ranked cues can be encoded as a single relational chunk and compete with the highest-validity cue; coalition recurrence, coherence, and accumulated conflict with the leading cue determine its accessibility. Conversely, when the opposing lower-cue coalition changes composition or fails to form a stable chunk, the advertised validity hierarchy remains accessible even with ten cues. Arbitration should therefore depend jointly on history-sensitive coalition learning, current coalition evidence, advertised validity dominance, and individual thresholds, with recency and hysteresis preventing complete trial-by-trial switching. Choices should arise from competition between a rank-based signal and a learned coalition signal, preferably preserving graded validity and consensus strength rather than reducing both signals immediately to signs. This mechanism can produce incomplete majority capture in the uniformly dense block of Experiment 1, very strong majority responding when a coherent anti-leading-cue coalition is learned in Experiment 2, lexical dominance in Experiment 5 despite nominally high load, and intermediate behavior in Experiments 4 and 6. It differs from pi_3 by rejecting raw load as the causal bottleneck and differs from pi_4 by learning cue-relationship structure rather than using only average block density. The new theory should make identifiable predictions by crossing cue count and density with coalition recurrence, coalition membership stability, block order, and abrupt coalition reversals while holding individual displays fixed.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.493)

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


---

### `pi_3` (overall score: 0.405)

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


---

### `pi_2` (overall score: 0.197)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4095 -> ACCEPTED
- iter 2: loss=0.5483 -> REJECTED
- iter 3: loss=0.5941 -> REJECTED
- iter 4: loss=0.5296 -> REJECTED
- iter 5: loss=0.3795 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.3795 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.3331 (var=0.0030) (Δ vs real +0.3321)
  - iter 2: 0.2404 (var=0.0038) (Δ vs real +0.2393)
  - iter 3: 0.2841 (var=0.0065) (Δ vs real +0.2831)
  - iter 4: 0.0504 (var=0.0013) (Δ vs real +0.0494)
  - iter 5 (most recent): 0.3700 (var=0.0030) (Δ vs real +0.3689)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)
- pi_4: 0.0091 (var=0.0018)

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
  - iter 1: 0.1500 (var=0.0240) (Δ vs real -0.1933)
  - iter 2: -0.4925 (var=0.0187) (Δ vs real -0.8358)
  - iter 3: -0.2842 (var=0.0217) (Δ vs real -0.6275)
  - iter 4: -0.8858 (var=0.0070) (Δ vs real -1.2292)
  - iter 5 (most recent): 0.3292 (var=0.0279) (Δ vs real -0.0142)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)
- pi_4: 0.2442 (var=0.0221)

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
  - iter 1: -1.0000 (var=0.0000) (Δ vs real -2.0000)
  - iter 2: -1.0000 (var=0.0000) (Δ vs real -2.0000)
  - iter 3: -1.0000 (var=0.0000) (Δ vs real -2.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5 (most recent): -1.0000 (var=0.0000) (Δ vs real -2.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)
- pi_4: 1.0000 (var=0.0000)

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
  - iter 1: 0.4550 (var=0.0150) (Δ vs real -0.0650)
  - iter 2: 0.2422 (var=0.0109) (Δ vs real -0.2778)
  - iter 3: 0.2450 (var=0.0148) (Δ vs real -0.2750)
  - iter 4: 0.0400 (var=0.0033) (Δ vs real -0.4800)
  - iter 5 (most recent): 0.4800 (var=0.0091) (Δ vs real -0.0400)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)
- pi_4: 0.5411 (var=0.0106)

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
  - iter 1: 0.4177 (var=0.0033) (Δ vs real -0.1595)
  - iter 2: 0.2166 (var=0.0031) (Δ vs real -0.3607)
  - iter 3: 0.2559 (var=0.0032) (Δ vs real -0.3214)
  - iter 4: 0.0123 (var=0.0002) (Δ vs real -0.5650)
  - iter 5 (most recent): 0.4493 (var=0.0030) (Δ vs real -0.1280)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6725 (var=0.0026)
- pi_3: 0.2905 (var=0.0033)
- pi_1: 0.1370 (var=0.0071)
- pi_2: 0.8314 (var=0.0161)

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
  - iter 1: 0.9131 (var=0.0028) (Δ vs real +0.0831)
  - iter 2: 0.7154 (var=0.0060) (Δ vs real -0.1146)
  - iter 3: 0.4477 (var=0.0059) (Δ vs real -0.3823)
  - iter 4: 0.1360 (var=0.0063) (Δ vs real -0.6940)
  - iter 5 (most recent): 0.7892 (var=0.0034) (Δ vs real -0.0408)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0035)
- pi_4: 0.1183 (var=0.0028)
- pi_1: 0.1504 (var=0.0063)
- pi_2: 0.8581 (var=0.0081)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Relational Coalition Consolidation theory proposes that people retain the advertised validity hierarchy while also learning recurring relationships among lower-ranked cues. On each encounter, the learner represents lower cues relative to the highest-validity discriminating cue and records which cues jointly oppose that leader. Repeated, internally coherent opposition patterns become relational chunks: their accessibility increases with recurrence, pairwise co-opposition, consensus, and repeated conflict with the leading cue. Accessibility changes gradually and exhibits hysteresis, so an established coalition remains available across occasional mismatching trials while an isolated coalition does not immediately displace rank-based processing. Decision evidence is a continuous competition between a graded rank signal, strengthened by advertised validity dominance, and a graded coalition signal reflecting the validity-weighted direction of lower cues. Coalition magnitude is compressed so that learned accessibility, rather than the raw size of the current tally margin, carries more of the influence. Cue count alone therefore does not cause pooling. Many cues can preserve lexical dominance when their coalitional membership rotates, whereas a smaller but recurrent and coherent anti-leader coalition can capture choice. An unusually dominant advertised leader additionally retains a bounded residual influence, preventing complete coalition capture without globally suppressing ordinary coalition learning. The account predicts order-dependent acquisition, persistence after coalition reversals, and different choices for identical displays following stable versus unstable coalition histories.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relational Coalition Consolidation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    learning_rate = float(parameters["learning_rate"])
    relation_prior = float(parameters["relation_prior"])
    recurrence_rate = float(parameters["recurrence_rate"])
    coalition_threshold = float(parameters["coalition_threshold"])
    gate_steepness = float(parameters["gate_steepness"])
    activation_retention = float(parameters["activation_retention"])
    hysteresis_strength = float(parameters["hysteresis_strength"])
    initial_access = float(parameters["initial_access"])
    access_floor = float(parameters["access_floor"])
    access_ceiling = float(parameters["access_ceiling"])
    dominance_gain = float(parameters["dominance_gain"])
    coalition_gain = float(parameters["coalition_gain"])
    consensus_exponent = float(parameters["consensus_exponent"])
    dominance_threshold = float(parameters["dominance_threshold"])
    dominance_steepness = float(parameters["dominance_steepness"])
    rank_safeguard = float(parameters["rank_safeguard"])
    sensitivity = float(parameters["evidence_sensitivity"])
    lapse = float(parameters["lapse"])

    cue_order = np.argsort(-validities, kind="stable")
    rank_position = np.empty(n_features, dtype=int)
    rank_position[cue_order] = np.arange(n_features)

    # The advertised validity reliability above chance supplies a graded,
    # bounded evidence scale. A small floor handles validity exactly 0.5.
    reliability = np.maximum(2.0 * validities - 1.0, 0.02)

    pair_opposition = np.zeros((n_features, n_features), dtype=float)
    pair_opportunity = np.zeros((n_features, n_features), dtype=float)
    prototypes = []
    activation = float(np.clip(initial_access, 0.0, 1.0))

    def describe(display):
        arr = np.asarray(display, dtype=float)
        if arr.ndim != 2 or arr.shape != (2, n_features):
            return None
        directions = np.sign(arr[0] - arr[1])
        leader = None
        for cue in cue_order:
            if directions[cue] != 0.0:
                leader = int(cue)
                break
        if leader is None:
            return {
                "directions": directions,
                "leader": None,
                "leader_sign": 0.0,
                "lower": np.array([], dtype=int),
                "opponents": np.array([], dtype=int),
                "mask": np.zeros(n_features, dtype=bool)
            }

        leader_sign = float(directions[leader])
        lower = np.asarray([
            j for j in range(n_features)
            if rank_position[j] > rank_position[leader] and directions[j] != 0.0
        ], dtype=int)
        opponents = np.asarray([
            j for j in lower if directions[j] == -leader_sign
        ], dtype=int)
        mask = np.zeros(n_features, dtype=bool)
        mask[opponents] = True
        return {
            "directions": directions,
            "leader": leader,
            "leader_sign": leader_sign,
            "lower": lower,
            "opponents": opponents,
            "mask": mask
        }

    def relational_strength(desc):
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]
        if lower.size == 0 or opponents.size < 2:
            return 0.0

        n_opp = float(opponents.size)
        n_support = float(lower.size - opponents.size)
        # Net coherence is zero when lower cues split evenly and one when a
        # unanimous lower coalition opposes the leading cue.
        coherence = max(0.0, (n_opp - n_support) / float(lower.size))

        pair_values = []
        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                numerator = pair_opposition[j, k] + 0.5 * relation_prior
                denominator = pair_opportunity[j, k] + relation_prior
                pair_values.append(numerator / max(denominator, 1e-12))
        pair_consistency = float(np.mean(pair_values)) if pair_values else 0.5

        # Similarity-based recurrence permits a learned coalition to tolerate
        # occasional missing members, rather than requiring exact repetition.
        recurrence_mass = 0.0
        current_size = int(np.sum(mask))
        for old_mask, old_weight in prototypes:
            intersection = int(np.sum(mask & old_mask))
            union = int(np.sum(mask | old_mask))
            similarity = (float(intersection) / float(union)) if union > 0 else 0.0
            recurrence_mass += old_weight * similarity ** 3
        recurrence = 1.0 - np.exp(-recurrence_rate * recurrence_mass)

        return float(coherence * pair_consistency * recurrence)

    def update_memory(desc):
        nonlocal prototypes
        if desc is None or desc["leader"] is None:
            return
        lower = desc["lower"]
        opponents = desc["opponents"]
        mask = desc["mask"]

        # Recency-weight all previously learned relations and prototypes.
        pair_opposition[:] *= (1.0 - learning_rate)
        pair_opportunity[:] *= (1.0 - learning_rate)
        prototypes = [
            (m, w * (1.0 - learning_rate))
            for m, w in prototypes
            if w * (1.0 - learning_rate) > 1e-6
        ]

        for u in range(lower.size):
            for v in range(u + 1, lower.size):
                j = int(lower[u])
                k = int(lower[v])
                pair_opportunity[j, k] += learning_rate
                pair_opportunity[k, j] += learning_rate

        for u in range(opponents.size):
            for v in range(u + 1, opponents.size):
                j = int(opponents[u])
                k = int(opponents[v])
                pair_opposition[j, k] += learning_rate
                pair_opposition[k, j] += learning_rate

        if opponents.size >= 2:
            prototypes.append((mask.copy(), learning_rate))

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Reconstruct the subject's latent coalition memory in actual trial order.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        desc_i = describe(np.vstack([a_i, b_i]))
        structure_i = relational_strength(desc_i)
        drive = (
            gate_steepness * (structure_i - coalition_threshold)
            + hysteresis_strength * (activation - 0.5) * 4.0
        )
        drive = float(np.clip(drive, -60.0, 60.0))
        target = 1.0 / (1.0 + np.exp(-drive))
        activation = (
            activation_retention * activation
            + (1.0 - activation_retention) * target
        )
        activation = float(np.clip(activation, 0.0, 1.0))
        update_memory(desc_i)

    current = describe(stim)
    if current is None or current["leader"] is None:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Viewing the current coalition retrieves matching relational chunks, but
    # activation changes only partially, preserving history and hysteresis.
    current_structure = relational_strength(current)
    current_drive = (
        gate_steepness * (current_structure - coalition_threshold)
        + hysteresis_strength * (activation - 0.5) * 4.0
    )
    current_drive = float(np.clip(current_drive, -60.0, 60.0))
    current_target = 1.0 / (1.0 + np.exp(-current_drive))
    activation = (
        activation_retention * activation
        + (1.0 - activation_retention) * current_target
    )
    activation = float(np.clip(activation, 0.0, 1.0))

    coalition_access = access_floor + (access_ceiling - access_floor) * activation
    coalition_access = float(np.clip(coalition_access, 0.0, 1.0))

    leader = int(current["leader"])
    leader_sign = float(current["leader_sign"])
    lower = current["lower"]

    # Rank evidence remains graded: a leader separated in advertised validity
    # from all currently discriminating lower cues receives a dominance bonus.
    if lower.size > 0:
        strongest_lower = float(np.max(reliability[lower]))
    else:
        strongest_lower = 0.0
    validity_dominance = max(
        0.0,
        (float(reliability[leader]) - strongest_lower)
        / max(float(reliability[leader]), 0.02)
    )
    rank_evidence = leader_sign * (1.0 + dominance_gain * validity_dominance)

    # A retrieved coalition pools lower directions without discarding their
    # advertised validities. A concave transform compresses differences among
    # current tally margins, leaving learned accessibility to carry more of
    # the coalition's magnitude while retaining graded directional evidence.
    if lower.size == 0:
        coalition_evidence = 0.0
    else:
        lower_weights = reliability[lower]
        raw_coalition_evidence = float(
            np.dot(lower_weights, current["directions"][lower])
            / max(float(np.sum(lower_weights)), 1e-12)
        )
        coalition_evidence = float(
            np.sign(raw_coalition_evidence)
            * np.abs(raw_coalition_evidence) ** consensus_exponent
        )

    decision_delta = (
        (1.0 - coalition_access) * rank_evidence
        + coalition_access * coalition_gain * coalition_evidence
    )

    # Only an unusually dominant advertised leader receives residual rank
    # protection. This bounded evidence-side safeguard is negligible under
    # ordinary validity hierarchies and does not suppress coalition learning.
    safeguard_logit = dominance_steepness * (
        validity_dominance - dominance_threshold
    )
    safeguard_logit = float(np.clip(safeguard_logit, -60.0, 60.0))
    safeguard_gate = 1.0 / (1.0 + np.exp(-safeguard_logit))
    safeguard_weight = float(np.clip(rank_safeguard * safeguard_gate, 0.0, 1.0))
    decision_delta = (
        (1.0 - safeguard_weight) * decision_delta
        + safeguard_weight * rank_evidence
    )

    logits = np.array([
        0.5 * sensitivity * decision_delta,
        -0.5 * sensitivity * decision_delta
    ], dtype=float)
    logits = logits - np.max(logits)
    core = np.exp(logits)
    core /= core.sum()

    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
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
- learning_rate: [0.08, 0.14]
- relation_prior: [0.8, 1.6]
- recurrence_rate: [2.5, 4.5]
- coalition_threshold: [0.16, 0.25]
- gate_steepness: [8.0, 13.0]
- activation_retention: [0.78, 0.88]
- hysteresis_strength: [0.45, 0.75]
- initial_access: [0.18, 0.30]
- access_floor: [0.34, 0.44]
- access_ceiling: [0.82, 0.94]
- dominance_gain: [1.0, 2.2]
- coalition_gain: [3.2, 4.8]
- consensus_exponent: [0.40, 0.60]
- dominance_threshold: [0.58, 0.68]
- dominance_steepness: [25.0, 40.0]
- rank_safeguard: [0.78, 0.94]
- evidence_sensitivity: [1.0, 1.8]
- lapse: [0.0, 0.06]

`rationale`: This is a minimal edit to the accepted iteration-1 model. Learning, prototype matching, activation, hysteresis, and the original convex arbitration are unchanged. First, the lower-cue weighted mean receives a concave signed transform. This preserves direction and graded evidence but compresses differences between weak and strong current tally margins, making expressed coalition influence depend relatively more on learned accessibility. It should reduce the spurious raw-tally modulation in Experiment 1 while strengthening weak but consolidated coalition signals in Experiments 2, 4, and 5; unanimous coalitions such as those prominent in Experiment 6 remain unchanged because magnitude one is fixed by the transform. Second, a sharply thresholded, bounded residual-rank safeguard is applied after the original arbitration. It is designed to activate only for an unusually large advertised validity gap, restoring lexical polarity in Experiment 3 without repeating the rejected global dominance suppression or universal additive log-odds rank term. Ordinary validity hierarchies retain the accepted base's strong coalition expression.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed structure-learning coalition family, but its arbitration is not sufficiently experiment-invariant. The largest failure is Experiment 3: humans are categorically aligned with the highest-validity discriminating cue (+1), whereas the model is categorically opposed to it (-1). Experiment 1 also fails strongly: the simulated TTB-agreement slope with tally direction is 0.333 versus an observed value near zero, showing excessive immediate sensitivity to lower-cue consensus. At the same time, coalition influence is too weak in Experiments 2 and 5: majority-oriented behavior is underestimated (0.150 versus 0.343 in Experiment 2; 0.418 versus 0.577 in Experiment 5). Experiment 6 goes in the opposite direction, with excessive majority capture (0.913 versus 0.830). Only Experiment 4 is reasonably close (0.455 versus 0.520). Thus, a global increase or decrease in coalition gain will not solve the pattern.
Rationale: Keep the relational-coalition mechanism, but recalibrate how learned structure and advertised validity dominance control access. The current high access floor (0.34–0.44), large coalition gain, and similarity-tolerant prototype retrieval allow current consensus or broadly overlapping/nested coalitions to influence choice before sufficiently diagnostic coalition learning has occurred. This likely produces the spurious tally slope in Experiment 1 and the complete majority reversal in Experiment 3. Make coalition influence more contingent on genuinely consolidated recurrence: lower the access floor substantially, reduce direct current-display activation, and require stronger membership stability or stricter prototype similarity rather than allowing pairwise overlap and nested subsets to generalize so freely. Crucially, incorporate advertised validity dominance into the accessibility gate, not merely as a modest additive bonus to rank evidence: a large leader–lower validity gap should raise the coalition threshold or suppress effective coalition access. This should restore lexical choice in Experiment 3 and moderate Experiment 6. To avoid worsening Experiments 2 and 5, compensate with a steeper recurrence-dependent transition or higher learned ceiling for stable coalitions under weak-to-moderate validity dominance. In short, separate low baseline/current-consensus influence from strong learned-coalition influence, while making dominance a nonlinear moderator. Also consider reducing hysteresis until activation is established, since the present self-reinforcement can preserve an accidentally activated coalition. The next fit should prioritize correcting Experiment 3 and flattening Experiment 1 without globally weakening the coalition signal needed in Experiments 2 and 5.

**Outcome of this advice:** iter 1 candidate loss=0.4095 -> iter 2 candidate loss=0.5483 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate is not experiment-invariant and was correctly rejected by the accept gate. Its modifications improve Experiment 1 only partially, reducing the excessive tally slope from 0.333 to 0.240, but the result remains far above the observed 0.001. More importantly, they severely damage coalition-sensitive behavior: Experiment 2 changes from modest majority preference (0.150) to strong lexical preference (-0.493 versus observed +0.343), Experiment 5 falls from 0.418 to 0.217 versus 0.577, and Experiment 4 falls from 0.455 to 0.242 versus 0.520. Experiment 6 also crosses from a small overestimate to an underestimate. Despite stronger dominance suppression and stricter matching, Experiment 3 remains categorically wrong (-1 versus +1). Thus, the previous package of lower baseline access, similarity exponent 8, weak current retrieval, and strong dominance suppression overconstrained coalition learning without fixing its most important false generalization.
Rationale: Because the previous recommendation produced a rejected candidate, do not repeat the broad push toward still stricter prototype matching, lower access, or stronger dominance suppression. Build on the unchanged accepted iteration-1 base and target a different in-family defect: latent coalition activation is currently a global license to pool all lower cues. Once activation is high, coalition_evidence averages every currently discriminating lower cue even when those cues do not match the coalition that produced activation. Hysteresis therefore transfers learned access to mismatching or nested displays, which can sustain the incorrect reversal in Experiment 3 and the tally slope in Experiment 1. Make expression of the coalition signal retrieval-specific while retaining latent hysteresis: multiply effective coalition influence by current prototype compatibility/coherence, or compute coalition evidence only over cues belonging to the retrieved prototype. Condition prototypes on the identity or rank of the leading discriminating cue so a coalition learned against one leader is not freely reused against another. Use a moderate similarity kernel near the accepted model rather than the rejected exponent-8 bottleneck, allowing stable coalitions in Experiments 2, 4, and 5 to remain learnable. In short, preserve the accepted model's learning strength and Experiment-6 fit, but separate persistent latent availability from display-specific coalition expression. This localized gating change is more diagnostic than globally weakening coalition access and should suppress inappropriate transfer in Experiments 1 and 3 without collapsing the valid coalition effects elsewhere.

**Outcome of this advice:** iter 2 candidate loss=0.5483 -> iter 3 candidate loss=0.5941 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is not experiment-invariant and was rejected with a worse loss (0.5941) than both the accepted base (0.4095) and the preceding rejected attempt. Leader-specific retrieval did not produce the intended selective correction. Experiment 3 remains categorically reversed (-1 versus observed +1), while Experiment 1 still shows a large spurious tally effect (0.284 versus 0.001) with increased between-subject variance. Meanwhile, valid coalition behavior is substantially suppressed: Experiment 2 is lexical rather than majority-oriented (-0.284 versus +0.343), Experiment 4 is too weakly majority-oriented (0.245 versus 0.520), Experiment 5 is too lexical (0.256 versus 0.577), and Experiment 6 has fallen sharply from the accepted base's 0.913 to 0.448 versus 0.830. Thus, the localized compatibility gate repeated the practical failure of the earlier stricter-retrieval package: it weakened coalition expression broadly without correcting Experiment 3.
Rationale: Build on the unchanged accepted iteration-1 base. Do not repeat my two rejected pushes toward stricter prototype matching, leader-specific retrieval suppression, lower access, or stronger accessibility gating. Instead, revise the continuous arbitration equation within the prescribed coalition-learning family. The present equation uses a convex switch, `(1-access)*rank + access*coalition`, so increasing coalition access mechanically extinguishes rank evidence. Moreover, rank evidence has an experiment-invariant baseline of 1, while coalition evidence is normalized to a bounded weighted mean. This places the two signals on incompatible scales and discards much of the advertised-validity information that should distinguish Experiment 3 from Experiments 2, 4, and 5. Use additive competition on common evidence units: retain rank evidence at full strength and add a learned coalition term with the opposite sign when appropriate. Derive rank strength from the leading cue's reliability or log-odds, and derive coalition strength from the accumulated lower-cue reliability with a controlled normalization or saturation, multiplied by learned accessibility. Let advertised validity dominance scale the rank-versus-coalition evidence ratio rather than suppressing retrieval. This should allow a highly dominant leader to remain decisive in Experiment 3, moderate the excessive majority capture of the accepted base in Experiment 6, and still permit a sufficiently reliable, consolidated coalition to win in Experiments 2, 4, and 5. Keep similarity and learning near the accepted base initially; tune evidence scaling, saturation, and temperature before introducing any new retrieval restriction.

**Outcome of this advice:** iter 3 candidate loss=0.5941 -> iter 4 candidate loss=0.5296 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed coalition-learning family, but it was rejected and is substantially less experiment-invariant than the accepted iteration-1 base. The additive common-unit arbitration successfully fixes Experiment 3, changing the model from categorical majority reversal to the observed lexical polarity (+1), and it nearly removes the spurious Experiment-1 tally slope (0.050 versus 0.001). However, this correction produces overwhelming lexical dominance everywhere else: Experiment 2 is -0.886 rather than +0.343, Experiment 4 is 0.040 rather than 0.520, Experiment 5 is 0.012 rather than 0.577, and Experiment 6 is 0.136 rather than 0.830. The low variances in Experiments 2, 4, and 5 indicate a systematic scaling failure rather than useful individual heterogeneity. Log-odds rank evidence plus a dominance bonus is simply too large for the square-root-normalized coalition term over the proposed coalition-gain range. Thus, the additive residual rank signal solved Experiment 3 by effectively reinstating a near-universal lexical rule, not by achieving selective coalition arbitration.
Rationale: Build again from the unchanged accepted iteration-1 base. My previous recommendation to replace its convex arbitration with fully additive log-odds competition was rejected, so do not repeat that equation or merely increase coalition_gain within it. Preserve the accepted base's stronger coalition expression, which was already reasonably close in Experiments 4 and 6 and much closer in Experiments 2 and 5. Make the Experiment-3 correction narrowly conditional instead: retain the convex rank–coalition arbitration generally, but add a bounded residual-rank floor or cap on coalition capture only when advertised validity dominance crosses a high, sharply nonlinear threshold. Apply this safeguard on the evidence side rather than globally suppressing accessibility, and tune the threshold so it is activated by Experiment 3's unusually dominant leading cue but not by the moderate hierarchies supporting coalition behavior in Experiments 2, 4, 5, and 6. This differs from the previously rejected broad dominance-gating package because it should leave ordinary coalition learning and retrieval unchanged.

For Experiment 1, avoid another hard compatibility gate or stricter prototype kernel, both of which were rejected. Instead, reduce the dependence of expressed coalition magnitude on the current display's raw tally. Let accumulated recurrence/co-opposition determine chunk strength, with the current display supplying primarily coalition direction and a soft match factor; do not average all currently discriminating lower cues as an immediate consensus signal. A soft, nonzero match should preserve learned effects without broadly shutting them down. The target is therefore the accepted model plus two localized changes: a high-dominance residual-rank safeguard for Experiment 3, and less raw current-tally modulation for Experiment 1. Do not adopt the current candidate's log-odds scaling, square-root accumulation, or universal additive rank term, since their joint effect is the severe lexical overcorrection across four experiments.

**Outcome of this advice:** iter 4 candidate loss=0.5296 -> iter 5 candidate loss=0.3795 -> the gate marked it ACCEPTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted candidate is a meaningful aggregate improvement and remains faithful to the prescribed structure-learning coalition family. It now fits Experiment 2 very closely (0.329 versus 0.343), and Experiments 4 and 6 are also reasonably close (0.480 versus 0.520; 0.789 versus 0.830). This confirms that the prior recommendation to preserve convex arbitration and make corrections locally, rather than imposing universal additive rank evidence, was productive. Nevertheless, two qualitative failures preclude an experiment-invariant account. In Experiment 3, humans categorically favor the highest-validity cue, while the candidate remains categorically reversed (-1 versus +1), so the proposed high-dominance safeguard is not actually activating strongly enough. In Experiment 1, the tally modulation is even larger than in the accepted iteration-1 base (0.370 versus an observed 0.001), showing that the concave transform amplifies rather than removes dependence on the current lower-cue tally. Because raising values in (0,1) to an exponent below one increases weak coalition evidence, this implementation strengthens immediate consensus signals even when coalition structure is poorly consolidated. Experiment 5 also remains moderately too lexical/insufficiently majority-oriented (0.449 versus 0.577), while its between-subject variance is higher than observed. Thus, a global reduction in coalition gain would worsen Experiment 5 and the currently good fits in Experiments 2, 4, and 6.
Rationale: Build on this newly accepted candidate, retaining its learning, recurrence, hysteresis, convex arbitration, and generally successful coalition strength. First, calibrate the bounded dominance safeguard so that it robustly crosses its activation threshold in Experiment 3. Inspect the actual validity_dominance values generated in each experiment and place a sharper threshold between Experiment 3 and the ordinary hierarchies, rather than relying on the current broad threshold range of 0.58–0.68. If the present normalized-gap statistic does not separate them, use a more discriminating advertised-validity contrast, such as the leader-to-best-lower reliability ratio or log-odds gap, solely inside the same sharply bounded safeguard. Increase the safeguard floor toward near-complete rank protection only above that threshold; do not restore the rejected universal additive rank term or broad dominance suppression.

Second, revise the current-evidence transform rather than globally weakening coalition access. The accepted concave exponent of 0.40–0.60 boosts every non-unanimous lower-cue margin and therefore explains the deterioration in Experiment 1. Coalition direction should be a soft blend of retrieved prototype-specific evidence and current lower-cue evidence, with current tally magnitude strongly attenuated when retrieval is weak or diffuse. When a coherent prototype is retrieved, preserve the strong coalition signal needed for Experiments 2, 4, and 6; when no particular coalition is retrieved, do not let the sign of the raw lower-cue average act as a generic pooling rule. This should be implemented as graded evidence weighting, not the hard compatibility gates or globally stricter matching previously rejected by the accept gate. A simpler minor alternative is to make the consensus exponent structure-dependent: convex or near-linear under weak recurrence, concave only after strong coalition consolidation. Finally, do not globally increase coalition_gain to fix Experiment 5, because the other majority-sensitive experiments are already close; any additional late majority influence should come from accumulated partial prototype evidence or retention, not raw display tally. The immediate priorities are flipping Experiment 3 without disturbing Experiments 2/4/6 and eliminating Experiment 1's generic tally slope.

**Outcome of this advice:** iter 5 candidate loss=0.3795 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Relational Coalition Consolidation theory proposes that people retain the advertised validity hierarchy while also learning recurring relationships among lower-ranked cues. On each encounter, the learner represents lower cues relative to the highest-validity discriminating cue and records which cues jointly oppose that leader. Repeated, internally coherent opposition patterns become relational chunks: their accessibility increases with recurrence, pairwise co-opposition, consensus, and repeated conflict with the leading cue. Accessibility changes gradually and exhibits hysteresis, so an established coalition remains available across occasional mismatching trials while an isolated coalition does not immediately displace rank-based processing. Decision evidence is a continuous competition between a graded rank signal, strengthened by advertised validity dominance, and a graded coalition signal reflecting the validity-weighted direction of lower cues. Weakly retrieved coalitions attenuate current consensus magnitude, whereas consolidated coalitions compress that magnitude and can behave as accessible relational chunks. Cue count alone therefore does not cause pooling. Many cues can preserve lexical dominance when their coalitional membership rotates, whereas a smaller but recurrent and coherent anti-leader coalition can capture choice. An unusually large advertised log-odds gap between the leader and the best lower cue additionally produces near-complete but bounded rank protection. The account predicts order-dependent acquisition, persistence after coalition reversals, and different choices for identical displays following stable versus unstable coalition histories.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Relational Coalition Consolidation expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"Stimulus has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=float).reshape(-1)\n    if validities.size != n_features:\n        raise ValueError(\"validities must contain one value per feature.\")\n\n    learning_rate = float(parameters[\"learning_rate\"])\n    relation_prior = float(parameters[\"relation_prior\"])\n    recurrence_rate = float(parameters[\"recurrence_rate\"])\n    coalition_threshold = float(parameters[\"coalition_threshold\"])\n    gate_steepness = float(parameters[\"gate_steepness\"])\n    activation_retention = float(parameters[\"activation_retention\"])\n    hysteresis_strength = float(parameters[\"hysteresis_strength\"])\n    initial_access = float(parameters[\"initial_access\"])\n    access_floor = float(parameters[\"access_floor\"])\n    access_ceiling = float(parameters[\"access_ceiling\"])\n    dominance_gain = float(parameters[\"dominance_gain\"])\n    coalition_gain = float(parameters[\"coalition_gain\"])\n    consensus_exponent = float(parameters[\"consensus_exponent\"])\n    dominance_threshold = float(parameters[\"dominance_threshold\"])\n    dominance_steepness = float(parameters[\"dominance_steepness\"])\n    rank_safeguard = float(parameters[\"rank_safeguard\"])\n    sensitivity = float(parameters[\"evidence_sensitivity\"])\n    lapse = float(parameters[\"lapse\"])\n\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    rank_position = np.empty(n_features, dtype=int)\n    rank_position[cue_order] = np.arange(n_features)\n\n    # The advertised validity reliability above chance supplies a graded,\n    # bounded evidence scale. A small floor handles validity exactly 0.5.\n    reliability = np.maximum(2.0 * validities - 1.0, 0.02)\n\n    pair_opposition = np.zeros((n_features, n_features), dtype=float)\n    pair_opportunity = np.zeros((n_features, n_features), dtype=float)\n    prototypes = []\n    activation = float(np.clip(initial_access, 0.0, 1.0))\n\n    def describe(display):\n        arr = np.asarray(display, dtype=float)\n        if arr.ndim != 2 or arr.shape != (2, n_features):\n            return None\n        directions = np.sign(arr[0] - arr[1])\n        leader = None\n        for cue in cue_order:\n            if directions[cue] != 0.0:\n                leader = int(cue)\n                break\n        if leader is None:\n            return {\n                \"directions\": directions,\n                \"leader\": None,\n                \"leader_sign\": 0.0,\n                \"lower\": np.array([], dtype=int),\n                \"opponents\": np.array([], dtype=int),\n                \"mask\": np.zeros(n_features, dtype=bool)\n            }\n\n        leader_sign = float(directions[leader])\n        lower = np.asarray([\n            j for j in range(n_features)\n            if rank_position[j] > rank_position[leader] and directions[j] != 0.0\n        ], dtype=int)\n        opponents = np.asarray([\n            j for j in lower if directions[j] == -leader_sign\n        ], dtype=int)\n        mask = np.zeros(n_features, dtype=bool)\n        mask[opponents] = True\n        return {\n            \"directions\": directions,\n            \"leader\": leader,\n            \"leader_sign\": leader_sign,\n            \"lower\": lower,\n            \"opponents\": opponents,\n            \"mask\": mask\n        }\n\n    def relational_strength(desc):\n        lower = desc[\"lower\"]\n        opponents = desc[\"opponents\"]\n        mask = desc[\"mask\"]\n        if lower.size == 0 or opponents.size < 2:\n            return 0.0\n\n        n_opp = float(opponents.size)\n        n_support = float(lower.size - opponents.size)\n        # Net coherence is zero when lower cues split evenly and one when a\n        # unanimous lower coalition opposes the leading cue.\n        coherence = max(0.0, (n_opp - n_support) / float(lower.size))\n\n        pair_values = []\n        for u in range(opponents.size):\n            for v in range(u + 1, opponents.size):\n                j = int(opponents[u])\n                k = int(opponents[v])\n                numerator = pair_opposition[j, k] + 0.5 * relation_prior\n                denominator = pair_opportunity[j, k] + relation_prior\n                pair_values.append(numerator / max(denominator, 1e-12))\n        pair_consistency = float(np.mean(pair_values)) if pair_values else 0.5\n\n        # Similarity-based recurrence permits a learned coalition to tolerate\n        # occasional missing members, rather than requiring exact repetition.\n        recurrence_mass = 0.0\n        current_size = int(np.sum(mask))\n        for old_mask, old_weight in prototypes:\n            intersection = int(np.sum(mask & old_mask))\n            union = int(np.sum(mask | old_mask))\n            similarity = (float(intersection) / float(union)) if union > 0 else 0.0\n            recurrence_mass += old_weight * similarity ** 3\n        recurrence = 1.0 - np.exp(-recurrence_rate * recurrence_mass)\n\n        return float(coherence * pair_consistency * recurrence)\n\n    def update_memory(desc):\n        nonlocal prototypes\n        if desc is None or desc[\"leader\"] is None:\n            return\n        lower = desc[\"lower\"]\n        opponents = desc[\"opponents\"]\n        mask = desc[\"mask\"]\n\n        # Recency-weight all previously learned relations and prototypes.\n        pair_opposition[:] *= (1.0 - learning_rate)\n        pair_opportunity[:] *= (1.0 - learning_rate)\n        prototypes = [\n            (m, w * (1.0 - learning_rate))\n            for m, w in prototypes\n            if w * (1.0 - learning_rate) > 1e-6\n        ]\n\n        for u in range(lower.size):\n            for v in range(u + 1, lower.size):\n                j = int(lower[u])\n                k = int(lower[v])\n                pair_opportunity[j, k] += learning_rate\n                pair_opportunity[k, j] += learning_rate\n\n        for u in range(opponents.size):\n            for v in range(u + 1, opponents.size):\n                j = int(opponents[u])\n                k = int(opponents[v])\n                pair_opposition[j, k] += learning_rate\n                pair_opposition[k, j] += learning_rate\n\n        if opponents.size >= 2:\n            prototypes.append((mask.copy(), learning_rate))\n\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n\n    # Reconstruct the subject's latent coalition memory in actual trial order.\n    for i in range(n_past):\n        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)\n        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)\n        if a_i.size != n_features or b_i.size != n_features:\n            continue\n        desc_i = describe(np.vstack([a_i, b_i]))\n        structure_i = relational_strength(desc_i)\n        drive = (\n            gate_steepness * (structure_i - coalition_threshold)\n            + hysteresis_strength * (activation - 0.5) * 4.0\n        )\n        drive = float(np.clip(drive, -60.0, 60.0))\n        target = 1.0 / (1.0 + np.exp(-drive))\n        activation = (\n            activation_retention * activation\n            + (1.0 - activation_retention) * target\n        )\n        activation = float(np.clip(activation, 0.0, 1.0))\n        update_memory(desc_i)\n\n    current = describe(stim)\n    if current is None or current[\"leader\"] is None:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Viewing the current coalition retrieves matching relational chunks, but\n    # activation changes only partially, preserving history and hysteresis.\n    current_structure = relational_strength(current)\n    current_drive = (\n        gate_steepness * (current_structure - coalition_threshold)\n        + hysteresis_strength * (activation - 0.5) * 4.0\n    )\n    current_drive = float(np.clip(current_drive, -60.0, 60.0))\n    current_target = 1.0 / (1.0 + np.exp(-current_drive))\n    activation = (\n        activation_retention * activation\n        + (1.0 - activation_retention) * current_target\n    )\n    activation = float(np.clip(activation, 0.0, 1.0))\n\n    coalition_access = access_floor + (access_ceiling - access_floor) * activation\n    coalition_access = float(np.clip(coalition_access, 0.0, 1.0))\n\n    leader = int(current[\"leader\"])\n    leader_sign = float(current[\"leader_sign\"])\n    lower = current[\"lower\"]\n\n    # Rank evidence remains graded: a leader separated in advertised validity\n    # from all currently discriminating lower cues receives a dominance bonus.\n    if lower.size > 0:\n        strongest_lower = float(np.max(reliability[lower]))\n    else:\n        strongest_lower = 0.0\n    validity_dominance = max(\n        0.0,\n        (float(reliability[leader]) - strongest_lower)\n        / max(float(reliability[leader]), 0.02)\n    )\n    rank_evidence = leader_sign * (1.0 + dominance_gain * validity_dominance)\n\n    # Current consensus is amplified only when a coherent relational chunk is\n    # retrieved. With weak or diffuse retrieval, a convex transform attenuates\n    # raw tally magnitude instead of allowing generic pooling. Consolidated\n    # retrieval smoothly restores the original concave chunk expression.\n    if lower.size == 0:\n        coalition_evidence = 0.0\n    else:\n        lower_weights = reliability[lower]\n        raw_coalition_evidence = float(\n            np.dot(lower_weights, current[\"directions\"][lower])\n            / max(float(np.sum(lower_weights)), 1e-12)\n        )\n        retrieval_logit = gate_steepness * (\n            current_structure - coalition_threshold\n        )\n        retrieval_logit = float(np.clip(retrieval_logit, -60.0, 60.0))\n        retrieval_strength = 1.0 / (1.0 + np.exp(-retrieval_logit))\n        effective_exponent = (\n            (1.0 - retrieval_strength) * 1.5\n            + retrieval_strength * consensus_exponent\n        )\n        coalition_evidence = float(\n            np.sign(raw_coalition_evidence)\n            * np.abs(raw_coalition_evidence) ** effective_exponent\n        )\n\n    decision_delta = (\n        (1.0 - coalition_access) * rank_evidence\n        + coalition_access * coalition_gain * coalition_evidence\n    )\n\n    # A sharply exceptional advertised-validity separation receives bounded\n    # rank protection. The log-odds gap better distinguishes a truly dominant\n    # leader from ordinary adjacent ranks than the normalized reliability gap.\n    eps = 1e-6\n    leader_validity = float(np.clip(validities[leader], eps, 1.0 - eps))\n    if lower.size > 0:\n        lower_validity = float(np.clip(np.max(validities[lower]), eps, 1.0 - eps))\n        leader_log_odds = np.log(leader_validity / (1.0 - leader_validity))\n        lower_log_odds = np.log(lower_validity / (1.0 - lower_validity))\n        advertised_gap = max(0.0, float(leader_log_odds - lower_log_odds))\n    else:\n        advertised_gap = float(\"inf\")\n\n    safeguard_logit = dominance_steepness * (\n        advertised_gap - dominance_threshold\n    )\n    safeguard_logit = float(np.clip(safeguard_logit, -60.0, 60.0))\n    safeguard_gate = 1.0 / (1.0 + np.exp(-safeguard_logit))\n    safeguard_weight = float(np.clip(rank_safeguard * safeguard_gate, 0.0, 1.0))\n    decision_delta = (\n        (1.0 - safeguard_weight) * decision_delta\n        + safeguard_weight * rank_evidence\n    )\n\n    logits = np.array([\n        0.5 * sensitivity * decision_delta,\n        -0.5 * sensitivity * decision_delta\n    ], dtype=float)\n    logits = logits - np.max(logits)\n    core = np.exp(logits)\n    core /= core.sum()\n\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "learning_rate": "[0.08, 0.14]",
    "relation_prior": "[0.8, 1.6]",
    "recurrence_rate": "[2.5, 4.5]",
    "coalition_threshold": "[0.16, 0.25]",
    "gate_steepness": "[8.0, 13.0]",
    "activation_retention": "[0.78, 0.88]",
    "hysteresis_strength": "[0.45, 0.75]",
    "initial_access": "[0.18, 0.30]",
    "access_floor": "[0.34, 0.44]",
    "access_ceiling": "[0.82, 0.94]",
    "dominance_gain": "[1.0, 2.2]",
    "coalition_gain": "[3.2, 4.8]",
    "consensus_exponent": "[0.40, 0.60]",
    "dominance_threshold": "[0.55, 0.85]",
    "dominance_steepness": "[18.0, 30.0]",
    "rank_safeguard": "[0.96, 0.995]",
    "evidence_sensitivity": "[1.0, 1.8]",
    "lapse": "[0.0, 0.06]"
  },
  "rationale": "This is a localized edit to the accepted candidate. Learning, prototype formation, recurrence, activation, hysteresis, access, and convex arbitration are unchanged. First, the fixed concave consensus transform is made retrieval-dependent. Weak current relational structure now yields a convex exponent of 1.5, attenuating non-unanimous raw lower-cue margins and targeting Experiment 1's spurious generic tally slope. As current structure crosses the existing coalition threshold, the exponent smoothly approaches the accepted concave range, preserving strong expression of consolidated coalitions in Experiments 2, 4, 5, and 6 without imposing a rejected hard compatibility gate. Second, only the safeguard statistic is changed: it uses the advertised log-odds gap between the leader and strongest discriminating lower cue, with a sharper near-complete bounded rank safeguard. This is intended to activate reliably for Experiment 3's exceptional advertised hierarchy while leaving ordinary coalition learning untouched. No universal additive rank term or global reduction in coalition gain is introduced."
}
```

## Usage

```json
{
  "input_tokens": 22798,
  "output_tokens": 5076
}
```
